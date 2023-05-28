//
// Created by ppwang on 2022/7/17.
//

#include "Hash3DAnchored.h"
#include <torch/torch.h>
#include "../Common.h"
#include "../Utils/StopWatch.h"
#include <iostream>

using Tensor = torch::Tensor;
#define __FILE_NAME__ "src/Field/Hash3DAnchored.cpp "

TORCH_LIBRARY(dec_hash3d_anchored, m)
{
  std::cout << "register Hash3DAnchoredInfo" << std::endl;
  m.class_<Hash3DAnchoredInfo>("Hash3DAnchoredInfo").def(torch::init());
}


Hash3DAnchored::Hash3DAnchored(GlobalDataPool* global_data_pool) {
  ScopeWatch dataset_watch("Hash3DAnchored::Hash3DAnchored");
  global_data_pool_ = global_data_pool;
  global_data_pool_->scene_field_ = reinterpret_cast<void*>(this);

  const auto& config = global_data_pool->config_["field"];

  pool_size_ = (1 << config["log2_table_size"].as<int>()) * N_LEVELS;
  std::cout << __FILE_NAME__ << ":" << __LINE__ << " pool_size_ = " << pool_size_ << std::endl; // 8388608

  mlp_hidden_dim_ = config["mlp_hidden_dim"].as<int>();
  std::cout << __FILE_NAME__ << ":" << __LINE__ << " mlp_hidden_dim_ = " << mlp_hidden_dim_ << std::endl; // 64
  mlp_out_dim_ = config["mlp_out_dim"].as<int>();
  std::cout << __FILE_NAME__ << ":" << __LINE__ << " mlp_out_dim_ = " << mlp_out_dim_ << std::endl; // 16
  n_hidden_layers_ = config["n_hidden_layers"].as<int>();
  std::cout << __FILE_NAME__ << ":" << __LINE__ << " n_hidden_layers_ = " << n_hidden_layers_ << std::endl; // 1

  // Feat pool
  feat_pool_ = (torch::rand({pool_size_, N_CHANNELS}, CUDAFloat) * .2f - 1.f) * 1e-4f;
  feat_pool_.requires_grad_(true);
  CHECK(feat_pool_.is_contiguous());

  n_volumes_ = global_data_pool->n_volumes_;  // 1547
  // Get prime numbers
  auto is_prim = [](int x) {
    for (int i = 2; i * i <= x; i++) {
      if (x % i == 0) return false;
    }
    return true;
  };

  std::vector<int> prim_selected;
  int min_local_prim = 1 << 28;
  int max_local_prim = 1 << 30;

  // find 3 * N_LEVELS(16) * n_volumes_ prime numbers. (ref: equation 5 from the paper).
  for (int i = 0; i < 3 * N_LEVELS * n_volumes_; i++) {
    int val;
    do {
      val = torch::randint(min_local_prim, max_local_prim, {1}, CPUInt).item<int>();
    }
    while (!is_prim(val));
    prim_selected.push_back(val);
  }

  CHECK_EQ(prim_selected.size(), 3 * N_LEVELS * n_volumes_);

  prim_pool_ = torch::from_blob(prim_selected.data(), 3 * N_LEVELS * n_volumes_, CPUInt).to(torch::kCUDA);
  prim_pool_ = prim_pool_.reshape({N_LEVELS, n_volumes_, 3}).contiguous();

  std::cout << __FILE_NAME__ << ":" << __LINE__ << " prim_pool_.shape = " << prim_pool_.sizes() << std::endl;
  // [16, 1547, 3] N_LEVELS = 16, n_volumes_ = 1547

  if (config["rand_bias"].as<bool>()) {
    std::cout << __FILE_NAME__ << ":" << __LINE__ << " use rand_bias" << std::endl;  // yes
    bias_pool_ = (torch::rand({ N_LEVELS * n_volumes_, 3 }, CUDAFloat) * 1000.f + 100.f).contiguous();
    std::cout << __FILE_NAME__ << ":" << __LINE__ << " bias_pool_.shape = " << bias_pool_.sizes() << std::endl; // [24752, 3]
  }
  else {
    std::cout << __FILE_NAME__ << ":" << __LINE__ << " do not use rand_bias" << std::endl;
    bias_pool_ = torch::zeros({ N_LEVELS * n_volumes_, 3 }, CUDAFloat).contiguous();
  }

  // Size of each level & each volume.
  {
    int local_size = pool_size_ / N_LEVELS;   // 8388608 / 16 = 524288 == 2 ^ 19
    local_size = (local_size >> 4) << 4;      // 524288
    feat_local_size_  = torch::full({ N_LEVELS }, local_size, CUDAInt).contiguous(); // torch.ones([N_LEVELS]) * local_size
    feat_local_idx_ = torch::cumsum(feat_local_size_, 0) - local_size;
    feat_local_idx_ = feat_local_idx_.to(torch::kInt32).contiguous();
    // [0, local_size, 2 * local_size, 3 * local_size, ...]
  }

  // MLP
  mlp_ = std::make_unique<TCNNWP>(global_data_pool_, N_LEVELS * N_CHANNELS, mlp_out_dim_, mlp_hidden_dim_, n_hidden_layers_);
}

Tensor Hash3DAnchored::AnchoredQuery(const Tensor& points, const Tensor& anchors) {
#ifdef PROFILE
  ScopeWatch watch(__func__);
#endif

  auto info = torch::make_intrusive<Hash3DAnchoredInfo>();

  query_points_ = ((points + 1.f) * .5f).contiguous();   // [-1, 1] -> [0, 1]
  query_volume_idx_ = anchors.contiguous();
  std::cout << __FILE_NAME__ << ":" << __LINE__ << " points.shape = " << points.sizes() << std::endl;   // [21638, 3]
  std::cout << __FILE_NAME__ << ":" << __LINE__ << " anchors.shape = " << anchors.sizes() << std::endl; //  [21638]
  
  info->hash3d_ = this;

  // torch::IValue is used to wrap an info object that contains a pointer to a Hash3DAnchored instance. This object is passed as an argument to the torch::autograd::Hash3DAnchoredFunction::apply function, which returns a list of IValues.
  Tensor feat = torch::autograd::Hash3DAnchoredFunction::apply(feat_pool_, torch::IValue(info))[0];  // [n_points, n_levels * n_channels];

  Tensor output = mlp_->Query(feat);
  output = output;
  return output;
}

int Hash3DAnchored::LoadStates(const std::vector<Tensor> &states, int idx) {
  feat_pool_.data().copy_(states[idx++]);
  prim_pool_ = states[idx++].clone().to(torch::kCUDA).contiguous();   // The size may changed.
  bias_pool_.data().copy_(states[idx++]);
  n_volumes_ = states[idx++].item<int>();

  mlp_->params_.data().copy_(states[idx++]);

  return idx;
}

std::vector<Tensor> Hash3DAnchored::States() {
  std::vector<Tensor> ret;
  ret.push_back(feat_pool_.data());
  ret.push_back(prim_pool_.data());
  ret.push_back(bias_pool_.data());
  ret.push_back(torch::full({1}, n_volumes_, CPUInt));

  ret.push_back(mlp_->params_.data());

  return ret;
}

std::vector<torch::optim::OptimizerParamGroup> Hash3DAnchored::OptimParamGroups() {
  // construct optimizers for the feat_pool_ and the mlp
  std::vector<torch::optim::OptimizerParamGroup> ret;


  float lr = global_data_pool_->learning_rate_;
  {
    auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
    opt->betas() = {0.9, 0.99};
    opt->eps() = 1e-15;

    std::vector<Tensor> params = { feat_pool_ };
    ret.emplace_back(std::move(params), std::move(opt));
  }

  {
    auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
    opt->betas() = {0.9, 0.99};
    opt->eps() = 1e-15;
    opt->weight_decay() = 1e-6;

    std::vector<Tensor> params;
    params.push_back(mlp_->params_);
    ret.emplace_back(std::move(params), std::move(opt));
  }

  return ret;
}

void Hash3DAnchored::Reset() {
  feat_pool_.data().uniform_(-1e-2f, 1e-2f);
  mlp_->InitParams();
}