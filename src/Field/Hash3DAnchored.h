//
// Created by ppwang on 2022/7/17.
//

#ifndef SANR_HASH3DANCHORED_H
#define SANR_HASH3DANCHORED_H

#pragma once
#include <torch/torch.h>

#include "TCNNWP.h"
#include "../Utils/GlobalDataPool.h"
#include "Field.h"

#define N_CHANNELS 2
#define N_LEVELS 16
// 1024
#define RES_FINE_POW_2 10.f
// 8
#define RES_BASE_POW_2 3.f

class Hash3DAnchored : public Field  {
/*
Hash table.
  - feat_pool_: the large hash table, a.k.a. pool.
  - prim_pool_: according to the f2-nerf paper, each leaf node has its own hash functions. The prime pool here stores the multipliers of the hash functions (equation 2).
  - bias_pool_: according to the f2-nerf paper, each leaf node has its own hash functions. The bias pool here stores the addends of the hash functions (equation 2).
  - feat_local_idx_: the starting indexes of each level in the hash table.
  - feat_local_size_: the length of each level in the hash table.
*/
  using Tensor = torch::Tensor;
public:
  Hash3DAnchored(GlobalDataPool* global_data_pool_);

  Tensor AnchoredQuery(const Tensor& points,           // [ n_points, 3 ]
                       const Tensor& anchors           // [ n_points, 3 ]
               ) override;

  int LoadStates(const std::vector<Tensor>& states, int idx) override;
  std::vector<Tensor> States() override;
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;
  void Reset() override;

  int pool_size_;
  int mlp_hidden_dim_, mlp_out_dim_, n_hidden_layers_;

  Tensor feat_pool_;   // [ pool_size_, n_channels_ ];
  Tensor prim_pool_;   // [ n_levels, 3 ];
  Tensor bias_pool_;   // [ n_levels * n_volumes, 3 ];
  Tensor feat_local_idx_;  // [ n_levels, ];
  Tensor feat_local_size_; // [ n_levels, ];

  std::unique_ptr<TCNNWP> mlp_;

  int n_volumes_;

  Tensor query_points_, query_volume_idx_;
};

class Hash3DAnchoredInfo : public torch::CustomClassHolder {
public:
  Hash3DAnchored* hash3d_ = nullptr;
};

namespace torch::autograd {

class Hash3DAnchoredFunction : public Function<Hash3DAnchoredFunction> {
/*
Function<T> is a template class that represents a PyTorch function. It is used to define custom autograd functions. The T is a template parameter that specifies the type of the function.
*/
public:
  static variable_list forward(AutogradContext *ctx,
                               Tensor feat_pool_,
                               IValue hash3d_info);

  static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

}

#endif //SANR_HASH3DANCHORED_H
