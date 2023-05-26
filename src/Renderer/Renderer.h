//
// Created by ppwang on 2022/5/7.
//

#ifndef SANR_RENDERER_H
#define SANR_RENDERER_H

#pragma once
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "../Utils/Pipe.h"
#include "../Utils/GlobalDataPool.h"
#include "../Field/FieldFactory.h"
#include "../Shader/ShaderFactory.h"
#include "../PtsSampler/PtsSamplerFactory.h"

/*
  RenderResult is a struct that defines the output of the rendering process. It consists of various Torch tensors such as colors, first_oct_dis, disparity, edge_feats, depth, weights, and idx_start_end. Each tensor probably represents a different component of the rendering output.
*/
struct RenderResult {
  using Tensor = torch::Tensor;
  Tensor colors;
  Tensor first_oct_dis;
  Tensor disparity;
  Tensor edge_feats;
  Tensor depth;
  Tensor weights;
  Tensor idx_start_end;
};

/*
  VolumeRenderInfo is a class that extends torch::CustomClassHolder, a base class provided by PyTorch for user-defined classes. This class contains a pointer to a SampleResultFlex (in PtsSampler.h) object.
*/
class VolumeRenderInfo : public torch::CustomClassHolder {
public:
  SampleResultFlex* sample_result;
};

/*
Renderer is a class that extends Pipe. It has various member functions and fields:

Render is a member function that takes in tensors representing the origins of rays (rays_o), directions of rays (rays_d), bounds, and emb_idx, and returns a RenderResult.

global_data_pool_, pts_sampler_, scene_field_, and shader_ are pointers to different objects. The unique_ptr is a smart pointer that retains sole ownership of an object through a pointer and destroys that object when the unique_ptr goes out of scope.

use_app_emb_ is a boolean variable which might indicate whether to use a certain type of embedding, and app_emb_ is a tensor that could store the embedding.

bg_color_type_ is an enumeration that indicates the type of background color. It could be white, black, or random noise.
*/
class Renderer : public Pipe {
  using Tensor = torch::Tensor;

  enum BGColorType { white, black, rand_noise };
public:
  Renderer(GlobalDataPool* global_data_pool, int n_images);
  RenderResult Render(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds, const Tensor& emb_idx);

  int LoadStates(const std::vector<Tensor>& states, int idx) override;
  std::vector<Tensor> States() override ;
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;

  GlobalDataPool* global_data_pool_;
  std::unique_ptr<PtsSampler> pts_sampler_;
  std::unique_ptr<Field> scene_field_;
  std::unique_ptr<Shader> shader_;

  bool use_app_emb_;
  Tensor app_emb_;

  BGColorType bg_color_type_ = BGColorType::rand_noise;

  SampleResultFlex sample_result_;
};

torch::Tensor FilterIdxBounds(const torch::Tensor& idx_bounds,
                              const torch::Tensor& mask);


#endif //SANR_RENDERER_H
