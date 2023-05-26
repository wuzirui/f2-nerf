//
// Created by ppwang on 2022/6/20.
//

#pragma once
#include <memory>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>
#include "../Utils/Pipe.h"
#include "../Utils/GlobalDataPool.h"
#include "../Common.h"

/*
SampleResultFlex: This struct represents the result of a sampling operation. It includes several torch tensors:

pts and dirs represent 3D points and directions respectively, each for n_all_pts points.
t and dt represent ray-depth and difference of ray-depth
anchors could represent reference points for each point in pts.
pts_idx_bounds might be used to store the near and far distance
first_oct_dis is likely related to some form of distance metric, but it's unclear without additional context.
*/
struct SampleResultFlex {
  using Tensor = torch::Tensor;
  Tensor pts;                           // [ n_all_pts, 3 ]
  Tensor dirs;                          // [ n_all_pts, 3 ]
  Tensor dt;                            // [ n_all_pts, 1 ]
  Tensor t;                             // [ n_all_pts, 1 ]
  Tensor anchors;                       // [ n_all_pts, 3 ]
  Tensor pts_idx_bounds;                // [ n_rays, 2 ] // start, end
  Tensor first_oct_dis;                 // [ n_rays, 1 ]
};

class PtsSampler : public Pipe {
  using Tensor = torch::Tensor;
public:
  PtsSampler() = default;
  virtual SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
  }
  virtual SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
  }
  virtual std::tuple<Tensor, Tensor> GetEdgeSamples(int n_pts) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor() };
  }

  /*
  updates the octree structure based on the sampled points, weights, and alpha values.
  */
  virtual void UpdateOctNodes(const SampleResultFlex& sample_result,
                              const Tensor& sampled_weights,
                              const Tensor& sampled_alpha) {
    CHECK(false) << "Not implemented";
  }

  GlobalDataPool* global_data_pool_ = nullptr;
};