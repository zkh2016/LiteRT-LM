// Copyright 2025 The LiteRT Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/preprocessor/minicpmv_pos_embed.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace litert::lm {
namespace {

// get_1d_sincos_pos_embed_from_grid(half_dim, pos) for a scalar position value.
// Writes half_dim values: [sin(pos*omega_0..), cos(pos*omega_0..)].
// omega[i] = 1 / 10000^(i / (half_dim/2)).
inline void Fill1dSincos(int half_dim, float pos, float* out) {
  const int q = half_dim / 2;  // number of frequencies
  for (int i = 0; i < q; ++i) {
    const float omega = 1.0f / std::pow(10000.0f, static_cast<float>(i) /
                                                       static_cast<float>(q));
    const float v = pos * omega;
    out[i] = std::sin(v);
    out[i + q] = std::cos(v);
  }
}

}  // namespace

std::vector<float> Compute2dSincosPosEmbed(int embed_dim, int grid_h,
                                           int grid_w) {
  const int half = embed_dim / 2;  // dims for the column half / row half
  std::vector<float> out(static_cast<size_t>(grid_h) * grid_w * embed_dim);
  for (int h = 0; h < grid_h; ++h) {
    for (int w = 0; w < grid_w; ++w) {
      float* row = out.data() +
                   (static_cast<size_t>(h) * grid_w + w) * embed_dim;
      // First half encodes the column index (w); second half the row (h).
      Fill1dSincos(half, static_cast<float>(w), row);
      Fill1dSincos(half, static_cast<float>(h), row + half);
    }
  }
  return out;
}

}  // namespace litert::lm
