// Copyright 2026 The ODML Authors.
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

std::vector<float> Compute2dSincosPosEmbed(int embed_dim, int grid_h,
                                           int grid_w) {
  const int half = embed_dim / 2;  // dims for the column half / row half
  const int q = half / 2;          // number of frequencies per 1d block
  // omega depends only on the frequency index i, not on (h, w); precompute it
  // once instead of recomputing pow() for every grid cell (~2.7x faster).
  std::vector<float> omega(q);
  for (int i = 0; i < q; ++i) {
    omega[i] = 1.0f / std::pow(10000.0f,
                               static_cast<float>(i) / static_cast<float>(q));
  }
  std::vector<float> out(static_cast<size_t>(grid_h) * grid_w * embed_dim);
  for (int h = 0; h < grid_h; ++h) {
    for (int w = 0; w < grid_w; ++w) {
      float* row = out.data() +
                   (static_cast<size_t>(h) * grid_w + w) * embed_dim;
      // First half encodes the column index (w); second half the row (h).
      const float fw = static_cast<float>(w), fh = static_cast<float>(h);
      for (int i = 0; i < q; ++i) {
        const float vw = fw * omega[i];
        row[i] = std::sin(vw);
        row[i + q] = std::cos(vw);
        const float vh = fh * omega[i];
        row[half + i] = std::sin(vh);
        row[half + i + q] = std::cos(vh);
      }
    }
  }
  return out;
}

}  // namespace litert::lm
