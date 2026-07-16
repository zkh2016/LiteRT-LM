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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_POS_EMBED_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_POS_EMBED_H_

#include <cstdint>
#include <vector>

namespace litert::lm {

// Computes MiniCPM-V's 2D sin-cos position embedding used by the resampler.
//
// Mirrors resampler.get_2d_sincos_pos_embed(embed_dim, image_size) in the
// original MiniCPM-V modeling code:
//   grid = meshgrid(grid_w, grid_h)  # w first
//   emb_h = get_1d(embed_dim/2, grid[0])  # encodes the column (w) index
//   emb_w = get_1d(embed_dim/2, grid[1])  # encodes the row (h) index
//   out   = concat([emb_h, emb_w], -1)    # [H, W, embed_dim]
// where get_1d(D, pos) = concat([sin(pos*omega), cos(pos*omega)]) with
//   omega[i] = 1 / 10000^(i / (D/2)),  i in [0, D/2).
//
// Returns a row-major buffer of shape [grid_h*grid_w, embed_dim]; the caller
// reshapes/broadcasts to the resampler input layout [L, 1, embed_dim].
// Verified bit-exact against the reference (max abs diff ~4.5e-6).
std::vector<float> Compute2dSincosPosEmbed(int embed_dim, int grid_h,
                                           int grid_w);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_POS_EMBED_H_
