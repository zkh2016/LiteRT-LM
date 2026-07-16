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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_IMAGE_PREPROCESS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_IMAGE_PREPROCESS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {

// MiniCPM-V image preprocessing (baseline: fixed 980x980, no slicing).
//
// Pipeline (matches the HF MiniCPMVImageProcessor for a single un-sliced
// image, and is bit-faithful to the reference PillowResize + normalize):
//   decode(stb) -> PillowResize BICUBIC to 980x980 -> (x/255 - 0.5)/0.5
//                -> HWC to CHW  => float [1, 3, 980, 980]
//
// norm_mean / norm_std default to 0.5 (from preprocessor_config.json), which
// maps [0,255] to [-1, 1]. image_size defaults to 980 (= 70 patches * 14).
struct MinicpmvPreprocessConfig {
  int image_size = 980;
  float norm_mean[3] = {0.5f, 0.5f, 0.5f};
  float norm_std[3] = {0.5f, 0.5f, 0.5f};
};

// Decodes raw image bytes (PNG/JPEG/... via stb_image) and returns a
// CHW float buffer of shape [1, 3, image_size, image_size].
absl::StatusOr<std::vector<float>> PreprocessImageFixed(
    const std::string& image_bytes, const MinicpmvPreprocessConfig& config = {});

// Same but reads the image from a file path.
absl::StatusOr<std::vector<float>> PreprocessImageFileFixed(
    const std::string& image_path, const MinicpmvPreprocessConfig& config = {});

// ---- Multi-slice (official) preprocessing ----
//
// MiniCPM-V official slicing: an image is split into a thumbnail (source
// image, resized to ~scale_resolution^2) plus grid_x*grid_y sub-images. Each
// slice is resized (edge divisible by patch_size), normalized, and packed via
// reshape_by_patch into a [3, patch, num_patches*patch] strip. The navit ViT
// consumes these strips with per-slice position_ids and a padding mask.
struct MinicpmvSliceConfig {
  int scale_resolution = 448;
  int patch_size = 14;
  int max_slice_nums = 9;
  int num_patches_per_side = 70;  // 980/14; navit position-embedding grid side
  int model_dim = 2560;           // resampler pos_embed dim
  // Path to the raw f32 [70,70,model_dim] resampler.pos_embed table dumped from
  // the model. Per-slice pos_embed = table[:tgt_h,:tgt_w]. Required.
  std::string pos_embed_table_path;
  float norm_mean[3] = {0.5f, 0.5f, 0.5f};
  float norm_std[3] = {0.5f, 0.5f, 0.5f};
};

// One preprocessed slice: reshape_by_patch strip + geometry.
struct MinicpmvSlice {
  std::vector<float> strip;     // [3, patch_size, num_patches*patch_size], CHW-patch
  int tgt_h = 0;                // patch rows  (H/patch)
  int tgt_w = 0;                // patch cols  (W/patch)
  int num_patches = 0;          // tgt_h * tgt_w
  std::vector<int64_t> position_ids;  // [num_patches], bucketized to the 70x70 grid
  std::vector<float> pos_embed;       // [num_patches, model_dim] sub-grid of the 70x70 table
};

// Result of slicing one image.
struct MinicpmvSliced {
  std::vector<MinicpmvSlice> slices;  // [thumbnail, sub-images... row-major]
  int grid_x = 0;                     // 0 if no slicing (thumbnail only)
  int grid_y = 0;
};

// Slices raw image bytes into the official thumbnail + sub-images and returns
// each slice's reshape_by_patch strip + position_ids. Bit-faithful to the HF
// MiniCPMVImageProcessor slicing (PIL bicubic, integer grid math).
absl::StatusOr<MinicpmvSliced> PreprocessImageSliced(
    const std::string& image_bytes, const MinicpmvSliceConfig& config = {});


}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_MINICPMV_IMAGE_PREPROCESS_H_
