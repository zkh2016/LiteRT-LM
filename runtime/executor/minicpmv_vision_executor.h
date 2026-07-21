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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MINICPMV_VISION_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MINICPMV_VISION_EXECUTOR_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"          // from @com_google_absl
#include "absl/strings/string_view.h"      // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"     // from @litert
#include "litert/cc/litert_model.h"           // from @litert
#include "litert/cc/litert_tensor_buffer.h"   // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/executor/vision_executor_settings.h"

namespace litert::lm {

// Vision executor for MiniCPM-V-4: chains a SigLIP encoder and a MiniCPM
// Resampler, both loaded as LiteRT CompiledModels.
//
// Unlike the generic VisionLiteRtCompiledModelExecutor (which feeds ONLY the
// encoder output to the adapter), the MiniCPM resampler requires a SECOND
// input, a 2D sin-cos position embedding. This executor computes that pos
// embedding internally and writes it into the resampler's second input buffer.
//
// Baseline: single image, fixed 980x980 (no slicing). The encoder produces
// [1, 4900, 1152]; the resampler produces [1, 64, model_dim] soft tokens.
class MinicpmvVisionExecutor : public VisionExecutor {
 public:
  static absl::StatusOr<std::unique_ptr<MinicpmvVisionExecutor>> Create(
      const VisionExecutorSettings& vision_executor_settings, Environment& env);

  // Encodes a preprocessed image tensor [1, 3, 980, 980] into vision soft
  // tokens [1, 64, model_dim].
  absl::StatusOr<ExecutorVisionData> Encode(
      const litert::TensorBuffer& input_image_tensor) override;

  // Not used by MiniCPM (no ViT-patchify positions input); returns error.
  absl::StatusOr<ExecutorVisionData> Encode(
      const absl::flat_hash_map<std::string, litert::TensorBuffer>&
          input_maps) override;

  absl::StatusOr<std::vector<int>> GetExpectedInputDimension() const override;

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override;

 private:
  MinicpmvVisionExecutor(Environment& env,
                         std::unique_ptr<ModelResources> resources,
                         CompiledModel encoder, CompiledModel resampler)
      : env_(env),
        resources_(std::move(resources)),
        encoder_(std::move(encoder)),
        resampler_(std::move(resampler)) {}

  Environment& env_;
  // Owns the mmap'd model buffers that back encoder_/resampler_ weights.
  // MUST be declared before the CompiledModels so it is destroyed AFTER them.
  std::unique_ptr<ModelResources> resources_;
  CompiledModel encoder_;    // SigLIP: [1,3,980,980] -> [1,num_patches,vision_dim]
  CompiledModel resampler_;  // [1,num_patches,vision_dim] + pos_embed -> [1,num_query,model_dim]

  // Cached 2D sin-cos position embedding, laid out as the resampler's second
  // input expects: [num_patches, 1, model_dim] (row-major grid, batch=1).
  std::vector<float> pos_embed_;  // size num_patches * model_dim

  // Persistent I/O buffers, created once at Create() and reused per Encode()
  // (matches the stock vision executor; creating fresh buffers each call
  // desyncs the already-prepared XNNPACK subgraph and crashes).
  std::vector<litert::TensorBuffer> encoder_in_;
  std::vector<litert::TensorBuffer> encoder_out_;
  std::vector<litert::TensorBuffer> resampler_in_;
  std::vector<litert::TensorBuffer> resampler_out_;
  // pos_embed written once into resampler_in_ (constant across images).
  bool resampler_pos_written_ = false;
  int resampler_feat_idx_ = 0;
  int resampler_pos_idx_ = 1;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MINICPMV_VISION_EXECUTOR_H_
