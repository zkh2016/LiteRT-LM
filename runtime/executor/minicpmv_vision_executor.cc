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

#include "runtime/executor/minicpmv_vision_executor.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"             // from @com_google_absl
#include "absl/status/status.h"            // from @com_google_absl
#include "absl/status/statusor.h"          // from @com_google_absl
#include "absl/strings/str_cat.h"          // from @com_google_absl
#include "absl/types/span.h"               // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"       // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"     // from @litert
#include "litert/cc/litert_model.h"           // from @litert
#include "litert/cc/litert_common.h"          // from @litert
#include "litert/cc/litert_options.h"         // from @litert
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"   // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/preprocessor/minicpmv_pos_embed.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {
absl::Status ApplyCpuOptions(
    const VisionExecutorSettings& settings, absl::string_view component_name,
    ::litert::Options& options) {
  LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
  cpu_options.SetNumThreads(4);
  auto default_xnn = TfLiteXNNPackDelegateOptionsDefault();
  cpu_options.SetXNNPackFlags(default_xnn.flags);
  // Give each sub-model (encoder / resampler) its OWN named XNNPACK weight
  // cache, mirroring the stock VisionLiteRtCompiledModelExecutor. Without this,
  // compiling in the same litert::Environment as the LLM lets the encoder's
  // XNNPACK delegate collide with the LLM's weight cache -> out-of-bounds read
  // (crash drifts across xnn_f32_vadd / vapproxgelu). SetCpuCacheOptions is a
  // no-op when the cache dir is ":nocache" / disabled.
  auto weight_cache_file = settings.GetWeightCacheFile(
      absl::StrCat(component_name, ExecutorSettingsBase::kXnnpackCacheSuffix),
      /*check_and_clean=*/true);
  RETURN_IF_ERROR(SetCpuCacheOptions(weight_cache_file, component_name,
                                     cpu_options));
  options.SetHardwareAccelerators(::litert::HwAccelerators::kCpu);
  return absl::OkStatus();
}
}  // namespace

using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::Options;

namespace {
// MiniCPM-V-4 fixed-980 baseline constants.
constexpr int kImageSize = 980;
constexpr int kPatchSize = 14;
constexpr int kGrid = kImageSize / kPatchSize;   // 70
constexpr int kNumPatches = kGrid * kGrid;       // 4900
constexpr int kVisionDim = 1152;
constexpr int kModelDim = 2560;
constexpr int kNumQuery = 64;
// Multi-slice (navit) constants.
constexpr int kMaxL = 1216;        // fixed padded patch count per slice (navit)
constexpr int kResamplerL = 4900;  // original resampler's fixed patch length
}  // namespace

absl::StatusOr<std::unique_ptr<MinicpmvVisionExecutor>>
MinicpmvVisionExecutor::Create(
    const VisionExecutorSettings& vision_executor_settings, Environment& env) {
  LITERT_ASSIGN_OR_RETURN(auto resources,
                          BuildLiteRtCompiledModelResources(
                              vision_executor_settings.GetModelAssets()));

  ASSIGN_OR_RETURN(
      auto encoder_buffer,
      resources->GetTFLiteModelBuffer(ModelType::kTfLiteVisionEncoder));
  ASSIGN_OR_RETURN(
      auto adapter_buffer,
      resources->GetTFLiteModelBuffer(ModelType::kTfLiteVisionAdapter));

  LITERT_ASSIGN_OR_RETURN(auto enc_options, Options::Create());
  LITERT_RETURN_IF_ERROR(ApplyCpuOptions(
      vision_executor_settings, VisionExecutorSettings::kEncoderName,
      enc_options));
  LITERT_ASSIGN_OR_RETURN(
      auto encoder,
      CompiledModel::Create(
          env,
          ::litert::BufferRef<uint8_t>(encoder_buffer.data(),
                                       encoder_buffer.size()),
          enc_options));
  LITERT_ASSIGN_OR_RETURN(auto res_options, Options::Create());
  LITERT_RETURN_IF_ERROR(ApplyCpuOptions(
      vision_executor_settings, VisionExecutorSettings::kAdapterName,
      res_options));
  LITERT_ASSIGN_OR_RETURN(
      auto resampler,
      CompiledModel::Create(
          env,
          ::litert::BufferRef<uint8_t>(adapter_buffer.data(),
                                       adapter_buffer.size()),
          res_options));

  auto executor = absl::WrapUnique(new MinicpmvVisionExecutor(
      env, std::move(resources), std::move(encoder), std::move(resampler),
      kNumPatches, kVisionDim, kModelDim, kNumQuery));

  // Precompute the 2D sin-cos position embedding for the fixed 70x70 grid.
  // Compute2dSincosPosEmbed returns [grid*grid, model_dim] row-major, which is
  // exactly the resampler's [num_patches, 1, model_dim] input flattened.
  executor->pos_embed_ = Compute2dSincosPosEmbed(kModelDim, kGrid, kGrid);

  // Create persistent I/O buffers once (reused per Encode); creating fresh
  // buffers each call desyncs the prepared XNNPACK subgraph and crashes.
  LITERT_ASSIGN_OR_RETURN(executor->encoder_in_,
                          executor->encoder_.CreateInputBuffers(0));
  LITERT_ASSIGN_OR_RETURN(executor->encoder_out_,
                          executor->encoder_.CreateOutputBuffers(0));
  LITERT_ASSIGN_OR_RETURN(executor->resampler_in_,
                          executor->resampler_.CreateInputBuffers(0));
  LITERT_ASSIGN_OR_RETURN(executor->resampler_out_,
                          executor->resampler_.CreateOutputBuffers(0));
  if (executor->resampler_in_.size() != 2) {
    return absl::InternalError(
        absl::StrCat("MiniCPM-V resampler expected 2 inputs but got ",
                     executor->resampler_in_.size()));
  }
  // Decide which resampler input is the feature vs the pos embedding.
  LITERT_ASSIGN_OR_RETURN(auto in0_type,
                          executor->resampler_in_[0].TensorType());
  const auto& in0_dims = in0_type.Layout().Dimensions();
  const int in0_last = in0_dims.empty() ? 0 : in0_dims[in0_dims.size() - 1];
  if (in0_last == kModelDim) {
    executor->resampler_feat_idx_ = 1;
    executor->resampler_pos_idx_ = 0;
  }
  // pos_embed is constant across images: write it once.
  LITERT_RETURN_IF_ERROR(
      executor->resampler_in_[executor->resampler_pos_idx_].Write<float>(
          absl::MakeConstSpan(executor->pos_embed_.data(),
                              executor->pos_embed_.size())));
  return executor;
}

absl::StatusOr<ExecutorVisionData> MinicpmvVisionExecutor::Encode(
    const litert::TensorBuffer& input_image_tensor) {
  // 1. Run the SigLIP encoder into feat [1,4900,1152], reusing the
  // persistent input buffer created in Create().
  LITERT_ASSIGN_OR_RETURN(auto image_span,
                          ReferTensorBufferAsSpan<float>(input_image_tensor));
  LITERT_ASSIGN_OR_RETURN(auto enc_in_size, encoder_in_[0].Size());
  LITERT_RETURN_IF_ERROR(encoder_in_[0].Write<float>(image_span));
  LITERT_RETURN_IF_ERROR(encoder_.Run(encoder_in_, encoder_out_));

  // 2. Feed the encoder feature into the resampler (pos_embed already written
  // once in Create) and run it. All buffers are persistent.
  LITERT_ASSIGN_OR_RETURN(auto feat_span,
                          ReferTensorBufferAsSpan<float>(encoder_out_[0]));
  LITERT_RETURN_IF_ERROR(
      resampler_in_[resampler_feat_idx_].Write<float>(feat_span));
  LITERT_RETURN_IF_ERROR(resampler_.Run(resampler_in_, resampler_out_));

  // Copy the resampler output into a fresh buffer to return (persistent
  // resampler_out_ is reused on the next Encode call).
  LITERT_ASSIGN_OR_RETURN(auto out_span,
                          ReferTensorBufferAsSpan<float>(resampler_out_[0]));
  LITERT_ASSIGN_OR_RETURN(auto ret_bufs, resampler_.CreateOutputBuffers(0));
  LITERT_RETURN_IF_ERROR(ret_bufs[0].Write<float>(out_span));
  return ExecutorVisionData(std::move(ret_bufs[0]),
                            /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<ExecutorVisionData> MinicpmvVisionExecutor::Encode(
    const absl::flat_hash_map<std::string, litert::TensorBuffer>& input_maps) {
  // Multi-slice navit path. The data processor packs all slices:
  //   "strips"       [N, 3, 14, kMaxL*14]  (each slice reshape_by_patch, padded)
  //   "position_ids" [N, kMaxL]            (int32; padded with 0)
  //   "num_patches"  [N]                   (int32; valid patch count per slice)
  // Each slice: navit SigLIP -> [kMaxL,1152] -> take valid -> pad to
  // kResamplerL -> original resampler -> [64,2560]. Concatenate to
  // [1, N*64, 2560].
  auto it_strips = input_maps.find("strips");
  auto it_pos = input_maps.find("position_ids");
  auto it_np = input_maps.find("num_patches");
  auto it_pe = input_maps.find("pos_embed");
  if (it_strips == input_maps.end() || it_pos == input_maps.end() ||
      it_np == input_maps.end() || it_pe == input_maps.end()) {
    return absl::InvalidArgumentError(
        "MiniCPM-V map Encode needs strips/position_ids/num_patches/pos_embed.");
  }
  LITERT_ASSIGN_OR_RETURN(auto strips_type, it_strips->second.TensorType());
  const auto& sdims = strips_type.Layout().Dimensions();
  const int num_slices = sdims.empty() ? 0 : sdims[0];
  if (num_slices <= 0) {
    return absl::InvalidArgumentError("MiniCPM-V: no slices in strips tensor.");
  }
  const size_t strip_elems = static_cast<size_t>(3) * 14 * kMaxL * 14;
  LITERT_ASSIGN_OR_RETURN(auto strips_span,
                          ReferTensorBufferAsSpan<float>(it_strips->second));
  LITERT_ASSIGN_OR_RETURN(auto pos_span,
                          ReferTensorBufferAsSpan<int32_t>(it_pos->second));
  LITERT_ASSIGN_OR_RETURN(auto np_span,
                          ReferTensorBufferAsSpan<int32_t>(it_np->second));
  // pos_embed [N, kMaxL, kModelDim]: per-slice sub-grid of the 70x70 table,
  // padded to kMaxL. Used to fill the resampler's position input per slice.
  LITERT_ASSIGN_OR_RETURN(auto pe_span,
                          ReferTensorBufferAsSpan<float>(it_pe->second));

  // Output accumulator: [num_slices * kNumQuery, kModelDim].
  std::vector<float> all_tokens(
      static_cast<size_t>(num_slices) * kNumQuery * kModelDim);

  // position_ids buffer is int64 in the navit tflite; build once, reuse.
  std::vector<int64_t> pos_i64(kMaxL);
  // 4D additive attention mask [1,1,kMaxL,kMaxL]; rebuilt per slice.
  std::vector<float> attn4d(static_cast<size_t>(kMaxL) * kMaxL);
  // resampler feature buffer (pad to kResamplerL).
  std::vector<float> res_feat(static_cast<size_t>(kResamplerL) * kVisionDim);
  std::vector<float> res_pos(static_cast<size_t>(kResamplerL) * kModelDim);

  for (int s = 0; s < num_slices; ++s) {
    const int valid = np_span[s];
    // ---- navit encoder ----
    // pixel: encoder_in_[0]
    LITERT_RETURN_IF_ERROR(encoder_in_[0].Write<float>(absl::MakeConstSpan(
        strips_span.data() + static_cast<size_t>(s) * strip_elems,
        strip_elems)));
    // position_ids (int32 -> int64): encoder_in_[1]
    for (int i = 0; i < kMaxL; ++i)
      pos_i64[i] = static_cast<int64_t>(pos_span[static_cast<size_t>(s) * kMaxL + i]);
    LITERT_RETURN_IF_ERROR(encoder_in_[1].Write<int64_t>(
        absl::MakeConstSpan(pos_i64.data(), pos_i64.size())));
    // attn mask [1,1,kMaxL,kMaxL]: rows valid, cols>=valid -> -inf. Actually
    // _prepare_4d_attention_mask masks KEY positions: mask[.,.,q,k] = 0 if
    // k<valid else large-negative, broadcast over q.
    const float kNeg = -3.0e38f;
    for (int q = 0; q < kMaxL; ++q) {
      float* row = attn4d.data() + static_cast<size_t>(q) * kMaxL;
      for (int k = 0; k < kMaxL; ++k) row[k] = (k < valid) ? 0.0f : kNeg;
    }
    LITERT_RETURN_IF_ERROR(encoder_in_[2].Write<float>(
        absl::MakeConstSpan(attn4d.data(), attn4d.size())));
    LITERT_RETURN_IF_ERROR(encoder_.Run(encoder_in_, encoder_out_));
    LITERT_ASSIGN_OR_RETURN(auto feat_span,
                            ReferTensorBufferAsSpan<float>(encoder_out_[0]));
    // feat is [1, kMaxL, 1152]; take first `valid` rows, pad to kResamplerL.
    std::fill(res_feat.begin(), res_feat.end(), 0.0f);
    std::copy(feat_span.data(),
              feat_span.data() + static_cast<size_t>(valid) * kVisionDim,
              res_feat.begin());
    // ---- original resampler ----
    // feat padded to kResamplerL.
    LITERT_RETURN_IF_ERROR(resampler_in_[resampler_feat_idx_].Write<float>(
        absl::MakeConstSpan(res_feat.data(), res_feat.size())));
    // per-slice pos_embed: take valid rows from pos_embed[s] (padded to kMaxL),
    // pad to kResamplerL. Layout: resampler pos input is [kResamplerL, 1, kModelDim].
    std::fill(res_pos.begin(), res_pos.end(), 0.0f);
    std::copy(pe_span.data() + static_cast<size_t>(s) * kMaxL * kModelDim,
              pe_span.data() + static_cast<size_t>(s) * kMaxL * kModelDim +
                  static_cast<size_t>(valid) * kModelDim,
              res_pos.begin());
    LITERT_RETURN_IF_ERROR(resampler_in_[resampler_pos_idx_].Write<float>(
        absl::MakeConstSpan(res_pos.data(), res_pos.size())));
    LITERT_RETURN_IF_ERROR(resampler_.Run(resampler_in_, resampler_out_));
    LITERT_ASSIGN_OR_RETURN(auto out_span,
                            ReferTensorBufferAsSpan<float>(resampler_out_[0]));
    std::copy(out_span.data(),
              out_span.data() + static_cast<size_t>(kNumQuery) * kModelDim,
              all_tokens.begin() + static_cast<size_t>(s) * kNumQuery * kModelDim);
  }

  // Build output TensorBuffer [1, num_slices*kNumQuery, kModelDim].
  const int total_tokens = num_slices * kNumQuery;
  ::litert::RankedTensorType out_type(
      ::litert::ElementType::Float32,
      ::litert::Layout(::litert::Dimensions({1, total_tokens, kModelDim})));
  LITERT_ASSIGN_OR_RETURN(
      auto out_buf,
      ::litert::TensorBuffer::CreateManaged(
          env_, ::litert::TensorBufferType::kHostMemory, out_type,
          static_cast<size_t>(total_tokens) * kModelDim * sizeof(float)));
  LITERT_RETURN_IF_ERROR(
      out_buf.Write<float>(absl::MakeConstSpan(all_tokens.data(), all_tokens.size())));
  return ExecutorVisionData(std::move(out_buf), /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<std::vector<int>>
MinicpmvVisionExecutor::GetExpectedInputDimension() const {
  return std::vector<int>{1, 3, kImageSize, kImageSize};
}

absl::StatusOr<VisionExecutorProperties>
MinicpmvVisionExecutor::GetVisionExecutorProperties() const {
  return absl::UnimplementedError("Not implemented.");
}

}  // namespace litert::lm
