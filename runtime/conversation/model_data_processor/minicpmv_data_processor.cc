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

#include "runtime/conversation/model_data_processor/minicpmv_data_processor.h"

#include <memory>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "absl/memory/memory.h"       // from @com_google_absl
#include "absl/status/status.h"       // from @com_google_absl
#include "absl/status/statusor.h"     // from @com_google_absl
#include "absl/strings/string_view.h" // from @com_google_absl
#include "absl/types/span.h"          // from @com_google_absl
#include "absl/container/flat_hash_map.h"
#include "runtime/components/preprocessor/minicpmv_image_preprocess.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using nlohmann::ordered_json;

// Builds a single-slice TensorBufferMap InputImage from one preprocessed slice.
// The executor's Encode(map) treats N=1 -> 64 vision tokens.
absl::StatusOr<InputImage> BuildSliceImage(const MinicpmvSlice& sl) {
  constexpr int kMaxL = kMinicpmvMaxPatchLen;
  constexpr int kModelDim = kMinicpmvModelDim;
  constexpr int kPatch = kMinicpmvPatchSize;
  const size_t strip_elems = static_cast<size_t>(3) * kPatch * kMaxL * kPatch;
  std::vector<float> strips(strip_elems, 0.0f);
  std::vector<int32_t> posids(kMaxL, 0);
  std::vector<int32_t> nps(1, sl.num_patches);
  std::vector<float> pes(static_cast<size_t>(kMaxL) * kModelDim, 0.0f);
  const int strip_w = sl.num_patches * kPatch;
  const int padded_w = kMaxL * kPatch;
  for (int r = 0; r < 3 * kPatch; ++r) {
    std::copy(sl.strip.data() + static_cast<size_t>(r) * strip_w,
              sl.strip.data() + static_cast<size_t>(r) * strip_w + strip_w,
              strips.data() + static_cast<size_t>(r) * padded_w);
  }
  for (int k = 0; k < sl.num_patches && k < kMaxL; ++k)
    posids[k] = static_cast<int32_t>(sl.position_ids[k]);
  std::copy(sl.pos_embed.data(),
            sl.pos_embed.data() + static_cast<size_t>(sl.num_patches) * kModelDim,
            pes.data());
  LITERT_ASSIGN_OR_RETURN(
      auto strips_t,
      CopyToTensorBuffer<float>(
          absl::MakeConstSpan(strips),
          ::litert::Dimensions({1, 3, kPatch, kMaxL * kPatch})));
  LITERT_ASSIGN_OR_RETURN(
      auto pos_t, CopyToTensorBuffer<int32_t>(absl::MakeConstSpan(posids),
                                              ::litert::Dimensions({1, kMaxL})));
  LITERT_ASSIGN_OR_RETURN(
      auto np_t, CopyToTensorBuffer<int32_t>(absl::MakeConstSpan(nps),
                                             ::litert::Dimensions({1})));
  LITERT_ASSIGN_OR_RETURN(
      auto pe_t, CopyToTensorBuffer<float>(
                     absl::MakeConstSpan(pes),
                     ::litert::Dimensions({1, kMaxL, kModelDim})));
  absl::flat_hash_map<std::string, ::litert::TensorBuffer> m;
  m.emplace("strips", std::move(strips_t));
  m.emplace("position_ids", std::move(pos_t));
  m.emplace("num_patches", std::move(np_t));
  m.emplace("pos_embed", std::move(pe_t));
  return InputImage(std::move(m));
}

}  // namespace

absl::StatusOr<std::unique_ptr<MinicpmvDataProcessor>>
MinicpmvDataProcessor::Create(MinicpmvDataProcessorConfig config,
                              const PromptTemplateCapabilities& capabilities) {
  return absl::WrapUnique(new MinicpmvDataProcessor(config, capabilities));
}

absl::StatusOr<ordered_json> MinicpmvDataProcessor::MessageToTemplateInput(
    const ordered_json& message) const {
  return message;
}

absl::StatusOr<ordered_json> MinicpmvDataProcessor::FormatTools(
    const ordered_json& tools) const {
  return tools;
}

absl::StatusOr<std::vector<InputData>>
MinicpmvDataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const ordered_json& messages,
    const MinicpmvDataProcessorArguments& args) const {
  // MiniCPM-V wraps each image as <image>{N x <unk>}</image>. We match the
  // "<image>" boundary as the image token; the actual number of vision
  // placeholder tokens is set downstream from the resampler output count.
  // The chat template (kMinicpmv case in model_type_utils.cc) renders each
  // image content item as the literal string "<image_soft_token>". We must
  // match THAT token (not "<image>") so ProcessMultimodalPrompt replaces it
  // with the N vision placeholder (-1) tokens; otherwise the literal string
  // leaks into the LLM input and is echoed back. Matches fastvlm's config.
  // MiniCPM-V wraps the image placeholders as:  <image>{64 x -1}</image>
  // The kMinicpmv chat template renders each image content item as the literal
  // "<image_soft_token>"; we match that as the image token and surround it with
  // the real MiniCPM markers <image>/</image> (tokenized as text). The matched
  // token is replaced downstream by N vision placeholder (-1) tokens = the
  // resampler output row count (64), which the embedding splice fills with the
  // vision features. NOTE: <image_soft_token> is NOT a MiniCPM vocab token; it
  // is only a boundary marker consumed here, never tokenized.
  // Multi-slice: emit the official structured layout with per-slice InputImages.
  //   <image_id>0</image_id><image>{thumbnail 64}</image>
  //   [<slice>{64}</slice> per sub-image, grid row-major, rows joined by \n]
  // Each InputImage is a single-slice map -> executor Encode(map) yields 64
  // vision (-1) placeholder tokens, spliced in emit order (thumbnail first).

  // 1. Extract the raw image bytes from messages (first image content item).
  std::string image_bytes;
  for (const auto& message : messages) {
    if (!message.contains("content") || !message["content"].is_array()) continue;
    for (const auto& item : message["content"]) {
      if (item.is_object() && item.contains("type") && item["type"] == "image") {
        if (item.contains("path")) {
          std::ifstream f(item["path"].get<std::string>(), std::ios::binary);
          std::ostringstream ss; ss << f.rdbuf(); image_bytes = ss.str();
        } else if (item.contains("bytes")) {
          image_bytes = item["bytes"].get<std::string>();
        }
        break;
      }
    }
    if (!image_bytes.empty()) break;
  }
  if (image_bytes.empty()) {
    return absl::InvalidArgumentError("MiniCPM-V: no image bytes in messages.");
  }

  // 2. Slice.
  MinicpmvSliceConfig scfg;
  ASSIGN_OR_RETURN(MinicpmvSliced sliced, PreprocessImageSliced(image_bytes, scfg));
  const int gx = sliced.grid_x, gy = sliced.grid_y;

  // 3. Split the rendered prompt at the single <image_soft_token> marker.
  std::string rendered(rendered_template_prompt);
  const std::string marker = "<image_soft_token>";
  size_t mpos = rendered.find(marker);
  std::string pre = (mpos == std::string::npos) ? rendered : rendered.substr(0, mpos);
  std::string post = (mpos == std::string::npos) ? "" : rendered.substr(mpos + marker.size());

  // 4. Build interleaved InputData.
  std::vector<InputData> out;
  auto push_text = [&](const std::string& t) {
    if (!t.empty()) out.emplace_back(InputText(t));
  };
  push_text(pre);
  // image_id + thumbnail
  push_text("<image_id>0</image_id><image>");
  ASSIGN_OR_RETURN(auto thumb, BuildSliceImage(sliced.slices[0]));
  out.emplace_back(std::move(thumb));
  push_text("</image>");
  // sub-slices (index 1..N-1) laid out grid row-major; rows joined by \n.
  int sub = 1;
  for (int r = 0; r < gy; ++r) {
    for (int col = 0; col < gx; ++col) {
      if (sub >= static_cast<int>(sliced.slices.size())) break;
      push_text("<slice>");
      ASSIGN_OR_RETURN(auto sl_img, BuildSliceImage(sliced.slices[sub]));
      out.emplace_back(std::move(sl_img));
      push_text("</slice>");
      ++sub;
    }
    if (r + 1 < gy) push_text("\n");
  }
  push_text(post);
  return out;
}

absl::StatusOr<Message> MinicpmvDataProcessor::ToMessageImpl(
    const Responses& responses,
    const MinicpmvDataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  ordered_json content = ordered_json::array(
      {{{"type", "text"}, {"text", std::string(response_text)}}});
  return ordered_json::object({{"role", "assistant"}, {"content", content}});
}

absl::Status MinicpmvDataProcessor::CloneStateImpl(
    const TypeSafeModelDataProcessor<MinicpmvDataProcessorConfig,
                                     MinicpmvDataProcessorArguments>& other) {
  return absl::OkStatus();
}

}  // namespace litert::lm
