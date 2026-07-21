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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MINICPMV_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MINICPMV_DATA_PROCESSOR_H_

#include <memory>
#include <string>
#include <vector>

#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/minicpmv_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Data processor for MiniCPM-V-4 (SigLIP + Resampler + Llama).
//
// Baseline: single image, fixed 980x980, no slicing. The prompt marker
// "(<image>./</image>)" is expanded to "<image>{64 x <unk>}</image>" and each
// <unk> becomes a vision-embedding placeholder token, which the LLM executor
// fills with a resampler soft-token row via EmbeddingLookupMultiModal.
class MinicpmvDataProcessor
    : public TypeSafeModelDataProcessor<MinicpmvDataProcessorConfig,
                                        MinicpmvDataProcessorArguments> {
 public:
  static absl::StatusOr<std::unique_ptr<MinicpmvDataProcessor>> Create(
      MinicpmvDataProcessorConfig config,
      const PromptTemplateCapabilities& capabilities);

  const MinicpmvDataProcessorConfig& GetConfig() const override {
    return config_;
  }

  absl::StatusOr<nlohmann::ordered_json> MessageToTemplateInput(
      const nlohmann::ordered_json& message) const override;

  absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const override;

  absl::string_view CodeFenceStart() const override { return ""; }
  absl::string_view CodeFenceEnd() const override { return ""; }

 private:
  explicit MinicpmvDataProcessor(
      MinicpmvDataProcessorConfig config,
      const PromptTemplateCapabilities& capabilities)
      : config_(config), capabilities_(capabilities) {}

  absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const MinicpmvDataProcessorArguments& args) const override;

  absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses,
      const MinicpmvDataProcessorArguments& args) const override;

  absl::Status CloneStateImpl(
      const TypeSafeModelDataProcessor<MinicpmvDataProcessorConfig,
                                       MinicpmvDataProcessorArguments>& other)
      override;

  MinicpmvDataProcessorConfig config_;
  PromptTemplateCapabilities capabilities_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MINICPMV_DATA_PROCESSOR_H_
