// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_SETTINGS_H_

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// Selects which concrete VisionExecutor the ResourceManager instantiates.
// ResourceManager sees only the VisionExecutorSettings (not the model type), so
// the model-specific choice is carried here, set from the LlmModelType oneof in
// EngineAdvancedImpl::Create. Add a new value per model that needs a bespoke
// vision executor rather than the stock VisionLiteRtCompiledModelExecutor.
enum class VisionExecutorKind {
  kStock = 0,   // VisionLiteRtCompiledModelExecutor (default)
  kMinicpmv,    // MinicpmvVisionExecutor
};

// The VisionExecutorSettings class is used to configure the VisionExecutor.
// It is used to configure the vision encoder and vision adapter models.
// Args:
//   - model_assets: The model assets to use for the vision encoder and vision
//   adapter models.
//   - encoder_backend: The backend to use for the vision encoder model.
//   - adapter_backend: The backend to use for the vision adapter model.
class VisionExecutorSettings : public ExecutorSettingsBase {
 public:
  static absl::StatusOr<VisionExecutorSettings> CreateDefault(
      const ModelAssets& model_assets, Backend encoder_backend,
      Backend adapter_backend);

  static constexpr absl::string_view kEncoderName = ".vision_encoder";
  static constexpr absl::string_view kAdapterName = ".vision_adapter";

  // Getter for encoder_backend.
  Backend GetEncoderBackend() const;
  // Setter for encoder_backend.
  absl::Status SetEncoderBackend(Backend backend);

  // Getter for scoped_encoder_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedEncoderCacheFile() const {
    return scoped_encoder_cache_file_;
  }

  // Setter for scoped_encoder_cache_file.
  void SetScopedEncoderCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_encoder_cache_file_ = std::move(cache_file);
  }

  // Getter for adapter_backend.
  Backend GetAdapterBackend() const;
  // Setter for adapter_backend.
  absl::Status SetAdapterBackend(Backend backend);
  VisionExecutorKind GetExecutorKind() const { return executor_kind_; }
  void SetExecutorKind(VisionExecutorKind kind) { executor_kind_ = kind; }

  // Getter for scoped_adapter_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedAdapterCacheFile() const {
    return scoped_adapter_cache_file_;
  }

  // Setter for scoped_adapter_cache_file.
  void SetScopedAdapterCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_adapter_cache_file_ = std::move(cache_file);
  }

  // Getter for scoped_encoder_program_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedEncoderProgramCacheFile()
      const {
    return scoped_encoder_program_cache_file_;
  }

  // Setter for scoped_encoder_program_cache_file.
  void SetScopedEncoderProgramCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_encoder_program_cache_file_ = std::move(cache_file);
  }

  // Getter for scoped_adapter_program_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedAdapterProgramCacheFile()
      const {
    return scoped_adapter_program_cache_file_;
  }

  // Setter for scoped_adapter_program_cache_file.
  void SetScopedAdapterProgramCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_adapter_program_cache_file_ = std::move(cache_file);
  }

  using ExecutorSettingsBase::GetWeightCacheFile;
  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetWeightCacheFile(absl::string_view suffix,
                     bool check_and_clean) const override;

  using ExecutorSettingsBase::GetProgramCacheFile;
  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetProgramCacheFile(absl::string_view suffix,
                      bool check_and_clean) const override;

 private:
  explicit VisionExecutorSettings(const ModelAssets& model_assets)
      : ExecutorSettingsBase(model_assets) {}

  // The backend to use for the vision encoder model.
  Backend encoder_backend_;

  // The backend to use for the vision adapter model.
  Backend adapter_backend_;

  // Which concrete VisionExecutor to build for this model.
  VisionExecutorKind executor_kind_ = VisionExecutorKind::kStock;

  // The cache file to use for the vision encoder model.
  std::shared_ptr<litert::lm::ScopedFile> scoped_encoder_cache_file_;

  // The cache file to use for the vision adapter model.
  std::shared_ptr<litert::lm::ScopedFile> scoped_adapter_cache_file_;

  // The program cache file to use for the vision encoder model.
  std::shared_ptr<litert::lm::ScopedFile> scoped_encoder_program_cache_file_;

  // The program cache file to use for the vision adapter model.
  std::shared_ptr<litert::lm::ScopedFile> scoped_adapter_program_cache_file_;
};

std::ostream& operator<<(std::ostream& os,
                         const VisionExecutorSettings& settings);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_SETTINGS_H_
