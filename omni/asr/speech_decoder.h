// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_SPEECH_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_SPEECH_DECODER_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert_lm::omni::asr {

// Abstract interface for decoding preprocessed audio features into tokens.
class SpeechDecoder {
 public:
  // Represents a decoded token ID and its optional timestamp.
  struct DecodedToken {
    int token_id;
    std::optional<int> timestamp_ms;
  };

  virtual ~SpeechDecoder() = default;

  // Decodes mel_features into tokens.
  virtual absl::StatusOr<std::vector<DecodedToken>> Decode(
      absl::Span<const float> mel_features) = 0;

  // Resets internal model decoder state (e.g. KV cache) for a new audio stream.
  virtual void Reset() = 0;
};

}  // namespace litert_lm::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_SPEECH_DECODER_H_
