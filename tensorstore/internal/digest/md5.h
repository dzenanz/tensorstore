// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_DIGEST_MD5_H_
#define TENSORSTORE_INTERNAL_DIGEST_MD5_H_

#include <stdint.h>

#include <array>
#include <string_view>

#include "absl/strings/cord.h"
#include <openssl/md5.h>

namespace tensorstore {
namespace internal {

/// MD5 digester.
class MD5Digester {
 public:
  MD5Digester() { MD5_Init(&ctx_); }

  void Write(std::string_view src) {
    MD5_Update(&ctx_, src.data(), src.size());
  }

  void Write(const absl::Cord& cord);

  using DigestType = std::array<uint8_t, MD5_DIGEST_LENGTH>;

  DigestType Digest();

 private:
  MD5_CTX ctx_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_DIGEST_MD5_H_
