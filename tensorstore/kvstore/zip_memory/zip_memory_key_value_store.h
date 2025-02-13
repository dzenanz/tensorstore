// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_ZIP_MEMORY_ZIP_MEMORY_KEY_VALUE_STORE_H_
#define TENSORSTORE_KVSTORE_ZIP_MEMORY_ZIP_MEMORY_KEY_VALUE_STORE_H_

/// \file
/// A key-value store backed by a zip file, which can be in memory or on disk.

#include "tensorstore/kvstore/kvstore.h"

namespace tensorstore {

/// Creates a new (unique) zip_memory KvStore.
///
/// \param atomic If `true`, atomic multi-key transactions are supported.  If
///     `false`, only single-key atomic transactions are supported.  Both
///     versions are exposed for testing implementations of transactional
///     operations.
kvstore::DriverPtr GetZipMemoryKeyValueStore();

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_ZIP_MEMORY_ZIP_MEMORY_KEY_VALUE_STORE_H_
