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

#include "tensorstore/kvstore/zip/zip_key_value_store.h"

#include <atomic>
#include <deque>
#include <filesystem>  // C++17
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

// in-memory zip test begins
#include "mz.h"
#include "mz_os.h"  // for MZ_VERSION_MADEBY
#include "mz_strm.h"
#include "mz_strm_buf.h"
#include "mz_strm_mem.h"
#include "mz_strm_os.h"  // for testing success
#include "mz_strm_split.h"
#include "mz_zip.h"

void test_stream_mem(void) {
  mz_zip_file file_info;
  void* read_mem_stream = nullptr;
  void* write_mem_stream = nullptr;
  void* os_stream = nullptr;
  void* zip_handle = nullptr;
  int32_t written = 0;
  int32_t read = 0;
  int32_t text_size = 0;
  int32_t buffer_size = 0;
  int32_t err = MZ_OK;
  const uint8_t* buffer_ptr = nullptr;
  char* password = "1234";
  char* text_name = "test";
  char* text_ptr = "test string";
  char temp[120];

  memset(&file_info, 0, sizeof(file_info));

  text_size = (int32_t)strlen(text_ptr);

  /* Write zip to memory stream */
  mz_stream_mem_create(&write_mem_stream);
  mz_stream_mem_set_grow_size(write_mem_stream, 128 * 1024);
  mz_stream_open(write_mem_stream, nullptr, MZ_OPEN_MODE_CREATE);

  mz_zip_create(&zip_handle);
  err = mz_zip_open(zip_handle, write_mem_stream, MZ_OPEN_MODE_WRITE);

  if (err == MZ_OK) {
    file_info.version_madeby = MZ_VERSION_MADEBY;
    file_info.compression_method = MZ_COMPRESS_METHOD_DEFLATE;
    file_info.filename = text_name;
    file_info.uncompressed_size = text_size;
    file_info.aes_version = MZ_AES_VERSION;

    err = mz_zip_entry_write_open(zip_handle, &file_info,
                                  MZ_COMPRESS_LEVEL_DEFAULT, 0, password);
    if (err == MZ_OK) {
      written = mz_zip_entry_write(zip_handle, text_ptr, text_size);
      if (written < MZ_OK) err = written;
      mz_zip_entry_close(zip_handle);
    }

    mz_zip_close(zip_handle);
  } else {
    err = MZ_INTERNAL_ERROR;
  }

  mz_zip_delete(&zip_handle);

  mz_stream_mem_get_buffer(write_mem_stream, (const void**)&buffer_ptr);
  mz_stream_mem_seek(write_mem_stream, 0, MZ_SEEK_END);
  buffer_size = (int32_t)mz_stream_mem_tell(write_mem_stream);

  if (err == MZ_OK) {
    /* Create a zip file on disk for inspection */
    mz_stream_os_create(&os_stream);
    mz_stream_os_open(os_stream, "mytest.zip",
                      MZ_OPEN_MODE_WRITE | MZ_OPEN_MODE_CREATE);
    mz_stream_os_write(os_stream, buffer_ptr, buffer_size);
    mz_stream_os_close(os_stream);
    mz_stream_os_delete(&os_stream);
  }

  if (err == MZ_OK) {
    /* Read from a memory stream */
    mz_stream_mem_create(&read_mem_stream);
    mz_stream_mem_set_buffer(read_mem_stream, (void*)buffer_ptr, buffer_size);
    mz_stream_open(read_mem_stream, nullptr, MZ_OPEN_MODE_READ);

    mz_zip_create(&zip_handle);
    err = mz_zip_open(zip_handle, read_mem_stream, MZ_OPEN_MODE_READ);

    if (err == MZ_OK) {
      err = mz_zip_goto_first_entry(zip_handle);
      if (err == MZ_OK) err = mz_zip_entry_read_open(zip_handle, 0, password);
      if (err == MZ_OK)
        read = mz_zip_entry_read(zip_handle, temp, sizeof(temp));

      MZ_UNUSED(read);

      mz_zip_entry_close(zip_handle);
      mz_zip_close(zip_handle);
    }

    mz_zip_delete(&zip_handle);

    mz_stream_mem_close(&read_mem_stream);
    mz_stream_mem_delete(&read_mem_stream);
    read_mem_stream = nullptr;
  }

  mz_stream_mem_close(write_mem_stream);
  mz_stream_mem_delete(&write_mem_stream);
  write_mem_stream = nullptr;
}

// in-memory zip test ends

namespace {
/// Which part of this path/url is file name, and which is key within it.
/// Returns true if zip filename was successfully determined.
bool getZipFileFromKey(const std::string& key, std::string& file_part,
                       std::string& key_part) {
  std::string::size_type pos = key.find(".zip");
  if (pos == std::string::npos) {
    // .zip not found
    file_part = "C:/a/tsTest.zip";  // for testing
    key_part = key;
    return false;
    // throw std::runtime_error("Key " + key +
    //                          " does not contain '.zip' substring");
  }
  file_part = key.substr(0, pos + 4);  // include .zip
  key_part = key.substr(pos + 5);      // skip separator

  // check whether this is a directory with weird .zip extension
  while (std::filesystem::is_directory(file_part)) {
    pos = key.find(".zip", pos);  // search again
    if (pos != std::string::npos) {
      file_part = key.substr(0, pos + 4);  // include .zip
      key_part = key.substr(pos + 5);      // skip separator
    }
  }

  return true;
}
}  // namespace

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal_kvstore::DeleteRangeEntry;
using ::tensorstore::internal_kvstore::kReadModifyWrite;
using ::tensorstore::kvstore::ReadResult;

TimestampedStorageGeneration GenerationNow(StorageGeneration generation) {
  return TimestampedStorageGeneration{std::move(generation), absl::Now()};
}

/// The actual data for a zip-based KeyValueStore.
///
/// This is a separate reference-counted object where: `ZipDriver` ->
/// `Context::Resource<ZipEncapsulatorResource>` -> `ZipEncapsulator`.
/// This allows the `ZipDriver` to retain a reference to the
/// `ZipEncapsulatorResource`, while also allowing an equivalent
/// `ZipDriver` to be constructed from the
/// `ZipEncapsulatorResource`.
struct ZipEncapsulator
    : public internal::AtomicReferenceCount<ZipEncapsulator> {
  using Ptr = internal::IntrusivePtr<ZipEncapsulator>;
  struct ValueWithGenerationNumber {
    absl::Cord value;  // payload bytes
    uint64_t generation_number;
    StorageGeneration generation() const {
      return StorageGeneration::FromUint64(generation_number);
    }
  };

  using Map = absl::btree_map<std::string, ValueWithGenerationNumber>;
  std::pair<Map::iterator, Map::iterator> Find(const std::string& inclusive_min,
                                               const std::string& exclusive_max)
      ABSL_SHARED_LOCKS_REQUIRED(mutex) {
    return {values.lower_bound(inclusive_min),
            exclusive_max.empty() ? values.end()
                                  : values.lower_bound(exclusive_max)};
  }

  std::pair<Map::iterator, Map::iterator> Find(const KeyRange& range)
      ABSL_SHARED_LOCKS_REQUIRED(mutex) {
    return Find(range.inclusive_min, range.exclusive_max);
  }

  absl::Mutex mutex;
  /// Next generation number to use when updating the value associated with a
  /// key.  Using a single per-store counter rather than a per-key counter
  /// ensures that creating a key, deleting it, then creating it again does
  /// not result in the same generation number being reused for a given key.
  uint64_t next_generation_number ABSL_GUARDED_BY(mutex) = 0;
  Map values ABSL_GUARDED_BY(mutex);

  /// ZipEncapsulator ivars.
  //mz_zip_file file_info ABSL_GUARDED_BY(mutex){};
  void* mem_stream ABSL_GUARDED_BY(mutex) = nullptr;
  void* file_stream ABSL_GUARDED_BY(mutex) = nullptr;
  void* buffered_stream ABSL_GUARDED_BY(mutex) = nullptr;
  void* split_stream ABSL_GUARDED_BY(mutex) = nullptr;
  void* zip_handle ABSL_GUARDED_BY(mutex) = nullptr;

  void closeZip() {
    if (zip_handle != nullptr) {
      mz_zip_close(zip_handle);
      mz_zip_delete(&zip_handle);
    }
    if (split_stream != nullptr) {
      mz_stream_split_close(split_stream);
      mz_stream_split_delete(&split_stream);
    }
    if (buffered_stream != nullptr) {
      mz_stream_buffered_delete(&buffered_stream);
    }
    if (file_stream != nullptr) {
      mz_stream_os_delete(&file_stream);
    }
    if (mem_stream != nullptr) {
      mz_stream_mem_close(mem_stream);
      mz_stream_mem_delete(&mem_stream);
    }
  }

  bool openZipFromFile(const char* fileName,
                       int32_t openMode = MZ_OPEN_MODE_READWRITE) {
    int32_t err = MZ_OK;

    mz_stream_os_create(&file_stream);
    mz_stream_buffered_create(&buffered_stream);
    mz_stream_split_create(&split_stream);

    mz_stream_set_base(buffered_stream, file_stream);
    mz_stream_set_base(split_stream, buffered_stream);
    mz_stream_open(split_stream, fileName, openMode);

    mz_zip_create(&zip_handle);
    err = mz_zip_open(zip_handle, split_stream, openMode);
    if (err != MZ_OK) {
      closeZip();
      return false;
    }
    return true;
  }

  bool openZipFileForWriting(const char* fileName,
      int32_t openMode = MZ_OPEN_MODE_READWRITE)
  {
    std::filesystem::path file(fileName);
    if (!std::filesystem::is_directory(file.parent_path()))
    {
      // create the directory first
      mz_dir_make(file.parent_path().string().c_str());
    }
    
    if (!std::filesystem::exists(file)) {
      openMode |= MZ_OPEN_MODE_CREATE;
    }
    else
    {
      openMode |= MZ_OPEN_MODE_APPEND;
    }

    return openZipFromFile(fileName, openMode);
  }

  bool findEntry(const std::string& filePath, mz_zip_file** file_info) {
    int32_t err = MZ_OK;
    err = mz_zip_goto_first_entry(zip_handle);

    if (err == MZ_OK) {
      err = mz_zip_entry_get_info(zip_handle, file_info);
    }

    // go through the list of files in the zip archive
    while (std::string((*file_info)->filename) != filePath) {
      if (err == MZ_OK) {
        err = mz_zip_goto_next_entry(zip_handle);
      } else {
        break;
      }
      if (err == MZ_OK) {
        err = mz_zip_entry_get_info(zip_handle, file_info);
      } else {
        break;
      }
    }
    return std::string((*file_info)->filename) == filePath;
  }
};

/// Defines the context resource (see `tensorstore/context.h`) that actually
/// owns the stored key/value pairs.
struct ZipEncapsulatorResource
    : public internal::ContextResourceTraits<ZipEncapsulatorResource> {
  constexpr static char id[] = "zip_encapsulator";
  struct Spec {};
  using Resource = ZipEncapsulator::Ptr;
  static Spec Default() { return {}; }
  static constexpr auto JsonBinder() { return jb::Object(); }
  static Result<Resource> Create(
      Spec, internal::ContextResourceCreationContext context) {
    test_stream_mem();
    return ZipEncapsulator::Ptr(new ZipEncapsulator);
  }
  static Spec GetSpec(const Resource&,
                      const internal::ContextSpecBuilder& builder) {
    test_stream_mem();
    return {};
  }
};

const internal::ContextResourceRegistration<ZipEncapsulatorResource>
    resource_registration;

/// Data members for `ZipDriverSpec`.
struct ZipDriverSpecData {
  Context::Resource<ZipEncapsulatorResource> zip_encapsulator;

  /// Make this type compatible with `tensorstore::ApplyMembers`.
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    // `x` is a reference to a `SpecData` object.  This function must invoke
    // `f` with a reference to each member of `x`.
    return f(x.zip_encapsulator);
  };

  /// Must specify a JSON binder.
  constexpr static auto default_json_binder = jb::Object(
      jb::Member(ZipEncapsulatorResource::id,
                 jb::Projection<&ZipDriverSpecData::zip_encapsulator>()));
};

class ZipDriverSpec
    : public internal_kvstore::RegisteredDriverSpec<ZipDriverSpec,
                                                    ZipDriverSpecData> {
 public:
  /// Specifies the string identifier under which the driver will be registered.
  static constexpr char id[] = "zip";

  Future<kvstore::DriverPtr> DoOpen() const override;

  Result<std::string> ToUrl(std::string_view path) const override {
    std::string encoded_path;
    internal::PercentEncodeUriPath(path, encoded_path);
    return tensorstore::StrCat(id, "://", encoded_path);
  }
};

/// Defines the "zip" KeyValueStore driver.
class ZipDriver
    : public internal_kvstore::RegisteredDriver<ZipDriver, ZipDriverSpec> {
 public:
  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  Future<const void> DeleteRange(KeyRange range) override;

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override;

  absl::Status ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                               size_t& phase, Key key,
                               ReadModifyWriteSource& source) override;

  absl::Status TransactionalDeleteRange(
      const internal::OpenTransactionPtr& transaction, KeyRange range) override;

  class TransactionNode;

  /// Returns a reference to the stored key value pairs.  The stored data is
  /// owned by the `Context::Resource` rather than directly by
  /// `ZipDriver` in order to allow it to live as long as the
  /// `Context` from which the `ZipDriver` was opened, and thereby
  /// allow an equivalent `ZipDriver` to be re-opened from the
  /// `Context`.
  ZipEncapsulator& data() { return **spec_.zip_encapsulator; }

  /// Obtains a `BoundSpec` representation from an open `Driver`.
  absl::Status GetBoundSpecData(ZipDriverSpecData& spec) const {
    // `spec` is returned via an out parameter rather than returned via a
    // `Result`, as that simplifies use cases involving composition via
    // inheritance.
    spec = spec_;
    return absl::Status();
  }

  /// In simple cases, such as the "zip" driver, the `Driver` can simply
  /// store a copy of the `BoundSpecData` as a member.
  SpecData spec_;
};

Future<kvstore::DriverPtr> ZipDriverSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<ZipDriver>();
  driver->spec_ = data_;
  return driver;
}

using BufferedReadModifyWriteEntry =
    internal_kvstore::AtomicMultiPhaseMutation::BufferedReadModifyWriteEntry;

class ZipDriver::TransactionNode
    : public internal_kvstore::AtomicTransactionNode {
  using Base = internal_kvstore::AtomicTransactionNode;

 public:
  using Base::Base;

  /// Commits a (possibly multi-key) transaction atomically.
  ///
  /// The commit involves two steps, both while holding a lock on the entire
  /// KeyValueStore:
  ///
  /// 1. Without making any modifications, validates that the underlying
  ///    KeyValueStore data matches the generation constraints specified in the
  ///    transaction.  If validation fails, the commit is retried, which
  ///    normally results in any modifications being "rebased" on top of any
  ///    modified values.
  ///
  /// 2. If validation succeeds, applies the modifications.
  void AllEntriesDone(
      internal_kvstore::SinglePhaseMutation& single_phase_mutation) override
      ABSL_NO_THREAD_SAFETY_ANALYSIS {
    if (!single_phase_mutation.remaining_entries_.HasError()) {
      auto& data = static_cast<ZipDriver&>(*this->driver()).data();
      TimestampedStorageGeneration generation;
      UniqueWriterLock lock(data.mutex);
      absl::Time commit_time = absl::Now();
      if (!ValidateEntryConditions(data, single_phase_mutation, commit_time)) {
        lock.unlock();
        internal_kvstore::RetryAtomicWriteback(single_phase_mutation,
                                               commit_time);
        return;
      }
      ApplyMutation(data, single_phase_mutation, commit_time);
      lock.unlock();
      internal_kvstore::AtomicCommitWritebackSuccess(single_phase_mutation);
    } else {
      internal_kvstore::WritebackError(single_phase_mutation);
    }
    MultiPhaseMutation::AllEntriesDone(single_phase_mutation);
  }

  /// Validates that the underlying `data` matches the generation constraints
  /// specified in the transaction.  No changes are made to the `data`.
  static bool ValidateEntryConditions(
      ZipEncapsulator& data,
      internal_kvstore::SinglePhaseMutation& single_phase_mutation,
      const absl::Time& commit_time) ABSL_SHARED_LOCKS_REQUIRED(data.mutex) {
    bool validated = true;
    for (auto& entry : single_phase_mutation.entries_) {
      if (!ValidateEntryConditions(data, entry, commit_time)) {
        validated = false;
      }
    }
    return validated;
  }

  static bool ValidateEntryConditions(ZipEncapsulator& data,
                                      internal_kvstore::MutationEntry& entry,
                                      const absl::Time& commit_time)
      ABSL_SHARED_LOCKS_REQUIRED(data.mutex) {
    if (entry.entry_type() == kReadModifyWrite) {
      return ValidateEntryConditions(
          data, static_cast<BufferedReadModifyWriteEntry&>(entry), commit_time);
    }
    auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
    // `DeleteRangeEntry` imposes no constraints itself, but the superseded
    // `ReadModifyWriteEntry` nodes may have constraints.
    bool validated = true;
    for (auto& deleted_entry : dr_entry.superseded_) {
      if (!ValidateEntryConditions(
              data, static_cast<BufferedReadModifyWriteEntry&>(deleted_entry),
              commit_time)) {
        validated = false;
      }
    }
    return validated;
  }

  static bool ValidateEntryConditions(ZipEncapsulator& data,
                                      BufferedReadModifyWriteEntry& entry,
                                      const absl::Time& commit_time)
      ABSL_SHARED_LOCKS_REQUIRED(data.mutex) {
    auto& stamp = entry.read_result_.stamp;
    auto if_equal = StorageGeneration::Clean(stamp.generation);
    if (StorageGeneration::IsUnknown(if_equal)) {
      assert(stamp.time == absl::InfiniteFuture());
      return true;
    }
    auto it = data.values.find(entry.key_);
    if (it == data.values.end()) {
      if (StorageGeneration::IsNoValue(if_equal)) {
        entry.read_result_.stamp.time = commit_time;
        return true;
      }
    } else if (if_equal == it->second.generation()) {
      entry.read_result_.stamp.time = commit_time;
      return true;
    }
    return false;
  }

  /// Applies the changes in the transaction to the stored `data`.
  ///
  /// It is assumed that the constraints have already been validated by
  /// `ValidateConditions`.
  static void ApplyMutation(
      ZipEncapsulator& data,
      internal_kvstore::SinglePhaseMutation& single_phase_mutation,
      const absl::Time& commit_time) ABSL_EXCLUSIVE_LOCKS_REQUIRED(data.mutex) {
    std::cout << "applying mutation" << std::endl;
    int* troublemaker = nullptr;
    // *troublemaker = 1;  // let's have the debugger break here too

    for (auto& entry : single_phase_mutation.entries_) {
      if (entry.entry_type() == kReadModifyWrite) {
        auto& rmw_entry = static_cast<BufferedReadModifyWriteEntry&>(entry);
        auto& stamp = rmw_entry.read_result_.stamp;
        stamp.time = commit_time;
        if (!StorageGeneration::IsDirty(
                rmw_entry.read_result_.stamp.generation)) {
          // Do nothing
        } else if (rmw_entry.read_result_.state == ReadResult::kMissing) {
          data.values.erase(rmw_entry.key_);
          stamp.generation = StorageGeneration::NoValue();
        } else {
          assert(rmw_entry.read_result_.state == ReadResult::kValue);
          auto& v = data.values[rmw_entry.key_];
          v.generation_number = data.next_generation_number++;
          v.value = std::move(rmw_entry.read_result_.value);
          stamp.generation = v.generation();
        }
      } else {
        auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
        auto it_range = data.Find(dr_entry.key_, dr_entry.exclusive_max_);
        data.values.erase(it_range.first, it_range.second);
      }
    }
  }
};

Future<ReadResult> ZipDriver::Read(Key key, ReadOptions options) {
  auto& data = this->data();
  absl::ReaderMutexLock lock(&data.mutex);
  std::string zipFileName, keyPart;
  getZipFileFromKey(key, zipFileName, keyPart);

  ReadResult result;

  if (!data.openZipFromFile(zipFileName.c_str(), MZ_OPEN_MODE_READ)) {
    throw std::runtime_error("Could not open " + zipFileName);
  }

  int32_t err = MZ_OK;
  mz_zip_file* file_info = nullptr;
  if (!data.findEntry(keyPart, &file_info)) {
    // Key not found.
    result.stamp = GenerationNow(StorageGeneration::NoValue());
    result.state = ReadResult::kMissing;
    return result;
  }

  // Key found.
  if (err == MZ_OK) {
    err = mz_zip_entry_read_open(data.zip_handle, 0, nullptr);
  }
  int32_t read = 0;
  size_t length = file_info->uncompressed_size;
  std::string buffer(length + 1, 0);  // debug-friendly zero terminator
  buffer.resize(length);  // but we don't want it to be a part of the string
  if (err == MZ_OK) {
    read = mz_zip_entry_read(data.zip_handle, &buffer[0],
                             file_info->uncompressed_size);
    if (read != file_info->uncompressed_size) {
      mz_zip_entry_close(data.zip_handle);
      data.closeZip();
      throw std::runtime_error("Could not read key " + keyPart + " from file " +
                               zipFileName + ". Read " + std::to_string(read) +
                               " bytes out of " + std::to_string(length));
    }
  }
  // TODO: keep the zip file open for subsequent reads
  mz_zip_entry_close(data.zip_handle);
  data.closeZip();

  absl::Cord value(std::move(buffer));

  ++data.next_generation_number;
  auto nextGen = StorageGeneration::FromUint64(data.next_generation_number);
  result.stamp = GenerationNow(nextGen);
  if (options.if_not_equal == nextGen ||
      (!StorageGeneration::IsUnknown(options.if_equal) &&
       options.if_equal != nextGen)) {
    // Generation associated with `key` matches `if_not_equal`.  Abort.
    return result;
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto byte_range,
                               options.byte_range.Validate(length));
  result.state = ReadResult::kValue;
  result.value = internal::GetSubCord(value, byte_range);
  return result;
}

Future<TimestampedStorageGeneration> ZipDriver::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  using ValueWithGenerationNumber = ZipEncapsulator::ValueWithGenerationNumber;
  auto& data = this->data();
  absl::WriterMutexLock lock(&data.mutex);

  std::string zipFileName, keyPart;
  getZipFileFromKey(key, zipFileName, keyPart);
  if (!data.openZipFileForWriting(zipFileName.c_str())) {
    throw std::runtime_error("Could not open " + zipFileName + " for writing");
  }

  int32_t err = MZ_OK;
  mz_zip_file* file_info = nullptr;
  if (!data.findEntry(keyPart, &file_info)) {
    // Key does not already exist.
    if (!StorageGeneration::IsUnknown(options.if_equal) &&
        !StorageGeneration::IsNoValue(options.if_equal)) {
      // Write is conditioned on there being an existing key with a
      // generation of `if_equal`.  Abort.
      return GenerationNow(StorageGeneration::Unknown());
    }
    if (!value) {
      // Delete was requested, but key already doesn't exist.
      return GenerationNow(StorageGeneration::NoValue());
    }

    // Add the key/value pair to the zip file
    mz_zip_file uninitialized_file_info;
    mz_zip_file file_info{};
    file_info.version_madeby = MZ_VERSION_MADEBY;
    file_info.compression_method = MZ_COMPRESS_METHOD_DEFLATE;
    file_info.filename = keyPart.c_str();
    file_info.uncompressed_size = value.has_value() ? value.value().size() : 0;
    file_info.aes_version = MZ_AES_VERSION;

    // we leave the compression to another abstraction layer, e.g. Zarr
    err = mz_zip_entry_write_open(data.zip_handle, &file_info,
                                  MZ_COMPRESS_METHOD_STORE, 0, nullptr);

    if (err == MZ_OK) {
      int32_t written = 0;
      if (value.has_value()) {
        written = mz_zip_entry_write(data.zip_handle, value.value().Flatten().data(),
                               file_info.uncompressed_size);
      } else {
        written = mz_zip_entry_write(data.zip_handle, nullptr, 0);
      }
      if (written < MZ_OK) {
        err = written;
      }
      mz_zip_entry_close(data.zip_handle);
    }

    data.closeZip();

    ++data.next_generation_number;
    auto nextGen = StorageGeneration::FromUint64(data.next_generation_number);
    return GenerationNow(nextGen);
  }


  auto& values = data.values;
  auto it = values.find(key);

  // Key already exists.
  if (!value) {
    // Delete request.
    values.erase(it);

    return GenerationNow(StorageGeneration::NoValue());
  }
  // Update
 
  // Set the generation number to the next unused generation number.
  it->second.generation_number = data.next_generation_number++;
  // Update the value.
  it->second.value = std::move(*value);
  return GenerationNow(it->second.generation());
}

Future<const void> ZipDriver::DeleteRange(KeyRange range) {
  auto& data = this->data();
  absl::WriterMutexLock lock(&data.mutex);
  if (!range.empty()) {
    auto it_range = data.Find(range);
    data.values.erase(it_range.first, it_range.second);
  }
  return absl::OkStatus();  // Converted to a ReadyFuture.
}

void ZipDriver::ListImpl(ListOptions options,
                         AnyFlowReceiver<absl::Status, Key> receiver) {
  auto& data = this->data();
  std::atomic<bool> cancelled{false};
  execution::set_starting(receiver, [&cancelled] {
    cancelled.store(true, std::memory_order_relaxed);
  });

  // Collect the keys.
  std::vector<Key> keys;
  {
    absl::ReaderMutexLock lock(&data.mutex);
    auto it_range = data.Find(options.range);
    for (auto it = it_range.first; it != it_range.second; ++it) {
      if (cancelled.load(std::memory_order_relaxed)) break;
      std::string_view key = it->first;
      keys.emplace_back(
          key.substr(std::min(options.strip_prefix_length, key.size())));
    }
  }

  // Send the keys.
  for (auto& key : keys) {
    if (cancelled.load(std::memory_order_relaxed)) break;
    execution::set_value(receiver, std::move(key));
  }
  execution::set_done(receiver);
  execution::set_stopping(receiver);
}

absl::Status ZipDriver::ReadModifyWrite(
    internal::OpenTransactionPtr& transaction, size_t& phase, Key key,
    ReadModifyWriteSource& source) {
  return internal_kvstore::AddReadModifyWrite<TransactionNode>(
      this, transaction, phase, std::move(key), source);
}

absl::Status ZipDriver::TransactionalDeleteRange(
    const internal::OpenTransactionPtr& transaction, KeyRange range) {
  return internal_kvstore::AddDeleteRange<TransactionNode>(this, transaction,
                                                           std::move(range));
}

// TODO: add support for dataURL, like here:
// https://github.com/InsightSoftwareConsortium/itk-wasm/blob/ea4654350076ee7ec6a570b542513ea56fa9baa1/packages/compress-stringify/compress-stringify.cxx#L100-L101
Result<kvstore::Spec> ParseZipUrl(std::string_view url) {
  auto parsed = internal::ParseGenericUri(url);
  assert(parsed.scheme == tensorstore::ZipDriverSpec::id);
  if (!parsed.query.empty()) {
    return absl::InvalidArgumentError("Query string not supported");
  }
  if (!parsed.fragment.empty()) {
    return absl::InvalidArgumentError("Fragment identifier not supported");
  }
  auto driver_spec = internal::MakeIntrusivePtr<ZipDriverSpec>();
  driver_spec->data_.zip_encapsulator =
      Context::Resource<ZipEncapsulatorResource>::DefaultSpec();
  return {std::in_place, std::move(driver_spec),
          internal::PercentDecode(parsed.authority_and_path)};
}

}  // namespace

kvstore::DriverPtr GetZipKeyValueStore() {
  auto ptr = internal::MakeIntrusivePtr<ZipDriver>();
  ptr->spec_.zip_encapsulator =
      Context::Default().GetResource<ZipEncapsulatorResource>().value();
  return ptr;
}

}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::ZipDriver)

// Registers the driver.
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::ZipDriverSpec>
    registration;

const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{tensorstore::ZipDriverSpec::id,
                            tensorstore::ParseZipUrl};
}  // namespace
