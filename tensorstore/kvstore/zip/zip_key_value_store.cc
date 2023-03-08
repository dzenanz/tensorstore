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
#include <limits>
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
#include "mz.h"
#include "mz_os.h"  // for MZ_VERSION_MADEBY
#include "mz_strm.h"
#include "mz_strm_buf.h"
#include "mz_strm_mem.h"
#include "mz_strm_os.h"  // for testing success
#include "mz_strm_split.h"
#include "mz_zip.h"
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

namespace {
using BufferInfo = struct {
  char* pointer;
  size_t size;
};

/// Which part of this path/url is file name, and which is key within it.
/// Returns true if zip filename was successfully determined.
bool getZipFileFromKey(const std::string& key, std::string& file_part,
                       std::string& key_part) {
  std::string::size_type pos = key.find(".zip");
  if (pos == std::string::npos) {
    // .zip not found
    file_part = "tsTest.zip";  // for testing
    key_part = key;
    return false;
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

/// Which part of this path is memory address, and which is key within it.
/// Returns true if memory address was successfully determined.
bool getMemoryInformationFromKey(const std::string& key,
                                 BufferInfo** bufferInfo, std::string& key_part,
                                 std::string& addressString) {
  std::string::size_type pos = key.find(".memory");
  if (pos == std::string::npos) {
    return false;
  }

  addressString = key.substr(0, pos);  // exclude .memory
  key_part = key.substr(pos + 8);      // skip separator
  size_t address = std::stoull(addressString);
  *bufferInfo = reinterpret_cast<BufferInfo*>(address);

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

  absl::Mutex mutex;
  /// TensorStore wants these generation numbers, this is used to provide them.
  uint64_t next_generation_number ABSL_GUARDED_BY(mutex) = 0;

  /// ZipEncapsulator ivars.
  // mz_zip_file file_info ABSL_GUARDED_BY(mutex){};
  void* mem_stream ABSL_GUARDED_BY(mutex) = nullptr;
  void* file_stream ABSL_GUARDED_BY(mutex) = nullptr;
  void* buffered_stream ABSL_GUARDED_BY(mutex) = nullptr;
  void* split_stream ABSL_GUARDED_BY(mutex) = nullptr;
  void* zip_handle ABSL_GUARDED_BY(mutex) = nullptr;
  BufferInfo* bufferInfo ABSL_GUARDED_BY(mutex){};
  bool memoryWasAllocated ABSL_GUARDED_BY(mutex) = false;
  std::string openedFileName ABSL_GUARDED_BY(mutex);

  ~ZipEncapsulator() {
    absl::WriterMutexLock lock(&mutex);  // Is this needed?
    closeZip();
  }

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
      if (memoryWasAllocated) {
        updateBufferInfo();
        // mz_stream_mem_delete would delete the output buffer too
        MZ_FREE(mem_stream);
        mem_stream = nullptr;
        memoryWasAllocated = false;
      } else {
        mz_stream_mem_delete(&mem_stream);
      }
    }
    openedFileName.resize(0);
  }

  bool openZipFromFile(const char* fileName, int32_t openMode) {
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

    openedFileName = fileName;
    return true;
  }

  bool openZipFileForWriting(const char* fileName, int32_t openMode) {
    std::filesystem::path file(fileName);
    if (!std::filesystem::is_directory(file.parent_path())) {
      // create the directory first
      mz_dir_make(file.parent_path().string().c_str());
    }

    openMode |= MZ_OPEN_MODE_CREATE;

    return openZipFromFile(fileName, openMode);
  }

  void updateBufferInfo() {
    if (mem_stream) {
      mz_stream_mem_get_buffer(mem_stream, (const void**)&bufferInfo->pointer);
      int32_t len;
      mz_stream_mem_get_buffer_length(mem_stream, &len);
      bufferInfo->size = len;
    }
  }

  bool openZipFromMemory(int32_t openMode) {
    int32_t err = MZ_OK;

    mz_stream_mem_create(&mem_stream);

    if (openMode == MZ_OPEN_MODE_READ) {
      if (bufferInfo->size > std::numeric_limits<int32_t>::max()) {
        closeZip();
        throw std::runtime_error("In-memory files of up 2GiB are supported.");
      }
      mz_stream_mem_set_buffer(mem_stream, bufferInfo->pointer,
                               bufferInfo->size);
      mz_stream_open(mem_stream, nullptr, MZ_OPEN_MODE_READ);
    } else  // Write
    {
      openMode |= MZ_OPEN_MODE_CREATE;
      mz_stream_mem_set_grow_size(mem_stream, 64 * 1024);  // 64 KiB
      mz_stream_open(mem_stream, nullptr, openMode);
      memoryWasAllocated = true;
    }

    mz_zip_create(&zip_handle);
    err = mz_zip_open(zip_handle, mem_stream, openMode);
    if (err != MZ_OK) {
      closeZip();
      return false;
    }
    updateBufferInfo();

    return true;
  }

  bool openZipViaKey(const std::string& key, std::string& key_part,
                     int32_t openMode) {
    std::string zipFileName;
    if (getZipFileFromKey(key, zipFileName, key_part)) {
      if (zipFileName == openedFileName) return true;  // already open

      if (!openedFileName.empty()) {
        closeZip();  // we need to close the old file
      }

      bool success;
      if (openMode == MZ_OPEN_MODE_READ) {
        success = openZipFromFile(zipFileName.c_str(), openMode);
      } else {
        success = openZipFileForWriting(zipFileName.c_str(), openMode);
      }

      if (!success) {
        throw std::runtime_error("Could not open " + zipFileName);
      }
      return true;
    }

    if (getMemoryInformationFromKey(key, &bufferInfo, key_part, zipFileName)) {
      if (zipFileName == openedFileName) return true;  // already open

      if (!openedFileName.empty()) {
        closeZip();  // we need to close the old file
      }

      if (!openZipFromMemory(openMode)) {
        throw std::runtime_error("Could not open " + key);
      }
      openedFileName = zipFileName;
      return true;
    }
    return false;
  }

  bool findEntry(const std::string& filePath, mz_zip_file** file_info) {
    int32_t err = MZ_OK;
    err = mz_zip_goto_first_entry(zip_handle);

    if (err == MZ_OK) {
      err = mz_zip_entry_get_info(zip_handle, file_info);
    }

    if (err != MZ_OK)  // could be MZ_END_OF_LIST
    {
      return false;
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

  int32_t addZipEntry(std::string key, std::optional<kvstore::Value> value) {
    mz_zip_file file_info;
    memset(&file_info, 0, sizeof(file_info));
    file_info.version_madeby = MZ_VERSION_MADEBY;
    file_info.compression_method = MZ_COMPRESS_METHOD_DEFLATE;
    file_info.filename = key.c_str();
    file_info.uncompressed_size = value.value().size();
    file_info.aes_version = MZ_AES_VERSION;

    // we leave the compression to another abstraction layer, e.g. Zarr
    int32_t err = mz_zip_entry_write_open(zip_handle, &file_info,
                                          MZ_COMPRESS_METHOD_STORE, 0, nullptr);

    if (err == MZ_OK) {
      int32_t written =
          mz_zip_entry_write(zip_handle, value.value().Flatten().data(),
                             file_info.uncompressed_size);

      if (written < MZ_OK) {
        err = written;
      }
      mz_zip_entry_close(zip_handle);
    }
    updateBufferInfo();

    return err;
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
    return ZipEncapsulator::Ptr(new ZipEncapsulator);
  }
  static Spec GetSpec(const Resource&,
                      const internal::ContextSpecBuilder& builder) {
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

    std::string keyPart;
    data.openZipViaKey(entry.key_, keyPart, MZ_OPEN_MODE_READ);

    int32_t err = MZ_OK;
    mz_zip_file* file_info = nullptr;
    if (!data.findEntry(keyPart, &file_info)) {
      // not found
      if (StorageGeneration::IsNoValue(if_equal)) {
        entry.read_result_.stamp.time = commit_time;
        return true;
      } else {
        return false;
      }
    }

    // found
    entry.read_result_.stamp.time = commit_time;
    return true;
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
          throw std::runtime_error("Erasing an entry is not implemented");
        } else {
          assert(rmw_entry.read_result_.state == ReadResult::kValue);
          WriteOptions options;
          auto writeResult = GetZipKeyValueStore().get()->Write(
              rmw_entry.key_, rmw_entry.read_result_.value, options);
          stamp = writeResult.value();
        }
      } else {
        throw std::runtime_error("Erasing entry range is not implemented");
      }
    }
  }
};

Future<ReadResult> ZipDriver::Read(Key key, ReadOptions options) {
  auto& data = this->data();
  absl::ReaderMutexLock lock(&data.mutex);

  std::string keyPart;
  if (!data.openZipViaKey(key, keyPart, MZ_OPEN_MODE_READ)) {
    throw std::runtime_error("Could not open " + key);
  }

  ReadResult result;

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
      throw std::runtime_error("Could not read key " + key + ". Read " +
                               std::to_string(read) + " bytes out of " +
                               std::to_string(length));
    }
  }
  mz_zip_entry_close(data.zip_handle);

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
  auto& data = this->data();
  absl::WriterMutexLock lock(&data.mutex);

  std::string keyPart;
  if (!data.openZipViaKey(key, keyPart, MZ_OPEN_MODE_READWRITE)) {
    throw std::runtime_error("Could not open " + key + " for writing");
  }

  int32_t err = MZ_OK;
  mz_zip_file* entry_file_info = nullptr;
  if (!data.findEntry(keyPart, &entry_file_info)) {
    // Key not found.
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
    err = data.addZipEntry(keyPart, value);

    ++data.next_generation_number;
    auto nextGen = StorageGeneration::FromUint64(data.next_generation_number);
    return GenerationNow(nextGen);
  }
  // Key already exists.
  if (!value) {
    // TODO: Erase
    return GenerationNow(StorageGeneration::NoValue());
  } else {
    // TODO: Update
    ++data.next_generation_number;
    auto nextGen = StorageGeneration::FromUint64(data.next_generation_number);
    return GenerationNow(nextGen);
  }
}

Future<const void> ZipDriver::DeleteRange(KeyRange range) {
  // throw std::runtime_error("Erasing key range is not implemented");
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
  int32_t err = MZ_OK;
  err = mz_zip_goto_first_entry(data.zip_handle);

  mz_zip_file* entry_file_info = nullptr;
  if (err == MZ_OK) {
    err = mz_zip_entry_get_info(data.zip_handle, &entry_file_info);
  }

  std::vector<Key> keys;
  // go through the list of files in the zip archive
  while (err == MZ_OK) {
    if (cancelled.load(std::memory_order_relaxed)) break;

    err = mz_zip_goto_next_entry(data.zip_handle);
    if (err == MZ_OK) {
      err = mz_zip_entry_get_info(data.zip_handle, &entry_file_info);
    }

    if (err == MZ_OK) {
      Key key(entry_file_info->filename);
      if (key >= options.range.inclusive_min &&
          key < options.range.exclusive_max) {
        keys.emplace_back(
            key.substr(std::min(options.strip_prefix_length, key.size())));
      }
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
  return Driver::ReadModifyWrite(transaction, phase, std::move(key), source);
}

absl::Status ZipDriver::TransactionalDeleteRange(
    const internal::OpenTransactionPtr& transaction, KeyRange range) {
  return Driver::TransactionalDeleteRange(transaction, std::move(range));
}

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
