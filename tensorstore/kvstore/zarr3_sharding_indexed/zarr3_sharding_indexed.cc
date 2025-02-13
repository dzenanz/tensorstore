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

#include "tensorstore/kvstore/zarr3_sharding_indexed/zarr3_sharding_indexed.h"

#include <stdint.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/cache_key/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/estimate_heap_usage/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/key.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/shard_format.h"
#include "tensorstore/serialization/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/result_sender.h"  // IWYU pragma: keep
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/garbage_collection/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace zarr3_sharding_indexed {

namespace {
using ::tensorstore::internal_kvstore::DeleteRangeEntry;
using ::tensorstore::internal_kvstore::kReadModifyWrite;

// Read-only KvStore adapter that maps read requests to suffix-length byte range
// requests in order to retrieve just the shard index.
//
// This is an implementation detail of `ShardIndexCache`, which relies on
// `KvsBackedCache`.
class ShardIndexKeyValueStore : public kvstore::Driver {
 public:
  explicit ShardIndexKeyValueStore(kvstore::DriverPtr base,
                                   int64_t index_size_in_bytes)
      : base_(std::move(base)), index_size_in_bytes_(index_size_in_bytes) {}

  Future<kvstore::ReadResult> Read(kvstore::Key key,
                                   kvstore::ReadOptions options) override {
    assert(options.byte_range == OptionalByteRangeRequest{});
    options.byte_range =
        OptionalByteRangeRequest::SuffixLength(index_size_in_bytes_);
    return MapFutureError(
        InlineExecutor{},
        [](const absl::Status& status) {
          return internal::ConvertInvalidArgumentToFailedPrecondition(status);
        },
        base_->Read(std::move(key), std::move(options)));
  }

  std::string DescribeKey(std::string_view key) override {
    return tensorstore::StrCat("shard index in ", base_->DescribeKey(key));
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final {
    // No-op
  }

  kvstore::Driver* base() { return base_.get(); }

 private:
  kvstore::DriverPtr base_;
  int64_t index_size_in_bytes_;
};

// Read-only shard index cache.
//
// This is used by ShardedKeYValueStore for read and list requests.
class ShardIndexCache
    : public internal::KvsBackedCache<ShardIndexCache, internal::AsyncCache> {
  using Base = internal::KvsBackedCache<ShardIndexCache, internal::AsyncCache>;

 public:
  using ReadData = ShardIndex;

  class Entry : public Base::Entry {
   public:
    using OwningCache = ShardIndexCache;

    size_t ComputeReadDataSizeInBytes(const void* read_data) override {
      const auto& cache = GetOwningCache(*this);
      return read_data
                 ? cache.shard_index_params().num_entries * sizeof(uint64_t) * 2
                 : 0;
    }

    std::string GetKeyValueStoreKey() override {
      return GetOwningCache(*this).base_kvstore_path_;
    }

    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override {
      GetOwningCache(*this).executor()(
          [this, value = std::move(value),
           receiver = std::move(receiver)]() mutable {
            std::shared_ptr<ReadData> read_data;
            if (value) {
              TENSORSTORE_ASSIGN_OR_RETURN(
                  auto shard_index,
                  DecodeShardIndex(*value,
                                   GetOwningCache(*this).shard_index_params()),
                  static_cast<void>(execution::set_error(receiver, _)));
              read_data = std::make_shared<ReadData>(std::move(shard_index));
            }
            execution::set_value(receiver, std::move(read_data));
          });
    }
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    ABSL_UNREACHABLE();
  }

  explicit ShardIndexCache(kvstore::DriverPtr base_kvstore,
                           std::string base_kvstore_path, Executor executor,
                           ShardIndexParameters&& params)
      : Base(kvstore::DriverPtr(new ShardIndexKeyValueStore(
            std::move(base_kvstore),
            params.index_codec_state->encoded_size()))),
        base_kvstore_path_(std::move(base_kvstore_path)),
        executor_(std::move(executor)),
        shard_index_params_(std::move(params)) {}

  ShardIndexKeyValueStore* shard_index_kvstore_driver() {
    return static_cast<ShardIndexKeyValueStore*>(this->Base::kvstore_driver());
  }

  kvstore::Driver* base_kvstore_driver() {
    return shard_index_kvstore_driver()->base();
  }
  const std::string& base_kvstore_path() const { return base_kvstore_path_; }
  const Executor& executor() { return executor_; }
  span<const Index> grid_shape() const {
    return span<const Index>(shard_index_params_.index_shape.data(),
                             shard_index_params_.index_shape.size() - 1);
  }

  const ShardIndexParameters& shard_index_params() const {
    return shard_index_params_;
  }

  std::string base_kvstore_path_;
  Executor executor_;
  ShardIndexParameters shard_index_params_;
};

/// Cache used to buffer writes.
///
/// Each cache entry correspond to a particular shard.  The entry key directly
/// encodes the uint64 shard number (via `memcpy`).
///
/// This cache is used only for writing, not for reading.  However, in order to
/// update existing non-empty shards, it does read the full contents of the
/// existing shard and store it within the cache entry.  This data is discarded
/// once writeback completes.
class ShardedKeyValueStoreWriteCache
    : public internal::KvsBackedCache<ShardedKeyValueStoreWriteCache,
                                      internal::AsyncCache> {
  using Base = internal::KvsBackedCache<ShardedKeyValueStoreWriteCache,
                                        internal::AsyncCache>;

 public:
  using ReadData = ShardEntries;

  explicit ShardedKeyValueStoreWriteCache(
      internal::CachePtr<ShardIndexCache> shard_index_cache)
      : Base(kvstore::DriverPtr(shard_index_cache->base_kvstore_driver())),
        shard_index_cache_(std::move(shard_index_cache)) {}

  class Entry : public Base::Entry {
   public:
    using OwningCache = ShardedKeyValueStoreWriteCache;

    size_t ComputeReadDataSizeInBytes(const void* data) override {
      return internal::EstimateHeapUsage(*static_cast<const ReadData*>(data));
    }

    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override {
      GetOwningCache(*this).executor()(
          [this, value = std::move(value),
           receiver = std::move(receiver)]() mutable {
            ShardEntries entries;
            const auto& shard_index_params =
                GetOwningCache(*this).shard_index_params();
            if (value) {
              TENSORSTORE_ASSIGN_OR_RETURN(
                  entries, DecodeShard(*value, shard_index_params),
                  static_cast<void>(execution::set_error(receiver, _)));
            } else {
              // Initialize empty shard.
              entries.entries.resize(shard_index_params.num_entries);
            }
            execution::set_value(
                receiver, std::make_shared<ShardEntries>(std::move(entries)));
          });
    }

    void DoEncode(std::shared_ptr<const ShardEntries> data,
                  EncodeReceiver receiver) override {
      // Can call `EncodeShard` synchronously without using our executor since
      // `DoEncode` is already guaranteed to be called from our executor.
      execution::set_value(
          receiver,
          EncodeShard(*data, GetOwningCache(*this).shard_index_params()));
    }

    std::string GetKeyValueStoreKey() override {
      return GetOwningCache(*this).base_kvstore_path();
    }
  };

  class TransactionNode : public Base::TransactionNode,
                          public internal_kvstore::AtomicMultiPhaseMutation {
   public:
    using OwningCache = ShardedKeyValueStoreWriteCache;
    using Base::TransactionNode::TransactionNode;

    absl::Mutex& mutex() override { return this->mutex_; }

    void PhaseCommitDone(size_t next_phase) override {}

    internal::TransactionState::Node& GetTransactionNode() override {
      return *this;
    }

    void Abort() override {
      this->AbortRemainingPhases();
      Base::TransactionNode::Abort();
    }

    std::string DescribeKey(std::string_view key) override {
      auto& cache = GetOwningCache(*this);
      return tensorstore::StrCat(
          DescribeInternalKey(key, cache.shard_index_params().grid_shape()),
          " in ",
          cache.kvstore_driver()->DescribeKey(cache.base_kvstore_path()));
    }

    // Computes a new decoded shard that takes into account all mutations to the
    // shard requested by this transaction.
    //
    // The implementation makes use of `AtomicMultiPhaseMutation`, which keeps
    // track of the mutations.
    //
    // Multiple concurrent calls to `DoApply` are not allowed.  That constraint
    // is respected by `KvsBackedCache`, which is the only caller of this
    // function.
    void DoApply(ApplyOptions options, ApplyReceiver receiver) override;

    // Starts or retries applying.
    void StartApply();

    // Called by `AtomicMultiPhaseMutation` as part of `DoApply` once all
    // individual read-modify-write mutations for this shard have computed their
    // conditional update.
    //
    // This determines if:
    //
    // (a) all conditional updates are compatible (all depend on the same shard
    //     generation);
    //
    // (b) the existing content of the shard will be required.  The existing
    //     content is not required if, and only if, all shard entries are
    //     unconditionally overwritten.
    //
    // Based on those two conditions, one of 3 actions are taken:
    //
    // 1. If conditional updates are inconsistent, calls `StartApply` to
    //    re-request writeback from each individual read-modify-write operation
    //    with a more recent staleness bound.
    //
    // 2. If the existing shard content is required, reads the existing shard
    //    content, and then calls `MergeForWriteback` once it is ready.
    //
    // 3. If all entries are overwritten unconditionally, just calls
    //    `MergeForWriteback` immediately.
    void AllEntriesDone(
        internal_kvstore::SinglePhaseMutation& single_phase_mutation) override;

    // Attempts to compute the new encoded shard state that merges any
    // mutations into the existing state, completing the `DoApply` operation.
    //
    // This is used by the implementation of `DoApply`.
    //
    // If all conditional mutations are based on a consistent existing state
    // generation, sends the new state to `apply_receiver_`.  Otherwise, calls
    // `StartApply` to retry with an updated existing state.
    void MergeForWriteback(bool conditional);

    // Called by `AtomicMultiPhaseMutation` if an error occurs for an individual
    // read-modify-write operation.
    //
    // This is used by the implementation of `DoApply`.
    //
    // This simply records the error in `apply_status_`, and then propagates it
    // to the `apply_receiver_` once `AllEntriesDone` is called.
    void RecordEntryWritebackError(
        internal_kvstore::ReadModifyWriteEntry& entry,
        absl::Status error) override {
      absl::MutexLock lock(&mutex_);
      if (apply_status_.ok()) {
        apply_status_ = std::move(error);
      }
    }

    void Revoke() override {
      Base::TransactionNode::Revoke();
      { UniqueWriterLock(*this); }
      // At this point, no new entries may be added and we can safely traverse
      // the list of entries without a lock.
      this->RevokeAllEntries();
    }

    void WritebackSuccess(ReadState&& read_state) override;
    void WritebackError() override;

    void InvalidateReadState() override;

    bool MultiPhaseReadsCommitted() override { return this->reads_committed_; }

    /// Handles transactional read requests for single chunks.
    ///
    /// Always reads the full shard, and then decodes the individual chunk
    /// within it.
    void Read(
        internal_kvstore::ReadModifyWriteEntry& entry,
        kvstore::ReadModifyWriteTarget::TransactionalReadOptions&& options,
        kvstore::ReadModifyWriteTarget::ReadReceiver&& receiver) override {
      this->AsyncCache::TransactionNode::Read(options.staleness_bound)
          .ExecuteWhenReady(WithExecutor(
              GetOwningCache(*this).executor(),
              [&entry, if_not_equal = std::move(options.if_not_equal),
               receiver = std::move(receiver)](
                  ReadyFuture<const void> future) mutable {
                if (!future.result().ok()) {
                  execution::set_error(receiver, future.result().status());
                  return;
                }
                execution::submit(HandleShardReadSuccess(entry, if_not_equal),
                                  receiver);
              }));
    }

    /// Called asynchronously from `Read` when the full shard is ready.
    static Result<kvstore::ReadResult> HandleShardReadSuccess(
        internal_kvstore::ReadModifyWriteEntry& entry,
        const StorageGeneration& if_not_equal) {
      auto& self = static_cast<TransactionNode&>(entry.multi_phase());
      kvstore::ReadResult read_result;
      std::shared_ptr<const ShardEntries> entries;
      {
        AsyncCache::ReadLock<ShardEntries> lock{self};
        read_result.stamp = lock.stamp();
        entries = lock.shared_data();
      }
      if (!StorageGeneration::IsUnknown(read_result.stamp.generation) &&
          read_result.stamp.generation == if_not_equal) {
        read_result.state = kvstore::ReadResult::kUnspecified;
      } else {
        auto entry_id = InternalKeyToEntryId(entry.key_);
        const auto& entry = entries->entries[entry_id];
        if (!entry) {
          read_result.state = kvstore::ReadResult::kMissing;
        } else {
          read_result.state = kvstore::ReadResult::kValue;
          read_result.value = *entry;
        }
        if (StorageGeneration::IsDirty(read_result.stamp.generation)) {
          // Add layer to generation in order to make it possible to
          // distinguish:
          //
          // 1. the shard being modified by a predecessor
          //    `ReadModifyWrite` operation on the underlying
          //    KeyValueStore.
          //
          // 2. the chunk being modified by a `ReadModifyWrite`
          //    operation attached to this transaction node.
          read_result.stamp.generation = StorageGeneration::AddLayer(
              std::move(read_result.stamp.generation));
        }
      }
      return read_result;
    }

    // Staleness bound for the current pending call to `DoApply`.
    absl::Time apply_staleness_bound_;

    // The receiver argument for the current pending call to `DoApply`.
    ApplyReceiver apply_receiver_;

    // Error status for the current pending call to `DoApply`.
    absl::Status apply_status_;
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

  const internal::CachePtr<ShardIndexCache>& shard_index_cache() const {
    return shard_index_cache_;
  }

  const Executor& executor() { return shard_index_cache()->executor(); }

  const ShardIndexParameters& shard_index_params() const {
    return shard_index_cache_->shard_index_params();
  }

  int64_t num_entries_per_shard() const {
    return shard_index_cache_->shard_index_params().num_entries;
  }
  const std::string& base_kvstore_path() const {
    return shard_index_cache_->base_kvstore_path();
  }

  internal::CachePtr<ShardIndexCache> shard_index_cache_;
};

void ShardedKeyValueStoreWriteCache::TransactionNode::InvalidateReadState() {
  Base::TransactionNode::InvalidateReadState();
  internal_kvstore::InvalidateReadState(phases_);
}

void ShardedKeyValueStoreWriteCache::TransactionNode::DoApply(
    ApplyOptions options, ApplyReceiver receiver) {
  apply_receiver_ = std::move(receiver);
  apply_staleness_bound_ = options.staleness_bound;
  apply_status_ = absl::OkStatus();

  GetOwningCache(*this).executor()([this] { this->StartApply(); });
}

void ShardedKeyValueStoreWriteCache::TransactionNode::StartApply() {
  RetryAtomicWriteback(phases_, apply_staleness_bound_);
}

void ShardedKeyValueStoreWriteCache::TransactionNode::AllEntriesDone(
    internal_kvstore::SinglePhaseMutation& single_phase_mutation) {
  if (!apply_status_.ok()) {
    execution::set_error(std::exchange(apply_receiver_, {}),
                         std::exchange(apply_status_, {}));
    return;
  }
  auto& self = *this;
  GetOwningCache(*this).executor()([&self] {
    TimestampedStorageGeneration stamp;
    bool mismatch = false;

    int64_t num_entries = 0;
    auto& cache = GetOwningCache(self);
    const int64_t num_entries_per_shard = cache.num_entries_per_shard();

    // Determine if the writeback result depends on the existing shard, and if
    // all entries are conditioned on the same generation.
    for (auto& entry : self.phases_.entries_) {
      if (entry.entry_type() != kReadModifyWrite) {
        auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
        auto [begin_id, end_id] = InternalKeyRangeToEntryRange(
            dr_entry.key_, dr_entry.exclusive_max_, num_entries_per_shard);
        num_entries += end_id - begin_id;
        continue;
      }
      auto& buffered_entry =
          static_cast<AtomicMultiPhaseMutation::BufferedReadModifyWriteEntry&>(
              entry);
      auto& entry_stamp = buffered_entry.read_result_.stamp;
      ++num_entries;
      if (StorageGeneration::IsConditional(entry_stamp.generation)) {
        if (!StorageGeneration::IsUnknown(stamp.generation) &&
            StorageGeneration::Clean(stamp.generation) !=
                StorageGeneration::Clean(entry_stamp.generation)) {
          mismatch = true;
          break;
        } else {
          stamp = entry_stamp;
        }
      }
    }

    if (mismatch) {
      // Retry with newer staleness bound to try to obtain consistent
      // conditions.
      self.apply_staleness_bound_ = absl::Now();
      self.StartApply();
      return;
    }
    if (!StorageGeneration::IsUnknown(stamp.generation) ||
        num_entries != num_entries_per_shard) {
      self.internal::AsyncCache::TransactionNode::Read(
              self.apply_staleness_bound_)
          .ExecuteWhenReady([&self](ReadyFuture<const void> future) {
            if (!future.result().ok()) {
              execution::set_error(std::exchange(self.apply_receiver_, {}),
                                   future.result().status());
              return;
            }
            GetOwningCache(self).executor()(
                [&self] { self.MergeForWriteback(/*conditional=*/true); });
          });
      return;
    }
    self.MergeForWriteback(/*conditional=*/false);
  });
}

void ShardedKeyValueStoreWriteCache::TransactionNode::MergeForWriteback(
    bool conditional) {
  TimestampedStorageGeneration stamp;
  ShardEntries new_entries;
  if (conditional) {
    // The new shard state depends on the existing shard state.  We will need to
    // merge the mutations with the existing chunks.  Additionally, any
    // conditional mutations must be consistent with `stamp.generation`.
    auto lock = internal::AsyncCache::ReadLock<ShardEntries>{*this};
    stamp = lock.stamp();
    new_entries = *lock.shared_data();
  } else {
    // The new shard state is guaranteed not to depend on the existing shard
    // state.  We will merge the mutations into an empty set of existing chunks.
    stamp = TimestampedStorageGeneration::Unconditional();
  }

  auto& cache = GetOwningCache(*this);
  const int64_t num_entries_per_shard = cache.num_entries_per_shard();
  const bool has_existing_entries = !new_entries.entries.empty();
  new_entries.entries.resize(num_entries_per_shard);
  // Indicates that inconsistent conditional mutations were observed.
  bool mismatch = false;
  // Indicates that the new shard state is not identical to the existing shard
  // state.
  bool changed = false;
  for (auto& entry : phases_.entries_) {
    if (entry.entry_type() != kReadModifyWrite) {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
      auto [begin_id, end_id] = InternalKeyRangeToEntryRange(
          dr_entry.key_, dr_entry.exclusive_max_, num_entries_per_shard);
      if (has_existing_entries) {
        for (EntryId id = begin_id; id < end_id; ++id) {
          new_entries.entries[id] = std::nullopt;
        }
      }
      changed = true;
      continue;
    }

    auto& buffered_entry =
        static_cast<internal_kvstore::AtomicMultiPhaseMutation::
                        BufferedReadModifyWriteEntry&>(entry);
    if (StorageGeneration::IsConditional(
            buffered_entry.read_result_.stamp.generation) &&
        StorageGeneration::Clean(
            buffered_entry.read_result_.stamp.generation) !=
            StorageGeneration::Clean(stamp.generation)) {
      // This mutation is conditional, and is inconsistent with a prior
      // conditional mutation or with `existing_chunks`.
      mismatch = true;
      break;
    }
    if (buffered_entry.read_result_.state ==
            kvstore::ReadResult::kUnspecified ||
        !StorageGeneration::IsInnerLayerDirty(
            buffered_entry.read_result_.stamp.generation)) {
      // This is a no-op mutation; ignore it, which has the effect of retaining
      // the existing entry with this id, if present.
      continue;
    }
    auto entry_id = InternalKeyToEntryId(buffered_entry.key_);
    auto& new_entry = new_entries.entries[entry_id];
    if (buffered_entry.read_result_.state == kvstore::ReadResult::kValue) {
      new_entry = buffered_entry.read_result_.value;
      changed = true;
    } else if (new_entry) {
      new_entry = std::nullopt;
      changed = true;
    } else if (!conditional) {
      changed = true;
    }
  }
  if (mismatch) {
    // We can't proceed, because the existing entries (if applicable) and the
    // conditional mutations are not all based on a consistent existing shard
    // generation.  Retry, requesting that all mutations be based on a new
    // up-to-date existing shard generation, which will normally lead to a
    // consistent set.  In the rare case where there are both concurrent
    // external modifications to the existing shard, and also concurrent
    // requests for updated writeback states, multiple retries may be required.
    // However, even in that case, the extra retries aren't really an added
    // cost, because those retries would still be needed anyway due to the
    // concurrent modifications.
    apply_staleness_bound_ = absl::Now();
    this->StartApply();
    return;
  }
  internal::AsyncCache::ReadState update;
  update.stamp = std::move(stamp);
  if (changed) {
    update.stamp.generation.MarkDirty();
  }
  update.data = std::make_shared<ShardEntries>(std::move(new_entries));
  execution::set_value(std::exchange(apply_receiver_, {}), std::move(update));
}

void ShardedKeyValueStoreWriteCache::TransactionNode::WritebackSuccess(
    ReadState&& read_state) {
  for (auto& entry : phases_.entries_) {
    if (entry.entry_type() != kReadModifyWrite) {
      internal_kvstore::WritebackSuccess(static_cast<DeleteRangeEntry&>(entry));
    } else {
      internal_kvstore::WritebackSuccess(
          static_cast<internal_kvstore::ReadModifyWriteEntry&>(entry),
          read_state.stamp);
    }
  }
  internal_kvstore::DestroyPhaseEntries(phases_);
  Base::TransactionNode::WritebackSuccess(std::move(read_state));
}

void ShardedKeyValueStoreWriteCache::TransactionNode::WritebackError() {
  internal_kvstore::WritebackError(phases_);
  internal_kvstore::DestroyPhaseEntries(phases_);
  Base::TransactionNode::WritebackError();
}

struct ShardedKeyValueStoreSpecData {
  Context::Resource<internal::CachePoolResource> cache_pool;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;
  kvstore::Spec base;
  std::vector<Index> grid_shape;
  internal_zarr3::ZarrCodecChainSpec index_codecs;
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ShardedKeyValueStoreSpecData,
                                          internal_json_binding::NoOptions,
                                          IncludeDefaults,
                                          ::nlohmann::json::object_t)

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.cache_pool, x.data_copy_concurrency, x.base, x.grid_shape,
             x.index_codecs);
  };
};

namespace jb = ::tensorstore::internal_json_binding;

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ShardedKeyValueStoreSpecData,
    jb::Object(
        jb::Member("base",
                   jb::Projection<&ShardedKeyValueStoreSpecData::base>()),
        jb::Member(
            "grid_shape",
            jb::Projection<&ShardedKeyValueStoreSpecData::grid_shape>(
                jb::Validate([](const auto& options,
                                auto* obj) { return ValidateGridShape(*obj); },
                             jb::ChunkShapeVector(nullptr)))),
        jb::Member("index_codecs",
                   jb::Projection<&ShardedKeyValueStoreSpecData::index_codecs>(
                       internal_zarr3::ZarrCodecChainJsonBinder<
                           /*Constraints=*/false>)),
        jb::Member(internal::CachePoolResource::id,
                   jb::Projection<&ShardedKeyValueStoreSpecData::cache_pool>()),
        jb::Member(
            internal::DataCopyConcurrencyResource::id,
            jb::Projection<
                &ShardedKeyValueStoreSpecData::data_copy_concurrency>())));

class ShardedKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<
          ShardedKeyValueStoreSpec, ShardedKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "zarr3_sharding_indexed";
  Future<kvstore::DriverPtr> DoOpen() const override;

  Result<kvstore::Spec> GetBase(std::string_view path) const override {
    return data_.base;
  }
};

class ShardedKeyValueStore
    : public internal_kvstore::RegisteredDriver<ShardedKeyValueStore,
                                                ShardedKeyValueStoreSpec> {
 public:
  // Constructs the key value store.
  //
  // Args:
  //   params: Parmaeters for opening.
  //   shared_cache_key: Cache key to use.  Set when opened directly from a JSON
  //     spec, but not used when opened by the zarr v3 driver.
  explicit ShardedKeyValueStore(ShardedKeyValueStoreParameters&& params,
                                std::string_view shared_cache_key = {});

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  absl::Status ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                               size_t& phase, Key key,
                               ReadModifyWriteSource& source) override;

  Result<internal::OpenTransactionPtr> GetImplicitTransaction(
      const Key& key) override;

  absl::Status TransactionalDeleteRange(
      const internal::OpenTransactionPtr& transaction, KeyRange range) override;

  Future<const void> DeleteRange(KeyRange range) override;

  std::string DescribeKey(std::string_view key) override;

  kvstore::SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final;

  Result<KvStore> GetBase(std::string_view path,
                          const Transaction& transaction) const override;

  kvstore::Driver* base_kvstore_driver() const {
    return shard_index_cache()->base_kvstore_driver();
  }
  const ShardIndexParameters& shard_index_params() const {
    return shard_index_cache()->shard_index_params();
  }
  const Executor& executor() const { return shard_index_cache()->executor(); }
  const std::string& base_kvstore_path() const {
    return shard_index_cache()->base_kvstore_path();
  }

  const internal::CachePtr<ShardIndexCache>& shard_index_cache() const {
    return write_cache_->shard_index_cache_;
  }

  absl::Status GetBoundSpecData(ShardedKeyValueStoreSpecData& spec) const;

  internal::CachePtr<ShardedKeyValueStoreWriteCache> write_cache_;

  struct DataForSpec {
    Context::Resource<internal::CachePoolResource> cache_pool_resource;
    Context::Resource<internal::DataCopyConcurrencyResource>
        data_copy_concurrency_resource;
    ZarrCodecChainSpec index_codecs;
  };
  std::unique_ptr<DataForSpec> data_for_spec_;
};

ShardedKeyValueStore::ShardedKeyValueStore(
    ShardedKeyValueStoreParameters&& params,
    std::string_view shared_cache_key) {
  write_cache_ = params.cache_pool->GetCache<ShardedKeyValueStoreWriteCache>(
      shared_cache_key, [&] {
        return std::make_unique<ShardedKeyValueStoreWriteCache>(
            params.cache_pool->GetCache<ShardIndexCache>("", [&] {
              return std::make_unique<ShardIndexCache>(
                  std::move(params.base_kvstore),
                  std::move(params.base_kvstore_path),
                  std::move(params.executor), std::move(params.index_params));
            }));
      });
}

/// Asynchronous state and associated methods for  `ShardedKeyValueStore::Read`.
struct ReadOperationState {
  internal::PinnedCacheEntry<ShardIndexCache> entry_;
  EntryId entry_id_;
  kvstore::ReadOptions options_;

  static auto OnShardIndexReadyCallback(
      std::unique_ptr<ReadOperationState> self) {
    auto& executor = GetOwningCache(*self->entry_).executor();
    return WithExecutor(executor, [self = std::move(self)](
                                      Promise<kvstore::ReadResult> promise,
                                      ReadyFuture<const void> future) mutable {
      OnShardIndexReady(std::move(self), std::move(promise));
    });
  }

  static Future<kvstore::ReadResult> Start(ShardedKeyValueStore& store,
                                           kvstore::Key&& key,
                                           kvstore::ReadOptions&& options) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        EntryId entry_id,
        KeyToEntryIdOrError(key, store.shard_index_params().grid_shape()));
    auto shard_index_cache_entry =
        GetCacheEntry(store.shard_index_cache(), std::string_view{});
    auto shard_index_read_future =
        shard_index_cache_entry->Read(options.staleness_bound);
    return PromiseFuturePair<kvstore::ReadResult>::LinkValue(
               OnShardIndexReadyCallback(std::unique_ptr<ReadOperationState>(
                   new ReadOperationState{std::move(shard_index_cache_entry),
                                          entry_id, std::move(options)})),
               std::move(shard_index_read_future))
        .future;
  }

  static void OnShardIndexReady(std::unique_ptr<ReadOperationState> self,
                                Promise<kvstore::ReadResult> promise) {
    ShardIndexEntry index_entry = ShardIndexEntry::Missing();
    TimestampedStorageGeneration stamp;
    kvstore::ReadResult::State state;
    {
      auto lock = internal::AsyncCache::ReadLock<ShardIndexCache::ReadData>(
          *self->entry_);
      stamp = lock.stamp();
      if (!StorageGeneration::IsNoValue(stamp.generation) &&
          (self->options_.if_not_equal == stamp.generation ||
           (!StorageGeneration::IsUnknown(self->options_.if_equal) &&
            self->options_.if_equal != stamp.generation))) {
        state = kvstore::ReadResult::kUnspecified;
      } else {
        if (lock.data()) {
          const auto& shard_index = *lock.data();
          index_entry = shard_index[self->entry_id_];
        }
        state = kvstore::ReadResult::kMissing;
      }
    }
    if (index_entry.IsMissing()) {
      promise.SetResult(kvstore::ReadResult{state, {}, std::move(stamp)});
      return;
    }
    assert(!StorageGeneration::IsUnknown(stamp.generation));
    auto& cache = GetOwningCache(*self->entry_);
    assert(self->options_.byte_range.SatisfiesInvariants());
    TENSORSTORE_RETURN_IF_ERROR(
        index_entry.Validate(self->entry_id_),
        static_cast<void>(
            promise.SetResult(self->entry_->AnnotateError(_,
                                                          /*reading=*/true))));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto validated_byte_range,
        self->options_.byte_range.Validate(index_entry.length),
        static_cast<void>(promise.SetResult(_)));
    if (validated_byte_range.inclusive_min ==
        validated_byte_range.exclusive_max) {
      // Zero-length read request, no need to issue actual read.
      promise.SetResult(kvstore::ReadResult{kvstore::ReadResult::kValue,
                                            absl::Cord(), std::move(stamp)});
      return;
    }
    kvstore::ReadOptions kvs_read_options;
    kvs_read_options.if_equal = stamp.generation;
    kvs_read_options.staleness_bound = self->options_.staleness_bound;
    kvs_read_options.byte_range =
        ByteRange{static_cast<int64_t>(index_entry.offset +
                                       validated_byte_range.inclusive_min),
                  static_cast<int64_t>(index_entry.offset +
                                       validated_byte_range.exclusive_max)};
    LinkValue(
        [self = std::move(self)](
            Promise<kvstore::ReadResult> promise,
            ReadyFuture<kvstore::ReadResult> future) mutable {
          OnValueReady(std::move(self), std::move(promise),
                       std::move(future.value()));
        },
        std::move(promise),
        cache.base_kvstore_driver()->Read(
            std::string(cache.base_kvstore_path()),
            std::move(kvs_read_options)));
  }

  static void OnValueReady(std::unique_ptr<ReadOperationState> self,
                           Promise<kvstore::ReadResult> promise,
                           kvstore::ReadResult&& value) {
    if (value.aborted()) {
      // Concurrent modification.  Retry.
      auto shard_index_read_future =
          self->entry_->Read(/*staleness_bound=*/value.stamp.time);
      LinkValue(OnShardIndexReadyCallback(std::move(self)), std::move(promise),
                std::move(shard_index_read_future));
      return;
    }
    promise.SetResult(std::move(value));
  }
};

Future<kvstore::ReadResult> ShardedKeyValueStore::Read(Key key,
                                                       ReadOptions options) {
  return ReadOperationState::Start(*this, std::move(key), std::move(options));
}

// Asynchronous operation state for `ShardedKeyValueStore::ListImpl`.
struct ListOperationState {
  AnyFlowReceiver<absl::Status, kvstore::Key> receiver_;
  internal::PinnedCacheEntry<ShardIndexCache> shard_index_cache_entry_;
  Promise<void> promise_;
  Future<void> future_;
  kvstore::ListOptions options_;

  static void Start(ShardedKeyValueStore& store, kvstore::ListOptions&& options,
                    AnyFlowReceiver<absl::Status, kvstore::Key>&& receiver) {
    options.range = KeyRangeToInternalKeyRange(
        options.range, store.shard_index_params().grid_shape());
    auto self = std::make_unique<ListOperationState>();
    self->receiver_ = std::move(receiver);
    auto [promise, future] = PromiseFuturePair<void>::Make(MakeResult());
    self->promise_ = std::move(promise);
    self->future_ = std::move(future);
    self->future_.Force();
    execution::set_starting(self->receiver_, [promise = self->promise_] {
      promise.SetResult(absl::CancelledError(""));
    });
    self->options_ = std::move(options);
    self->shard_index_cache_entry_ =
        GetCacheEntry(store.shard_index_cache(), std::string_view{});
    auto shard_index_read_future =
        self->shard_index_cache_entry_->Read(self->options_.staleness_bound);
    auto* self_ptr = self.get();
    LinkValue(
        WithExecutor(store.executor(),
                     [self = std::move(self)](Promise<void> promise,
                                              ReadyFuture<const void> future) {
                       self->OnShardIndexReady();
                     }),
        self_ptr->promise_, std::move(shard_index_read_future));
  }

  void OnShardIndexReady() {
    auto shard_index =
        internal::AsyncCache::ReadLock<ShardIndex>(*shard_index_cache_entry_)
            .shared_data();
    if (!shard_index) {
      // Shard is not present, no entries to list.
      return;
    }
    const auto& shard_index_params =
        GetOwningCache(*shard_index_cache_entry_).shard_index_params();
    span<const Index> grid_shape = shard_index_params.grid_shape();
    auto start_index = InternalKeyToEntryId(options_.range.inclusive_min);
    auto end_index = InternalKeyToEntryId(options_.range.exclusive_max);
    for (EntryId i = start_index; i < end_index; ++i) {
      auto index_entry = (*shard_index)[i];
      if (index_entry.IsMissing()) continue;
      auto key = EntryIdToKey(i, grid_shape);
      key.erase(0, options_.strip_prefix_length);
      execution::set_value(receiver_, std::move(key));
    }
  }

  ~ListOperationState() {
    auto& r = promise_.raw_result();
    if (r.ok()) {
      execution::set_done(receiver_);
    } else {
      execution::set_error(receiver_, r.status());
    }
    execution::set_stopping(receiver_);
  }
};

void ShardedKeyValueStore::ListImpl(
    ListOptions options, AnyFlowReceiver<absl::Status, Key> receiver) {
  ListOperationState::Start(*this, std::move(options), std::move(receiver));
}

Future<TimestampedStorageGeneration> ShardedKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  return internal_kvstore::WriteViaTransaction(
      this, std::move(key), std::move(value), std::move(options));
}

absl::Status ShardedKeyValueStore::ReadModifyWrite(
    internal::OpenTransactionPtr& transaction, size_t& phase, Key key,
    ReadModifyWriteSource& source) {
  // Convert key to internal representation used in transaction data
  // structures.
  TENSORSTORE_ASSIGN_OR_RETURN(
      EntryId entry_id,
      KeyToEntryIdOrError(key, shard_index_params().grid_shape()));
  key = EntryIdToInternalKey(entry_id);
  // We only use the single cache entry with an empty string.
  auto entry = GetCacheEntry(write_cache_, std::string_view{});
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, GetWriteLockedTransactionNode(*entry, transaction));
  node->ReadModifyWrite(phase, std::move(key), source);
  if (!transaction) {
    // User did not specify a transaction.  Return the implicit transaction
    // that was created.
    transaction.reset(node.unlock()->transaction());
  }
  return absl::OkStatus();
}

Result<internal::OpenTransactionPtr>
ShardedKeyValueStore::GetImplicitTransaction(const Key& key) {
  auto entry = GetCacheEntry(write_cache_, std::string_view{});
  internal::OpenTransactionPtr transaction;
  TENSORSTORE_ASSIGN_OR_RETURN(auto node,
                               GetTransactionNode(*entry, transaction));
  return transaction;
}

absl::Status ShardedKeyValueStore::TransactionalDeleteRange(
    const internal::OpenTransactionPtr& transaction, KeyRange range) {
  range = KeyRangeToInternalKeyRange(range, shard_index_params().grid_shape());
  auto entry = GetCacheEntry(write_cache_, std::string_view{});
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, GetWriteLockedTransactionNode(*entry, transaction));
  node->DeleteRange(std::move(range));
  return absl::OkStatus();
}

Future<const void> ShardedKeyValueStore::DeleteRange(KeyRange range) {
  range = KeyRangeToInternalKeyRange(range, shard_index_params().grid_shape());
  internal::OpenTransactionPtr transaction;
  auto entry = GetCacheEntry(write_cache_, std::string_view{});
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, GetWriteLockedTransactionNode(*entry, transaction));
  node->DeleteRange(std::move(range));
  return node->transaction()->future();
}

std::string ShardedKeyValueStore::DescribeKey(std::string_view key) {
  return tensorstore::StrCat(
      zarr3_sharding_indexed::DescribeKey(key,
                                          shard_index_params().grid_shape()),
      " in ", base_kvstore_driver()->DescribeKey(base_kvstore_path()));
}

kvstore::SupportedFeatures ShardedKeyValueStore::GetSupportedFeatures(
    const KeyRange& key_range) const {
  return base_kvstore_driver()->GetSupportedFeatures(
      KeyRange::Singleton(base_kvstore_path()));
}

Result<KvStore> ShardedKeyValueStore::GetBase(
    std::string_view path, const Transaction& transaction) const {
  return KvStore(kvstore::DriverPtr(base_kvstore_driver()), base_kvstore_path(),
                 transaction);
}
}  // namespace
}  // namespace zarr3_sharding_indexed

namespace garbage_collection {
template <>
struct GarbageCollection<zarr3_sharding_indexed::ShardedKeyValueStore> {
  static void Visit(GarbageCollectionVisitor& visitor,
                    const zarr3_sharding_indexed::ShardedKeyValueStore& value) {
    garbage_collection::GarbageCollectionVisit(visitor,
                                               *value.base_kvstore_driver());
  }
};
}  // namespace garbage_collection

namespace zarr3_sharding_indexed {

absl::Status ShardedKeyValueStore::GetBoundSpecData(
    ShardedKeyValueStoreSpecData& spec) const {
  if (!data_for_spec_) {
    return absl::UnimplementedError("");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(spec.base.driver,
                               base_kvstore_driver()->GetBoundSpec());
  spec.base.path = base_kvstore_path();
  spec.data_copy_concurrency = data_for_spec_->data_copy_concurrency_resource;
  spec.cache_pool = data_for_spec_->cache_pool_resource;
  spec.index_codecs = data_for_spec_->index_codecs;
  const auto& shard_index_params = this->shard_index_params();
  spec.grid_shape.assign(shard_index_params.index_shape.begin(),
                         shard_index_params.index_shape.end() - 1);
  return absl::OkStatus();
}

Future<kvstore::DriverPtr> ShardedKeyValueStoreSpec::DoOpen() const {
  ShardIndexParameters index_params;
  TENSORSTORE_RETURN_IF_ERROR(
      index_params.Initialize(data_.index_codecs, data_.grid_shape));
  return MapFutureValue(
      InlineExecutor{},
      [spec = internal::IntrusivePtr<const ShardedKeyValueStoreSpec>(this),
       index_params =
           std::move(index_params)](kvstore::KvStore& base_kvstore) mutable
      -> Result<kvstore::DriverPtr> {
        std::string cache_key;
        internal::EncodeCacheKey(
            &cache_key, base_kvstore.driver, base_kvstore.path,
            spec->data_.data_copy_concurrency, spec->data_.grid_shape,
            spec->data_.index_codecs);
        ShardedKeyValueStoreParameters params;
        params.base_kvstore = std::move(base_kvstore.driver);
        params.base_kvstore_path = std::move(base_kvstore.path);
        params.executor = spec->data_.data_copy_concurrency->executor;
        params.cache_pool = *spec->data_.cache_pool;
        params.index_params = std::move(index_params);
        auto driver = internal::MakeIntrusivePtr<ShardedKeyValueStore>(
            std::move(params), cache_key);
        driver->data_for_spec_.reset(new ShardedKeyValueStore::DataForSpec{
            spec->data_.cache_pool,
            spec->data_.data_copy_concurrency,
            spec->data_.index_codecs,
        });
        return driver;
      },
      kvstore::Open(data_.base));
}

kvstore::DriverPtr GetShardedKeyValueStore(
    ShardedKeyValueStoreParameters&& parameters) {
  return kvstore::DriverPtr(new ShardedKeyValueStore(std::move(parameters)));
}

}  // namespace zarr3_sharding_indexed
}  // namespace tensorstore

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::zarr3_sharding_indexed::ShardedKeyValueStoreSpec>
    registration;
}  // namespace
