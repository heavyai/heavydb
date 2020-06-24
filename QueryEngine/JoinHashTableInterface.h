/*
 * Copyright 2019 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef QUERYENGINE_JOINHASHTABLEINTERFACE_H
#define QUERYENGINE_JOINHASHTABLEINTERFACE_H

#include <llvm/IR/Value.h>
#include <cstdint>
#include <set>
#include <string>
#include "Analyzer/Analyzer.h"
#include "ColumnarResults.h"
#include "CompilationOptions.h"
#include "DataMgr/Allocators/ThrustAllocator.h"
#include "Descriptors/RowSetMemoryOwner.h"
#include "HashJoinRuntime.h"

class TooManyHashEntries : public std::runtime_error {
 public:
  TooManyHashEntries()
      : std::runtime_error("Hash tables with more than 2B entries not supported yet") {}
};

class TableMustBeReplicated : public std::runtime_error {
 public:
  TableMustBeReplicated(const std::string& table_name)
      : std::runtime_error("Hash join failed: Table '" + table_name +
                           "' must be replicated.") {}
};

class HashJoinFail : public std::runtime_error {
 public:
  HashJoinFail(const std::string& reason) : std::runtime_error(reason) {}
};

class FailedToFetchColumn : public HashJoinFail {
 public:
  FailedToFetchColumn()
      : HashJoinFail("Not enough memory for columns involved in join") {}
};

class FailedToJoinOnVirtualColumn : public HashJoinFail {
 public:
  FailedToJoinOnVirtualColumn() : HashJoinFail("Cannot join on rowid") {}
};

struct HashJoinMatchingSet {
  llvm::Value* elements;
  llvm::Value* count;
  llvm::Value* slot;
};

struct DecodedJoinHashBufferEntry {
  std::vector<int64_t> key;
  std::set<int32_t> payload;

  bool operator<(const DecodedJoinHashBufferEntry& other) const {
    return std::tie(key, payload) < std::tie(other.key, other.payload);
  }

  bool operator==(const DecodedJoinHashBufferEntry& other) const {
    return key == other.key && payload == other.payload;
  }
};  // struct DecodedJoinHashBufferEntry

using DecodedJoinHashBufferSet = std::set<DecodedJoinHashBufferEntry>;

using InnerOuter = std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>;

class JoinHashTableInterface {
 public:
  virtual int64_t getJoinHashBuffer(const ExecutorDeviceType device_type,
                                    const int device_id = 0) const noexcept = 0;

  virtual size_t getJoinHashBufferSize(const ExecutorDeviceType device_type,
                                       const int device_id = 0) const
      noexcept = 0;  // bytes

  virtual std::string toString(const ExecutorDeviceType device_type,
                               const int device_id = 0,
                               bool raw = false) const = 0;

  virtual std::string toStringFlat64(const ExecutorDeviceType device_type,
                                     const int device_id) const;

  virtual std::string toStringFlat32(const ExecutorDeviceType device_type,
                                     const int device_id) const;

  virtual DecodedJoinHashBufferSet toSet(const ExecutorDeviceType device_type,
                                         const int device_id) const = 0;

  virtual llvm::Value* codegenSlot(const CompilationOptions&, const size_t) = 0;

  virtual HashJoinMatchingSet codegenMatchingSet(const CompilationOptions&,
                                                 const size_t) = 0;

  virtual int getInnerTableId() const noexcept = 0;

  virtual int getInnerTableRteIdx() const noexcept = 0;

  enum class HashType : int { OneToOne, OneToMany, ManyToMany };

  virtual HashType getHashType() const noexcept = 0;

  virtual bool layoutRequiresAdditionalBuffers(
      JoinHashTableInterface::HashType layout) const noexcept = 0;

  static std::string getHashTypeString(HashType ht) noexcept {
    const char* HashTypeStrings[3] = {"OneToOne", "OneToMany", "ManyToMany"};
    return HashTypeStrings[static_cast<int>(ht)];
  };

  virtual Data_Namespace::MemoryLevel getMemoryLevel() const noexcept = 0;

  virtual int getDeviceCount() const noexcept = 0;

  virtual size_t offsetBufferOff() const noexcept = 0;

  virtual size_t countBufferOff() const noexcept = 0;

  virtual size_t payloadBufferOff() const noexcept = 0;

  JoinColumn fetchJoinColumn(
      const Analyzer::ColumnVar* hash_col,
      const std::vector<Fragmenter_Namespace::FragmentInfo>& fragment_info,
      const Data_Namespace::MemoryLevel effective_memory_level,
      const int device_id,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      ThrustAllocator& dev_buff_owner,
      std::vector<std::shared_ptr<void>>& malloc_owner,
      Executor* executor,
      ColumnCacheMap* column_cache);

 public:
  //! Decode hash table into a std::set for easy inspection and validation.
  static DecodedJoinHashBufferSet toSet(
      size_t key_component_count,  // number of key parts
      size_t key_component_width,  // width of a key part
      size_t entry_count,          // number of hashable entries
      const int8_t* ptr1,          // hash entries
      const int8_t* ptr2,          // offsets
      const int8_t* ptr3,          // counts
      const int8_t* ptr4,          // payloads (rowids)
      size_t buffer_size);

  //! Decode hash table into a human-readable string.
  static std::string toString(
      const std::string& type,         // perfect, keyed, or geo
      const std::string& layout_type,  // one-to-one, one-to-many, many-to-many
      size_t key_component_count,      // number of key parts
      size_t key_component_width,      // width of a key part
      size_t entry_count,              // number of hashable entries
      const int8_t* ptr1,              // hash entries
      const int8_t* ptr2,              // offsets
      const int8_t* ptr3,              // counts
      const int8_t* ptr4,              // payloads (rowids)
      size_t buffer_size,
      bool raw = false);

  //! Make hash table from an in-flight SQL query's parse tree etc.
  static std::shared_ptr<JoinHashTableInterface> getInstance(
      const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const HashType preferred_hash_type,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor);

  //! Make hash table from named tables and columns (such as for testing).
  static std::shared_ptr<JoinHashTableInterface> getSyntheticInstance(
      std::string_view table1,
      std::string_view column1,
      std::string_view table2,
      std::string_view column2,
      const Data_Namespace::MemoryLevel memory_level,
      const HashType preferred_hash_type,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor);

  //! Make hash table from named tables and columns (such as for testing).
  static std::shared_ptr<JoinHashTableInterface> getSyntheticInstance(
      const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
      const Data_Namespace::MemoryLevel memory_level,
      const HashType preferred_hash_type,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor);

};  // class JoinHashTableInterface

std::ostream& operator<<(std::ostream& os, const DecodedJoinHashBufferEntry& e);

std::ostream& operator<<(std::ostream& os, const DecodedJoinHashBufferSet& s);

std::shared_ptr<Analyzer::ColumnVar> getSyntheticColumnVar(std::string_view table,
                                                           std::string_view column,
                                                           int rte_idx,
                                                           Executor* executor);

#endif  // QUERYENGINE_JOINHASHTABLEINTERFACE_H
