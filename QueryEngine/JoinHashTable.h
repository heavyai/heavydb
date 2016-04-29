/*
 * @file    JoinHashTable.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_JOINHASHTABLE_H
#define QUERYENGINE_JOINHASHTABLE_H

#include "ExpressionRange.h"
#include "../Analyzer/Analyzer.h"
#include "../Chunk/Chunk.h"
#include "../Fragmenter/Fragmenter.h"

#include <llvm/IR/Value.h>
#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <memory>
#include <mutex>

class Executor;

class JoinHashTable {
 public:
  static std::shared_ptr<JoinHashTable> getInstance(const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
                                                    const Catalog_Namespace::Catalog& cat,
                                                    const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                                                    const Data_Namespace::MemoryLevel memory_level,
                                                    const int device_count,
                                                    Executor* executor);

  int64_t getJoinHashBuffer(const ExecutorDeviceType device_type, const int device_id) {
#ifdef HAVE_CUDA
    return device_type == ExecutorDeviceType::CPU ? reinterpret_cast<int64_t>(&cpu_hash_table_buff_[0])
                                                  : gpu_hash_table_buff_[device_id];
#else
    CHECK(device_type == ExecutorDeviceType::CPU);
    return reinterpret_cast<int64_t>(&cpu_hash_table_buff_[0]);
#endif
  }

 private:
  JoinHashTable(const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
                const Analyzer::ColumnVar* col_var,
                const Catalog_Namespace::Catalog& cat,
                const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                const Data_Namespace::MemoryLevel memory_level,
                const ExpressionRange& col_range,
                Executor* executor)
      : qual_bin_oper_(qual_bin_oper),
        cat_(cat),
        query_infos_(query_infos),
        memory_level_(memory_level),
        col_range_(col_range),
        executor_(executor) {
    CHECK(col_range.getType() == ExpressionRangeType::Integer);
  }

  int reify(const int device_count);
  int initHashTableForDevice(const int8_t* col_buff,
                             const size_t num_elements,
                             const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols,
                             const Data_Namespace::MemoryLevel effective_memory_level,
                             const int device_id);
  llvm::Value* codegenSlot(const bool hoist_literals);

  std::shared_ptr<Analyzer::BinOper> qual_bin_oper_;
  const Catalog_Namespace::Catalog& cat_;
  const std::vector<Fragmenter_Namespace::TableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  std::vector<int32_t> cpu_hash_table_buff_;
  std::mutex cpu_hash_table_buff_mutex_;
#ifdef HAVE_CUDA
  std::vector<CUdeviceptr> gpu_hash_table_buff_;
#endif
  ExpressionRange col_range_;
  Executor* executor_;

  friend class Executor;
};

#endif  // QUERYENGINE_JOINHASHTABLE_H
