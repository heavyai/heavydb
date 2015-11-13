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
#include "../Fragmenter/Fragmenter.h"

#include <llvm/IR/Value.h>
#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <memory>

class Executor;

class JoinHashTable {
 public:
  static std::shared_ptr<JoinHashTable> getInstance(const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
                                                    const Catalog_Namespace::Catalog& cat,
                                                    const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                                    const Data_Namespace::MemoryLevel memory_level,
                                                    const Executor* executor);

 private:
  JoinHashTable(const Analyzer::ColumnVar* col_var,
                const Catalog_Namespace::Catalog& cat,
                const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                const Data_Namespace::MemoryLevel memory_level,
                const ExpressionRange& col_range,
                const Executor* executor)
      : col_var_(col_var),
        cat_(cat),
        query_infos_(query_infos),
        memory_level_(memory_level),
        col_range_(col_range),
        executor_(executor) {
    CHECK(col_range.type == ExpressionRangeType::Integer);
#ifdef HAVE_CUDA
    gpu_hash_table_buff_ = 0;
#endif
  }

  int reify();
  llvm::Value* codegenSlot(llvm::Value*, const Executor*);

  const Analyzer::ColumnVar* col_var_;
  const Catalog_Namespace::Catalog& cat_;
  const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  std::vector<int64_t> cpu_hash_table_buff_;
#ifdef HAVE_CUDA
  CUdeviceptr gpu_hash_table_buff_;
#endif
  ExpressionRange col_range_;
  const Executor* executor_;

  friend class Executor;
};

#endif  // QUERYENGINE_JOINHASHTABLE_H
