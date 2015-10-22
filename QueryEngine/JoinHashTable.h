/*
 * @file    JoinHashTable.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_JOINHASHTABLE_H
#define QUERYENGINE_JOINHASHTABLE_H

#include "../Analyzer/Analyzer.h"
#include "../Fragmenter/Fragmenter.h"

#include <llvm/IR/Value.h>

class Executor;

class JoinHashTable {
 public:
  JoinHashTable(const Analyzer::ColumnVar* col_var,
                const Catalog_Namespace::Catalog& cat,
                const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                const Data_Namespace::MemoryLevel memory_level)
      : col_var_(col_var), cat_(cat), query_infos_(query_infos), memory_level_(memory_level) {}

 private:
  llvm::Value* reify(llvm::Value*, const Executor*);

  const Analyzer::ColumnVar* col_var_;
  const Catalog_Namespace::Catalog& cat_;
  const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  std::vector<int64_t> cpu_hash_table_buff_;

  friend class Executor;
};

#endif  // QUERYENGINE_JOINHASHTABLE_H
