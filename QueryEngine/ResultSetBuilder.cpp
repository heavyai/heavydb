/*
 * Copyright 2020 OmniSci, Inc.
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

/**
 * @file    ResultSetBuilder.cpp
 * @author
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2020 OmniSci, Inc.,  All rights reserved.
 */

#include "ResultSetBuilder.h"
#include "RelAlgTranslator.h"

ResultSet* ResultSetBuilder::makeResultSet(
    const std::vector<TargetInfo>& targets,
    const ExecutorDeviceType device_type,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const Executor* executor) {
  return new ResultSet(targets,
                       device_type,
                       query_mem_desc,
                       row_set_mem_owner,
                       executor->getCatalog(),
                       executor->blockSize(),
                       executor->gridSize());
}

ResultSetDefaultBuilder::ResultSetDefaultBuilder(
    const QueryMemoryDescriptor& _query_mem_desc,
    const std::shared_ptr<RowSetMemoryOwner> _row_set_mem_owner)
    : query_mem_desc(_query_mem_desc), row_set_mem_owner(_row_set_mem_owner) {}

ResultSet* ResultSetDefaultBuilder::build() {
  ResultSet* st = ResultSetBuilder::makeResultSet(std::vector<TargetInfo>{},
                                                  ExecutorDeviceType::CPU,
                                                  query_mem_desc,
                                                  row_set_mem_owner,
                                                  nullptr);
  return st;
}

ResultSetLogicalValuesBuilder::ResultSetLogicalValuesBuilder(
    const std::vector<TargetInfo>& _targets,
    const QueryMemoryDescriptor& _query_mem_desc,
    const std::shared_ptr<RowSetMemoryOwner> _row_set_mem_owner)
    : logical_values(nullptr)
    , targets(_targets)
    , device_type(ExecutorDeviceType::CPU)
    , query_mem_desc(_query_mem_desc)
    , row_set_mem_owner(_row_set_mem_owner)
    , executor(nullptr) {}

ResultSetLogicalValuesBuilder::ResultSetLogicalValuesBuilder(
    const RelLogicalValues* _logical_values,
    const std::vector<TargetInfo>& _targets,
    const ExecutorDeviceType _device_type,
    const QueryMemoryDescriptor& _query_mem_desc,
    const std::shared_ptr<RowSetMemoryOwner> _row_set_mem_owner,
    const Executor* _executor)
    : logical_values(_logical_values)
    , targets(_targets)
    , device_type(_device_type)
    , query_mem_desc(_query_mem_desc)
    , row_set_mem_owner(_row_set_mem_owner)
    , executor(_executor) {}

ResultSet* ResultSetLogicalValuesBuilder::build() {
  ResultSet* rs = ResultSetBuilder::makeResultSet(
      targets, device_type, query_mem_desc, row_set_mem_owner, executor);

  if (logical_values && logical_values->hasRows()) {
    CHECK_EQ(logical_values->getRowsSize(), logical_values->size());

    auto storage = rs->allocateStorage();
    auto buff = storage->getUnderlyingBuffer();

    for (size_t i = 0; i < logical_values->getNumRows(); i++) {
      std::vector<std::shared_ptr<Analyzer::Expr>> row_literals;
      int8_t* ptr = buff + i * query_mem_desc.getRowSize();

      for (size_t j = 0; j < logical_values->getRowsSize(); j++) {
        auto rex_literal =
            dynamic_cast<const RexLiteral*>(logical_values->getValueAt(i, j));
        CHECK(rex_literal);
        const auto expr = RelAlgTranslator::translateLiteral(rex_literal);
        const auto constant = std::dynamic_pointer_cast<Analyzer::Constant>(expr);
        CHECK(constant);

        if (constant->get_is_null()) {
          CHECK(!targets[j].sql_type.is_varlen());
          *reinterpret_cast<int64_t*>(ptr) = inline_int_null_val(targets[j].sql_type);
        } else {
          const auto ti = constant->get_type_info();
          const auto datum = constant->get_constval();

          // Initialize the entire 8-byte slot
          *reinterpret_cast<int64_t*>(ptr) = EMPTY_KEY_64;

          const auto sz = ti.get_size();
          CHECK_GE(sz, int(0));
          std::memcpy(ptr, &datum, sz);
        }
        ptr += 8;
      }
    }
  }

  return rs;
}
