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
                       executor ? executor->getDataMgr() : nullptr,
                       executor ? executor->getBufferProvider() : nullptr,
                       executor ? executor->blockSize() : 0,
                       executor ? executor->gridSize() : 0);
}

void ResultSetBuilder::addVarlenBuffer(ResultSet* result_set,
                                       std::vector<std::string>& varlen_storage) {
  CHECK(result_set->serialized_varlen_buffer_.size() == 0);

  // init with an empty vector
  result_set->serialized_varlen_buffer_.emplace_back(std::vector<std::string>());

  // copy the values into the empty vector
  result_set->serialized_varlen_buffer_.front().assign(varlen_storage.begin(),
                                                       varlen_storage.end());
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

    std::vector<std::string> separate_varlen_storage;

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

          if (ti.is_string()) {
            // get string from datum and push to vector
            separate_varlen_storage.push_back(*(datum.stringval));

            // store the index/offset in ResultSet's storage
            //   (the # of the string in the varlen_storage, not strLen)
            *reinterpret_cast<int64_t*>(ptr) =
                static_cast<int64_t>(separate_varlen_storage.size() - 1);

          } else {
            // Initialize the entire 8-byte slot
            *reinterpret_cast<int64_t*>(ptr) = EMPTY_KEY_64;

            const auto sz = ti.get_size();
            CHECK_GE(sz, int(0));
            std::memcpy(ptr, &datum, sz);
          }
        }
        ptr += 8;
      }
    }

    // store the varlen data (ex. strings) into the ResultSet
    if (separate_varlen_storage.size()) {
      ResultSetBuilder::addVarlenBuffer(rs, separate_varlen_storage);
      rs->setSeparateVarlenStorageValid(true);
    }
  }

  return rs;
}

// static
ResultSet* ResultSetLogicalValuesBuilder::create(
    std::vector<TargetMetaInfo>& label_infos,
    std::vector<RelLogicalValues::RowValues>& logical_values) {
  // check to see if number of columns matches (at least the first row)
  size_t numCols =
      logical_values.size() ? logical_values.front().size() : label_infos.size();
  CHECK_EQ(label_infos.size(), numCols);

  size_t numRows = logical_values.size();

  QueryMemoryDescriptor query_mem_desc(/*executor=*/nullptr,
                                       /*entry_count=*/numRows,
                                       QueryDescriptionType::Projection,
                                       /*is_table_function=*/false);

  // target_infos -> defines the table columns
  std::vector<TargetInfo> target_infos;
  SQLTypeInfo str_ti(kTEXT, true, kENCODING_NONE);
  for (size_t col = 0; col < numCols; col++) {
    SQLTypeInfo colType = label_infos[col].get_type_info();
    target_infos.push_back(TargetInfo{false, kSAMPLE, colType, colType, true, false});
    query_mem_desc.addColSlotInfo({std::make_tuple(colType.get_size(), 8)});
  }

  std::shared_ptr<RelLogicalValues> rel_logical_values =
      std::make_shared<RelLogicalValues>(label_infos, logical_values);

  const auto row_set_mem_owner =
      std::make_shared<RowSetMemoryOwner>(nullptr, Executor::getArenaBlockSize());

  // Construct ResultSet
  ResultSet* rsp(ResultSetLogicalValuesBuilder(rel_logical_values.get(),
                                               target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr)
                     .build());

  return rsp;
}
