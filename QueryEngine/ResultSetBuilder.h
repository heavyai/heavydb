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
 * @file    ResultSetBuilder.h
 * @author
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2020 OmniSci, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_RESULTSETBUILDER_H
#define QUERYENGINE_RESULTSETBUILDER_H

#include "RelAlgDagBuilder.h"
#include "ResultSet.h"

// ********************************************************
// * Usage
//
//  SomeClass* cls;
//  ResultSetSomeClassBuilder builder(cls);
//  // builder ... further config as requried
//  ResultSet* set = builder.build();
//
// ********************************************************

class ResultSetBuilder {
 protected:
  ResultSet* makeResultSet(const std::vector<TargetInfo>& targets,
                           const ExecutorDeviceType device_type,
                           const QueryMemoryDescriptor& query_mem_desc,
                           const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                           const Executor* executor);
  void addVarlenBuffer(ResultSet* result_set, std::vector<std::string>& varlen_storage);

 public:
  virtual ResultSet* build() = 0;
};

class ResultSetDefaultBuilder : public ResultSetBuilder {
 private:
  const QueryMemoryDescriptor& query_mem_desc;
  const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner;

 public:
  ResultSetDefaultBuilder(const QueryMemoryDescriptor& query_mem_desc,
                          const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  ResultSet* build();
};

class ResultSetLogicalValuesBuilder : public ResultSetBuilder {
 private:
  const RelLogicalValues* logical_values;
  const std::vector<TargetInfo>& targets;
  const ExecutorDeviceType device_type;
  const QueryMemoryDescriptor& query_mem_desc;
  const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner;
  const Executor* executor;

 public:
  ResultSetLogicalValuesBuilder(
      const std::vector<TargetInfo>& targets,
      const QueryMemoryDescriptor& query_mem_desc,
      const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  ResultSetLogicalValuesBuilder(
      const RelLogicalValues* logical_values,
      const std::vector<TargetInfo>& targets,
      const ExecutorDeviceType device_type,
      const QueryMemoryDescriptor& query_mem_desc,
      const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const Executor* executor);

  ResultSet* build();

  // A simplified/common pre-packaged use case for this builder that creates a ResultSet
  //     ExecutorDeviceType is "CPU"
  //     QueryMemoryDescriptor is "Projection"
  //     RowSetMemoryOwner is default
  static ResultSet* create(std::vector<TargetMetaInfo>& label_infos,
                           std::vector<RelLogicalValues::RowValues>& logical_values);
};

#endif  // QUERYENGINE_RESULTSETBUILDER_H
