/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    ResultSetBuilder.h
 * @brief   Basic constructors and methods of the row set interface.
 *
 */

#ifndef QUERYENGINE_RESULTSETBUILDER_H
#define QUERYENGINE_RESULTSETBUILDER_H

#include "RelAlgDag.h"
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

  ResultSet* build() override;
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

  ResultSet* build() override;

  // A simplified/common pre-packaged use case for this builder that creates a ResultSet
  //     ExecutorDeviceType is "CPU"
  //     QueryMemoryDescriptor is "Projection"
  //     RowSetMemoryOwner is default
  static ResultSet* create(std::vector<TargetMetaInfo>& label_infos,
                           std::vector<RelLogicalValues::RowValues>& logical_values);
};

#endif  // QUERYENGINE_RESULTSETBUILDER_H
