/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"

#include "QueryRenderer/AggregationContext.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"
#include "QueryRenderer/Data/Transforms/Enums.h"
#include "QueryRenderer/Data/Transforms/Types.h"
#include "QueryRenderer/Data/Transforms/XformOp.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Interop/InteropBufferMgr.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

//
// class QuerySourceDataTable
//
// Data table implementation that references another data table implementation
// used by XFormOps (vega transforms) to reference the original query data table
//
class QuerySourceDataTable : public BaseDataTable, public BaseQueryDataTableSQLJSON {
 public:
  QuerySourceDataTable(QueryRendererContext& ctx,
                       const std::string& name,
                       const JSONLocation& json_loc);
  ~QuerySourceDataTable() override = default;

  // from BaseQueryDataTable
  // implemented
  bool hasAttribute(const std::string& attr_name) const final;
  SQLTypeInfo getAttributeTypeInfo(const std::string& attr_name) const final;
  QueryDataType getAttributeType(const std::string& attr_name) const final;
  gfx::BufferAttrType getAttributeBufferType(const std::string& attr_name) const final;
  gfx::BufferLayoutShPtr getAttributeBufferLayout(const std::string& attr_name) final;

  // not implemented
  bool hasData() const final;
  std::set<std::string> getAllAttrNames() const final;
  QueryLayoutBufferWkPtr getAttributeDataBuffer(const GpuId gpu_id,
                                                const std::string& attr_name) final;
  std::map<GpuId, QueryLayoutBufferWkPtr> getAttributeDataBuffers(
      const std::string& attr_name) final;
  std::vector<GpuId> getUsedGpuIds() const final;

  // local
  BaseDataTableShPtr getSourceDataRef() const { return data_; }
  std::string getSourceTableName() const;

  OpType getAttributeOpType(const std::string& attr_name) const;
  bool isVectorAttribute(const std::string& attr_name) const;

  template <typename T,
            typename std::enable_if<std::integral_constant<
                bool,
                std::is_arithmetic_v<T> ||
                    std::is_same_v<std::string, typename std::remove_cv<T>::type>>::
                                        value>::type* = nullptr>
  std::vector<T> getTypedVectorData(const std::string& attr) const {
    CHECK(xform_);
    InteropBufferMgr mapped_buffers(ctx_.getCudaMgr(), ctx_.getGlobalContext());

    RUNTIME_EX_ASSERT(xform_->hasOutput(attr),
                      createJSONRefError("Cannot evaluate operator for \"" + attr +
                                         "\". The output does not exist."));

    auto op = xform_->getOutputOp(attr);
    CHECK(op);
    if (op->isVectorOp()) {
      auto eval_result = op->evaluateVector<T>(name_, &mapped_buffers);
      LOG_IF(INFO, !eval_result.is_execution_complete)
          << "Transform evaluation for output \"" << attr << "\" is incomplete.";
      return eval_result.vector;
    }
    auto eval_result = op->evaluateValue<T>(name_, &mapped_buffers);
    LOG_IF(INFO, !eval_result.is_execution_complete)
        << "Transform evaluation for output \"" << attr << "\" is incomplete.";
    return {eval_result.value};
  }

  template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
  std::vector<T> getTypedVectorData(const std::vector<std::string>& attrs) const {
    CHECK(xform_);
    std::vector<T> rtn_vals;

    InteropBufferMgr mapped_buffers(ctx_.getCudaMgr(), ctx_.getGlobalContext());
    for (const auto& attr : attrs) {
      RUNTIME_EX_ASSERT(xform_->hasOutput(attr),
                        createJSONRefError("Cannot evaluate operator for \"" + attr +
                                           "\". The output does not exist."));

      auto op = xform_->getOutputOp(attr);
      CHECK(op);
      RUNTIME_EX_ASSERT(
          !op->isVectorOp(),
          createJSONRefError(
              "Cannot evaluate operator for \"" + attr +
              "\". It is an operator resulting in a array, which is not supported."));

      auto eval_result = op->evaluateValue<T>(name_, &mapped_buffers);
      LOG_IF(INFO, !eval_result.is_execution_complete)
          << "Transform evaluation for output \"" << attr << "\" is incomplete.";

      rtn_vals.push_back(eval_result.value);
    }

    return rtn_vals;
  }

 private:
  BaseDataTableShPtr data_;
  XformShPtr xform_;
  rapidjson::Pointer json_source_path_;

  bool isInternalCacheUpToDate() final;
  bool updateSqlFromJSONObj(const JSONLocation& json_loc) final;
  bool updateFromJSONObjInternal(const JSONLocation& json_loc,
                                 bool do_execute_query) final;
  bool queueQuery(const JSONLocation* json_loc,
                  const heavyai::InSituFlags insitu_flags) final;
  bool update() final;
  void postRunQuery(bool did_query_execute) final {}

  void markAggXformsDirtyAfterRenderStep() {
    CHECK(xform_);
    xform_->markAggOpsDirtyAfterRenderStep(name_);
  }

  void postJSONUpdate() final {
    CHECK(xform_);
    xform_->clearNonPublicFacingOpDataAfterVegaUpdate();
  }

  friend class QueryRendererContext;
  friend class AggregationContext;
};

// specializations
template <>
std::vector<std::string> QuerySourceDataTable::getTypedVectorData<std::string>(
    const std::string& attr) const;

}  // namespace QueryRenderer
