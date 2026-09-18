/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"

#include "QueryRenderer/Data/Transforms/Enums.h"
#include "QueryRenderer/Data/Transforms/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

class BaseXform : public BaseDataTable {
 public:
  ~BaseXform() override {}

  bool hasAttribute(const std::string& attr_name) const final;
  std::set<std::string> getAllAttrNames() const final;
  QueryLayoutBufferWkPtr getAttributeDataBuffer(const GpuId gpu_id,
                                                const std::string& attr_name) final;
  std::map<GpuId, QueryLayoutBufferWkPtr> getAttributeDataBuffers(
      const std::string& attr_name) final;

  SQLTypeInfo getAttributeTypeInfo(const std::string& attr_name) const final;
  QueryDataType getAttributeType(const std::string& attr_name) const final;
  gfx::BufferAttrType getAttributeBufferType(const std::string& attr_name) const final;
  gfx::BufferLayoutShPtr getAttributeBufferLayout(const std::string& attr_name) final;
  std::vector<GpuId> getUsedGpuIds() const final;

  virtual XformType getXformType() const = 0;
  virtual bool hasOutput(const std::string& output) const = 0;
  virtual const XformOpShPtr getOutputOp(const std::string& output) const = 0;
  virtual std::set<std::string> getAllOutputNames() const = 0;

  std::string getSourceDataTableName() const;
  const BaseDataTableShPtr getInputDataTable() const { return data_; }

  const QueryDataLayoutShPtr getOutputsDataLayout() const {
    CHECK(data_layout_);
    return data_layout_;
  }

  virtual void initialize(const XformShPtr& ptr, const JSONLocation& obj_loc) = 0;

 protected:
  BaseXform(const QueryRendererContext& ctx, const BaseDataTableShPtr& data)
      : BaseDataTable(DataInputFormat::kTransformed, DataOutputFormat::kRows)
      , ctx_{ctx}
      , data_{data}
      , initialized_{false} {}
  const QueryRendererContext& ctx_;
  BaseDataTableShPtr data_;
  QueryDataLayoutShPtr data_layout_;
  bool initialized_;

 private:
  void initializeInternal(const XformShPtr& ptr, const JSONLocation& obj_loc);
  bool update() final;
  void markAggOpsDirtyAfterRenderStep(const std::string& evaluator_name);
  virtual void markAggOpsDirtyAfterRenderStepInternal(
      const std::string& evaluator_name) = 0;

  void clearNonPublicFacingOpDataAfterVegaUpdate();
  virtual void clearNonPublicFacingOpDataAfterVegaUpdateInternal() {}

  static void iterateThruXformHierarchy(
      const BaseDataTableShPtr& data,
      std::function<void(const BaseQueryDataTableSQLJSON*)> reached_head_node_cb,
      std::function<void(BaseXform*)> reached_xform_node_cb);
  friend class QuerySourceDataTable;
  friend class XformOp;
  friend XformShPtr createTransform(const QueryRendererContext&,
                                    const BaseDataTableShPtr&,
                                    const JSONLocation&);
};

}  // namespace QueryRenderer
