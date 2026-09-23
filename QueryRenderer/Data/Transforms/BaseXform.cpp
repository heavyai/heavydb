/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/BaseXform.h"

#include "GfxDriver/Resources/BufferLayout.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"
#include "QueryRenderer/QueryDataLayout.h"

namespace QueryRenderer {

void BaseXform::initializeInternal(const XformShPtr& ptr, const JSONLocation& obj_loc) {
  CHECK(ptr.get() == this);
  initialize(ptr, obj_loc);
  initialized_ = true;
}

bool BaseXform::hasAttribute(const std::string& attr_name) const {
  CHECK(data_layout_);
  return data_layout_->hasAttribute(attr_name);
}

std::set<std::string> BaseXform::getAllAttrNames() const {
  CHECK(data_layout_);
  auto attr_vec = data_layout_->getAllAttrNames();
  return std::set<std::string>(attr_vec.begin(), attr_vec.end());
}

QueryLayoutBufferWkPtr BaseXform::getAttributeDataBuffer(const GpuId gpu_id,
                                                         const std::string& attr_name) {
  CHECK(false) << "To be implemented";
  return QueryLayoutBufferWkPtr();
}

std::map<GpuId, QueryLayoutBufferWkPtr> BaseXform::getAttributeDataBuffers(
    const std::string& attr_name) {
  CHECK(false) << "To be implemented";
  return {};
}

SQLTypeInfo BaseXform::getAttributeTypeInfo(const std::string& attr_name) const {
  CHECK(data_layout_);
  return data_layout_->getAttrSQLTypeInfoRef(attr_name);
}

QueryDataType BaseXform::getAttributeType(const std::string& attr_name) const {
  CHECK(data_layout_);
  return convertToQueryDataType(data_layout_->getAttrSQLTypeInfoRef(attr_name));
}

gfx::BufferAttrType BaseXform::getAttributeBufferType(
    const std::string& attr_name) const {
  CHECK(data_layout_);
  return data_layout_->getBufferLayout()->getAttributeType(attr_name);
}

gfx::BufferLayoutShPtr BaseXform::getAttributeBufferLayout(const std::string& attr_name) {
  RUNTIME_EX_ASSERT(data_layout_,
                    "Cannot get the layout for attribute \"" + attr_name +
                        "\". The xform data table has no data.");

  RUNTIME_EX_ASSERT(
      data_layout_->hasAttribute(attr_name),
      "The attribute \"" + attr_name +
          "\" does not exist in the xform data table. Cannot get a layout.");

  return data_layout_->getBufferLayout();
}

std::vector<GpuId> BaseXform::getUsedGpuIds() const {
  CHECK(false) << "To be implemented";
  return {};
}

bool BaseXform::update() {
  CHECK(false) << "To be implemented";
  return false;
}

void BaseXform::iterateThruXformHierarchy(
    const BaseDataTableShPtr& data,
    std::function<void(const BaseQueryDataTableSQLJSON*)> reached_head_node_cb,
    std::function<void(BaseXform*)> reached_xform_node_cb) {
  CHECK(data);
  auto curr_data = data.get();

  while (curr_data) {
    auto data_json = dynamic_cast<const BaseQueryDataTableSQLJSON*>(curr_data);
    if (data_json) {
      if (reached_head_node_cb) {
        reached_head_node_cb(data_json);
      }
      break;
    }
    auto xform_json = dynamic_cast<BaseXform*>(curr_data);
    CHECK(xform_json);
    if (reached_xform_node_cb) {
      reached_xform_node_cb(xform_json);
    }
    curr_data = xform_json->data_.get();
  }
}

std::string BaseXform::getSourceDataTableName() const {
  std::string rtn;
  iterateThruXformHierarchy(
      data_, [&rtn](auto const data_json) { rtn = data_json->getName(); }, nullptr);
  CHECK(rtn.size() != 0);
  return rtn;
}

void BaseXform::markAggOpsDirtyAfterRenderStep(const std::string& evaluator_name) {
  iterateThruXformHierarchy(data_, nullptr, [&evaluator_name](auto const xform_json) {
    xform_json->markAggOpsDirtyAfterRenderStepInternal(evaluator_name);
  });
  markAggOpsDirtyAfterRenderStepInternal(evaluator_name);
}

void BaseXform::clearNonPublicFacingOpDataAfterVegaUpdate() {
  iterateThruXformHierarchy(data_, nullptr, [](auto const xform_json) {
    xform_json->clearNonPublicFacingOpDataAfterVegaUpdateInternal();
  });
  clearNonPublicFacingOpDataAfterVegaUpdateInternal();
}

}  // namespace QueryRenderer
