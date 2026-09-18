/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/QuerySourceDataTable.h"

#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/AggregationContext.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"
#include "QueryRenderer/Data/Transforms/Utils.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

using ::gfx::BufferAttrType;
using ::gfx::BufferLayoutShPtr;
using ::gfx::InterleavedBufferLayout;
using ::gfx::InterleavedBufferLayoutShPtr;

QuerySourceDataTable::QuerySourceDataTable(QueryRendererContext& ctx,
                                           const std::string& name,
                                           const JSONLocation& json_loc)
    : BaseDataTable(DataInputFormat::kSourced, DataOutputFormat::kRows)
    , BaseQueryDataTableSQLJSON(ctx, name, json_loc, RenderQuerySpecialtyType::kNone) {
  RENDER_LOG_SCOPE();
}

bool QuerySourceDataTable::hasData() const {
  CHECK(false) << "to be implemented";
  return false;
}

bool QuerySourceDataTable::hasAttribute(const std::string& attr_name) const {
  CHECK(xform_);
  return xform_->hasAttribute(attr_name);
}

std::set<std::string> QuerySourceDataTable::getAllAttrNames() const {
  CHECK(false) << "to be implemented";
  return {};
}

QueryLayoutBufferWkPtr QuerySourceDataTable::getAttributeDataBuffer(
    const GpuId gpu_id,
    const std::string& attr_name) {
  CHECK(false) << "to be implemented";
  return QueryLayoutBufferWkPtr();
}

std::map<GpuId, QueryLayoutBufferWkPtr> QuerySourceDataTable::getAttributeDataBuffers(
    const std::string& attr_name) {
  CHECK(false) << "to be implemented";
  return {};
}

SQLTypeInfo QuerySourceDataTable::getAttributeTypeInfo(
    const std::string& attr_name) const {
  CHECK(xform_);
  return xform_->getAttributeTypeInfo(attr_name);
}

QueryDataType QuerySourceDataTable::getAttributeType(const std::string& attr_name) const {
  CHECK(xform_);
  return xform_->getAttributeType(attr_name);
}

BufferAttrType QuerySourceDataTable::getAttributeBufferType(
    const std::string& attr_name) const {
  CHECK(xform_);
  return xform_->getAttributeBufferType(attr_name);
}

BufferLayoutShPtr QuerySourceDataTable::getAttributeBufferLayout(
    const std::string& attr_name) {
  CHECK(xform_);
  return xform_->getAttributeBufferLayout(attr_name);
}

std::vector<GpuId> QuerySourceDataTable::getUsedGpuIds() const {
  CHECK(false) << "to be implemented";
  return {};
}

std::string QuerySourceDataTable::getSourceTableName() const {
  auto json_data = dynamic_cast<const BaseQueryDataTableSQLJSON*>(data_.get());
  CHECK(json_data);
  return json_data->getName();
}

OpType QuerySourceDataTable::getAttributeOpType(const std::string& attr_name) const {
  CHECK(xform_);
  RUNTIME_EX_ASSERT(hasAttribute(attr_name),
                    createJSONRefError("Cannot get operator type for \"" + attr_name +
                                       "\". Output does not exist."));

  auto op = xform_->getOutputOp(attr_name);
  CHECK(op);
  return op->getOpType();
}

bool QuerySourceDataTable::isVectorAttribute(const std::string& attr_name) const {
  CHECK(xform_);
  RUNTIME_EX_ASSERT(hasAttribute(attr_name),
                    createJSONRefError("Cannot check whether \"" + attr_name +
                                       "\" is an array. The output does not exist."));
  auto op = xform_->getOutputOp(attr_name);
  CHECK(op);
  return op->isVectorOp();
}

bool QuerySourceDataTable::isInternalCacheUpToDate() {
  return true;  // noop, we are not maintaining an internal cache in any way that could be
                // affected by an unchanging vega
}

bool QuerySourceDataTable::updateSqlFromJSONObj(const JSONLocation& json_loc) {
  return false;  // nothing to do
}

bool QuerySourceDataTable::queueQuery(const JSONLocation* json_loc,
                                      const heavyai::InSituFlags insitu_flags) {
  return false;
}

bool QuerySourceDataTable::updateFromJSONObjInternal(const JSONLocation& json_loc,
                                                     bool do_execute_query) {
  RENDER_LOG_SCOPE();
  const auto source_loc = json_loc.getMember(JSONSchema_v1::Data::kSourceProp);
  RUNTIME_EX_ASSERT(source_loc.isValid() && source_loc.isString(),
                    RapidJSONUtils::createJsonParseError(
                        source_loc.isValid() ? source_loc : json_loc,
                        "Source data object \"" + name_ + "\" must contain an \"" +
                            std::string(JSONSchema_v1::Data::kSourceProp) +
                            "\" property and it must be a string"));

  auto source_name = std::string(source_loc.getString());
  if (!ctx_.isJSONCacheUpToDate(json_source_path_, source_loc)) {
    data_ = ctx_.getDataTable(source_name);

    RUNTIME_EX_ASSERT(
        data_,
        RapidJSONUtils::createJsonParseError(
            source_loc,
            "Source data reference \"" + source_name + "\" does not exist in the vega."));
  }
  json_source_path_ = source_loc.getPathRef();

  CHECK(data_);
  auto json_data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
  CHECK(json_data);

  const auto xform_loc = json_loc.getMember(JSONSchema_v1::Data::kTransformProp);
  RUNTIME_EX_ASSERT(xform_loc.isValid() && xform_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        xform_loc.isValid() ? xform_loc : json_loc,
                        "Source data object \"" + name_ + "\" must contain a \"" +
                            std::string(JSONSchema_v1::Data::kTransformProp) +
                            "\" property and it must be an array."));

  xform_ = createTransform(ctx_, data_, xform_loc);
  return false;  // no query to run
}

bool QuerySourceDataTable::update() {
  return false;  // noop. No additional gpu resources needed here
}

// specializations
template <>
std::vector<std::string> QuerySourceDataTable::getTypedVectorData<std::string>(
    const std::string& attr) const {
  CHECK(xform_);
  InteropBufferMgr mapped_buffers(ctx_.getCudaMgr(), ctx_.getGlobalContext());

  RUNTIME_EX_ASSERT(xform_->hasOutput(attr),
                    createJSONRefError("Cannot evaluate operator for \"" + attr +
                                       "\". The output does not exist."));

  auto op = xform_->getOutputOp(attr);
  CHECK(op);
  RUNTIME_EX_ASSERT(
      op->isVectorOp(),
      createJSONRefError("Cannot evaluate operator for \"" + attr +
                         "\". The output is not an array. Only array outputs are "
                         "currently supported for evaluating to string."));

  auto eval_result = op->evaluateVector<std::string>(name_, &mapped_buffers);
  LOG_IF(INFO, !eval_result.is_execution_complete)
      << "Transform evaluation for output \"" << attr << "\" is incomplete.";
  return eval_result.vector;
}

}  // namespace QueryRenderer
