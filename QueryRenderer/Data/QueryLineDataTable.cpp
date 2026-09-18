/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/QueryLineDataTable.h"

#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/Enums/DataFormatType.h"
#include "QueryRenderer/Data/LineDataTableGpuResources.h"
#include "QueryRenderer/Data/Parsers/CrossSectionFormatJson.h"
#include "QueryRenderer/Data/Parsers/SqlQueryLineFormatJson.h"
#include "QueryRenderer/Data/QueryDataTableQueues.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "Shared/StringTransform.h"

namespace QueryRenderer {

using ::gfx::BufferAttrType;
using ::gfx::BufferLayoutShPtr;
using ::gfx::ColorRGBA;
using ::gfx::IndirectDrawIndexData;
using ::gfx::IndirectDrawVertexData;
using ::gfx::InterleavedBufferLayout;
using ::gfx::SequentialBufferLayout;
using ::gfx::ShaderBlockLayout;
using ::gfx::ShaderBlockLayoutShPtr;
using ::gfx::ShaderBlockType;

SqlQueryLineDataTableJSON::SqlQueryLineDataTableJSON(QueryRendererContext& ctx,
                                                     const std::string& name,
                                                     const JSONLocation& json_loc)
    : BaseLineDataTable(DataInputFormat::kSQL)
    , BaseQueryDataTableSQLJSON(ctx, name, json_loc, RenderQuerySpecialtyType::kLines) {
  RENDER_LOG_SCOPE();
}

std::vector<GpuId> SqlQueryLineDataTableJSON::getUsedGpuIds() const {
  const auto vbo_layout = getVboQueryDataLayout();
  CHECK(vbo_layout);
  // Filter out GpuData with no vertices to handle empty result sets
  return gpu_resources_->getGpuDataMap().getUsedGpuIds(
      [&](GpuId gpu_id, const LineDataTablePerGpuData& gpu_data) {
        return gpu_data.vbo && gpu_data.vbo->numVertices(vbo_layout);
      });
}

bool SqlQueryLineDataTableJSON::hasData() const {
  return gpu_resources_->hasVerticesForLayout(getVboQueryDataLayout());
}

bool SqlQueryLineDataTableJSON::hasAttribute(const std::string& attr_name) const {
  const auto vbo_layout = getVboQueryDataLayout();
  if (vbo_layout && vbo_layout->hasAttribute(attr_name)) {
    return true;
  }
  const auto ssbo_layout = getSsboQueryDataLayout();
  return ssbo_layout && ssbo_layout->hasAttribute(attr_name);
}

std::set<std::string> SqlQueryLineDataTableJSON::getAllAttrNames() const {
  std::set<std::string> rtn;
  const auto vbo_layout = getVboQueryDataLayout();
  if (vbo_layout) {
    auto attrs = vbo_layout->getAllAttrNames();
    std::copy(attrs.begin(), attrs.end(), std::inserter(rtn, rtn.end()));
  }
  const auto ssbo_layout = getSsboQueryDataLayout();
  if (ssbo_layout) {
    auto attrs = ssbo_layout->getAllAttrNames();
    std::copy(attrs.begin(), attrs.end(), std::inserter(rtn, rtn.end()));
  }

  return rtn;
}

QueryLayoutBufferWkPtr SqlQueryLineDataTableJSON::getAttributeDataBuffer(
    const GpuId gpu_id,
    const std::string& attr_name) {
  auto const& gpu_data = gpu_resources_->getGpuDataMap().getData(gpu_id);
  const auto vbo_layout = getVboQueryDataLayout();
  const auto ssbo_layout = getSsboQueryDataLayout();
  if (gpu_data.vbo && gpu_data.vbo->hasAttribute(attr_name, *vbo_layout)) {
    return gpu_data.vbo;
  } else if (gpu_data.ssbo && gpu_data.ssbo->hasAttribute(attr_name, *ssbo_layout)) {
    return gpu_data.ssbo;
  } else {
    THROW_RUNTIME_EX(createJSONRefError("Cannot get buffer for \"" + attr_name +
                                        "\". Attribute does not exist."));
  }

  return QueryLayoutBufferWkPtr();
}

std::map<GpuId, QueryLayoutBufferWkPtr>
SqlQueryLineDataTableJSON::getAttributeDataBuffers(const std::string& attr_name) {
  std::map<GpuId, QueryLayoutBufferWkPtr> rtn;
  std::map<GpuId, QueryLayoutBufferWkPtr>::iterator inserted_itr;

  if (hasData()) {
    // Note: not checking that the data layout ptrs exist here because that would already
    // be done in hasData()
    const auto vbo_layout = getVboQueryDataLayout();
    const auto ssbo_layout = getSsboQueryDataLayout();
    gpu_resources_->getGpuDataMap().visitData([&](GpuId gpu_id,
                                                  LineDataTablePerGpuData& gpu_data) {
      if (gpu_data.vbo && gpu_data.vbo->hasAttribute(attr_name, *vbo_layout)) {
        inserted_itr = rtn.emplace(gpu_id, gpu_data.vbo).first;
      } else if (gpu_data.ssbo && gpu_data.ssbo->hasAttribute(attr_name, *ssbo_layout)) {
        inserted_itr = rtn.emplace(gpu_id, gpu_data.ssbo).first;
      } else {
        RUNTIME_EX_ASSERT(
            !gpu_data.ssbo,
            createJSONRefError("Cannot get buffer for \"" + attr_name +
                               "\". Attribute does not exist in the line data."));
        // if we reach here, the data is empty, or in other words possible empty query
        // Note: we're only checking for the existence of the uniform buffer above
        // because the vbo may be populated due to a cache.
        // If the ubo doesn't exist, that means the query returned 0 results.
        return false;
      }

      CHECK(!rtn.begin()->second.expired() && !inserted_itr->second.expired() &&
            rtn.begin()->second.lock()->getQueryBufferType() ==
                inserted_itr->second.lock()->getQueryBufferType());
      return true;
    });
  }

  return rtn;
}

SQLTypeInfo SqlQueryLineDataTableJSON::getAttributeTypeInfo(
    const std::string& attr_name) const {
  const auto vbo_layout = getVboQueryDataLayout();
  const auto ssbo_layout = getSsboQueryDataLayout();
  if (vbo_layout && vbo_layout->hasAttribute(attr_name)) {
    return vbo_layout->getAttrSQLTypeInfoRef(attr_name);
  }

  if (ssbo_layout && ssbo_layout->hasAttribute(attr_name)) {
    return ssbo_layout->getAttrSQLTypeInfoRef(attr_name);
  }

  RUNTIME_EX_ASSERT(vbo_layout || ssbo_layout,
                    createJSONRefError("Cannot get type for \"" + attr_name +
                                       "\". The line vega data table has no data."));
  THROW_RUNTIME_EX(createJSONRefError("Cannot get type for \"" + attr_name +
                                      "\". The attribute does not exist."));
  return SQLTypeInfo();
}

QueryDataType SqlQueryLineDataTableJSON::getAttributeType(
    const std::string& attr_name) const {
  return convertToQueryDataType(getAttributeBufferType(attr_name));
}

BufferAttrType SqlQueryLineDataTableJSON::getAttributeBufferType(
    const std::string& attr_name) const {
  const auto vbo_layout = getVboQueryDataLayout();
  const auto ssbo_layout = getSsboQueryDataLayout();
  const auto attr_buffer_layout =
      (vbo_layout && vbo_layout->hasAttribute(attr_name)
           ? vbo_layout->getBufferLayout()
           : (ssbo_layout && ssbo_layout->hasAttribute(attr_name)
                  ? ssbo_layout->getBufferLayout()
                  : nullptr));
  if (attr_buffer_layout) {
    return attr_buffer_layout->getAttributeType(attr_name);
  }

  RUNTIME_EX_ASSERT(
      vbo_layout || ssbo_layout,
      createJSONRefError("Cannot get type for \"" + attr_name +
                         "\". The line table has not been initialized with data yet."));
  THROW_RUNTIME_EX(
      createJSONRefError("Cannot get type for \"" + attr_name +
                         "\". The attributedoes not exist in the line data."));
  return BufferAttrType::kInt;
}

BufferLayoutShPtr SqlQueryLineDataTableJSON::getAttributeBufferLayout(
    const std::string& attr_name) {
  const auto vbo_layout = getVboQueryDataLayout();
  const auto ssbo_layout = getSsboQueryDataLayout();
  const auto attr_buffer_layout =
      (vbo_layout && vbo_layout->hasAttribute(attr_name)
           ? vbo_layout->getBufferLayout()
           : (ssbo_layout && ssbo_layout->hasAttribute(attr_name)
                  ? ssbo_layout->getBufferLayout()
                  : nullptr));
  if (attr_buffer_layout) {
    return attr_buffer_layout;
  }

  RUNTIME_EX_ASSERT(vbo_layout || ssbo_layout,
                    createJSONRefError("Cannot get the layout for \"" + attr_name +
                                       "\". The line vega data table has no data."));

  THROW_RUNTIME_EX(createJSONRefError("Cannot get layout for \"" + attr_name +
                                      "\". The attribute does not exist."));
  return nullptr;
}

QueryDataLayoutShPtr SqlQueryLineDataTableJSON::getVboQueryDataLayout() const {
  auto rtn = BaseQueryDataTableSQLJSON::getVboQueryDataLayout();
  if (rtn) {
    return rtn;
  }

  if (gpu_resources_->getGpuDataMap().isEmpty()) {
    return nullptr;
  }

  auto* gpu_data = gpu_resources_->getGpuDataMap().getFirstData();
  CHECK(gpu_data);
  RUNTIME_EX_ASSERT(gpu_data->vbo != nullptr,
                    createJSONRefError("Cannot get vertex data layout. The line data has "
                                       "no vertex data defined."));

  auto result_buffer = dynamic_cast<QueryVertexBuffer*>(gpu_data->vbo.get());
  CHECK(result_buffer);

  return result_buffer->getQueryDataLayout();
}

QueryDataLayoutShPtr SqlQueryLineDataTableJSON::getSsboQueryDataLayout() const {
  auto rtn = BaseQueryDataTableSQLJSON::getSsboQueryDataLayout();
  if (rtn) {
    return rtn;
  }

  if (gpu_resources_->getGpuDataMap().isEmpty()) {
    return nullptr;
  }

  auto* gpu_data = gpu_resources_->getGpuDataMap().getFirstData();
  CHECK(gpu_data);

  if (gpu_data->ssbo == nullptr) {
    return nullptr;
  }

  auto result_buffer = dynamic_cast<QueryShaderStorageBuffer*>(gpu_data->ssbo.get());
  CHECK(result_buffer);

  return result_buffer->getQueryDataLayout();
}

bool SqlQueryLineDataTableJSON::updateSqlFromJSONObj(const JSONLocation& json_loc) {
  RENDER_LOG_SCOPE();
  query_sql_.updateFromJSONObj(json_loc, ctx_.doHitTest(), name_);

  return true;
}

bool SqlQueryLineDataTableJSON::updateFromJSONObjInternal(const JSONLocation& json_loc,
                                                          bool do_execute_query) {
  RENDER_LOG_SCOPE();
  const auto format_loc = json_loc.getMember(JSONSchema_v1::Data::kFormatProp);
  RUNTIME_EX_ASSERT(
      format_loc.isValid() && (format_loc.isObject() || format_loc.isString()),
      RapidJSONUtils::createJsonParseError(
          (format_loc.isValid() ? format_loc : json_loc),
          "Line data object \"" + name_ + "\" must contain a \"" +
              std::string(JSONSchema_v1::Data::kFormatProp) +
              "\" property and it must be an object or a string"));

  if (format_loc.isObject()) {
    auto format_type_loc = format_loc.getMember(JSONSchema_v1::Data::kTypeProp);
    CHECK(format_type_loc.isValid());
    CHECK(format_type_loc.isString());

    switch (get_data_format_from_string(to_lower(format_type_loc.getString()))) {
      case DataFormatType::kCrossSection1d:
        CrossSectionFormatJson::validate(format_loc);
        break;
      case DataFormatType::kLines:
        SqlQueryLineFormatJson::validate(format_loc);
        break;
      case DataFormatType::kRasterMesh2d:
      case DataFormatType::kCrossSection2d:
      case DataFormatType::kUnknown:
        CHECK(false) << format_type_loc.getString();
        break;
    }
    return do_execute_query;
  }

  return true;
}

bool SqlQueryLineDataTableJSON::queueQuery(const JSONLocation* json_loc,
                                           const heavyai::InSituFlags insitu_flags) {
  RENDER_LOG_SCOPE() << "is_update_pending: " << is_update_pending_;
  bool will_query_run = false;
  if (is_update_pending_) {
    // first run the sql query, then initialize the resources
    auto insitu_flags = SqlQueryLineFormatJson::ShouldAttemptInSituRender(*json_loc)
                            ? heavyai::InSituFlags::kInSitu
                            : heavyai::InSituFlags::kForcedNonInSitu;

    will_query_run = ctx_.getDataTableQueues().addToQueryQueue(
        shared_from_this(), *json_loc, std::nullopt, insitu_flags);

    is_update_pending_ = false;
  }
  return will_query_run;
}

void SqlQueryLineDataTableJSON::postRunQuery(bool did_query_execute) {
  RENDER_LOG_SCOPE() << "did_query_execute: " << did_query_execute;
  bool use_index_buffer = false;  // TODO(adb): read / set from somewhere?

  // now initialize resources
  gpu_resources_->initGpuResourcesFromBuffers(
      ctx_.getGlobalContext(), use_index_buffer, name_);
}

bool SqlQueryLineDataTableJSON::update() {
  auto json_obj = ctx_.getJSONObj(json_path_);
  CHECK(json_obj.isValid());
  return queueQuery(&json_obj, render_query_result_.getInSituFlags());
}

};  // namespace QueryRenderer
