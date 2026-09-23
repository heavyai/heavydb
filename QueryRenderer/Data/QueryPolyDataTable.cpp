/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/QueryPolyDataTable.h"

#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/filesystem.hpp>
#include <boost/geometry.hpp>

#include "Analyzer/Analyzer.h"
#include "DataMgr/DataMgr.h"
#include "GfxDriver/Colors/ColorRGBA.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/BufferLayout.h"
#include "GfxDriver/Resources/ShaderBlockLayout.h"
#include "QueryRenderer/Data/PolyDataTableGpuResources.h"
#include "QueryRenderer/Data/QueryDataTableQueues.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/QueryRendererContext.h"

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

namespace {

std::string build_poly_render_query(const std::string& poly_table_name,
                                    const std::string& facts_table_name,
                                    const std::string& filter_expr,
                                    const std::string& agg_expr,
                                    const std::string& facts_key,
                                    const std::string& polys_key) {
  std::stringstream ss;
  ss << "SELECT " << poly_table_name << ".rowid, " << agg_expr << " FROM "
     << facts_table_name << ", " << poly_table_name << " WHERE " << filter_expr
     << (filter_expr.empty() ? "" : " AND ") << facts_table_name << "." << facts_key
     << " = " << poly_table_name << "." << polys_key << " GROUP BY " << poly_table_name
     << ".rowid;";
  return ss.str();
}

std::string transform_to_poly_render_query(const std::string& query_str,
                                           const std::string& poly_table_name,
                                           const std::string& polys_key) {
  auto result = query_str;
  {
    boost::regex aliased_group_expr{R"(\s+([^\s]+)\s+as\s+([^(\s|,)]+))",
                                    boost::regex::extended | boost::regex::icase};
    boost::smatch what;
    std::string what1, what2;
    if (boost::regex_search(result, what, aliased_group_expr)) {
      what1 = std::string(what[1]);
      what2 = std::string(what[2]);
      result.replace(what.position(), what.length(), " " + what1);
    } else {
      what1 = std::string(what[1]);
      what2 = std::string(what[2]);
    }
    boost::ireplace_all(result, what2, what1);
  }
  std::string groupby_expr;
  {
    boost::regex group_expr{R"(group\s+by\s+([^(\s|;|,)]+))",
                            boost::regex::extended | boost::regex::icase};
    boost::smatch what;
    CHECK(boost::regex_search(result, what, group_expr));
    groupby_expr = what[1];
    boost::ireplace_all(result, std::string(what[1]), poly_table_name + ".rowid");
  }
  CHECK(!groupby_expr.empty());
  const auto join_filter = groupby_expr + " = " + poly_table_name + "." + polys_key;
  {
    boost::regex where_expr(R"(\s+where\s+(.*)\s+group\s+by)",
                            boost::regex::extended | boost::regex::icase);
    boost::smatch what_where;
    boost::regex from_expr{R"(\s+from\s+([^\s]+)\s+)",
                           boost::regex::extended | boost::regex::icase};
    boost::smatch what_from;
    if (boost::regex_search(result, what_where, where_expr)) {
      result.replace(what_where.position(),
                     what_where.length(),
                     " WHERE " + what_where[1] + " AND " + join_filter + " GROUP BY");
      CHECK(boost::regex_search(result, what_from, from_expr));
      result.replace(what_from.position(),
                     what_from.length(),
                     " FROM " + std::string(what_from[1]) + ", " + poly_table_name + " ");
    } else {
      CHECK(boost::regex_search(result, what_from, from_expr));
      result.replace(what_from.position(),
                     what_from.length(),
                     " FROM " + std::string(what_from[1]) + ", " + poly_table_name +
                         " WHERE " + join_filter + " ");
    }
  }
  return result;
}
}  // namespace

SqlQueryPolyDataTableJSON::SqlQueryPolyDataTableJSON(QueryRendererContext& ctx,
                                                     const std::string& name,
                                                     const JSONLocation& json_loc)
    : BasePolyDataTable(DataInputFormat::kSQL)
    , BaseQueryDataTableSQLJSON(ctx, name, json_loc, RenderQuerySpecialtyType::kPolys)
    , query_type_changed_{true} {
  RENDER_LOG_SCOPE();
}

bool SqlQueryPolyDataTableJSON::hasData() const {
  return gpu_resources_->hasVerticesForLayout(getVboQueryDataLayout());
}

bool SqlQueryPolyDataTableJSON::hasAttribute(const std::string& attr_name) const {
  const auto vbo_layout = getVboQueryDataLayout();
  if (vbo_layout && vbo_layout->hasAttribute(attr_name)) {
    return true;
  }
  const auto ssbo_layout = getSsboQueryDataLayout();
  return ssbo_layout && ssbo_layout->hasAttribute(attr_name);
}

std::set<std::string> SqlQueryPolyDataTableJSON::getAllAttrNames() const {
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

QueryLayoutBufferWkPtr SqlQueryPolyDataTableJSON::getAttributeDataBuffer(
    const GpuId gpu_id,
    const std::string& attr_name) {
  auto const& gpu_data = gpu_resources_->getGpuDataMap().getData(gpu_id);
  const auto vbo_layout = getVboQueryDataLayout();
  const auto ssbo_layout = getSsboQueryDataLayout();
  CHECK(vbo_layout);
  CHECK(ssbo_layout);
  if (gpu_data.vbo && gpu_data.vbo->hasAttribute(attr_name, *vbo_layout)) {
    return gpu_data.vbo;
  } else if (gpu_data.ssbo && gpu_data.ssbo->hasAttribute(attr_name, *ssbo_layout)) {
    return gpu_data.ssbo;
  } else {
    THROW_RUNTIME_EX(
        createJSONRefError("Attribute \"" + attr_name + "\" does not exist."));
  }
  return QueryLayoutBufferWkPtr();
}

std::map<GpuId, QueryLayoutBufferWkPtr>
SqlQueryPolyDataTableJSON::getAttributeDataBuffers(const std::string& attr_name) {
  std::map<GpuId, QueryLayoutBufferWkPtr> rtn;
  std::map<GpuId, QueryLayoutBufferWkPtr>::iterator inserted_itr;
  if (hasData()) {
    // Note: not checking that the data layout ptrs exist here because that would
    // already be done in hasData()
    const auto vbo_layout = getVboQueryDataLayout();
    const auto ssbo_layout = getSsboQueryDataLayout();
    CHECK(vbo_layout);
    CHECK(ssbo_layout);
    gpu_resources_->getGpuDataMap().visitData([&](GpuId gpu_id,
                                                  PolyDataTablePerGpuData& gpu_data) {
      if (gpu_data.vbo && gpu_data.vbo->hasAttribute(attr_name, *vbo_layout)) {
        inserted_itr = rtn.emplace(gpu_id, gpu_data.vbo).first;
      } else if (gpu_data.ssbo && gpu_data.ssbo->hasAttribute(attr_name, *ssbo_layout)) {
        inserted_itr = rtn.emplace(gpu_id, gpu_data.ssbo).first;
      } else {
        RUNTIME_EX_ASSERT(
            !gpu_data.ssbo,
            createJSONRefError("Cannot get data buffer for \"" + attr_name +
                               "\". The attribute does not exist in the poly data."));
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

SQLTypeInfo SqlQueryPolyDataTableJSON::getAttributeTypeInfo(
    const std::string& attr_name) const {
  const auto vbo_layout = getVboQueryDataLayout();
  if (vbo_layout && vbo_layout->hasAttribute(attr_name)) {
    return vbo_layout->getAttrSQLTypeInfoRef(attr_name);
  }

  const auto ssbo_layout = getSsboQueryDataLayout();
  if (ssbo_layout && ssbo_layout->hasAttribute(attr_name)) {
    return ssbo_layout->getAttrSQLTypeInfoRef(attr_name);
  }

  RUNTIME_EX_ASSERT(vbo_layout || ssbo_layout,
                    createJSONRefError("Cannot get type for \"" + attr_name +
                                       "\". The poly vega data table has no data."));
  THROW_RUNTIME_EX(createJSONRefError("Cannot get type for \"" + attr_name +
                                      "\". The attribute does not exist."));
}

QueryDataType SqlQueryPolyDataTableJSON::getAttributeType(
    const std::string& attr_name) const {
  return convertToQueryDataType(getAttributeBufferType(attr_name));
}

BufferAttrType SqlQueryPolyDataTableJSON::getAttributeBufferType(
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
                         "\". The poly table has not been initialized with data yet."));
  THROW_RUNTIME_EX(
      createJSONRefError("Cannot get type for \"" + attr_name +
                         "\". The attribute does not exist in the line data."));
}

BufferLayoutShPtr SqlQueryPolyDataTableJSON::getAttributeBufferLayout(
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
                                       "\". The poly vega data table has no data."));

  THROW_RUNTIME_EX(createJSONRefError("Cannot get layout for \"" + attr_name +
                                      "\". The attribute does not exist."));
}

bool SqlQueryPolyDataTableJSON::updateSqlFromJSONObj(const JSONLocation& json_loc) {
  RENDER_LOG_SCOPE();
  // Copy the table infos before they are changed in updateFromJSONObjAndQueueQuery().
  // Used for extra validation
  //
  // NOTE: This is copying a shared ptr. The original shared_ptr will be reset in the
  // the call to updateFromJSONObj(), so this should still be ok.

  // TODO(croot): remove this if we parse/validate the sql in
  // BaseQueryDataTableSQLJSON::updateFromJSONObj()
  curr_table_info_ = query_sql_.getAllTableInfoPtr();
  CHECK(curr_table_info_);

  auto curr_query_str = query_sql_.getSqlQueryStr();
  auto curr_query_opts = query_sql_.getRenderQueryOptions();

  query_sql_.updateFromJSONObj(json_loc, ctx_.doHitTest(), name_);
  CHECK(query_sql_.getAllTableInfoPtr());

  query_sql_.getRenderQueryOptions().setFlags(
      RenderQueryOptions::FlagBits::kInjectRowIdForPPLL);

  bool execute_query = (curr_query_str != query_sql_.getSqlQueryStr() ||
                        curr_query_opts != query_sql_.getRenderQueryOptions());
  return execute_query;
}

bool SqlQueryPolyDataTableJSON::updateFromJSONObjInternal(const JSONLocation& json_loc,
                                                          bool do_execute_query) {
  const auto key_loc = json_loc.getMember(JSONSchema_v1::Data::kPolysKeyProp);
  if (key_loc.isValid()) {
    RUNTIME_EX_ASSERT(
        key_loc.isString(),
        RapidJSONUtils::createJsonParseError(
            key_loc,
            "Poly data object \"" + name_ + "\" has a \"" +
                std::string(JSONSchema_v1::Data::kPolysKeyProp) +
                "\" property, but it is not a string. It must be a string"));
    if (!do_execute_query) {
      do_execute_query = polys_key_ != key_loc.getString();
    }
    polys_key_ = key_loc.getString();

    auto const& all_table_info = query_sql_.getAllTableInfoRef();
    RUNTIME_EX_ASSERT(
        !all_table_info.phys_tables.empty() &&
            all_table_info.phys_tables[0].table_name.table_name.empty(),
        RapidJSONUtils::createJsonParseError(
            key_loc,
            "Poly data object \"" + name_ + "\" has a \"" +
                std::string(JSONSchema_v1::Data::kPolysKeyProp) +
                "\" property and therefore requires a \"dbTableName\" string property."));

  } else {
    if (polys_key_.size()) {
      do_execute_query = true;
    }
    polys_key_ = "";
  }

  // A "factsKey" indicates options to build up your query in the json
  // TODO(croot): deperecate this?
  const auto facts_loc = json_loc.getMember(JSONSchema_v1::Data::kFactsKeyProp);
  if (facts_loc.isValid()) {
    RUNTIME_EX_ASSERT(
        !polys_key_.empty(),
        RapidJSONUtils::createJsonParseError(
            facts_loc,
            "Poly data object \"" + name_ + "\" has a \"" +
                std::string(JSONSchema_v1::Data::kFactsKeyProp) +
                "\" property and therefore requires a \"polyKeys\" string property."));

    RUNTIME_EX_ASSERT(
        facts_loc.isString(),
        RapidJSONUtils::createJsonParseError(
            facts_loc,
            "Poly data object \"" + name_ + "\" has a \"" +
                std::string(JSONSchema_v1::Data::kFactsKeyProp) +
                "\" property, but it is not a string. It must be a string"));
    if (!do_execute_query) {
      do_execute_query = facts_key_ != facts_loc.getString();
    }
    facts_key_ = facts_loc.getString();

    const auto facts_table_loc =
        json_loc.getMember(JSONSchema_v1::Data::kFactsTableNameProp);
    RUNTIME_EX_ASSERT(facts_table_loc.isValid() && facts_table_loc.isString(),
                      RapidJSONUtils::createJsonParseError(
                          facts_table_loc.isValid() ? facts_table_loc : json_loc,
                          "Poly data object \"" + name_ + "\" has a \"" +
                              std::string(JSONSchema_v1::Data::kFactsKeyProp) +
                              "\" property and therefore requires a \"" +
                              std::string(JSONSchema_v1::Data::kFactsTableNameProp) +
                              "\" string property."));
    if (!do_execute_query) {
      do_execute_query = facts_table_name_ != facts_table_loc.getString();
    }
    facts_table_name_ = facts_table_loc.getString();

    const auto agg_loc = json_loc.getMember(JSONSchema_v1::Data::kAggExprProp);
    if (agg_loc.isValid()) {
      RUNTIME_EX_ASSERT(
          agg_loc.isString(),
          RapidJSONUtils::createJsonParseError(
              agg_loc,
              "Poly data object \"" + name_ + "\" has a \"" +
                  std::string(JSONSchema_v1::Data::kAggExprProp) +
                  "\" property, but it is not a string. It must be a string."));
      if (!do_execute_query) {
        do_execute_query = agg_expr_ != agg_loc.getString();
      }
      agg_expr_ = agg_loc.getString();
    } else {
      if (agg_expr_.size()) {
        do_execute_query = true;
      }
      agg_expr_ = "";
    }

    const auto filter_loc = json_loc.getMember(JSONSchema_v1::Data::kFilterExprProp);
    if (filter_loc.isValid()) {
      RUNTIME_EX_ASSERT(
          filter_loc.isString(),
          RapidJSONUtils::createJsonParseError(
              filter_loc,
              "Poly data object \"" + name_ + "\" has a \"" +
                  std::string(JSONSchema_v1::Data::kFilterExprProp) +
                  "\" property, but it is not a string. It must be a string."));
      if (!do_execute_query) {
        do_execute_query = filter_expr_ != filter_loc.getString();
      }
      filter_expr_ = filter_loc.getString();
    } else {
      if (filter_expr_.size()) {
        do_execute_query = true;
      }
      filter_expr_ = "";
    }

  } else {
    if (facts_key_.size()) {
      do_execute_query = true;
    }
    facts_key_ = "";
    agg_expr_ = "";
    filter_expr_ = "";
    facts_table_name_ = "";
  }

  if (do_execute_query) {
    // TODO(croot): do we only need to build the override sql
    // when executeQuery is true, or do we need to build it when
    // (force || !_isInternalCacheUpToDate()) also?
    // I think it only needs to be built only when executeQuery is true
    if (query_sql_.hasExecutableSql() && !polys_key_.empty()) {
      if (!facts_key_.empty()) {
        sql_query_str_override_ =
            build_poly_render_query(query_sql_.getPrimaryTableName(),
                                    facts_table_name_,
                                    filter_expr_,
                                    agg_expr_,
                                    facts_key_,
                                    polys_key_);
      } else {
        sql_query_str_override_ = transform_to_poly_render_query(
            query_sql_.getSqlQueryStr(), query_sql_.getPrimaryTableName(), polys_key_);
      }
    } else {
      sql_query_str_override_ = OptionalStr();
    }

    // now parse the query to extract the table names. This is needed
    // before the _runQueryAndInitResources() call below to check that
    // the query and it's associated poly table is properly cached.

    // TODO(scb): refactor this once the ParseQuery step is formalized for all Marks
    auto render_query_runner = ctx_.getRenderQueryRunner();
    if (render_query_runner) {
      auto render_timer = ctx_.getRenderTimer();
      const auto& sql_str_to_use = getSqlStrToUseRef();
      auto render_parse_info =
          render_query_runner->executeQueryParse(*render_timer,
                                                 sql_str_to_use,
                                                 &json_loc,
                                                 query_sql_.getRenderQueryOptions(),
                                                 render_query_type_);

      auto const& sql_selected_tables = render_parse_info.getSqlSelectedTables();
      RUNTIME_EX_ASSERT(!sql_selected_tables.phys_tables.empty(),
                        "Physical table references are required to render polys. Logical "
                        "poly tables are not supported.");

      auto all_table_info_ptr = query_sql_.getAllTableInfoPtr();
      CHECK(all_table_info_ptr);
      auto const& phys_tables = all_table_info_ptr->phys_tables;
      LOG_IF(WARNING,
             !phys_tables.empty() && phys_tables[0].table_name !=
                                         sql_selected_tables.phys_tables[0].table_name)
          << "The primary table to render \""
          << sql_selected_tables.phys_tables[0].table_name
          << "\" does not match the dbTableName json attribute: "
          << phys_tables[0].table_name << ". Using the table referenced in the query.";

      query_sql_.setAllTableInfo(
          std::make_shared<SQLSelectedTableContainer>(std::move(sql_selected_tables)));
    }
  } else {
    // This branch gets hit with xform ops (and possibly other cases)
    auto phys_tables = query_sql_.getAllTableInfoRef().phys_tables;
    LOG_IF(WARNING, !phys_tables.empty() && phys_tables != curr_table_info_->phys_tables)
        << "The primary poly table to render \""
        << curr_table_info_->phys_tables[0].table_name
        << "\" does not match the \"dbTableName\" json attribute: "
        << phys_tables[0].table_name << ". Using the table referenced in the query.";
    query_sql_.setAllTableInfo(curr_table_info_);

    // return force || !isInternalCacheUpToDate();

    // isInternalCacheUpToDate() always returns false so we *always* return true,
    // resulting in query execution. I'm not sure this was intended but it's currently
    // necessary to duplicate previous behavior (several tests fail otherwise, mostly due
    // to XFormOps and QuerySourceDataTable)
    return true;
  }
  // clear the stashed curr_table_info_ which is just to preserve the value prior to
  // updateSqlFromJSONObj being called
  curr_table_info_ = nullptr;

  return do_execute_query;
}

const std::string& SqlQueryPolyDataTableJSON::getSqlStrToUseRef() const {
  return (sql_query_str_override_ ? *sql_query_str_override_
                                  : query_sql_.getSqlQueryStr());
}

bool SqlQueryPolyDataTableJSON::queueQuery(const JSONLocation* json_loc,
                                           const heavyai::InSituFlags insitu_flags) {
  RENDER_LOG_SCOPE();
  bool will_query_run = false;
  if (is_update_pending_) {
    will_query_run = ctx_.getDataTableQueues().addToQueryQueue(
        shared_from_this(), *json_loc, sql_query_str_override_, insitu_flags);
    is_update_pending_ = false;
  }
  return will_query_run;
}

void SqlQueryPolyDataTableJSON::postRunQuery(bool did_query_execute) {
  RENDER_LOG_SCOPE() << "did_query_execute: " << did_query_execute;
  gpu_resources_->initGpuResourcesFromBuffers(ctx_.getGlobalContext(), name_);
}

bool SqlQueryPolyDataTableJSON::update() {
  // TODO(croot): only need to execute the query when
  // the query has changed -- which is currently handled in
  // BaseQueryDataTableSQLJSON::updateFromJSONObjAndQueueQuery, but what if the
  // tables themselves have changed? Should re-run query in that case here.
  // TODO(scb): is the above comment still meaningful?
  bool will_query_run =
      queueQuery(nullptr,
                 // TODO(croot): this argument is probably going to go away
                 // sometime soon as a result of moving all the poly column
                 // injection logic to the ExecuteRenderInterface layer
                 render_query_result_.getInSituFlags());

  return will_query_run;
}

void SqlQueryPolyDataTableJSON::resetStateFlags() {
  BaseQueryDataTableSQLJSON::resetStateFlags();
  query_type_changed_ = false;
}

}  // namespace QueryRenderer
