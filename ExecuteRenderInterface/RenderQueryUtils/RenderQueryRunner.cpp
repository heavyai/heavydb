/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/RenderQueryRunner.h"

#include "ExecuteRenderInterface/RenderQueryUtils/NonInsituQueryClassifier.h"
#include "ExecuteRenderInterface/RenderQueryUtils/RelScanTree.h"
#include "ExecuteRenderInterface/RenderQueryUtils/RenderQueryResultProcessor.h"
#include "ExecuteRenderInterface/RenderQueryUtils/RenderRelAlgUtils.h"
#include "ExecuteRenderInterface/RenderQueryUtils/SQLSelectedTablesBuilder.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryEngine/RelAlgDag.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryRenderer/Data/Enums/DataFormatType.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "Shared/measure.h"
#include "ThriftHandler/DBHandler.h"
namespace QueryRenderer {

namespace {

const StringDictionaryProxy& get_string_dict_proxy_for_column(
    const Catalog_Namespace::Catalog& catalog,
    const QueryDataLayout& query_data_layout,
    const std::string& column_name,
    const ResultSet& results) {
  auto const& col_type_info = query_data_layout.getAttrSQLTypeInfoRef(column_name);
  if (!col_type_info.is_string() || col_type_info.get_compression() != kENCODING_DICT) {
    throw std::runtime_error(
        "The column alias \"" + column_name +
        "\" is not a dictionary-encoded string column. Cannot get string ids.");
  }
  const auto& dict_key = col_type_info.getStringDictKey();

  auto row_set_mem_owner = results.getRowSetMemOwner();
  if (!row_set_mem_owner) {
    throw std::runtime_error(
        "A RowSetMemoryOwner object is not defined which is required to retrieve the "
        "string dictionary proxy. Cannot get string ids for column alias \"" +
        column_name + "\".");
  }

  StringDictionaryProxy* sdp(nullptr);
  if (!dict_key.dict_id) {
    sdp = row_set_mem_owner->getLiteralStringDictProxy();
  } else {
    // NOTE: the below call to RowSetMemOwner::getOrAddStringDictProxy is not currently
    // lock protected. Usually this call is done through the
    // Executor::getStringDictionaryProxy() method, which does its own locking before
    // calling RowSetMemOwner::getOrAddStringDictProxy, but we don't have direct acess to
    // an executor at this point, so we are doing our own direct call. However it is not
    // lock protected. If this code should ever be called from multiple threads, we would
    // need to lock this appropriately.
    sdp = row_set_mem_owner->getOrAddStringDictProxy(dict_key, false);
  }

  if (!sdp) {
    throw std::runtime_error("The string dictionary for dictionary (dict_id: " +
                             std::to_string(dict_key.dict_id) +
                             ", db_id: " + std::to_string(dict_key.db_id) +
                             ")"
                             " is undefined. Cannot get string ids for column alias \"" +
                             column_name + "\".");
  }

  return *sdp;
}

}  // namespace

bool RenderQueryRunner::use_parallel_executors_ = false;

RenderQueryRunner::RenderQueryRunner(DBHandler& db_handler,
                                     RenderHandler::Impl& render_handler,
                                     QueryRenderManager& render_manager,
                                     const RenderSessionKey& render_session_key)
    : render_handler_(render_handler)
    , render_manager_(render_manager)
    , render_info_(render_session_key, RenderQueryOptions())
    , render_query_exec_opts_({db_handler.allow_multifrag_,
                               db_handler.allow_loop_joins_,
                               g_enable_watchdog,
                               g_enable_dynamic_watchdog,
                               g_dynamic_watchdog_time_limit,
                               db_handler.system_parameters_.gpu_input_mem_limit,
                               db_handler.jit_debug_}) {
  RENDER_LOG_SCOPE();
}

void RenderQueryRunner::setUseParallelExecutors(const bool use_parallel_executors) {
  use_parallel_executors_ = use_parallel_executors;
}

void RenderQueryRunner::notifyQueryExecutionComplete() const {
  RENDER_LOG_SCOPE();
  if (render_info_.render_allocator_map_ptr) {
    render_info_.render_allocator_map_ptr->prepForRendering(nullptr);
  }
}

RenderQueryParseData RenderQueryRunner::executeQueryParse(
    RenderQueryExecuteTimer& render_timer,
    const std::string& query_str,
    const JSONLocation*,
    const RenderQueryOptions& query_opts,
    const RenderQuerySpecialtyType render_query_type) {
  RENDER_LOG_SCOPE();
  CHECK(render_query_type == RenderQuerySpecialtyType::kPolys)
      << "pre-parse of the render-queries are currently only supported for POLYS";

  // TODO(croot):
  // If we do a pre-parse of the query, we don't need to re-gather data, like the
  // referenced tables, during the execution of the query. We should find a way
  // that these parse-steps can be ignored during execution (and re-use the
  // RenderQueryParseData generated here during the execution)
  render_info_.reset(query_opts, heavyai::InSituFlags::kInSitu);
  render_info_.setRenderQueryStr(query_str);

  // TODO(croot): if we parse and execute, we're duplicating the parse_to_ra(...)
  // right now. This could be averted by moving query_ra outside and capture it in
  // the lambda be we need to be careful about the mult-layer case, so the
  // query_ra needs to be cleared out after every execution to clear it out before
  // the parse/execution of the next layer's query
  RenderHandler::Impl::QueryLockState query_lock_state;
  auto clock_begin = timer_start();
  auto query_ra = render_handler_.parseToRelAlgAndAcquireLocks(
      query_str, render_info_, &query_lock_state);
  render_timer.query_parse_time_ms += timer_stop(clock_begin);

  clock_begin = timer_start();
  auto query_data_layout =
      getDataLayoutFromRelAlgDag(buildRelAlgDagAndClassifyRender(render_info_, query_ra))
          .second;
  render_timer.query_execution_time_ms += timer_stop(clock_begin);

  // TODO(croot): note that the following is passing the query_data_layout as the
  // ssbo layout when a lines/polys render. Could there ever be a situation where
  // the data from the query for lines/polys could go in the vbo? If so, we need
  // to properly populate the vbo here then too.
  const bool is_notpolyline_render = render_query_type == RenderQuerySpecialtyType::kNone;
  return {
      RenderQueryStructure(render_info_.targets,
                           getReferencedTablesPostQueryRun(render_query_type),
                           render_info_.getInSituFlags()),
      RenderQueryBufferLayouts((is_notpolyline_render ? query_data_layout : nullptr),
                               (is_notpolyline_render ? nullptr : query_data_layout))};
}

namespace {

struct MeshValidateInputState {
  RenderHandler::Impl& render_handler;
  const std::shared_ptr<const Catalog_Namespace::SessionInfo>& session_info;
  const std::string& original_raster_query_str;

  ExecutionResult sqlExecute(const std::string& query_string) {
    return render_handler.sqlExecute(session_info, query_string);
  }
};

/**
 * Gets a numeric value from a TargetValue. It is assumed that A) the value is a Scalar
 * and B) the caller already knows the underlying type of the TargetValue
 */
template <typename T>
T get_numeric_value_from_column(const TargetValue& target_value) {
  auto const* scalar_value = boost::get<ScalarTargetValue>(&target_value);
  CHECK(scalar_value != nullptr);
  auto const* value = boost::get<T>(scalar_value);
  CHECK(value != nullptr);
  return *value;
}

auto* get_bigint_from_column = &get_numeric_value_from_column<int64_t>;

/**
 * Gets a numeric value from a TargetValue by casting the underlying type. It is assumed
 * the TargetValue is scalar.
 * NOTE: this does not do any bounds checking.
 */
template <typename T>
T get_numeric_column(const TargetValue& value) {
  auto const* scalar_value = boost::get<ScalarTargetValue>(&value);
  CHECK(scalar_value != nullptr);
  return boost::apply_visitor(
      [](auto&& val) {
        using ValType = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<ValType, NullableString>) {
          CHECK(false);
          return T{0};
        } else {
          // TODO(croot): do bounds checking?
          return static_cast<T>(val);
        }
      },
      *scalar_value);
}

/**
 * Executes a validation prequery with proper exception throws in case of an error.
 */
ExecutionResult execute_validate_prequery(MeshValidateInputState& state,
                                          const JSONLocation& json_loc,
                                          const std::string& query_string) {
  RENDER_LOG_SCOPE();
  try {
    return state.sqlExecute(query_string);
  } catch (const std::exception& e) {
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        json_loc,
        "Exception running prequery to validate raster data for query '" +
            state.original_raster_query_str + "'. " + e.what()));
  }
}

/**
 * Gets a TargetValue for a specific column at a specific row in a result set
 * By default, decimal columns are converted to double
 */
inline TargetValue getColumnFromRow(const ResultSet& results,
                                    const size_t row_idx,
                                    const size_t col_idx,
                                    const bool convert_decimal_to_double = true) {
  // NOTE: the size_t type for row/col idx is needed to disambiguate
  // the getRowAt() method. The intent here is to call the getRowAt() method that returns
  // a single TargetValue. That method has a 'decimal_to_double' argument that controls
  // whether to convert decimals to doubles, which is generally desirable to do here when
  // materializing mesh properties.
  return results.getRowAt(row_idx, col_idx, false, convert_decimal_to_double);
}

using CreateErrorStreamFunc =
    std::function<std::stringstream(const MeshValidateInputState&)>;

/**
 * utility function that creates a stringstream for raster2d validation errors and
 * initializes the stream with a common error prefix.
 */
std::stringstream create_raster2d_error_stream(const MeshValidateInputState& state) {
  std::stringstream error_ss;
  error_ss << "Invalid 2d raster mesh. The result of render query '"
           << state.original_raster_query_str << "'";
  return error_ss;
}

/**
 * Gets the size of a single dimension of a 2d/3d raster grid. It is assumed the
 * dimension_size_query string only results in a min/max to verify that the dimension is
 * uniform across the domain. This function then checks that the min/max are equal,
 * verifying uniformity.
 */
uint32_t get_dimension_size_from_query(CreateErrorStreamFunc create_error_stream,
                                       MeshValidateInputState& state,
                                       const JSONLocation& json_loc,
                                       const std::string& dimension_size_query,
                                       const char dimension_tag) {
  auto dimension_size_result =
      execute_validate_prequery(state, json_loc, dimension_size_query).getRows();
  CHECK(dimension_size_result);
  CHECK_EQ(dimension_size_result->rowCount(), 1u);
  auto const min_dimension_size =
      get_bigint_from_column(getColumnFromRow(*dimension_size_result, 0, 0));
  auto const max_dimension_size =
      get_bigint_from_column(getColumnFromRow(*dimension_size_result, 0, 1));

  if (min_dimension_size != max_dimension_size) {
    auto error_ss = create_error_stream(state);
    error_ss << " is irregular in the " << dimension_tag << " dimension. One row is size "
             << min_dimension_size << " , another is size " << max_dimension_size << ".";
    THROW_RUNTIME_EX(error_ss.str());
  }

  // TODO(croot): should we put limits on the size of the raster that we can process?
  CHECK_LE(min_dimension_size, std::numeric_limits<uint32_t>::max());

  return static_cast<uint32_t>(min_dimension_size);
}

std::pair<RasterMeshMetadata, std::string> validate_raster_mesh2d(
    MeshValidateInputState& state,
    const JSONLocation& format_loc) {
  RasterMeshMetadata raster_mesh_metadata;
  RasterMeshFormatJson::parse(raster_mesh_metadata, format_loc);
  auto const coords_loc = format_loc.getMember(JSONSchema_v1::Data::kCoordsProp);

  auto const count_star_query =
      "SELECT /*+ keep_result */ count(*) as total_count FROM (" +
      state.original_raster_query_str + ")";

  auto const y_dimension_size_query =
      "SELECT /*+ keep_result */ min(total_count), max(total_count) from (" +
      count_star_query + " GROUP BY " + raster_mesh_metadata.x_coord_name + ")";

  auto const x_dimension_size_query =
      "SELECT /*+ keep_result */ min(total_count), max(total_count) from (" +
      count_star_query + " GROUP BY " + raster_mesh_metadata.y_coord_name + ")";

  auto count_star_result =
      execute_validate_prequery(state, coords_loc, count_star_query).getRows();
  CHECK(count_star_result);
  CHECK_EQ(count_star_result->rowCount(), 1u);
  auto const total_raster_size =
      get_bigint_from_column(getColumnFromRow(*count_star_result, 0, 0));

  auto& raster_size_x = raster_mesh_metadata.width;
  auto& raster_size_y = raster_mesh_metadata.height;

  if (total_raster_size > 0) {
    raster_size_x = get_dimension_size_from_query(
        create_raster2d_error_stream, state, coords_loc, x_dimension_size_query, 'X');
    raster_size_y = get_dimension_size_from_query(
        create_raster2d_error_stream, state, coords_loc, y_dimension_size_query, 'Y');

    const uint64_t calculated_total_size = raster_size_x * raster_size_y;

    if (calculated_total_size != static_cast<uint64_t>(total_raster_size)) {
      auto error_ss = create_raster2d_error_stream(state);
      error_ss
          << " would not be planar. The resolution of the raster is calculated to be "
          << raster_size_x << "x" << raster_size_y << " = " << calculated_total_size
          << ", but there are " << total_raster_size
          << " total pixels resulting from the query)";
      THROW_RUNTIME_EX(error_ss.str());
    }
  }

  return {raster_mesh_metadata,
          state.original_raster_query_str + " ORDER BY " +
              raster_mesh_metadata.y_coord_name + ", " +
              raster_mesh_metadata.x_coord_name + " ASC"};
}

std::string validate_cross_section1d(MeshValidateInputState& state,
                                     const JSONLocation& format_loc) {
  CrossSectionMetadata cross_section_metadata;
  CrossSectionFormatJson::parse(cross_section_metadata, format_loc);

  auto const& x_coord = cross_section_metadata.x_coord_name;
  auto const& y_coord = cross_section_metadata.y_coord_name;
  auto const& line = cross_section_metadata.linestring;

  // get num_points
  RUNTIME_EX_ASSERT(format_loc.hasMember(JSONSchema_v1::Data::kCrossSectionNumPointsProp),
                    "Cross Section 1D num_points property not present");
  auto const num_points_loc =
      format_loc.getMember(JSONSchema_v1::Data::kCrossSectionNumPointsProp);
  auto const num_points = num_points_loc.getInt();
  RUNTIME_EX_ASSERT(num_points >= 2, "Cross Section 1D num_points must be at least 2");

  // derive the ST_DWithin distance
  auto const spacing_x = (line[1][0] - line[0][0]) / static_cast<double>(num_points);
  auto const spacing_y = (line[1][1] - line[0][1]) / static_cast<double>(num_points);
  auto const within_distance = std::sqrt(std::pow(spacing_x, 2) + std::pow(spacing_y, 2));

  // adjust the query to filter using an ST_DWITHIN(cross_section_line, st_point(x, y))
  // clause to only grab points within a distance of the cross-section line, and wrap
  // the result in the CS1D table function to output the final LINESTRING
  auto const line_x1 = std::to_string(line[0][0]);
  auto const line_y1 = std::to_string(line[0][1]);
  auto const line_x2 = std::to_string(line[1][0]);
  auto const line_y2 = std::to_string(line[1][1]);
  return "SELECT * FROM TABLE(tf_cross_section_1d(raster => CURSOR(SELECT * FROM (" +
         state.original_raster_query_str +
         ") WHERE ST_DWITHIN(ST_GeomFromText('LINESTRING(" + line_x1 + " " + line_y1 +
         ", " + line_x2 + " " + line_y2 + ")'), ST_Point(" + x_coord + ", " + y_coord +
         "), " + std::to_string(within_distance) + ")), line_x1 => " + line_x1 +
         ", line_y1 => " + line_y1 + ", line_x2 => " + line_x2 + ", line_y2 => " +
         line_y2 + ", num_points => " + std::to_string(num_points) + "))";
}

std::pair<RasterMeshMetadata, std::string> validate_cross_section2d(
    MeshValidateInputState& state,
    const JSONLocation& format_loc) {
  CrossSectionMetadata cross_section_metadata;
  CrossSectionFormatJson::parse(cross_section_metadata, format_loc);

  auto const& x_coord = cross_section_metadata.x_coord_name;
  auto const& y_coord = cross_section_metadata.y_coord_name;
  auto const& line = cross_section_metadata.linestring;

  // get num_points_x|y
  auto const num_points_x_loc =
      format_loc.getMember(JSONSchema_v1::Data::kCrossSectionNumPointsXProp);
  auto const num_points_y_loc =
      format_loc.getMember(JSONSchema_v1::Data::kCrossSectionNumPointsYProp);
  auto const num_points_x = num_points_x_loc.getInt();
  auto const num_points_y = num_points_y_loc.getInt();
  RUNTIME_EX_ASSERT(num_points_x >= 2,
                    "Cross Section 2D num_points_x must be at least 2");
  RUNTIME_EX_ASSERT(num_points_y >= 2,
                    "Cross Section 2D num_points_y must be at least 2");

  // get dwithin_distance
  RUNTIME_EX_ASSERT(
      format_loc.hasMember(JSONSchema_v1::Data::kCrossSectionDWithinDistanceProp),
      "Cross Section 2D dwithin_distance property not present");
  auto const dwithin_distance_loc =
      format_loc.getMember(JSONSchema_v1::Data::kCrossSectionDWithinDistanceProp);
  auto const dwithin_distance = dwithin_distance_loc.getDouble();
  RUNTIME_EX_ASSERT(dwithin_distance > 0.0,
                    "Cross Section 2D dwithin_distance must be a positive value");

  // convert metadata for rendering as mesh2d
  RasterMeshMetadata raster_mesh_metadata;
  raster_mesh_metadata.x_coord_name = x_coord;
  raster_mesh_metadata.y_coord_name = y_coord;
  raster_mesh_metadata.width = num_points_x;
  raster_mesh_metadata.height = num_points_y;

  // adjust the query to filter using an ST_DWITHIN(cross_section_line, st_point(x, y))
  // clause to only grab points within a distance of the cross-section line, and wrap
  // the result in the CS2D table function to output the final mesh vertex values
  // and ensure sorted in row-major order
  auto const line_x1 = std::to_string(line[0][0]);
  auto const line_y1 = std::to_string(line[0][1]);
  auto const line_x2 = std::to_string(line[1][0]);
  auto const line_y2 = std::to_string(line[1][1]);
  return {raster_mesh_metadata,
          "SELECT * FROM TABLE(tf_cross_section_2d(raster => CURSOR(SELECT * FROM (" +
              state.original_raster_query_str +
              ") WHERE ST_DWITHIN(ST_GeomFromText('LINESTRING(" + line_x1 + " " +
              line_y1 + ", " + line_x2 + " " + line_y2 + ")'), ST_Point(" + x_coord +
              ", " + y_coord + "), " + std::to_string(dwithin_distance) +
              ")), line_x1 => " + line_x1 + ", line_y1 => " + line_y1 + ", line_x2 => " +
              line_x2 + ", line_y2 => " + line_y2 + ", num_points_x => " +
              std::to_string(num_points_x) + ", num_points_y => " +
              std::to_string(num_points_y) + ", dwithin_distance => " +
              std::to_string(dwithin_distance) + ")) ORDER BY y, x ASC"};
}

std::optional<std::string> validate_data_format(RasterMeshMetadata& raster_mesh_metadata,
                                                MeshValidateInputState state,
                                                const JSONLocation& data_loc) {
  std::optional<std::string> query_str_override;
  const auto format_loc = data_loc.getMember(JSONSchema_v1::Data::kFormatProp);
  CHECK(format_loc.isValid());

  if (format_loc.isObject()) {
    auto format_type_loc = format_loc.getMember(JSONSchema_v1::Data::kTypeProp);
    CHECK(format_type_loc.isValid());
    CHECK(format_type_loc.isString());

    auto const format_type = to_lower(format_type_loc.getString());

    auto data_format = get_data_format_from_string(format_type);

    switch (data_format) {
      case DataFormatType::kRasterMesh2d:
        std::tie(raster_mesh_metadata, query_str_override) =
            validate_raster_mesh2d(state, format_loc);
        break;
      case DataFormatType::kCrossSection1d:
        query_str_override = validate_cross_section1d(state, format_loc);
        break;
      case DataFormatType::kCrossSection2d:
        std::tie(raster_mesh_metadata, query_str_override) =
            validate_cross_section2d(state, format_loc);
        break;
      case DataFormatType::kLines:
      case DataFormatType::kUnknown:
        break;
    }
  }

  return query_str_override;
}

}  // namespace

RenderQueryExecuteData RenderQueryRunner::executeQuery(
    RenderQueryExecuteTimer& render_timer,
    const std::string& query_str,
    const JSONLocation* data_loc,
    const RenderQueryOptions& query_opts,
    const RenderQuerySpecialtyType render_query_type,
    const heavyai::InSituFlags insitu_flags) {
  RENDER_LOG_SCOPE();
  // reset any layouts from prior executions
  render_info_.reset(query_opts, insitu_flags);

  auto query_str_override = query_str;
  if (render_query_type == RenderQuerySpecialtyType::kLines ||
      render_query_type == RenderQuerySpecialtyType::kMesh2d) {
    CHECK(data_loc);
    auto const optional_query_str = validate_data_format(
        raster_mesh_metadata_,
        {render_handler_, render_info_.getSessionInfoPtr(), query_str},
        *data_loc);
    if (optional_query_str) {
      query_str_override = *optional_query_str;
    }
  }
  render_info_.setRenderQueryStr(query_str_override);

#ifndef HAVE_CUDA
  // no CUDA at all
  render_info_.disableCudaBuffers();
#endif

  // TODO(croot): if we parse and execute, we're duplicating the
  // parse_to_ra(...) right now. This could be averted by moving query_ra
  // outside and capture it in the lambda be we need to be careful about the
  // mult-layer case, so the query_ra needs to be cleared out after every
  // execution to clear it out before the parse/execution of the next layer's
  // query
  RenderHandler::Impl::QueryLockState query_lock_state;
  auto clock_begin = timer_start();
  auto query_ra = render_handler_.parseToRelAlgAndAcquireLocks(
      query_str_override, render_info_, &query_lock_state);
  render_timer.query_parse_time_ms += timer_stop(clock_begin);

  // NOTE: render allocator only really needs to be initialized for in-situ queries, but
  // we won't know that until after the query is executed. Any hit initializing the
  // allocator for all queries should be negligible.
  // TODO(croot): could look into lazy initialization.
  initRenderAllocator();

  auto render_query_execute_data =
      executeRenderQueryImpl(render_timer,
                             query_ra,
                             buildRelAlgDagAndClassifyRender(render_info_, query_ra),
                             std::move(query_lock_state),
                             data_loc,
                             render_query_type);

  if (query_opts.requiresPhysicalTables() &&
      render_query_execute_data.getSqlSelectedPhysicalTables().empty()) {
    throw std::runtime_error(
        "Cannot complete render_vega request because physical tables are required, but "
        "there are no physical tables found in the query \"" +
        query_str + "\"." +
        (query_opts.isHitTestingEnabled()
             ? " Physical tables are required because hit-testing is enabled."
             : ""));
  }

  return render_query_execute_data;
}

std::vector<int32_t> RenderQueryRunner::getStringIds(
    const QueryDataLayout& query_data_layout,
    const std::string& column_name,
    const std::vector<std::string>& column_values_to_convert,
    const ResultSet& results,
    const bool warn) const {
  if (!column_values_to_convert.size()) {
    return {};
  }

  auto const& sdp =
      get_string_dict_proxy_for_column(render_info_.getSessionInfo().getCatalog(),
                                       query_data_layout,
                                       column_name,
                                       results);

  int32_t string_id;
  std::vector<int32_t> string_ids;
  for (const auto& column_value : column_values_to_convert) {
    string_id = sdp.getIdOfStringNoGeneration(column_value);
    if (string_id == StringDictionary::INVALID_STR_ID) {
      const auto error_str = "The string \"" + column_value +
                             "\" does not have a valid id in the dictionary-encoded "
                             "string column aliased by \"" +
                             column_name + "\".";
      if (warn) {
        LOG(WARNING) << error_str;
      } else {
        throw std::runtime_error(error_str);
      }
    }
    string_ids.push_back(string_id);
  }
  return string_ids;
}

std::vector<std::string> RenderQueryRunner::getStringsFromIds(
    const QueryDataLayout& query_data_layout,
    const std::string& column_name,
    const std::vector<int32_t>& column_value_ids,
    const ResultSet& results) const {
  if (!column_value_ids.size()) {
    return {};
  }

  auto const& sdp =
      get_string_dict_proxy_for_column(render_info_.getSessionInfo().getCatalog(),
                                       query_data_layout,
                                       column_name,
                                       results);

  std::vector<std::string> stringvals;
  for (const auto& id : column_value_ids) {
    stringvals.push_back(sdp.getString(id));
  }
  return stringvals;
}

std::unique_ptr<RelAlgDag> RenderQueryRunner::buildRelAlgDagAndClassifyRender(
    RenderInfo& render_info,
    const std::string& query_ra) {
  RENDER_LOG_SCOPE();
  auto rel_alg_dag = RelAlgDagBuilder::buildDag(query_ra, false);

  // build an alternate bi-directional tree representation of the dag for easier
  // dag traversal isolated to paths with a direct line-of-sight to RelScan nodes without
  // going through aggregates
  auto rel_scan_tree = RelScanTree::create(*rel_alg_dag);

  // we need to do a scan of the ra tree to determine if we need to set or force the query
  // non-insitu. Queries should only be forced non-insitu for the purposes of
  // hit-testing (rebuilding the query for hit-testing would not work or would otherwise
  // be slow). There are a handful of cases where queries should run non-insitu regardless
  // of whether hit-testing is enabled or not (i.e. aggregate queries or queries w/ window
  // functions or cursor-less table functions)
  NonInsituQueryClassifier::classify(render_info, *rel_alg_dag, rel_scan_tree.get());

  // Alter the RA for render. Do this before any flattening/optimizations are done to
  // the tree.
  RenderRelAlgUtils::alterRAForRender(rel_scan_tree.get(), render_info);

  RelAlgDagBuilder::optimizeDag(*rel_alg_dag);

  return rel_alg_dag;
}

ExecutionResult RenderQueryRunner::validateQuery(
    RenderHandler::Impl& render_handler,
    RenderInfo& render_info,
    std::unique_ptr<RelAlgDag> rel_alg_dag,
    const RenderQueryExecutionOptions& render_query_exec_opts) {
  RENDER_LOG_SCOPE();
  ExecutionResult result;

  auto timer = DEBUG_TIMER(__func__);

  std::optional<logger::ThreadLocalIds> parent_thread_local_ids;
  auto validate_func =
      std::make_unique<std::packaged_task<void(size_t, std::string, std::string)>>(
          [&](const size_t executor_index,
              const std::string& query_session,
              const std::string& submitted_time_str) {
            std::optional<logger::LocalIdsScopeGuard> lisg;
            if (parent_thread_local_ids) {
              lisg = parent_thread_local_ids->setNewThreadId();
            }
            auto executor = Executor::getExecutor(
                executor_index,
                render_handler.db_handler_->jit_debug_ ? "/tmp" : "",
                render_handler.db_handler_->jit_debug_ ? "mapdquery" : "",
                render_handler.db_handler_->system_parameters_);
            const auto& session_info = render_info.getSessionInfo();
            const auto [compilation_opts, execution_opts] =
                render_query_exec_opts.convertToRelAlgExecutorOptions(session_info, true);
            RelAlgExecutor rel_alg_executor(
                executor.get(), std::move(rel_alg_dag), session_info);
            result = rel_alg_executor.executeRelAlgQuery(
                compilation_opts, execution_opts, false, false, &render_info);
          });
  auto const query_session = render_info.getSessionInfo().get_session_id();
  // let's not register this validation query session to query queue
  // i.e., executor->enrollQuerySession(...)
  if (use_parallel_executors_) {
    parent_thread_local_ids = logger::thread_local_ids();
    auto validate_task = std::make_shared<QueryDispatchQueue::Task>(
        std::move(validate_func), query_session, "");
    render_handler.dispatch_query_task(validate_task);
    auto result_future = validate_task->execute_rel_alg_task->get_future().share();
    result_future.get();
  } else {
    (*validate_func)(Executor::UNITARY_EXECUTOR_ID, query_session, "");
  }
  return result;
}

ExecutionResult RenderQueryRunner::executeQuery(
    RenderHandler::Impl& render_handler,
    RenderInfo& render_info,
    std::unique_ptr<RelAlgDag> rel_alg_dag,
    const RenderQueryExecutionOptions& render_query_exec_opts) {
  RENDER_LOG_SCOPE();
  const auto& session_info = render_info.getSessionInfo();
  CompilationOptions compilation_opts;
  ExecutionOptions execution_opts;
  std::tie(compilation_opts, execution_opts) =
      render_query_exec_opts.convertToRelAlgExecutorOptions(session_info, false);

  if (compilation_opts.device_type == ExecutorDeviceType::CPU &&
      render_info.useCudaBuffers()) {
    render_info.forceNonInSitu();
  }

  auto timer = DEBUG_TIMER(__func__);

  ExecutionResult result;
  std::optional<logger::ThreadLocalIds> parent_thread_local_ids;
  auto query_execute_func =
      std::make_unique<std::packaged_task<void(size_t, std::string, std::string)>>(
          [&](auto const executor_index,
              const std::string& query_session,
              const std::string& submitted_time_str) {
            std::optional<logger::LocalIdsScopeGuard> lisg;
            if (parent_thread_local_ids) {
              lisg = parent_thread_local_ids->setNewThreadId();
            }
            auto executor = Executor::getExecutor(
                executor_index,
                render_handler.db_handler_->jit_debug_ ? "/tmp" : "",
                render_handler.db_handler_->jit_debug_ ? "mapdquery" : "",
                render_handler.db_handler_->system_parameters_);
            RelAlgExecutor rel_alg_executor(
                executor.get(), std::move(rel_alg_dag), session_info);
            result = rel_alg_executor.executeRelAlgQuery(
                compilation_opts, execution_opts, false, false, &render_info);
          });
  auto const query_session = render_info.getSessionInfo().get_session_id();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  CHECK(executor);
  if (g_enable_runtime_query_interrupt && !query_session.empty()) {
    executor->enrollQuerySession(
        query_session,
        hide_sensitive_data_from_query(render_info.getRenderQueryStr()),
        "",
        Executor::UNITARY_EXECUTOR_ID,
        QuerySessionStatus::QueryStatus::PENDING_QUEUE);
  }
  if (use_parallel_executors_) {
    parent_thread_local_ids = logger::thread_local_ids();
    auto query_execute_task = std::make_shared<QueryDispatchQueue::Task>(
        std::move(query_execute_func), query_session, "");
    render_handler.dispatch_query_task(query_execute_task);
    auto future = query_execute_task->execute_rel_alg_task->get_future().share();
    future.get();
  } else {
    (*query_execute_func)(Executor::UNITARY_EXECUTOR_ID, query_session, "");
  }
  return result;
}

RenderQueryExecuteData RenderQueryRunner::executeRenderQueryImpl(
    RenderQueryExecuteTimer& render_timer,
    const std::string& query_ra,
    std::unique_ptr<RelAlgDag> rel_alg_dag,
    RenderHandler::Impl::QueryLockState&& query_lock_state,
    const JSONLocation* data_loc,
    const RenderQuerySpecialtyType render_query_type) {
  RENDER_LOG_SCOPE();
  // now run the query
  auto rtnData = executeRenderQuery(render_timer,
                                    std::move(rel_alg_dag),
                                    std::move(query_lock_state),
                                    data_loc,
                                    render_query_type);

  return rtnData;
}

void RenderQueryRunner::initRenderAllocator() {
  if (!render_info_.render_allocator_map_ptr) {
    render_info_.render_allocator_map_ptr =
        std::make_unique<RenderAllocatorMap>(&render_manager_);
  }
}

SQLSelectedTableContainer RenderQueryRunner::getReferencedTablesPostQueryRun(
    const RenderQuerySpecialtyType render_query_type) {
  RENDER_LOG_SCOPE();
  // maintains the list of tables/views used in the query
  SQLSelectedTableContainer sql_selected_tables;

  // builds the list of tables/views and resolves views
  SQLSelectedTablesBuilder table_builder{*this, sql_selected_tables};

  for (const auto& table_name : render_info_.table_names) {
    const auto cat =
        Catalog_Namespace::SysCatalog::instance().getCatalog(table_name.db_name);
    CHECK(cat);
    const auto table_meta = cat->getMetadataForTable(table_name.table_name);
    CHECK(table_meta) << "Cant get metadata for table: \"" << table_name.table_name
                      << "\" in catalog: \"" << table_name.db_name << "\"";
    table_builder.push_back(*cat, *table_meta);
  }

  return sql_selected_tables;
}

std::pair<ExecutionResult, QueryDataLayoutShPtr>
RenderQueryRunner::getDataLayoutFromRelAlgDag(std::unique_ptr<RelAlgDag> rel_alg_dag) {
  RENDER_LOG_SCOPE();
  auto result = validateQuery(
      render_handler_, render_info_, std::move(rel_alg_dag), render_query_exec_opts_);
  return std::make_pair(
      result,
      std::make_shared<QueryDataLayout>(&render_info_.getSessionInfo().getCatalog(),
                                        render_info_.targets,
                                        QueryDataLayout::LayoutType::kVertexInterleaved));
}

RenderQueryExecuteData RenderQueryRunner::executeRenderQuery(
    RenderQueryExecuteTimer& render_timer,
    std::unique_ptr<RelAlgDag> rel_alg_dag,
    RenderHandler::Impl::QueryLockState&& query_lock_state,
    const JSONLocation* data_loc,
    const RenderQuerySpecialtyType render_query_type) {
  RENDER_LOG_SCOPE();

  // TODO(croot): have a utility function to execute queries, probably in DBHandler
  // somewhere so that we don't have two execution paths. Right now there are 2, the
  // execution path for sql_execute, and the execution path for render_vega when there
  // are sql queries. A single place would be preferred so that there is only one place
  // to add new code and to keep the two paths in sync when executing queries (the flush
  // in execute_leaf_render_query would've been caught immediately had that been done)
  //
  // The two code paths were split because DBHandler::sql_query_impl was designed to
  // work in the Thrift object space, where here we're not. So a proper utility function
  // would act as another layer of abstraction that DBHandler::sql_query_impl would
  // call.

  ExecutionResult exe_result;
  SQLSelectedTableContainer sql_selected_tables;
  std::chrono::steady_clock::time_point result_processor_clock;
  {
    // NOTE: locks will be released upon exit of this block
    auto execute_locks = std::move(query_lock_state);

    auto execution_clock = timer_start();
    exe_result = executeQuery(
        render_handler_, render_info_, std::move(rel_alg_dag), render_query_exec_opts_);

    const auto& results = exe_result.getRows();

    // reduce execution time by the time spent during queue waiting
    render_timer.query_execution_time_ms +=
        timer_stop(execution_clock) - results->getQueueTime();
    render_timer.queue_time_ms += results->getQueueTime();

    // start clock for query result processing
    result_processor_clock = timer_start();

    // need to call getReferencedTablesPostQueryRun under execution locks to keep the
    // catalog in-sync with what was used in query execution
    sql_selected_tables = getReferencedTablesPostQueryRun(render_query_type);
  }

  auto num_rows =
      RenderResultProcessor::processExecuteQueryResults(render_manager_,
                                                        exe_result,
                                                        data_loc,
                                                        render_query_type,
                                                        render_info_,
                                                        &raster_mesh_metadata_);

  render_timer.render_time_ms += timer_stop(result_processor_clock);

  return {RenderQueryBufferLayouts(render_info_.getQueryVboLayout(),
                                   render_info_.getQuerySsboLayout()),
          RenderQueryOutput(
              RenderQueryStructure(render_info_.targets,
                                   std::move(sql_selected_tables),
                                   render_info_.getInSituFlags()),
              RenderQueryResult(
                  exe_result.getRows(), num_rows, render_info_.getInSituFlags()))};
}

}  // namespace QueryRenderer
