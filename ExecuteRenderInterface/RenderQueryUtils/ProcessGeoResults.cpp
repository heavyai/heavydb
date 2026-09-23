/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/ProcessGeoResults.h"

#include <vector>

#include "ExecuteRenderInterface/RenderQueryUtils/ProcessResultsUtils.h"
#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustGeo.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryRenderer/Cache/LineMgr.h"
#include "QueryRenderer/Cache/PolyMgr.h"
#include "QueryRenderer/Data/Parsers/SqlQueryLineFormatJson.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/Interop/QueryBuffer.h"
#include "QueryRenderer/QueryRenderManager.h"
#include "QueryRenderer/Utils/ProfUtils.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContext.h"
#include "Shared/ShapeDrawData.h"
#include "Shared/measure.h"
#include "Shared/scope.h"
#include "Shared/thread_count.h"

#ifdef HAVE_CUDA
#include "CudaMgr/CudaMgr.h"
#include "GfxInterop/Utils/CudaErrorCheck.h"
#endif  // HAVE_CUDA

namespace QueryRenderer {

inline auto build_line_poly_vbo_layout(
    EncodingType coords_encoding,
    const QueryDataLayout::LayoutType vertex_layout_to_use =
        QueryDataLayout::LayoutType::kVertexInterleaved) {
  // the actual data type for the coords in the VBO
  // uncompressed = double
  // compressed = int32_t
  auto const sql_type =
      (coords_encoding == kENCODING_GEOINT) ? SQLTypes::kINT : SQLTypes::kDOUBLE;
  auto layout = std::make_shared<QueryDataLayout>(
      std::vector<QueryDataLayout::AttrAliasInfo>(
          {QueryDataLayout::AttrAliasInfo("x", SQLTypeInfo(sql_type, true)),
           QueryDataLayout::AttrAliasInfo("y", SQLTypeInfo(sql_type, true))}),
      vertex_layout_to_use,
      sql_type_to_render_type);
  return layout;
}

struct LineVertexBufferVisitor : public boost::static_visitor<double> {
  double operator()(const float& f) const { return static_cast<double>(f); }
  double operator()(const double& d) const { return d; }
  double operator()(const int64_t& i) const { return static_cast<double>(i); }
  double operator()(const NullableString& n) const {
    throw std::runtime_error(
        "Cannot convert NullableString to line vertex buffer data. Possible column "
        "validation failure.");
    return 0;
  }
};

using GeoLineCoordsAndSizes = std::pair<std::shared_ptr<std::vector<double>>,
                                        std::shared_ptr<std::vector<int32_t>>>;
struct GeoLineCoordsAndSizesBufferVisitor
    : public boost::static_visitor<GeoLineCoordsAndSizes> {
  template <class T>
  GeoLineCoordsAndSizes operator()(const T& tv) const {
    if constexpr (std::is_same_v<T, GeoMultiLineStringTargetValue>) {
      return {tv.coords, tv.linestring_sizes};
    } else if constexpr (std::is_same_v<T, GeoLineStringTargetValue>) {
      return {tv.coords, nullptr};
    }
    throw std::runtime_error(
        "Invalid geo target value type when trying to extract linestring coords and "
        "sizes. Expecting a LINESTRING/MULTILINESTRING.");
  }
};

void set_line_vertex_buffer_data_entry(
    const std::shared_ptr<std::vector<double>>& vertexDataShPtr,
    const SqlQueryLineFormatJson::LineVertexDataTemplate& vertexDataTemplate,
    const std::vector<TargetValue>& row,
    const std::vector<TargetMetaInfo>& targets,
    const size_t num_duplicate_entries) {
  CHECK_EQ(row.size(), targets.size());

  auto DataFromTargetValue = [](const ScalarTargetValue* scalar_tv) {
    return boost::apply_visitor(LineVertexBufferVisitor(), *scalar_tv);
  };
  auto GeoLineCoordsAndSizesFromTargetValue = [](const GeoTargetValue* geo_tv) {
    const auto& checked_geo_tv = geo_tv->get();
    return boost::apply_visitor(GeoLineCoordsAndSizesBufferVisitor(), checked_geo_tv);
  };

  //
  // LINESTRING/MULTILINESTRING
  // will always be interleaved, and never mixed with other columns
  // this code is directly equivalent to that in CopyVertexDataFunctor in ThrustLines
  //

  if (num_duplicate_entries == 2 &&
      vertexDataTemplate.target_column_indices.size() == 1) {
    auto const& tv = row[vertexDataTemplate.target_column_indices[0]];
    auto const* geo_tv = boost::get<GeoTargetValue>(&tv);
    if (geo_tv && geo_tv->is_initialized()) {
      // get coords and sizes
      auto const [geo_coords_arr_ptr, geo_linesizes_arr_ptr] =
          GeoLineCoordsAndSizesFromTargetValue(geo_tv);

      // coords mandatory, sizes optional
      CHECK(geo_coords_arr_ptr);

      // how many line sizes
      uint32_t num_linesizes;
      if (geo_linesizes_arr_ptr) {
        num_linesizes = geo_linesizes_arr_ptr->size();
      } else {
        num_linesizes = 1u;
      }

      // how many verts
      uint32_t num_verts = geo_coords_arr_ptr->size() / 2;

      // build lines
      uint32_t coord_index{};
      uint32_t last_line = num_linesizes - 1;
      for (uint32_t i = 0; i < num_linesizes; i++) {
        // this line size
        uint32_t linesize;
        if (geo_linesizes_arr_ptr) {
          // this line's actual size
          linesize = (*geo_linesizes_arr_ptr)[i];
        } else {
          // only one line, size = num_verts
          linesize = num_verts;
        }

        // repeat first vert
        vertexDataShPtr->push_back((*geo_coords_arr_ptr)[coord_index]);
        vertexDataShPtr->push_back((*geo_coords_arr_ptr)[coord_index + 1]);

        // copy verts
        for (uint32_t j = 0; j < linesize; j++) {
          vertexDataShPtr->push_back((*geo_coords_arr_ptr)[coord_index++]);
          vertexDataShPtr->push_back((*geo_coords_arr_ptr)[coord_index++]);
        }

        // repeat last vert
        vertexDataShPtr->push_back((*geo_coords_arr_ptr)[coord_index - 2]);
        vertexDataShPtr->push_back((*geo_coords_arr_ptr)[coord_index - 1]);

        // separator (between lines only)
        if (i < last_line) {
          static constexpr double kSeparatorValue = -std::numeric_limits<double>::max();
          vertexDataShPtr->push_back(kSeparatorValue);
          vertexDataShPtr->push_back(kSeparatorValue);
        }
      }

      // done
      return;
    }
  }

  //
  // more general scalar/array data (interleaved or sequential)
  //

  int ctr = num_duplicate_entries - 1;
  size_t offset = vertexDataShPtr->size();
  for (auto col_idx : vertexDataTemplate.target_column_indices) {
    const auto tv = row[col_idx];
    // TODO(adb): Use a top-level visitor for reading the TargetValue type
    auto const* geo_tv = boost::get<GeoTargetValue>(&tv);
    if (geo_tv) {
      throw std::runtime_error("Unexpected geo column in line render data template");
    }
    const ScalarTargetValue* scalar_tv = boost::get<ScalarTargetValue>(&tv);
    if (scalar_tv) {
      vertexDataShPtr->push_back(DataFromTargetValue(scalar_tv));
      if (ctr == 0) {
        for (size_t i = 0; i < num_duplicate_entries; i++) {
          vertexDataShPtr->push_back((*vertexDataShPtr)[offset + i]);
        }
      }
      ctr--;
      continue;
    }
    const auto array_tv = boost::get<ArrayTargetValue>(&tv);
    if (array_tv && array_tv->is_initialized()) {
      const auto& scalar_tv_vector = array_tv->get();
      for (const ScalarTargetValue& scalar_tv : scalar_tv_vector) {
        vertexDataShPtr->push_back(DataFromTargetValue(&scalar_tv));
        if (ctr == 0) {
          for (size_t i = 0; i < num_duplicate_entries; i++) {
            vertexDataShPtr->push_back((*vertexDataShPtr)[offset + i]);
          }
        }
        ctr--;
      }
    }
  }
  // Duplicate the last num_duplicate_vertices vertices (in reverse order)
  // only if we added any, and if we did, we must have added AT LEAST that many
  if (vertexDataShPtr->size() > offset) {
    CHECK_GE(vertexDataShPtr->size() - offset, num_duplicate_entries);
    for (size_t i = 0; i < num_duplicate_entries; i++) {
      vertexDataShPtr->push_back(*(vertexDataShPtr->end() - num_duplicate_entries));
    }
  }
}

uint64_t process_lines_non_in_situ(QueryRenderManager& render_manager,
                                   const ExecutionResult& results,
                                   const JSONLocation* data_loc,
                                   RenderInfo& render_info) {
  RENDER_LOG_SCOPE();

  const int gpu_idx = render_manager.getLeastSubscribedGpuId();
  VLOG(1) << "Selected gpu " << gpu_idx << " for non-insitu lines render";

  const auto& rows = results.getRows();
  rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  const auto entry_count = rows->entryCount();
  const auto row_count = rows->rowCount(entry_count > kMinRowCountWorthMultiThreading);

  const auto& resultTargets = results.getTargetsMeta();
  const auto rowid_status = get_rowid_status(resultTargets, render_info);

  CHECK(data_loc && data_loc->isValid());
  const auto name_loc = data_loc->getMember(JSONSchema_v1::Data::kNameProp);
  CHECK(name_loc.isValid() && name_loc.isString());
  std::string vega_data_table_name = name_loc.getString();

  // TODO(adb): Determine whether or not to use indices based on the vega specification
  bool use_index_buffer = false;

  SqlQueryLineFormatJson::QueryColumnInfoMap query_column_info_map;
  unsigned int query_column_index_ctr = 0;
  for (const auto& target : resultTargets) {
    const auto type_info = target.get_type_info();
    if (type_info.is_array() &&
        !(type_info.get_subtype() == kFLOAT || type_info.get_subtype() == kDOUBLE)) {
      const auto msg =
          std::string("Unable to complete line rendering request. Array column " +
                      target.get_resname() + " is not of type `float` or `double`.");
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
    if (type_info.is_geometry() &&
        !(type_info.get_type() == kLINESTRING ||
          type_info.get_type() == kMULTILINESTRING || type_info.get_type() == kPOINT)) {
      const auto msg =
          std::string("Unable to complete line rendering request. Geo column " +
                      target.get_resname() + " (" + type_info.get_type_name() +
                      ") is not of type `LINESTRING`, `MULTILINESTRING` or `POINT`.");
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
    query_column_info_map.insert(
        std::make_pair(target.get_resname(),
                       std::make_pair(query_column_index_ctr, type_info.get_type())));
    query_column_index_ctr++;
  }

  // Parse vega specification to get vertex layout and indices of vertex columns in
  // query
  CHECK(data_loc->hasMember(JSONSchema_v1::Data::kFormatProp));
  auto vertex_layout = SqlQueryLineFormatJson::SetVertexLayout(*data_loc);
  const auto lineVertexQueryTargets = SqlQueryLineFormatJson::SetVertexQueryTargets(
      *data_loc, query_column_info_map, vertex_layout);

  // Populate data query indices
  std::vector<unsigned int> data_query_indices;
  std::set<unsigned int> vertex_query_col_index_set(
      lineVertexQueryTargets.primary_vertex_query_indices.begin(),
      lineVertexQueryTargets.primary_vertex_query_indices.end());
  vertex_query_col_index_set.insert(
      lineVertexQueryTargets.secondary_vertex_query_indices.begin(),
      lineVertexQueryTargets.secondary_vertex_query_indices.end());
  for (unsigned int i = 0; i < resultTargets.size(); i++) {
    auto itr = vertex_query_col_index_set.find(i);
    if (itr == vertex_query_col_index_set.end()) {
      data_query_indices.push_back(i);
    }
  }

  std::vector<SqlQueryLineFormatJson::LineDrawBufferData> lineDrawBufferDataVec;

  std::pair<SqlQueryLineFormatJson::LineVertexDataTemplate,
            SqlQueryLineFormatJson::LineVertexDataTemplate>
      vertexDataTemplate{{lineVertexQueryTargets.primary_vertex_query_indices,
                          2 * DefaultNumBytesPerColumnType},
                         {lineVertexQueryTargets.secondary_vertex_query_indices,
                          2 * DefaultNumBytesPerColumnType}};

  auto data_query_result =
      get_render_data_template(resultTargets,
                               render_info.targets,
                               data_query_indices,
                               {},  // targets to ignore
                               {},  // extra target aliases
                               QueryDataLayout::LayoutType::kStorage,
                               row_count,
                               rowid_status,
                               true,   // uses a result set
                               true);  // allocate local row data buffer

  std::vector<gfx::IndirectDrawVertexData> indvboData(row_count);
  std::vector<gfx::IndirectDrawIndexData> indiboData;
  if (use_index_buffer) {
    indiboData.resize(row_count);
  }

  const auto do_work =
      [&rowid_status,
       &resultTargets,
       &data_query_result,
       &indvboData,
       &indiboData,
       &vertexDataTemplate,
       vertex_layout,
       use_index_buffer](const std::vector<TargetValue>& crt_row,
                         const size_t row_idx,
                         const size_t resultrow_entry_idx,
                         size_t& vertex_ctr,
                         SqlQueryLineFormatJson::LineDrawBufferData& lineDrawBufferData) {
        if (vertex_layout == QueryDataLayout::LayoutType::kVertexSequential) {
          set_line_vertex_buffer_data_entry(lineDrawBufferData.primary_vertices_ptr,
                                            vertexDataTemplate.first,
                                            crt_row,
                                            resultTargets,
                                            /*num_duplicate_entries=*/1);

          set_line_vertex_buffer_data_entry(lineDrawBufferData.secondary_vertices_ptr,
                                            vertexDataTemplate.second,
                                            crt_row,
                                            resultTargets,
                                            /*num_duplicate_entries=*/1);
        } else {
          set_line_vertex_buffer_data_entry(lineDrawBufferData.primary_vertices_ptr,
                                            vertexDataTemplate.first,
                                            crt_row,
                                            resultTargets,
                                            /*num_duplicate_entries=*/2);
        }

        set_non_in_situ_render_data_entry(data_query_result,
                                          crt_row,
                                          resultTargets,
                                          row_idx,
                                          resultrow_entry_idx,
                                          rowid_status,
                                          data_query_result.align_bytes);

        const auto numVerts = (lineDrawBufferData.primary_vertices_ptr->size() +
                               lineDrawBufferData.secondary_vertices_ptr->size()) /
                                  2 -
                              vertex_ctr;
        indvboData[row_idx] = gfx::IndirectDrawVertexData(numVerts, 1, -1);

        if (use_index_buffer) {
          // TODO(adb): support indices and indirect draw index buffer
          indiboData[row_idx] = gfx::IndirectDrawIndexData();
        }

        vertex_ctr += numVerts;
      };

  if (!DISABLE_MULTI_THREADING && !rows->isTruncated() &&
      entry_count > kMinRowCountWorthMultiThreading) {
    const size_t worker_count = cpu_threads();
    std::vector<size_t> worker_vertex_counts(worker_count, 0);
    std::vector<std::vector<size_t>> worker_rowid_assignment(worker_count);
    size_t approx_count_per_worker = row_count / worker_count;
    {
      // Since the workload can exhibit high skew, each thread returns a measure of
      // whether or not work was done along with the parsed draw buffer data. This
      // approach avoids extra cuMemCpy calls later.
      std::vector<
          std::future<std::pair<bool, SqlQueryLineFormatJson::LineDrawBufferData>>>
          buffer_threads;
      std::atomic<size_t> row_idx{0};
      for (size_t i = 0,
                  start_entry = 0,
                  stride = (entry_count + worker_count - 1) / worker_count;
           i < worker_count && start_entry < entry_count;
           ++i, start_entry += stride) {
        const auto end_entry = std::min(start_entry + stride, entry_count);
        buffer_threads.push_back(std::async(
            std::launch::async,
            [&rows,
             &do_work,
             &lineVertexQueryTargets,
             &worker_vertex_counts,
             &worker_rowid_assignment,
             &row_idx,
             &approx_count_per_worker,
             parent_thread_local_ids = logger::thread_local_ids()](
                const size_t start, const size_t end, const size_t worker_idx) {
              logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
              worker_rowid_assignment[worker_idx].reserve(approx_count_per_worker);

              SqlQueryLineFormatJson::LineDrawBufferData lineDrawBufferData;
              lineDrawBufferData.primary_vertices_ptr =
                  std::make_shared<std::vector<double>>();
              lineDrawBufferData.primary_vertices_ptr->reserve(
                  approx_count_per_worker *
                  lineVertexQueryTargets.primary_vertex_query_indices.size());

              lineDrawBufferData.secondary_vertices_ptr =
                  std::make_shared<std::vector<double>>();
              lineDrawBufferData.secondary_vertices_ptr->reserve(
                  approx_count_per_worker *
                  lineVertexQueryTargets.secondary_vertex_query_indices.size());
              for (size_t i = start; i < end; ++i) {
                const auto crt_row = rows->getRowAtNoTranslations(i);
                if (!crt_row.empty()) {
                  size_t cur_row_idx = row_idx.fetch_add(1);
                  worker_rowid_assignment[worker_idx].push_back(cur_row_idx);
                  do_work(crt_row,
                          cur_row_idx,
                          i,
                          worker_vertex_counts[worker_idx],
                          lineDrawBufferData);
                }
              }

              if ((lineDrawBufferData.primary_vertices_ptr->size() == 0) &&
                  (lineDrawBufferData.secondary_vertices_ptr->size() == 0)) {
                return std::make_pair(false,
                                      SqlQueryLineFormatJson::LineDrawBufferData{});
              }
              return std::make_pair(true, lineDrawBufferData);
            },
            start_entry,
            end_entry,
            i));
      }
      lineDrawBufferDataVec.reserve(worker_count);

      for (auto& child : buffer_threads) {
        const auto& ret = child.get();
        if (ret.first) {
          lineDrawBufferDataVec.push_back(ret.second);
        }
      }
    }

    // Now properly set the indvbo / indibo vertex counts in a second pass
    {
      std::vector<std::future<void>> buffer_threads;
      for (size_t i = 0; i < worker_count; i++) {
        buffer_threads.push_back(std::async(
            std::launch::async,
            [&indvboData,
             &worker_rowid_assignment,
             &worker_vertex_counts,
             parent_thread_local_ids =
                 logger::thread_local_ids()](const size_t worker_idx) {
              logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();

              // vertex_ctr_offset is the number of vertices processed by all
              // other workers that came before this one
              size_t vertex_ctr_offset = 0;
              for (size_t i = 0; i < worker_idx; i++) {
                vertex_ctr_offset += worker_vertex_counts[i];
              }

              for (const auto& row_idx : worker_rowid_assignment[worker_idx]) {
                indvboData[row_idx].first_vertex = vertex_ctr_offset;
                vertex_ctr_offset += indvboData[row_idx].vertex_count;
              }
            },
            i));
      }

      for (auto& child : buffer_threads) {
        child.get();
      }
    }
  } else {
    lineDrawBufferDataVec.emplace_back();
    auto& lineDrawBufferData = lineDrawBufferDataVec[0];
    lineDrawBufferData.primary_vertices_ptr = std::make_shared<std::vector<double>>();
    lineDrawBufferData.primary_vertices_ptr->reserve(
        row_count * lineVertexQueryTargets.primary_vertex_query_indices.size());

    lineDrawBufferData.secondary_vertices_ptr = std::make_shared<std::vector<double>>();
    lineDrawBufferData.secondary_vertices_ptr->reserve(
        row_count * lineVertexQueryTargets.secondary_vertex_query_indices.size());
    size_t row_idx = 0;
    size_t vertex_ctr = 0;
    rows->moveToBegin();
    while (true) {
      const auto crt_row = rows->getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      do_work(crt_row,
              row_idx,
              rows->getCurrentRowBufferIndex(),
              vertex_ctr,
              lineDrawBufferData);
      indvboData[row_idx].first_vertex = vertex_ctr - indvboData[row_idx].vertex_count;

      row_idx++;
    }
  }

  // If the buffer layout is interleaved, secondaryVertexData.data will be size 0
  size_t num_vertex_bytes = 0;
  for (const auto& workerVertexIndexData : lineDrawBufferDataVec) {
    num_vertex_bytes += (workerVertexIndexData.primary_vertices_ptr->size() +
                         workerVertexIndexData.secondary_vertices_ptr->size()) *
                        DefaultNumBytesPerColumnType;
  }

  LineTableByteData lineByteData(
      {num_vertex_bytes,
       0,
       data_query_result.data.size(),
       row_count * sizeof(gfx::IndirectDrawVertexData),
       use_index_buffer ? row_count * sizeof(gfx::IndirectDrawIndexData) : 0});

  auto vertLayout = build_line_poly_vbo_layout(kENCODING_NONE, vertex_layout);

  auto& line_mgr = render_manager.getLineMgr();
  line_mgr.bufferLineData(vega_data_table_name,
                          lineDrawBufferDataVec,
                          vertex_layout,
                          data_query_result.data,
                          lineByteData,
                          indvboData,
                          indiboData,
                          data_query_result.render_data_layout,
                          vertLayout,
                          use_index_buffer,
                          DefaultNumBytesPerColumnType,
                          gpu_idx);

  render_info.setQueryVboLayout(vertLayout);
  render_info.setQuerySsboLayout(data_query_result.render_data_layout);

  return row_count;
}

enum DependentColumns { XMIN = 0, XMAX, YMIN, YMAX, MAX_DEPENDENCIES };
using DependentIdxArray = std::array<int, DependentColumns::MAX_DEPENDENCIES>;

/**
 * A utility struct that describes the structure for the column slots of a query output
 * buffer (QOB) that are reserved for poly-related data when rendering a non-insitu poly
 * query. An instance of this struct will map out the layout of the geo slots in the QOB.
 * For more info on the layout here, see the comments ahead of the PolyResultBufferInfo
 * class.
 */
struct QOBGeoSlotDescriptor {
  // total # of 8-byte slots for geo-related data in the qob
  int32_t num_qob_geo_slots;

  // the byte stride of the 3 geo-related slots in the qob
  // 8 bytes/slot
  int32_t qob_geo_stride;

  // 1st geo slot - coords offset per row - required
  int32_t qob_geo_coords_slot;
  int32_t qob_geo_coords_slot_offset;

  // 2nd geo slot - rings offset per row - required
  int32_t qob_geo_rings_slot;
  int32_t qob_geo_rings_slot_offset;

  // 3rd geo slot - # rings for the row - required
  int32_t qob_geo_num_rings_slot;
  int32_t qob_geo_num_rings_slot_offset;

  QOBGeoSlotDescriptor()
      : num_qob_geo_slots{0}
      , qob_geo_stride{0}
      , qob_geo_coords_slot{-1}
      , qob_geo_coords_slot_offset{-1}
      , qob_geo_rings_slot{-1}
      , qob_geo_rings_slot_offset{-1}
      , qob_geo_num_rings_slot{-1}
      , qob_geo_num_rings_slot_offset{-1} {
    auto add_slot = [&](auto& slot_idx, auto& slot_offset) {
      slot_idx = num_qob_geo_slots++;
      slot_offset = DefaultNumBytesPerColumnType * slot_idx;
    };
    add_slot(qob_geo_coords_slot, qob_geo_coords_slot_offset);
    add_slot(qob_geo_rings_slot, qob_geo_rings_slot_offset);
    add_slot(qob_geo_num_rings_slot, qob_geo_num_rings_slot_offset);
    qob_geo_stride = DefaultNumBytesPerColumnType * num_qob_geo_slots;
  }
};

template <class T>
struct is_poly_target_value_ptr
    : std::integral_constant<
          bool,
          std::is_same_v<GeoPolyTargetValuePtr, typename std::remove_cv<T>::type> ||
              std::is_same_v<GeoMultiPolyTargetValuePtr,
                             typename std::remove_cv<T>::type>> {};

/**
 * Handles per-row visitation of geo target values from results sets when the geo return
 * type is set to GeoTargetValueGpuPtr or GeoTargetValuePtr. In both of these modes, any
 * geo data is returned as pointers to the original buffer data and not materialized
 * coordinates. This visitor maintains those pointers to be populated into buffers at a
 * later point.
 */
class PolyPtrVisitor : public ::boost::static_visitor<>, public ::boost::noncopyable {
 public:
  /**
   * @param target_vals Array of target values for all the used columns in a result set.
   * @param entry_idx The entry index in the result set for the current row.
   * @param polygeo_idx The index in the target_vals array for the geo column to be
   *                    rendered.
   * @param coord_byte_sz The byte size for the geo coordinates in the geo column to be
   *                      rendered. If compressed - 4 bytes, otherwise 8.
   */
  explicit PolyPtrVisitor(std::vector<TargetValue>&& target_vals,
                          QOBGeoSlotDescriptor& slot_descriptor,
                          const size_t entry_idx,
                          const int polygeo_idx,
                          const int coord_byte_sz)
      : row_target_vals_(std::move(target_vals))
      , entry_idx_(entry_idx)
      , coord_byte_sz_(coord_byte_sz) {
    auto& tv = row_target_vals_[polygeo_idx];
    const auto geo_tv = ::boost::get<GeoTargetValuePtr>(&tv);
    CHECK(geo_tv);
    ::boost::apply_visitor(*this, *geo_tv);
  }

  /**
   * @brief Operator called when boost::apply_vistor is performed. Extracts the geo
   * pointers from the target value.
   */
  template <typename T,
            typename std::enable_if<is_poly_target_value_ptr<T>::value>::type* = nullptr>
  void operator()(T& poly_tv) {
    // // get ring_sizes
    CHECK(poly_tv.ring_sizes_data);
    CHECK(poly_tv.coords_data);

    // assumes the target value being accessed won't be touched again. Anything requiring
    // access to the geo data should use the APIs on this visitor object
    coords_ = std::move(poly_tv.coords_data);
    ring_sizes_ = std::move(poly_tv.ring_sizes_data);
  }

  template <typename T,
            typename std::enable_if<!is_poly_target_value_ptr<T>::value>::type* = nullptr>
  void operator()(T& other_geo_tv) {
    CHECK(false) << typeid(T).name();
  }

  inline const int8_t* getCoordsPtr() const { return coords_->pointer; }
  inline size_t getNumCoords() const {
    // checks that there is valid data in both the rings and coords.
    // If the rings is invalid, we'll force no coords.
    return ring_sizes_->length ? coords_->length / coord_byte_sz_ : 0;
  }
  inline const int8_t* getRingSizesPtr() const { return ring_sizes_->pointer; }
  inline const int32_t* getRingSizes() const {
    return reinterpret_cast<int32_t*>(ring_sizes_->pointer);
  }
  inline size_t getNumRingSizes() const {
    // checks that there is valid data in both the rings and coords.
    // If the coords is invalid, we'll force no rings.
    return coords_->length ? ring_sizes_->length / sizeof(int32_t) : 0;
  }

  inline size_t getEntryIdx() const { return entry_idx_; }
  inline const std::vector<TargetValue>& getTargetValuesRef() const {
    return row_target_vals_;
  }

 private:
  std::vector<TargetValue> row_target_vals_;
  const size_t entry_idx_;
  const int coord_byte_sz_;
  std::shared_ptr<VarlenDatum> coords_;
  std::shared_ptr<VarlenDatum> ring_sizes_;
};

/**
 * @brief Maintains the visitation state per thread when iterating through a polygon
 * result set.
 */
struct PolyResultSetThreadState {
  /**
   * @param in_start_idx Start entry index of the result set this thread visits.
   * @param in_end_idx End entry index of the result set this thread visits.
   */
  PolyResultSetThreadState(const size_t in_start_idx, const size_t in_end_idx)
      : start_idx(in_start_idx)
      , end_idx(in_end_idx)
      , coords_count(0)
      , rings_count(0)
      , gpu_idx(-1)
      , coords_offset(0)
      , rings_offset(0)
      , row_offset(0) {}

  const size_t start_idx;  // start idx of the result set for this thread block
  const size_t end_idx;    // end idx of the result set

  size_t coords_count;  // total poly coords in this thread block
  size_t rings_count;   // total rings in this thread block

  int gpu_idx;           // gpu idx configured for this thread block
  size_t coords_offset;  // an offset into the final full coord array where this
                         // thread-block's coord data will begin
  size_t rings_offset;   // an offset into the final rings array where this thread block's
                         // ring data will begin
  size_t row_offset;     // an offset into the final query-output-buffer array where this
                         // thread block's ssbo data will begin

  std::map<size_t, size_t>
      visited_rows;  // the result set row indices this thread-block visited

  /*
   * @brief Adds a visited row with poly data to this thread-block
   */
  void addRow(const size_t row_idx,
              const size_t entry_idx,
              const PolyPtrVisitor& poly_row) {
    auto curr_num_coords = poly_row.getNumCoords();
    auto curr_num_rings = poly_row.getNumRingSizes();
    // only add the row if it has renderable data
    if (curr_num_coords > 0 && curr_num_rings > 0) {
      coords_count += curr_num_coords;
      rings_count += curr_num_rings;
      visited_rows.emplace(row_idx, entry_idx);
    }
  }

  /**
   * @brief Returns the number of rows visited by this thread-block
   */
  inline size_t rowCount() const { return visited_rows.size(); }
};

/**
 * @brief A baseline struct for visiting ScalarTargetValue boost::variant objects.
 * Only operates on the arithmetic values of ScalarTargetValue objects.
 */
struct QueryOutputBufferDataVisitor : ::boost::static_visitor<> {
  QueryOutputBufferDataVisitor(uint8_t* data_ptr) : curr_data_ptr(data_ptr) {}

  uint8_t* curr_data_ptr;

  template <typename T,
            typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
  void operator()(const T& scalar) {
    // TODO(croot): handle this with specializations or additial compile-time const-expr
    // or something
    constexpr auto type_sz =
        sizeof(typename std::remove_cv<typename std::remove_reference<T>::type>::type);
    static_assert(type_sz == 8 || type_sz == 4, "Requires a type size of 8 or 4");
    auto data_ptr = reinterpret_cast<T*>(curr_data_ptr + (type_sz == 8 ? 0 : 4));
    *data_ptr = scalar;
    curr_data_ptr += type_sz;
  }

  template <typename T,
            typename std::enable_if<!std::is_arithmetic<T>::value>::type* = nullptr>
  void operator()(T& scalar) {
    CHECK(false) << typeid(T).name();
  }
};

/**
 * Template specialization for floats which currently need to be converted to a double.
 */
template <>
void QueryOutputBufferDataVisitor::operator()<float>(const float& scalar) {
  auto data_ptr = reinterpret_cast<double*>(curr_data_ptr);
  *data_ptr = static_cast<double>(scalar);
  curr_data_ptr += sizeof(double);
}

/**
 * @brief Maintains the per-gpu state of polygonal data used to build render buffers.
 * Maintains the per-gpu state of the data used to build polygon render buffers.
 * This polygonal data is usually built while iterating a result set.
 * Post collection of the data, the render buffers can be initialized with a series of
 * methods that allocate the buffers, run thrust operations to populate the buffers from
 * the initial data, and then release the buffers for rendering.
 *
 * The initial data built before the thrust operations has a pre-defined layout, which
 * looks like the following:
 *
 * [Query-output-buffer (qob)][coords per row][rings per row]
 *
 * And the layout of the QOB looks like the following:
 *
 */
// clang-format off
/*
 * [--8bytes--] [--8bytes--] [--8bytes--]   [--0 or 8bytes--]    [...8bytes per attr...]
 * [Coords Ptr] [PRings Ptr]  [# PRings]   [Rest of the attrs]
 */
// clang-format on
/*
 * Coords ptr: byte offset into the coords data for this row
 * PRings ptr: byte offset into the rings data for this row
 * # PRings: Number of poly rings for this row. The number of vertices/coords from this
 *           row can be derived from the rings.
 */
class PolyResultBufferInfo {
 public:
  const int gpu_idx;

  // sums for coords, rings, and rows in these buffers
  uint64_t gpu_coord_count{0};
  uint64_t gpu_ring_count{0};
  uint64_t gpu_row_count{0};

  /**
   * @param in_render_manager Render manager reference
   * @param vega_data_table_name The source polygon data table name. Used to grab the
   * right buffers at a later time in the render loop.
   * @param in_gpu_idx Gpu index for these buffers.
   * @param in_qob_stride Full byte stride for the query output buffer.
   * @param is_geo_data_on_device If true, the geo data from the result set is currently
   *                              on the gpu. In this case we can avoid doing any internal
   *                              buffer magic and just keep that data on the gpu and
   *                              access it later.
   * @param coord_byte_sz Byte size of the coordinate data. If compressed, this is 4,
   *                      otherwise 8.
   */
  explicit PolyResultBufferInfo(QueryRenderManager& in_render_manager,
                                QOBGeoSlotDescriptor& in_geo_slot_descriptor,
                                const std::string& in_vega_data_table_name,
                                const int in_gpu_idx,
                                const size_t in_qob_stride,
                                const bool is_geo_data_on_device,
                                const int coord_byte_sz)
      : gpu_idx(in_gpu_idx)
      , render_manager(in_render_manager)
      , geo_slot_descriptor(in_geo_slot_descriptor)
      , vega_data_table_name(in_vega_data_table_name)
      , qob_stride(in_qob_stride)
      , _is_geo_data_on_device(is_geo_data_on_device)
      , _coord_byte_sz(coord_byte_sz) {}

  ~PolyResultBufferInfo() {
    if (render_allocator) {
      // If we have a render allocator, it means we allocated something on it, so mark it
      // complete.
      render_allocator->markChunkComplete();
    }
  }

  /**
   * Allocates the initial data block for this gpu. Should be called after all the data
   * has been collected. Should also be called on the main thread.
   *
   * This is not thread safe.
   */
  void initializeBufferAndOffsets() {
    // be sure to byte align to 8 bytes. This is necessary in case the render allocator
    // buffer is used for other layers in a multi-layer render. Keeping it 8-byte aligned
    // will avoid misaligned address errors. See: https://jira.omnisci.com/browse/BE-3669
    auto qob_bytes = gpu_row_count * qob_stride;  // QOB
    qob_bytes += qob_bytes % 8;

    uint64_t coords_bytes{0}, rings_bytes{0};

    if (!_is_geo_data_on_device) {
      // if geo data is not on the gpu, then we need to build up the coordinate/rings
      // buffers to bus to gpu.
      coords_bytes = gpu_coord_count * _coord_byte_sz;  // coords
      coords_bytes += coords_bytes % 8;

      rings_bytes = gpu_ring_count * sizeof(int32_t);  // rings
      rings_bytes += rings_bytes % 8;
    }

    coords_buffer_ptr_offset = qob_bytes;
    rings_buffer_ptr_offset = coords_buffer_ptr_offset + coords_bytes;

    // all together now
    all_data.resize(qob_bytes + coords_bytes + rings_bytes, 0);

    CHECK_EQ(all_data.size() % 8, 0u);
  }

  static inline void applyCoordsPtrOnDevice(const int qob_geo_coords_slot_offset,
                                            uint8_t* row_data_ptr,
                                            const PolyPtrVisitor& poly_row) {
    // adds the device ptr where the coords for a particular row can be accessed.

    // in this case, the geo data is already sitting on the device, so what we
    // received from the result set is the actual device ptr, so just use that.
    auto coords_offset_ptr =
        reinterpret_cast<int64_t*>(row_data_ptr + qob_geo_coords_slot_offset);
    *coords_offset_ptr = reinterpret_cast<int64_t>(poly_row.getCoordsPtr());
  }

  static inline void applyCoordsPtr(const int qob_geo_coords_slot_offset,
                                    uint8_t* row_data_ptr,
                                    const size_t coords_idx,
                                    const int coord_byte_sz) {
    // add the coords buffer offset to the QOB
    auto coords_offset_ptr =
        reinterpret_cast<int64_t*>(row_data_ptr + qob_geo_coords_slot_offset);
    *coords_offset_ptr = static_cast<int64_t>(coords_idx * coord_byte_sz);
  }

  static inline void applyRingSizesPtrOnDevice(const int qob_geo_rings_slot_offset,
                                               uint8_t* row_data_ptr,
                                               const PolyPtrVisitor& poly_row) {
    // adds the device ptr where the rings for a particular row can be accessed.

    // in this case, the geo data is already sitting on the device, so what we received
    // from the result set is the actual device ptr for the rings, so we can just use
    // that.
    auto rings_offset_ptr =
        reinterpret_cast<int64_t*>(row_data_ptr + qob_geo_rings_slot_offset);
    *rings_offset_ptr = reinterpret_cast<int64_t>(poly_row.getRingSizesPtr());
  }

  static inline void applyRingSizesPtr(const int qob_geo_rings_slot_offset,
                                       uint8_t* row_data_ptr,
                                       const size_t rings_idx) {
    // add the rings buffer offset to the QOB
    auto rings_offset_ptr =
        reinterpret_cast<int64_t*>(row_data_ptr + qob_geo_rings_slot_offset);
    *rings_offset_ptr = static_cast<int64_t>(
        rings_idx *
        sizeof(std::remove_pointer_t<decltype(PolyBufferRawPtrs::ring_sizes)>));
  }

  static inline void applyNumRings(const int qob_geo_num_rings_slot_offset,
                                   uint8_t* row_data_ptr,
                                   const PolyPtrVisitor& poly_row) {
    // add the # of rings to the QOB
    auto num_rings_ptr =
        reinterpret_cast<int64_t*>(row_data_ptr + qob_geo_num_rings_slot_offset);
    *num_rings_ptr = static_cast<int64_t>(poly_row.getNumRingSizes());
  }

  static inline void applyResultSetColumns(
      const int qob_geo_stride,
      uint8_t* row_data_ptr,
      const PolyPtrVisitor& poly_row,
      const std::unordered_set<int>& ssbo_col_idxs,
      const std::vector<TargetMetaInfo>& result_targets,
      const size_t entry_idx,
      const RowIdStatus& rowid_status) {
    // add the additional column data from the row to the QOB
    auto& target_values = poly_row.getTargetValuesRef();
    QueryOutputBufferDataVisitor visitor(row_data_ptr + qob_geo_stride);
    for (auto col_idx : ssbo_col_idxs) {
      auto col_target = result_targets[col_idx];
      auto& col_tv = target_values[col_idx];
      auto scalar_tv = ::boost::get<ScalarTargetValue>(&col_tv);
      if (!scalar_tv) {
        throw std::runtime_error(
            "Only scalar target values are currently supported in poly rendering. "
            "Column \"" +
            col_target.get_resname() + "\" is of type " +
            col_target.get_type_info().get_type_name());
      }

      if (col_idx == rowid_status.rowid_idx) {
        // For all non-insitu queries, we need to replace any explicit rowids in the query
        // with the entry index of the results for proper hit-testing
        ScalarTargetValue new_rowid_value{static_cast<int64_t>(entry_idx)};
        ::boost::apply_visitor(visitor, new_rowid_value);
      } else {
        ::boost::apply_visitor(visitor, *scalar_tv);
      }
    }

    if (rowid_status.add_rowid) {
      auto rowid_ptr = reinterpret_cast<int64_t*>(visitor.curr_data_ptr);
      *rowid_ptr = static_cast<int64_t>(entry_idx);
    }
  }

  template <typename T>
  static inline void applyCoordsData(int8_t* coords,
                                     const PolyPtrVisitor& poly_row,
                                     size_t& coords_idx) {
    // Add the coords, either compressed or uncompressed
    const auto output_coords = reinterpret_cast<T*>(coords);
    const auto row_coords = reinterpret_cast<const T*>(poly_row.getCoordsPtr());
    const auto num_coords = poly_row.getNumCoords();
    for (size_t i = 0; i < num_coords; ++i) {
      output_coords[coords_idx++] = row_coords[i];
    }
  }

  static inline void applyRingSizesData(int32_t* ring_sizes,
                                        const PolyPtrVisitor& poly_row,
                                        size_t& rings_idx) {
    // Add the rings
    const auto row_rings = poly_row.getRingSizes();
    const auto num_rings = poly_row.getNumRingSizes();
    for (size_t i = 0; i < num_rings; ++i) {
      ring_sizes[rings_idx++] = row_rings[i];
    }
  }

  /**
   * @brief applys the per-row data for each thread-state to the render buffers to be
   * bussed to a device for later handling.
   * @param state Thread state holding results from a poly render query.
   * @param poly_row_data The full vector of all visited rows from the query.
   * @param ssbo_col_idxs The indices of all the columns from the query that will go to
   *                      the ssbo.
   * @param result_targets A vector of all the meta info for the resuling columns of the
   *                       query.
   * @param rowid_status struct holding on to state determining the current status of a
   *                     rowid column(s)
   */
  void applyThreadStateData(
      const PolyResultSetThreadState& state,
      const std::vector<std::unique_ptr<PolyPtrVisitor>>& poly_row_data,
      const std::unordered_set<int>& ssbo_col_idxs,
      const std::vector<TargetMetaInfo>& result_targets,
      const RowIdStatus& rowid_status) {
    if (!all_data.size()) {
      // empty data on this gpu, so early out.
      return;
    }

    // get the buffer ptrs for the qob, coords, and rings
    auto buffer_ptrs = _getBufferPtrs();

    size_t coords_idx{state.coords_offset};  // current coords index
    size_t rings_idx{state.rings_offset};    // current rings index

    // qob pointer for this row
    uint8_t* row_data_ptr = buffer_ptrs.qob + state.row_offset * qob_stride;

    // Add the row data for every visited row by the thread.
    if (_is_geo_data_on_device) {
      // If the poly geo data is already on the device, then we only need to add the
      // coords/rings ptrs and not worry about bussing any of the actual coords/rings data
      for (auto& row_info : state.visited_rows) {
        size_t row_idx, entry_idx;
        std::tie(row_idx, entry_idx) = row_info;
        const auto& poly_row = *poly_row_data[row_idx];

        applyCoordsPtrOnDevice(
            geo_slot_descriptor.qob_geo_coords_slot_offset, row_data_ptr, poly_row);
        applyRingSizesPtrOnDevice(
            geo_slot_descriptor.qob_geo_rings_slot_offset, row_data_ptr, poly_row);
        applyNumRings(
            geo_slot_descriptor.qob_geo_num_rings_slot_offset, row_data_ptr, poly_row);
        applyResultSetColumns(geo_slot_descriptor.qob_geo_stride,
                              row_data_ptr,
                              poly_row,
                              ssbo_col_idxs,
                              result_targets,
                              entry_idx,
                              rowid_status);

        row_data_ptr += qob_stride;
      }
    } else {
      // poly geo data is on cpu, so we need to build out the coords/poly buffers to be
      // bussed to gpu for render processing.

      // support GEOINT compressed and uncompressed geo data.
      CHECK(_coord_byte_sz == 8 || _coord_byte_sz == 4) << _coord_byte_sz;
      auto apply_coords_func =
          _coord_byte_sz == 8 ? applyCoordsData<double> : applyCoordsData<int32_t>;
      for (auto& row_info : state.visited_rows) {
        size_t row_idx, entry_idx;
        std::tie(row_idx, entry_idx) = row_info;
        const auto& poly_row = *poly_row_data[row_idx];

        applyCoordsPtr(geo_slot_descriptor.qob_geo_coords_slot_offset,
                       row_data_ptr,
                       coords_idx,
                       _coord_byte_sz);
        applyRingSizesPtr(
            geo_slot_descriptor.qob_geo_rings_slot_offset, row_data_ptr, rings_idx);
        applyNumRings(
            geo_slot_descriptor.qob_geo_num_rings_slot_offset, row_data_ptr, poly_row);

        applyResultSetColumns(geo_slot_descriptor.qob_geo_stride,
                              row_data_ptr,
                              poly_row,
                              ssbo_col_idxs,
                              result_targets,
                              entry_idx,
                              rowid_status);

        // apply coords and rings to the buffers.
        apply_coords_func(buffer_ptrs.coords, poly_row, coords_idx);
        applyRingSizesData(buffer_ptrs.ring_sizes, poly_row, rings_idx);

        row_data_ptr += qob_stride;
      }
    }
  }

  /**
   * Initializes the render allocator for this gpu and busses the buffer memory to the
   * device. Should be called after applyThreadStateData()
   *
   * This is thread safe.
   */
  void initializeRenderAllocator(RenderAllocatorMap& render_allocator_map) {
    if (all_data.size()) {
      CHECK(!render_allocator);
      render_allocator = render_allocator_map.getRenderAllocator(gpu_idx);
      CHECK(render_allocator);
      auto data_sz = all_data.size();
      render_allocator->alloc(data_sz);
      render_allocator_map.bufferData(
          reinterpret_cast<int8_t*>(all_data.data()), data_sz, gpu_idx);

      qob_base_ptr =
          render_allocator->getBasePtr() + render_allocator->getCurrentChunkOffset();
    }
  }

  /**
   * Runs the first thrust stage. This stage is ultimately used to determine the size of
   * the render buffers that should be allocated. Should be called after
   * initializeRenderAllocator().
   *
   * This is thread safe.
   */
  void runThrustStage1(DataMgrThrustContext&& thrust_context,
                       const EncodingType geo_enc_type,
                       const int32_t qob_rowid_slot) {
    if (qob_base_ptr) {
      CHECK(!pdc);
      pdc = std::make_unique<PolygonDataConverter>(std::move(thrust_context));

      pdc->ConvertStage1(
          gpu_row_count,
          qob_base_ptr,
          qob_stride,
          geo_slot_descriptor.qob_geo_rings_slot,
          qob_rowid_slot,
          stage1_row_count,
          stage1_vert_count,
          stage1_poly_count,
          _is_geo_data_on_device
              ? nullptr
              : qob_base_ptr + rings_buffer_ptr_offset);  // only adds rings offset if
                                                          // data is not already on device
    }
  }

  /**
   * Allocates the render buffers for the poly data. This should be called after
   * runThrustStage1.
   *
   * This is not thread safe until a render_manager lock is removed.
   */
  void initializePolyRenderBuffers(const size_t ssbo_align_bytes) {
    // should be run after stage 1 and before stage 2 and run on the main thread (for
    // now)

    if (qob_base_ptr && stage1_vert_count && stage1_row_count && stage1_poly_count) {
      // allocate cache buffers now that we know what size they need to be
      PolyTableByteData polyByteData(
          // allocate space for uncompressed verts. If verts are compressed, they will be
          // uncompressed in the 2nd thrust stage
          {stage1_vert_count * sizeof(double) * 2,
           stage1_row_count * sizeof(gfx::IndirectDrawVertexData),
           stage1_poly_count * sizeof(gfx::IndirectDrawVertexData),
           ssbo_align_bytes * stage1_row_count,
           0});

      // allocate row ids
      polyByteData.num_poly_rowids_bytes = stage1_poly_count * sizeof(uint32_t);

      // create the in-situ buffers
      auto& poly_mgr = render_manager.getPolyMgr();
      buffer_ptrs = poly_mgr.createPolyTableInSituBuffers(
          vega_data_table_name, gpu_idx, polyByteData);

      // switch the buffers we just created to CUDA mode and get their handles
      buffer_memory_descriptors =
          poly_mgr.getPolyTableInSituBufferDescriptors(buffer_ptrs, gpu_idx);
    }
  }

  /**
   * Runs the second thrust stage. This transforms the original polygonal data to
   * renderable form and stores it in the render buffers initialized from
   * initializePolyRenderBuffers().
   *
   * This is thread safe
   */
  void runThrustStage2(
      const std::vector<uint32_t>& per_row_data_buffer_col_idxs,
      const std::vector<uint32_t>& per_row_data_ssbo_offsets,
      const std::vector<gfx::BufferAttrType>& per_row_data_buffer_col_types,
      const size_t ssbo_align_bytes,
      const EncodingType geo_enc_type) {
    if (qob_base_ptr && stage1_vert_count && stage1_row_count && stage1_poly_count) {
      // outputs of thrust buffer builder stage 2
      std::vector<uint32_t> num_rows_per_batch;
      std::vector<uint32_t> num_polys_per_batch;

      CHECK(pdc);

      // run stage 2
      // @TODO simplify argument list with structs
      if (_is_geo_data_on_device) {
        pdc->ConvertStage2(stage1_row_count,
                           qob_base_ptr,
                           qob_stride,
                           geo_slot_descriptor.qob_geo_coords_slot,
                           geo_slot_descriptor.qob_geo_rings_slot,
                           geo_enc_type,
                           per_row_data_buffer_col_idxs,
                           per_row_data_ssbo_offsets,
                           per_row_data_buffer_col_types,
                           ssbo_align_bytes,
                           buffer_memory_descriptors,
                           num_rows_per_batch,
                           num_polys_per_batch,
                           nullptr,
                           nullptr);
      } else {
        pdc->ConvertStage2(stage1_row_count,
                           qob_base_ptr,
                           qob_stride,
                           geo_slot_descriptor.qob_geo_coords_slot,
                           geo_slot_descriptor.qob_geo_rings_slot,
                           geo_enc_type,
                           per_row_data_buffer_col_idxs,
                           per_row_data_ssbo_offsets,
                           per_row_data_buffer_col_types,
                           ssbo_align_bytes,
                           buffer_memory_descriptors,
                           num_rows_per_batch,
                           num_polys_per_batch,
                           qob_base_ptr + coords_buffer_ptr_offset,
                           qob_base_ptr + rings_buffer_ptr_offset);
      }

      // package and store the batch info
      poly_draw_batch_info = std::make_unique<PolyDrawBatchInfo>(
          std::move(num_rows_per_batch), std::move(num_polys_per_batch));
    }
  }

  /**
   * Releases the poly render buffers for rendering by unmapping the interop buffer.
   * This should be called after runThrustStage2().
   *
   * This is not thread safe until locks in QueryRenderManager are cleaned up.
   */
  void releasePolyRenderBuffers(std::shared_ptr<QueryDataLayout> vert_layout,
                                std::shared_ptr<QueryDataLayout> ssbo_layout) {
    if (qob_base_ptr) {
      auto& poly_mgr = render_manager.getPolyMgr();
      if (poly_draw_batch_info) {
        // attach this also
        poly_mgr.setPolyTableInSituBuffersPolyDrawBatchInfo(
            vega_data_table_name, gpu_idx, std::move(poly_draw_batch_info));
      }

      // unmap buffers for rendering
      poly_mgr.releasePolyTableInSituBuffersForRendering(
          buffer_ptrs, gpu_idx, vert_layout, ssbo_layout);
    }
  }

 private:
  struct PolyBufferRawPtrs {
    uint8_t* qob;
    int8_t* coords;  // coords, will be casted later on to handle compressed/uncompressed
    int32_t* ring_sizes;
  };
  /**
   * Gets the buffer ptrs for all the sections - qob, coords, rings
   */
  PolyBufferRawPtrs _getBufferPtrs() {
    auto qob = all_data.data();
    return PolyBufferRawPtrs{
        qob,
        reinterpret_cast<int8_t*>(qob + coords_buffer_ptr_offset),   // uncasted coords
        reinterpret_cast<int32_t*>(qob + rings_buffer_ptr_offset)};  // rings
  }

  QueryRenderManager& render_manager;
  QOBGeoSlotDescriptor& geo_slot_descriptor;
  const std::string& vega_data_table_name;
  const size_t qob_stride;
  const bool _is_geo_data_on_device;
  const int _coord_byte_sz;

  // Render allocator the data here will be bussed to.
  RenderAllocator* render_allocator{nullptr};

  // the start pointer of the buffered data from the render allocator
  int8_t* qob_base_ptr{nullptr};

  // functional object for running the thrust polygon conversions.
  std::unique_ptr<PolygonDataConverter> pdc;

  // the render buffer ptrs
  PolyBufferPtrs buffer_ptrs;

  // the CUDA memory descriptors for the above render buffers - needed for thrust
  PolyBufferMemoryDescriptors buffer_memory_descriptors;

  // poly draw batch info
  PolyDrawBatchInfoUqPtr poly_draw_batch_info;

  // outputs of thrust buffer build stage 1:

  // number of non-empty rows
  uint32_t stage1_row_count = 0;
  // total number of verts across all non-empty rows
  uint32_t stage1_vert_count = 0;
  // total number of polys (rings) across all non-empty rows
  uint32_t stage1_poly_count = 0;

  // byte offset to start of coords data
  uint64_t coords_buffer_ptr_offset{0};

  // byte offset to start of rings data
  uint64_t rings_buffer_ptr_offset{0};

  std::vector<uint8_t> all_data;  // byte vector containing all data, includes ssbo
                                  // data, coords bytes and rings bytes
};                                // namespace

/**
 * @brief Runs a multi-threaded operation over a container. All the elements of the
 * container are equally distributed over the total number of threads available.
 * @param first Container iterator to start multi-threaded iteration from.
 * @param last Container iterator to end iteration at.
 * @param f Unary function to apply to every element in the container between first and
 *          last.
 */
template <class InputIt, class UnaryFunction>
inline void multithread_container(InputIt first, InputIt last, UnaryFunction f) {
  const size_t num_elems = std::distance(first, last);
  if (!DISABLE_MULTI_THREADING && num_elems > 1) {
    const auto worker_count = static_cast<size_t>(cpu_threads());
    const auto elems_per_thread = num_elems / worker_count;
    auto remainder = static_cast<int64_t>(num_elems % worker_count);
    std::vector<std::future<void>> threads;

    // now launch the threads
    size_t elem_diff = elems_per_thread + (remainder > 0 ? 1 : 0);
    while (first != last && elem_diff > 0) {
      auto last_thread_item = first;
      for (size_t i = 0; i < elem_diff; ++i) {
        // need to find the end itr for this thread using this approach in order to
        // support LegacyForwardIterator containers, which don't support the '+=' operator
        ++last_thread_item;
      }
      threads.push_back(std::async(
          std::launch::async,
          [&, parent_thread_local_ids = logger::thread_local_ids()](InputIt curr_item,
                                                                    InputIt end_item) {
            logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
            for (; curr_item != end_item; ++curr_item) {
              f(&(*curr_item));
            }
          },
          first,
          last_thread_item));
      if (remainder == 1) {
        // we've used up all of the remainder, so use the evenly-divided
        // count the rest of the way
        --elem_diff;
      }
      --remainder;
      first = last_thread_item;
    }

    for (auto& child : threads) {
      child.get();
    }
  } else {
    for (; first != last; ++first) {
      f(&(*first));
    }
  }
}

/**
 * @brief Runs a single-threaded operation over a container.
 * @param first Container iterator to start iteration from.
 * @param last Container iterator to end iteration at.
 * @param f Unary function to apply to every element in the container between first and
 *          last.
 */
template <class InputIt, class UnaryFunction>
inline void singlethread_container(InputIt first, InputIt last, UnaryFunction f) {
  for (; first != last; ++first) {
    f(&(*first));
  }
}

/**
 * @brief Builds polygon render buffers from a non-insitu query.
 * Builds polygon render buffers from a non-insitu query result set.
 * @param render_manager Render manager singleton.
 * @param rows The result set from running the query.
 * @param data_loc The json location for the data block in the vega that defined the
 * query.
 * @param render_info The render info object that was passed thru the query
 * engine. Contains useful metadata about the query execution and result.
 * @param result_targets The target meta info for the resulting columns of the result
 * set.
 * @param polygeo_idx The result_targets index of the location of the polygon data
 * column to be rendered.
 * @param rowid_idx The result_targets index of the rowid column, if it exists.
 */
uint64_t process_polygons_non_in_situ(QueryRenderManager& render_manager,
                                      const std::shared_ptr<ResultSet>& rows,
                                      const JSONLocation& data_loc,
                                      RenderInfo& render_info,
                                      const std::vector<TargetMetaInfo>& result_targets,
                                      const int polygeo_idx) {
  CHECK_GE(polygeo_idx, 0);
  CHECK_LE(static_cast<size_t>(polygeo_idx), result_targets.size());

  // Get the name of the vega data block. This is used as a key in a map to store the
  // poly render buffers in the end.
  const auto name_loc = data_loc.getMember(JSONSchema_v1::Data::kNameProp);
  const std::string vega_data_table_name = name_loc.getString();

  uint64_t num_rows{0};

  // set the appropriate poly return type from the result set. Setting to GpuPtr here to
  // indicate that if any of the geo data is still sitting on gpu to be fetched, leave it
  // there on the gpu, and just return the device pts
  // TODO(croot): Look into direct-memory-access fetching rather than going thru
  // boost::variant
  rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValueGpuPtr);

  const auto rowid_status = get_rowid_status(result_targets, render_info);

  // Capture target column indices for any physical poly columns we need for rendering.
  // These will also be used to tell get_render_data_template() to ignore them when
  // building out the ssbo.
  // TODO(croot): handle the hard coded suffixes better
  const auto poly_column_prefix = result_targets[polygeo_idx].get_resname();
  std::unordered_set<int> ssbo_col_idxs;
  std::unordered_set<unsigned int> physical_column_target_indices_to_ignore{
      static_cast<unsigned int>(polygeo_idx)};
  DependentIdxArray dependent_idxs;
  dependent_idxs.fill(-1);

  std::map<std::string, int> dependent_render_cols;
  for (int i = 0; i < static_cast<int>(result_targets.size()); i++) {
    const std::string& tn = result_targets[i].get_resname();
    auto itr = dependent_render_cols.find(tn);
    if (itr != dependent_render_cols.end()) {
      physical_column_target_indices_to_ignore.emplace(i);
      dependent_idxs[itr->second] = i;
    } else if (i != polygeo_idx) {
      ssbo_col_idxs.insert(i);
    }
  }
  if (dependent_render_cols.size() > 0) {
    for (size_t i = 0; i < dependent_idxs.size(); ++i) {
      if (dependent_idxs[i] < 0) {
        throw std::runtime_error("Poly render-dependent column \"" + std::to_string(i) +
                                 "\" not found");
      }
    }
  }

  // the target column indices we need
  std::vector<unsigned int> target_column_indices(result_targets.size());
  std::iota(target_column_indices.begin(), target_column_indices.end(), 0);

  // Build out the template for the ssbo buffer.
  auto data_query_result = get_render_data_template(
      result_targets,
      render_info.targets,
      target_column_indices,
      physical_column_target_indices_to_ignore,
      {},  // extra target aliases
      QueryDataLayout::LayoutType::kStorage,
      0,
      rowid_status,
      true,    // uses a result set
      false);  // do not allocate a local buffer to place the data
  auto ssbo_layout = data_query_result.render_data_layout->getBufferLayout();
  CHECK(ssbo_layout);
  // prepare to capture this
  int32_t qob_rowid_slot = -1;

  // the simple data we need to pass to Stage 2 of the thrust process to describe the
  // SSBO We need the output ssbo index slot, the byte offset into the ssbo, and the
  // attr type.
  std::vector<uint32_t> per_row_data_buffer_col_idxs;
  std::vector<uint32_t> per_row_data_ssbo_offsets;
  std::vector<gfx::BufferAttrType> per_row_data_buffer_col_types;

  QOBGeoSlotDescriptor geo_slot_descriptor;

  uint32_t curr_buf_idx = geo_slot_descriptor.num_qob_geo_slots;
  const auto addSsboMetaData = [&](const std::string& name) {
    auto offset = ssbo_layout->getAttributeByteOffset(name);
    auto buffer_attr_type = ssbo_layout->getAttributeType(name);
    if (name == kRowIdColumnName) {
      // capture rowid QOB slot
      qob_rowid_slot = curr_buf_idx;
    }
    per_row_data_buffer_col_idxs.push_back(curr_buf_idx++);
    per_row_data_ssbo_offsets.push_back(offset);
    switch (buffer_attr_type) {
      case gfx::BufferAttrType::kInt:
      case gfx::BufferAttrType::kUint:
      case gfx::BufferAttrType::kInt64:
      case gfx::BufferAttrType::kUint64:
      case gfx::BufferAttrType::kFloat:
      case gfx::BufferAttrType::kDouble:
        per_row_data_buffer_col_types.push_back(buffer_attr_type);
        break;
      default:
        // @TODO simon.eves
        // do we need to support any other types?
        throw std::runtime_error(
            "Internal Error: In-Situ Geo SSBO has unsupported attr '" + name +
            "' of type " + gfx::to_string(buffer_attr_type));
    }
  };

  // then the other attrs in buffer_col_idx order (implicit from being a std::set)
  for (auto idx : ssbo_col_idxs) {
    addSsboMetaData(result_targets[idx].get_resname());
  }
  if (rowid_status.add_rowid) {
    addSsboMetaData(kRowIdColumnName);
  }

  // polys always require rowid
  if (qob_rowid_slot < 0) {
    throw std::runtime_error(
        "Failed to find '" + std::string(kRowIdColumnName) +
        "' column in poly query result, which is required for poly rendering");
  }

  // now we know how many actual buffer columns, we can compute the overall stride
  uint32_t query_output_buffer_stride =
      (ssbo_col_idxs.size() + (rowid_status.add_rowid ? 1 : 0)) *
          DefaultNumBytesPerColumnType +
      geo_slot_descriptor.qob_geo_stride;

  // build out the initial vector that stores the per-row data from the result set
  const auto entry_count = rows->entryCount();
  std::vector<std::unique_ptr<PolyPtrVisitor>> poly_row_data(entry_count);

  // check whether we may be gathering geo data that is already on the gpu. In this case
  // we can avoid fetching the data from gpu, and re-bussing, and instead, render
  // directly from that data already on gpu. NOTE: As of 05/23/2019 the result set only
  // holds onto this data on a single gpu. If that ever changes, we'll need to adjust
  // this logic.
  // TODO(croot): If the geo data sitting on gpu is large, we may want to consider
  // re-distributing that across all avalable gpus.
  auto is_geo_data_on_device = rows->isGeoColOnGpu(polygeo_idx);
  auto result_device_id = rows->getDeviceId();

#if HAVE_CUDA
  //
  // This is a workaround for the ongoing issue of ResultSets sometimes containing
  // device address references (Geo*TargetValuePtr) that do not match the device
  // ID that they report. This previously resulted in on-device (aka "hybrid") mode
  // here invoking the Thrust passes on the wrong GPU, which in turn resulted in
  // a fatal CUDA Error 700.
  //
  // If the ResultSet reports that the polygon data is on a GPU, we ignore the
  // device ID reported by the ResultSet. Instead, we read the first row from the
  // ResultSet and pull out the actual device address from the Geo*TargetValuePtr
  // datum. Then we look for this address in the CudaMgr device memory allocation
  // map in order to determine which GPU the polygon data is actually on. Then we
  // override the device ID so that the subsequent Thrust operations, and indeed
  // the polygon render itself, run on that GPU instead.
  //
  // simon.eves 2/6/24
  //
  // Original comment from Chris Root:
  //
  // this is a fail safe in the event that the device_id is not 0. If the device_id is >
  // 0, illegal memory access crashes occur in stage2 of the thrust poly building code.
  // Initial debugging seems to point to a GfxDriver->cuda device pairing issue or
  // initialize steps that fail if not on the first device. Stage 1 works fine, which
  // means the data in the QOB on the render allocator and the result-set data on the
  // gpu is fine. It crashes in stage2 when writing to buffers allocated by
  // GfxDriver. If the server is forced to be only single-gpu, or the device_id = 0,
  // then this all works fine. Here's an example query using the default render test
  // data that triggers a device_id > 0 with a multi-gpu-configured server:
  //
  // SELECT contributor_state  AS key0,
  //        Sum(amount)        AS color,
  //        Sample(geo_column) AS geo_column
  // FROM   contributions,
  //        us_states
  // WHERE  ( contributions.contributor_state = us_states.stusps )
  //        AND ( us_states.stusps = 'WY' )
  // GROUP  BY key0;
  //
  // It's only when the device_id != 0 that this issue is hit.
  // Until that is resolved, forcing the data off gpu by setting the geo return type to
  // GeoTargetValuePtr.
  //
  // NOTE: forcing this can be a big bottleneck for large poly tables as the data is
  // fetched off gpu row-by-row.
  //

  if (is_geo_data_on_device && rows->rowCount() > 0) {
    // report original value
    VLOG(1) << "Determining actual GPU residency for on-device geo data:";
    VLOG(1) << "  ResultSet claims device_id = " << result_device_id;

    // get allocation map
    auto* data_mgr = render_manager.getDataMgr();
    CHECK(data_mgr);
    auto* cuda_mgr = data_mgr->getCudaMgr();

    // prepare to capture superset of device IDs
    std::set<int> device_ids;

    auto const start_time = timer_start();

    // iterate entire result set and check each row coords pointer
    rows->moveToBegin();
    while (true) {
      auto row = rows->getNextRow(false, false);
      if (row.empty()) {
        break;
      }
      // get the datum and cast to geo
      auto const tv = row[polygeo_idx];
      auto const geo_tv = ::boost::get<GeoTargetValuePtr>(&tv);
      if (geo_tv) {
        // then attempt to cast to POLYGON or MULTIPOLYGON
        auto const poly_tv = ::boost::get<GeoPolyTargetValuePtr>(geo_tv);
        auto const multipoly_tv = ::boost::get<GeoMultiPolyTargetValuePtr>(geo_tv);
        if (poly_tv || multipoly_tv) {
          // get coords
          auto const coords =
              multipoly_tv ? multipoly_tv->coords_data : poly_tv->coords_data;
          if (coords && !coords->is_null) {
            // coord datum is valid, get device pointer and size
            auto const coords_ptr = reinterpret_cast<CUdeviceptr>(coords->pointer);
            auto const coords_len = coords->length;
            // get device ID for this allocation
            // THIS WILL CURRENTLY CHECK IF A MATCHING ALLOCATION IS NOT FOUND!
            auto const device_id =
                cuda_mgr->getDeviceNumFromDevicePtr(coords_ptr, coords_len);
            // store in the set
            device_ids.insert(device_id);
          }
        }
      }
    }

    // how did we do?
    if (device_ids.size() == 1) {
      // we found everything all on one GPU
      auto const single_device_id = *device_ids.begin();
      if (result_device_id != single_device_id) {
        result_device_id = single_device_id;
        VLOG(1) << "  Overriding ResultSet device_id to " << result_device_id;
      }
    } else {
      // we found a state we can't handle, and need to revert to slow mode
      if (device_ids.size() > 1) {
        std::stringstream ss;
        for (auto const& id : device_ids) {
          ss << " " << id;
        }
        VLOG(1) << "  Found coords device addresses on more than one device_id!"
                << ss.str();
      } else if (device_ids.size() == 0) {
        VLOG(1) << "  Failed to find CUDA allocation for any coords device addresses!";
      }
      VLOG(1) << "  Reverting to slow mode!";
      is_geo_data_on_device = false;
      rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValuePtr);
    }

    // reset ResultSet
    rows->moveToBegin();

    auto const scan_ms = timer_stop(start_time);
    VLOG(1) << "  Took " << scan_ms << "ms";
  }
#endif  // HAVE_CUDA

#ifdef HAVE_CUDA
  auto cuda_mgr = render_manager.getCudaMgr();
  CHECK(cuda_mgr);
  // get the CUDA contexts and num gpus. This will be used later in building the buffers
  auto& cuda_device_contexts = cuda_mgr->getDeviceContexts();

  // get num gpus to use. If geo data is already on a device, then forcing number of gpus
  // to 1 as the result set only supports single-device pairing.
  const auto num_gpus = is_geo_data_on_device ? 1 : cuda_device_contexts.size();
#else
  // no cuda means all queries run on cpu currently, so is_geo_on_device should always be
  // false
  CHECK(!is_geo_data_on_device);
  const auto num_gpus = render_manager.getNumGpus();
#endif  // HAVE_CUDA

  // a gpu idx -> poly data map
  std::unordered_map<int, PolyResultBufferInfo> gpu_idx_to_poly_data_map;
  // compression is lost during serialization. Grab the geo type from the result set,
  // which preserves this state.
  const auto& geo_col_sql_type = result_targets[polygeo_idx].get_type_info();

  // validate coords encoding
  // we only support uncompressed and compressed(32) for now
  const auto coords_encoding = geo_col_sql_type.get_compression();
  const auto coords_comp_param = geo_col_sql_type.get_comp_param();
  if (!(coords_encoding == kENCODING_NONE ||
        (coords_encoding == kENCODING_GEOINT && coords_comp_param == 32))) {
    throw std::runtime_error("MULTIPOLYGON geo column '" + poly_column_prefix +
                             "' has unsupported compression type " +
                             geo_col_sql_type.to_string() +
                             ". It must be 'NONE' or 'COMPRESSED(32)')");
  }
  const auto num_coord_bytes =
      coords_encoding == kENCODING_GEOINT ? coords_comp_param / 8 : sizeof(double);

  // now that we know the coords encoding, get the vertex data layout
  auto vertLayout = build_line_poly_vbo_layout(coords_encoding);

  // per-thread callback for capturing a row from the result set
  const auto do_thread_state_capture_row_data =
      [&](PolyResultSetThreadState& thread_state,
          std::vector<TargetValue>&& crt_row,
          const size_t row_idx,
          const size_t entry_idx) {
        poly_row_data[row_idx] = std::make_unique<PolyPtrVisitor>(std::move(crt_row),
                                                                  geo_slot_descriptor,
                                                                  entry_idx,
                                                                  polygeo_idx,
                                                                  num_coord_bytes);
        thread_state.addRow(row_idx, entry_idx, *poly_row_data[row_idx]);
      };

  // callback to reduce all the thread states built in the per-thread callback
  const auto do_thread_state_reduction =
      [&](std::vector<PolyResultSetThreadState>& thread_states) {
        const int states_per_gpu =
            num_gpus > thread_states.size()
                // all thread states on a single-gpu - TODO(croot):
                // should this be 1 thread-state per gpu yet?
                ? static_cast<int>(thread_states.size())
                // divide thread states across all available gpus
                : static_cast<int>(thread_states.size() / num_gpus);

        // set state_gpu_cnt = states_per_gpu + 1 to force an initialization step the
        // first time thru the for loop below
        int state_gpu_cnt{static_cast<int>(states_per_gpu) + 1};
        int gpu_idx{-1};

        // curr_stats = undefined. Don't touch until it is properly initialized in the for
        // loop
        PolyResultBufferInfo* curr_stats{nullptr};

        // assign the thread-states to a gpu, and build up the per-gpu stats to assist
        // with building the buffers
        for (auto& thread_state : thread_states) {
          if (state_gpu_cnt > states_per_gpu) {
            gpu_idx = is_geo_data_on_device ? result_device_id : (gpu_idx + 1) % num_gpus;

            // now curr_stats is defined
            curr_stats = &gpu_idx_to_poly_data_map
                              .emplace(std::piecewise_construct,
                                       std::forward_as_tuple(gpu_idx),
                                       std::forward_as_tuple(render_manager,
                                                             geo_slot_descriptor,
                                                             vega_data_table_name,
                                                             gpu_idx,
                                                             query_output_buffer_stride,
                                                             is_geo_data_on_device,
                                                             num_coord_bytes))
                              .first->second;

            state_gpu_cnt = 0;  // reset
          }
          CHECK(curr_stats);
          CHECK_EQ(thread_state.gpu_idx, -1);
          thread_state.gpu_idx = gpu_idx;
          thread_state.coords_offset = curr_stats->gpu_coord_count;
          thread_state.rings_offset = curr_stats->gpu_ring_count;
          thread_state.row_offset = curr_stats->gpu_row_count;

          curr_stats->gpu_coord_count += thread_state.coords_count;
          curr_stats->gpu_ring_count += thread_state.rings_count;
          auto row_count = thread_state.rowCount();
          curr_stats->gpu_row_count += row_count;
          num_rows += row_count;

          state_gpu_cnt++;
        }

        // need to initialize the per-gpu buffers. This needs to be done in the main
        // thread to avoid memory contention seg-faults.
        for (auto& item : gpu_idx_to_poly_data_map) {
          item.second.initializeBufferAndOffsets();
        }

        multithread_container(thread_states.begin(),
                              thread_states.end(),
                              [&](const PolyResultSetThreadState* thread_state) {
                                auto itr =
                                    gpu_idx_to_poly_data_map.find(thread_state->gpu_idx);
                                CHECK(itr != gpu_idx_to_poly_data_map.end());
                                itr->second.applyThreadStateData(*thread_state,
                                                                 poly_row_data,
                                                                 ssbo_col_idxs,
                                                                 result_targets,
                                                                 rowid_status);
                              });

        // now bus the built data buffers to the gpu. From there, we'll build out the poly
        // buffers, using groups and such

        // NOTE: Needed to use a PolyGpuCounts* argument for this lambda for std::async to
        // work. If you use a non-const reference argument to a function called by
        // std::async, it won't compile
        CHECK(render_info.render_allocator_map_ptr)
            << "Render allocator map not initialized.";
        multithread_container(gpu_idx_to_poly_data_map.begin(),
                              gpu_idx_to_poly_data_map.end(),
                              [&](decltype(gpu_idx_to_poly_data_map)::value_type* item) {
                                item->second.initializeRenderAllocator(
                                    *(render_info.render_allocator_map_ptr));
                              });
      };

  // Process the result set and capture row data in a thread-state object.
  START_RENDERER_TIMER(capture_vertex_info_timer);
  const auto row_count = executor_process_result_rows<PolyResultSetThreadState>(
      *rows, do_thread_state_capture_row_data, do_thread_state_reduction);
  poly_row_data.resize(row_count);
  poly_row_data.shrink_to_fit();
  STOP_RENDERER_TIMER(
      capture_vertex_info_timer, nullptr, INFO, "    capture vertex info");

  // TODO(croot): the following per-gpu code to build out the gpu buffers via thrust is
  // done multi-threaded. However, the code is broken out into stages because some of
  // the intermediate steps can not be multi-threaded - namely any render_manager
  // methods. Until those methods can be made thread-safe, the stages will need to be
  // broken up awkwardly like this.

  // run thrust stage 1
  multithread_container(gpu_idx_to_poly_data_map.begin(),
                        gpu_idx_to_poly_data_map.end(),
                        [&](decltype(gpu_idx_to_poly_data_map)::value_type* item) {
#ifdef HAVE_CUDA
                          CHECK_RENDER_CUDA_ERRORS(
                              cuCtxSetCurrent(cuda_device_contexts[item->second.gpu_idx]),
                              item->second.gpu_idx);
#endif  // HAVE_CUDA
                          item->second.runThrustStage1(
                              DataMgrThrustContext(ThrustAllocator(
                                  render_manager.getDataMgr(), item->second.gpu_idx)),
                              geo_col_sql_type.get_compression(),
                              qob_rowid_slot);
                        });

  // NOTE: adding the release scope guard first in case the initialize call fails within
  // that first loop
  ScopeGuard releaseBuffers = [&] {
    // release the render buffers.
    for (auto& item : gpu_idx_to_poly_data_map) {
      item.second.releasePolyRenderBuffers(vertLayout,
                                           data_query_result.render_data_layout);
    }
  };
  // initialize poly render buffers from results of stage 1
  for (auto& item : gpu_idx_to_poly_data_map) {
    item.second.initializePolyRenderBuffers(data_query_result.align_bytes);
  }

  // run thrust stage 2
  multithread_container(gpu_idx_to_poly_data_map.begin(),
                        gpu_idx_to_poly_data_map.end(),
                        [&](decltype(gpu_idx_to_poly_data_map)::value_type* item) {
#ifdef HAVE_CUDA
                          CHECK_RENDER_CUDA_ERRORS(
                              cuCtxSetCurrent(cuda_device_contexts[item->second.gpu_idx]),
                              item->second.gpu_idx);
#endif  // HAVE_CUDA
                          item->second.runThrustStage2(
                              per_row_data_buffer_col_idxs,
                              per_row_data_ssbo_offsets,
                              per_row_data_buffer_col_types,
                              data_query_result.align_bytes,
                              geo_col_sql_type.get_compression());
                        });

  render_info.setQueryVboLayout(vertLayout);
  render_info.setQuerySsboLayout(data_query_result.render_data_layout);

  return num_rows;
}

template <SQLTypes geo_type, SQLTypes... geo_types>
inline std::string print_geo_types(const std::string& separator) {
  return (toString(geo_type) + ... + (separator.data() + toString(geo_types)));
}

template <SQLTypes... geo_types>
int get_unique_geo_column_index_from_targets(
    const std::vector<TargetMetaInfo>& result_targets) {
  int geo_column_idx{-1};
  for (auto i = 0u; i < result_targets.size(); ++i) {
    auto const curr_type = result_targets[i].get_type_info().get_type();
    auto const is_valid_type = (... || (curr_type == geo_types));
    if (is_valid_type) {
      if (geo_column_idx >= 0) {
        throw std::runtime_error(
            "Found more than one " + print_geo_types<geo_types...>("/") +
            " column in query result. This is currently unsupported.");
      }
      geo_column_idx = i;
    }
  }
  if (geo_column_idx < 0) {
    throw std::runtime_error("Query result does not contain a " +
                             print_geo_types<geo_types...>(" or ") + " column. A " +
                             print_geo_types<geo_types...>("/") + " column is required.");
  }

  return geo_column_idx;
}

auto get_unique_poly_column_index_from_targets =
    get_unique_geo_column_index_from_targets<kPOLYGON, kMULTIPOLYGON>;
auto get_unique_line_column_index_from_targets =
    get_unique_geo_column_index_from_targets<kLINESTRING, kMULTILINESTRING>;

uint64_t process_polygons_non_in_situ(QueryRenderManager& render_manager,
                                      const ExecutionResult& results,
                                      const JSONLocation* data_loc,
                                      RenderInfo& render_info) {
  RENDER_LOG_SCOPE();
  CHECK(data_loc && data_loc->isValid());

  const auto& rows = results.getRows();
  CHECK(rows);

  LOG(INFO) << "process_polygons: start";
  START_RENDERER_TIMER(total_timer);

  // get the column index for the poly geo column
  const auto& result_targets = results.getTargetsMeta();
  auto const polygeo_idx =
      get_unique_poly_column_index_from_targets(results.getTargetsMeta());

  auto num_rows = process_polygons_non_in_situ(
      render_manager, rows, *data_loc, render_info, result_targets, polygeo_idx);

  STOP_RENDERER_TIMER(total_timer, nullptr, INFO, "total");
  LOG(INFO) << "process_polygons: end " << num_rows << " rows";

  return num_rows;
}

uint64_t process_geo_in_situ(QueryRenderManager& render_manager,
                             const InSituGeoRenderType in_situ_geo_render_type,
                             const ExecutionResult& results,
                             const JSONLocation* data_loc,
                             RenderInfo& render_info) {
  RENDER_LOG_SCOPE();
  CHECK(render_info.isInSitu());
  CHECK(data_loc && data_loc->isValid());

  const auto name_loc = data_loc->getMember(JSONSchema_v1::Data::kNameProp);
  const std::string vega_data_table_name = name_loc.getString();

#ifndef NDEBUG
  LOG(INFO) << "process_geo: start";
#endif

  START_RENDERER_TIMER(total_timer);

  // derive TargetMetaInfo from RenderInfo::targets as the ExecutionSet ResultSet is
  // empty
  // @TODO (croot) the one in the ResultSet should not be empty, so fix that
  std::vector<TargetMetaInfo> result_targets;
  for (const auto& t : render_info.targets) {
    const std::string& name = t->get_resname();
    const auto& ti = t->get_expr()->get_type_info();
    result_targets.emplace_back(name, ti);
  }

  // scan the query result columns for ONE geo column of the appropriate type
  int geo_column_index{-1};
  switch (in_situ_geo_render_type) {
    case InSituGeoRenderType::kPOLYGONS:
      geo_column_index = get_unique_poly_column_index_from_targets(result_targets);
      break;
    case InSituGeoRenderType::kLINES:
      geo_column_index = get_unique_line_column_index_from_targets(result_targets);
      break;
    default:
      CHECK(false) << toString(in_situ_geo_render_type);
  }
  auto const geo_column_name = result_targets[geo_column_index].get_resname();
  auto const geo_column_type_info = result_targets[geo_column_index].get_type_info();

  // determine if the coords column is compressed
  // we only support uncompressed and compressed(32) for now
  // @TODO support more
  EncodingType coords_encoding = geo_column_type_info.get_compression();
  int coords_comp_param = geo_column_type_info.get_comp_param();
  CHECK(coords_encoding == kENCODING_NONE ||
        (coords_encoding == kENCODING_GEOINT && coords_comp_param == 32))
      << "Geo column '" << geo_column_name
      << "' has unsupported compression type (must be 'NONE' or 'GEOINT(32)')";

  // @TODO (simon) combine these sets of indices into a struct to be passed around

  // get query buffer column indices for the
  // mandatory columns, and all the others
  int32_t coords_ptr_buffer_col_idx = -1;
  int32_t ring_sizes_ptr_buffer_col_idx = -1;

  // NOTE: this needs to default to 0 as it's required to be 0 in the non-multi linestring
  // case
  int32_t linestring_sizes_ptr_buffer_col_idx = 0;

  int32_t rowid_buffer_col_idx = -1;
  std::map<uint32_t, std::string> other_buffer_col_idx_to_name;

  // also capture target column indices for these
  // needed by get_render_data_template()
  // the first one is mandatory
  // the others are captured so we can tell get_render_data_template() to ignore them
  unsigned int geo_target_col_idx = 0;

  // first buffer column is 'key' which we skip
  int32_t buffer_col_idx = 1;
  for (size_t i = 0; i < result_targets.size(); i++) {
    const auto& t = result_targets[i];
    const auto& ti = t.get_type_info();
    if (t.get_resname() == geo_column_name) {
      // capture the geo (coords, ring_sizes, and [if multi] poly_rings)
      switch (in_situ_geo_render_type) {
        case InSituGeoRenderType::kLINES:
          CHECK(IS_GEO_LINE(ti.get_type()));
          coords_ptr_buffer_col_idx = buffer_col_idx++;
          buffer_col_idx++;  // skip over coords_num
          if (geo_column_type_info.get_type() == kMULTILINESTRING) {
            linestring_sizes_ptr_buffer_col_idx = buffer_col_idx++;
            buffer_col_idx++;  // skip over linestring_sizes_num
          }
          geo_target_col_idx = (unsigned int)i;
          break;
        case InSituGeoRenderType::kPOLYGONS:
          CHECK(IS_GEO_POLY(ti.get_type()));
          coords_ptr_buffer_col_idx = buffer_col_idx++;
          buffer_col_idx++;  // skip over coords_num
          ring_sizes_ptr_buffer_col_idx = buffer_col_idx++;
          buffer_col_idx++;  // skip over ring_sizes_num
          geo_target_col_idx = (unsigned int)i;
          if (geo_column_type_info.get_type() == kMULTIPOLYGON) {
            buffer_col_idx += 2;  // skip over poly_rings_ptr and poly_rings_num
          }
          break;
        default:
          break;
      }
    } else {
      // capture rowid
      if (t.get_resname() == kRowIdColumnName) {
        rowid_buffer_col_idx = buffer_col_idx;
      }
      // capture other column buffer col name and idx
      CHECK(!ti.is_array());
      other_buffer_col_idx_to_name[buffer_col_idx] = t.get_resname();
      buffer_col_idx++;
    }
  }

  // check we've got everything we need
  if (coords_ptr_buffer_col_idx < 0) {
    throw std::runtime_error("Failed to find '" + geo_column_name +
                             "_coords' in query result");
  }
  if (in_situ_geo_render_type == InSituGeoRenderType::kLINES) {
    if (geo_column_type_info.get_type() == kMULTILINESTRING &&
        linestring_sizes_ptr_buffer_col_idx == 0) {
      throw std::runtime_error("Failed to find '" + geo_column_name +
                               "_linestring_sizes' in query result");
    }
  } else if (in_situ_geo_render_type == InSituGeoRenderType::kPOLYGONS) {
    if (ring_sizes_ptr_buffer_col_idx < 0) {
      throw std::runtime_error("Failed to find '" + geo_column_name +
                               "_ring_sizes' in query result");
    }
    if (rowid_buffer_col_idx < 0) {
      // ppll polys always require rowid
      throw std::runtime_error(
          "Failed to find '" + std::string(kRowIdColumnName) +
          "' column in poly query result, which is required for poly rendering");
    }
  }

  // now we know how many actual buffer columns, we can compute the overall stride
  uint32_t query_output_buffer_stride = buffer_col_idx * DefaultNumBytesPerColumnType;

  // the target column indices we need
  std::vector<unsigned int> target_column_indices(result_targets.size());
  std::iota(target_column_indices.begin(), target_column_indices.end(), 0);

  // get destination for per-row shading data
  // we need the QueryDataLayout, size, and alignment
  // but we don't allocate a local buffer (as the SSBO is populated directly)
  // we only need one of these even for multi-GPU if we don't let it allocate
  // a buffer and compute the num_data_bytes ourselves

  std::unordered_set<unsigned int> target_column_indices_to_ignore;
  switch (in_situ_geo_render_type) {
    case InSituGeoRenderType::kPOLYGONS:
    case InSituGeoRenderType::kLINES:
      target_column_indices_to_ignore = {geo_target_col_idx};
      break;
    default:
      break;
  }

  auto data_query_result =
      get_render_data_template(result_targets,
                               render_info.targets,
                               target_column_indices,
                               target_column_indices_to_ignore,
                               {},  // extra target aliases
                               QueryDataLayout::LayoutType::kStorage,
                               0,  // not allocating buffer so this can be zero
                               RowIdStatus(result_targets.size(), render_info),
                               false,   // in-situ render, so does not use a result set
                               false);  // do NOT allocate local per-row buffer

  // get the SSBO layout for the per-row buffer
  CHECK(data_query_result.render_data_layout);

  auto ssbo_layout = data_query_result.render_data_layout->getBufferLayout();
  CHECK(ssbo_layout);

  // the simple data we need to pass to Stage 2 to describe the SSBO
  std::vector<uint32_t> per_row_data_buffer_col_idxs;
  std::vector<uint32_t> per_row_data_ssbo_offsets;
  std::vector<gfx::BufferAttrType> per_row_data_buffer_col_types;

  // then the other attrs in buffer_col_idx order (implicit from being a std::set)
  for (const auto& o : other_buffer_col_idx_to_name) {
    const auto& buffer_col_idx = o.first;
    const auto& name = o.second;
    auto offset = ssbo_layout->getAttributeByteOffset(name);
    auto buffer_attr_type = ssbo_layout->getAttributeType(name);
    per_row_data_buffer_col_idxs.push_back(buffer_col_idx);
    per_row_data_ssbo_offsets.push_back(offset);
    switch (buffer_attr_type) {
      case gfx::BufferAttrType::kInt:
      case gfx::BufferAttrType::kUint:
      case gfx::BufferAttrType::kInt64:
      case gfx::BufferAttrType::kUint64:
      case gfx::BufferAttrType::kFloat:
      case gfx::BufferAttrType::kDouble:
        per_row_data_buffer_col_types.push_back(buffer_attr_type);
        break;
      default:
        // @TODO simon.eves
        // do we need to support any other types?
        throw std::runtime_error(
            "Internal Error: In-Situ Geo SSBO has unsupported attr '" + name +
            "' of type " + gfx::to_string(buffer_attr_type));
    }
  }

  // vertex data layout
  auto vertLayout = build_line_poly_vbo_layout(coords_encoding);

  auto data_mgr = render_manager.getDataMgr();
  CHECK(data_mgr);
#ifdef HAVE_CUDA
  // get the CUDA contexts
  auto& cudaDeviceContexts = data_mgr->getCudaMgr()->getDeviceContexts();
  int numGpus = (int)cudaDeviceContexts.size();
#else
  int numGpus = render_manager.getNumGpus();
#endif  // HAVE_CUDA

  //
  // per GPU from here on?
  //

  struct InSituThrustResults {
    uint32_t in_row_count = 0;    // number of rows from query
    uint32_t out_row_count = 0;   // number of non-empty rows
    uint32_t out_vert_count = 0;  // total number of verts across all non-empty rows
    uint32_t out_item_count = 0;  // total number of polys/rings all non-empty rows
    InSituThrustResults& operator+=(const InSituThrustResults& rhs) {
      in_row_count += rhs.in_row_count;
      out_row_count += rhs.out_row_count;
      out_vert_count += rhs.out_vert_count;
      out_item_count += rhs.out_item_count;
      return *this;
    }
  };

  struct InSituThrustState {
    int gpu_id;
    InSituThrustResults results;
    std::unique_ptr<LineDataConverter> line_data_converter;
    std::unique_ptr<PolygonDataConverter> poly_data_converter;
    int8_t* query_output_buffer_base_ptr{nullptr};
    LineBufferPtrs line_buffer_ptrs;
    PolyBufferPtrs poly_buffer_ptrs;
    LineBufferMemoryDescriptors line_buffer_memory_descriptors;
    PolyBufferMemoryDescriptors poly_buffer_memory_descriptors;
  };

  //
  // thrust stage 1 task
  //

  auto execute_thrust_stage_1 = [&](InSituThrustState* state) {
    // get the actual query output buffer size and pointer
    CHECK(render_info.render_allocator_map_ptr);
    RenderAllocator* ra =
        render_info.render_allocator_map_ptr->getRenderAllocator(state->gpu_id);
    CHECK(ra);
    const auto query_output_buffer_offset = ra->getCurrentChunkOffset();
    const size_t num_query_output_buffer_bytes = ra->getCurrentChunkSize();
    state->query_output_buffer_base_ptr = ra->getBasePtr() + query_output_buffer_offset;
    ScopeGuard chunk_finish = [ra] { ra->markChunkComplete(); };

    if (!num_query_output_buffer_bytes) {
      // This means there was no data written to the buffer via the render allocator
      // so skip
      return;
    }

    // now we can deduce the number of rows
    // size should a whole multiple of the stride
    // @TODO make this work for all query types
    // currently only works for simple whole-table projections
    // also breaks with multi-fragment tables
    CHECK_EQ(num_query_output_buffer_bytes % query_output_buffer_stride, 0u)
        << "NQOBB=" << num_query_output_buffer_bytes
        << ", QOBS=" << query_output_buffer_stride;
    state->results.in_row_count =
        num_query_output_buffer_bytes / query_output_buffer_stride;

    // no query output rows from this GPU?
    if (state->results.in_row_count == 0) {
      return;
    }

#ifdef HAVE_CUDA
    // set the context for this GPU
    CHECK_RENDER_CUDA_ERRORS(cuCtxSetCurrent(cudaDeviceContexts[state->gpu_id]),
                             state->gpu_id);
#endif  // HAVE_CUDA

    // create a ThrustContext to pass to the data converter
    DataMgrThrustContext thrust_context(ThrustAllocator(data_mgr, state->gpu_id));

    switch (in_situ_geo_render_type) {
      case InSituGeoRenderType::kLINES:
        // make a line data converter
        state->line_data_converter =
            std::make_unique<LineDataConverter>(std::move(thrust_context));

        // run stage 1
        START_RENDERER_TIMER(thrust_stage1_timer);
        state->line_data_converter->ConvertStage1(state->results.in_row_count,
                                                  state->query_output_buffer_base_ptr,
                                                  query_output_buffer_stride,
                                                  coords_ptr_buffer_col_idx,
                                                  linestring_sizes_ptr_buffer_col_idx,
                                                  coords_encoding,
                                                  state->results.out_row_count,
                                                  state->results.out_vert_count,
                                                  state->results.out_item_count);
        STOP_RENDERER_TIMER(thrust_stage1_timer, nullptr, INFO, " Thrust Stage 1");
        break;
      case InSituGeoRenderType::kPOLYGONS:
        // make a poly data converter
        state->poly_data_converter =
            std::make_unique<PolygonDataConverter>(std::move(thrust_context));

        // run stage 1
        START_RENDERER_TIMER(thrust_stage1_timer);
        state->poly_data_converter->ConvertStage1(state->results.in_row_count,
                                                  state->query_output_buffer_base_ptr,
                                                  query_output_buffer_stride,
                                                  ring_sizes_ptr_buffer_col_idx,
                                                  rowid_buffer_col_idx,
                                                  state->results.out_row_count,
                                                  state->results.out_vert_count,
                                                  state->results.out_item_count);
        STOP_RENDERER_TIMER(thrust_stage1_timer, nullptr, INFO, " Thrust Stage 1");
        break;
    }

#ifndef NDEBUG
    LOG(INFO) << "DEBUG: GPU " << state->gpu_id
              << ", query: " << state->results.in_row_count
              << " rows, output: " << state->results.out_row_count << " rows, "
              << state->results.out_vert_count << " verts, "
              << state->results.out_item_count << " items";
#endif
  };

  //
  // create buffers task
  //

  // NOTE: num_rows does not need to be atomic because it is incremented in a
  // single-thread-only lambda. But using std::atomic as a safety precaution
  // in case the lambda is changed to be parallel at some point
  std::atomic<uint64_t> num_rows{0};
  auto execute_create_buffers = [&](InSituThrustState* state) {
    // anything to do?
    num_rows.fetch_add(state->results.out_row_count);
    if (state->results.out_row_count == 0) {
      return;
    }

    switch (in_situ_geo_render_type) {
      case InSituGeoRenderType::kLINES: {
        // allocate cache buffers now that we know what size they need to be
        auto const coord_value_size =
            (coords_encoding == kENCODING_GEOINT) ? sizeof(int32_t) : sizeof(double);
        LineTableByteData lineByteData(
            {state->results.out_vert_count * coord_value_size * 2,
             0,
             state->results.out_row_count * data_query_result.align_bytes,
             state->results.out_row_count * sizeof(gfx::IndirectDrawVertexData),
             0});

        auto& line_mgr = render_manager.getLineMgr();
        // create the in-situ buffers
        state->line_buffer_ptrs = line_mgr.createLineTableInSituBuffers(
            vega_data_table_name, state->gpu_id, lineByteData);

        // switch the buffers we just created to CUDA mode and get their handles
        state->line_buffer_memory_descriptors =
            line_mgr.getLineTableInSituBufferDescriptors(state->line_buffer_ptrs,
                                                         state->gpu_id);
      } break;
      case InSituGeoRenderType::kPOLYGONS: {
        // allocate cache buffers now that we know what size they need to bex
        auto const coord_value_size =
            (coords_encoding == kENCODING_GEOINT) ? sizeof(int32_t) : sizeof(double);
        PolyTableByteData polyByteData(
            {state->results.out_vert_count * coord_value_size * 2,
             state->results.out_row_count * sizeof(gfx::IndirectDrawVertexData),
             state->results.out_item_count * sizeof(gfx::IndirectDrawVertexData),
             data_query_result.align_bytes * state->results.out_row_count,
             0});

        // allocate rowids
        polyByteData.num_poly_rowids_bytes =
            state->results.out_item_count * sizeof(uint32_t);

        // create the in-situ buffers
        auto& poly_mgr = render_manager.getPolyMgr();
        state->poly_buffer_ptrs = poly_mgr.createPolyTableInSituBuffers(
            vega_data_table_name, state->gpu_id, polyByteData);

        // switch the buffers we just created to CUDA mode and get their handles
        state->poly_buffer_memory_descriptors =
            poly_mgr.getPolyTableInSituBufferDescriptors(state->poly_buffer_ptrs,
                                                         state->gpu_id);
      } break;
    }
  };

  //
  // thrust stage 2 task
  //

  auto execute_thrust_stage_2 = [&](InSituThrustState* state) {
    // anything to do?
    if (state->results.out_row_count == 0) {
      return;
    }

#ifdef HAVE_CUDA
    // set the context for this GPU
    CHECK_RENDER_CUDA_ERRORS(cuCtxSetCurrent(cudaDeviceContexts[state->gpu_id]),
                             state->gpu_id);
#endif  // HAVE_CUDA

    switch (in_situ_geo_render_type) {
      case InSituGeoRenderType::kLINES: {
        // run stage 2
        CHECK(state->line_data_converter);
        START_RENDERER_TIMER(thrust_stage2_timer);
        state->line_data_converter->ConvertStage2(state->query_output_buffer_base_ptr,
                                                  query_output_buffer_stride,
                                                  coords_ptr_buffer_col_idx,
                                                  linestring_sizes_ptr_buffer_col_idx,
                                                  coords_encoding,
                                                  per_row_data_buffer_col_idxs,
                                                  per_row_data_ssbo_offsets,
                                                  per_row_data_buffer_col_types,
                                                  data_query_result.align_bytes,
                                                  state->line_buffer_memory_descriptors);
        STOP_RENDERER_TIMER(thrust_stage2_timer, nullptr, INFO, "  Thrust Stage 2");
      } break;
      case InSituGeoRenderType::kPOLYGONS: {
        // outputs of stage 2
        std::vector<uint32_t> num_rows_per_batch;
        std::vector<uint32_t> num_polys_per_batch;

        // run stage 2
        CHECK(state->poly_data_converter);
        START_RENDERER_TIMER(thrust_stage2_timer);
        // @TODO simplify argument list with structs
        state->poly_data_converter->ConvertStage2(state->results.out_row_count,
                                                  state->query_output_buffer_base_ptr,
                                                  query_output_buffer_stride,
                                                  coords_ptr_buffer_col_idx,
                                                  ring_sizes_ptr_buffer_col_idx,
                                                  coords_encoding,
                                                  per_row_data_buffer_col_idxs,
                                                  per_row_data_ssbo_offsets,
                                                  per_row_data_buffer_col_types,
                                                  data_query_result.align_bytes,
                                                  state->poly_buffer_memory_descriptors,
                                                  num_rows_per_batch,
                                                  num_polys_per_batch);
        STOP_RENDERER_TIMER(thrust_stage2_timer, nullptr, INFO, "  Thrust Stage 2");

        // package the batch info
        auto poly_draw_batch_info = std::make_unique<PolyDrawBatchInfo>(
            std::move(num_rows_per_batch), std::move(num_polys_per_batch));

        // attach it to the table
        auto& poly_mgr = render_manager.getPolyMgr();
        poly_mgr.setPolyTableInSituBuffersPolyDrawBatchInfo(
            vega_data_table_name, state->gpu_id, std::move(poly_draw_batch_info));
      } break;
    }
  };

  //
  // release buffers task
  //

  auto execute_release_buffers = [&](InSituThrustState* state) {
    switch (in_situ_geo_render_type) {
      case InSituGeoRenderType::kLINES: {
        auto& line_mgr = render_manager.getLineMgr();
        line_mgr.releaseLineTableInSituBuffersForRendering(
            state->line_buffer_ptrs,
            state->gpu_id,
            vertLayout,
            data_query_result.render_data_layout);
      } break;
      case InSituGeoRenderType::kPOLYGONS: {
        auto& poly_mgr = render_manager.getPolyMgr();
        poly_mgr.releasePolyTableInSituBuffersForRendering(
            state->poly_buffer_ptrs,
            state->gpu_id,
            vertLayout,
            data_query_result.render_data_layout);
      } break;
    }
  };

  //
  // Run Thrust tasks on one thread per GPU
  // Buffer create/release cannot be threaded
  //

  std::vector<InSituThrustState> states(numGpus);
  int gpu_id = 0;
  for (auto& state : states) {
    state.gpu_id = gpu_id++;
  }

#ifndef NDEBUG
  InSituThrustResults total_results;
#endif

  // run tasks
  {
    ScopeGuard always_release_buffers = [&]() {
      // even if thrust_stage_2 throws
      singlethread_container(states.begin(), states.end(), execute_release_buffers);
    };
    multithread_container(states.begin(), states.end(), execute_thrust_stage_1);
    singlethread_container(states.begin(), states.end(), execute_create_buffers);
    multithread_container(states.begin(), states.end(), execute_thrust_stage_2);
  }

#ifndef NDEBUG
  // sum results
  for (int gpu_id = 0; gpu_id < numGpus; gpu_id++) {
    total_results += states[gpu_id].results;
  }

  if (total_results.in_row_count == 0) {
    LOG(INFO) << "process_geo: Query returned no result rows on any GPU";
  }
  if (total_results.out_row_count == 0) {
    LOG(INFO) << "process_geo: Query returned no non-empty rows";
  }
  if (total_results.out_vert_count == 0 || total_results.out_item_count == 0) {
    LOG(INFO) << "process_geo: Query returned no renderable geometry";
  }
#endif

  // update the render query data
  render_info.setQueryVboLayout(vertLayout);
  render_info.setQuerySsboLayout(data_query_result.render_data_layout);

  STOP_RENDERER_TIMER(total_timer, nullptr, INFO, "total");

#ifndef NDEBUG
  LOG(INFO) << "process_geo: end " << num_rows << " rows";
#endif

  return num_rows;
}

}  // namespace QueryRenderer
