/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "QueryRenderer/Interface/RenderQueryInterfaceDeclarations.h"
#include "QueryRenderer/Interface/SqlSelectedTableInfo.h"
#include "Shared/Rendering/InSituFlags.h"

namespace QueryRenderer {

class QueryDataLayout;
using QueryDataLayoutShPtr = std::shared_ptr<QueryDataLayout>;

/**
 * Composition of data defining the structure of a render query.
 * Right now this includes the list of tables selected by the SQL and the target
 * expressions defining the final column outputs. This data is accessible after parsing
 * the query.
 */
struct RenderQueryStructure : public heavyai::InSituFlagsOwnerInterface {
  RenderQueryStructure(TargetEntries in_target_entries,
                       SQLSelectedTableContainer in_sql_selected_tables,
                       const heavyai::InSituFlags in_insitu_flags)
      : heavyai::InSituFlagsOwnerInterface(in_insitu_flags)
      , output_target_entries{std::move(in_target_entries)}
      , sql_selected_tables{std::move(in_sql_selected_tables)} {}

  TargetEntries output_target_entries;
  SQLSelectedTableContainer sql_selected_tables;
};

/**
 * Utility macro for adding RenderQueryStructure data getters for structs/classes that
 * are composed of a nested RenderQueryStructure instance.
 */
#define ADD_RENDER_QUERY_STRUCTURE_INTERFACE(render_query_structure)       \
  const SQLSelectedTableContainer& getSqlSelectedTables() const {          \
    return render_query_structure.sql_selected_tables;                     \
  }                                                                        \
                                                                           \
  const PhysicalTableInfoContainer& getSqlSelectedPhysicalTables() const { \
    return render_query_structure.sql_selected_tables.phys_tables;         \
  }                                                                        \
                                                                           \
  const TargetEntries& getOutputTargetEntries() const {                    \
    return render_query_structure.output_target_entries;                   \
  }

/**
 * Composition of the 2 possible buffer layouts for render queries, the vbo &
 * ssbo layouts. The layouts are used when transitioning the buffers from query to
 * render and define the contents of the respective render buffers
 */
struct RenderQueryBufferLayouts {
  RenderQueryBufferLayouts() : vbo_layout{nullptr}, ssbo_layout{nullptr} {}

  RenderQueryBufferLayouts(QueryDataLayoutShPtr in_vbo_layout,
                           QueryDataLayoutShPtr in_ssbo_layout)
      : vbo_layout{std::move(in_vbo_layout)}, ssbo_layout{std::move(in_ssbo_layout)} {}

  QueryDataLayoutShPtr vbo_layout = nullptr;
  QueryDataLayoutShPtr ssbo_layout = nullptr;

  void reset() {
    vbo_layout.reset();
    ssbo_layout.reset();
  }
};

/**
 * Composition of various elements describing the result of executing the render query.
 * This includes a result set, the total number of rows (valid + invalid), and
 * insitu-related data.
 */
class RenderQueryResult : public heavyai::InSituFlagsOwnerInterface {
 public:
  RenderQueryResult()
      : heavyai::InSituFlagsOwnerInterface(heavyai::InSituFlags::kInSitu)
      , result_set{nullptr}
      , num_rows{0} {}

  RenderQueryResult(std::shared_ptr<ResultSet> in_result_set,
                    uint64_t in_num_rows,
                    const heavyai::InSituFlags in_insitu_flags)
      : heavyai::InSituFlagsOwnerInterface(in_insitu_flags)
      , result_set{std::move(in_result_set)}
      , num_rows{in_num_rows} {}

  // the result set from the render query.
  // NOTE: this would be an empty result set if has_insitu_results = true
  std::shared_ptr<ResultSet> result_set = nullptr;

  // The total number of rows resulting from the render query execution.
  // This can include both valid and invalid rows (invalid rows can be found in in-situ
  // renders since query engine currently does not write results to a query output buffer
  // in a packed form)
  // NOTE: the number of resulting rows cannot be extracted from the result_rows item
  // above if the render was performed in-situ as it would be an empty ResultSet.
  uint64_t num_rows = 0;

  ResultSet* getResultSetPtr() { return result_set.get(); }
  const ResultSet* getResultSetPtr() const { return result_set.get(); }
  size_t getResultSetUseCount() const { return result_set.use_count(); }

  void clear() {
    result_set = nullptr;
    num_rows = 0;
    insitu_flags_ = heavyai::InSituFlags::kInSitu;
  }
};

/**
 * Utility macro for adding RenderQueryResult data getters for structs/classes that
 * are composed of a nested RenderQueryResult instance.
 */
#define ADD_RENDER_QUERY_RESULT_INTERFACE(render_query_result) \
  ResultSet* getResultSetPtr() {                               \
    return render_query_result.getResultSetPtr();              \
  }                                                            \
  const ResultSet* getResultSetPtr() const {                   \
    return render_query_result.getResultSetPtr();              \
  }                                                            \
                                                               \
  size_t getResultSetUseCount() const {                        \
    return render_query_result.getResultSetUseCount();         \
  }                                                            \
                                                               \
  bool isInSitu() const {                                      \
    return render_query_result.isInSitu();                     \
  }                                                            \
  bool couldRunInSitu() const {                                \
    return render_query_result.couldRunInSitu();               \
  }                                                            \
  uint64_t numResultRows() const {                             \
    return render_query_result.num_rows;                       \
  }

/**
 * Composition of data that describes the full output of executing a render query,
 * including the query's structure and results.
 */
struct RenderQueryOutput {
  RenderQueryOutput(RenderQueryStructure in_render_query_structure,
                    RenderQueryResult in_render_query_result)
      : render_query_structure{std::move(in_render_query_structure)}
      , render_query_result{std::move(in_render_query_result)} {}

  RenderQueryStructure render_query_structure;
  RenderQueryResult render_query_result;

  ADD_RENDER_QUERY_STRUCTURE_INTERFACE(render_query_structure);
  ADD_RENDER_QUERY_RESULT_INTERFACE(render_query_result);
};

/**
 * Composition of data that results from parsing a render query. This includes the
 * structure of the render query and its respective expected buffer layouts.
 */
struct RenderQueryParseData {
  RenderQueryParseData(RenderQueryStructure in_render_query_structure,
                       RenderQueryBufferLayouts in_render_buffer_layouts)
      : render_query_structure{std::move(in_render_query_structure)}
      , render_buffer_layouts{std::move(in_render_buffer_layouts)} {}

  RenderQueryStructure render_query_structure;
  RenderQueryBufferLayouts render_buffer_layouts;

  ADD_RENDER_QUERY_STRUCTURE_INTERFACE(render_query_structure);
};

/**
 * Composition of data that results from fully executing a render query. This includes the
 * full render query output and its respective bufer layouts.
 */
struct RenderQueryExecuteData {
  RenderQueryExecuteData(RenderQueryBufferLayouts in_render_buffer_layouts,
                         RenderQueryOutput in_render_query_output)
      : render_buffer_layouts{std::move(in_render_buffer_layouts)}
      , render_query_output{std::move(in_render_query_output)} {}

  RenderQueryBufferLayouts render_buffer_layouts;
  RenderQueryOutput render_query_output;

  ADD_RENDER_QUERY_STRUCTURE_INTERFACE(render_query_output.render_query_structure);
  ADD_RENDER_QUERY_RESULT_INTERFACE(render_query_output.render_query_result);
};

}  // namespace QueryRenderer
