/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/RenderQueryResultProcessor.h"

#include "ExecuteRenderInterface/RenderQueryUtils/ProcessGeoResults.h"
#include "ExecuteRenderInterface/RenderQueryUtils/ProcessRasterMeshResults.h"
#include "ExecuteRenderInterface/RenderQueryUtils/ProcessRowResults.h"
#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"

namespace QueryRenderer {

uint64_t RenderResultProcessor::processExecuteQueryResults(
    QueryRenderManager& render_manager,
    const ExecutionResult& execute_results,
    const JSONLocation* data_loc,
    const RenderQuerySpecialtyType render_query_type,
    RenderInfo& render_info,
    RasterMeshMetadata* raster_mesh_metadata) {
  RENDER_LOG_SCOPE();
  uint64_t num_rows{0};
  if (render_info.isInSitu()) {
    switch (render_query_type) {
      case RenderQuerySpecialtyType::kPolys: {
        // render POLYGON, MULTIPOLYGON using Thrust pre-processing
        // @TODO implement POLYGON, MULTIPOLYGON as rows with mesh shader
        num_rows = process_geo_in_situ(render_manager,
                                       InSituGeoRenderType::kPOLYGONS,
                                       execute_results,
                                       data_loc,
                                       render_info);
        break;
      }
      case RenderQuerySpecialtyType::kLines: {
        // render LINESTRING, MULTILINESTRING using Thrust pre-processing
        num_rows = process_geo_in_situ(render_manager,
                                       InSituGeoRenderType::kLINES,
                                       execute_results,
                                       data_loc,
                                       render_info);
        break;
      }
      case RenderQuerySpecialtyType::kNone: {
        // render scalar point, POINT, MULTIPOINT direct
        num_rows = process_rows_in_situ(render_info);
        break;
      }
      case RenderQuerySpecialtyType::kMesh2d: {
        // Unreachable because all mesh2d rendering should be forced non-insitu
        UNREACHABLE();
        break;
      }
    }
  } else {
    // non-in-situ data
    switch (render_query_type) {
      case RenderQuerySpecialtyType::kPolys: {
        num_rows = process_polygons_non_in_situ(
            render_manager, execute_results, data_loc, render_info);
        break;
      }
      case RenderQuerySpecialtyType::kLines: {
        num_rows = process_lines_non_in_situ(
            render_manager, execute_results, data_loc, render_info);
        break;
      }
      case RenderQuerySpecialtyType::kNone: {
        num_rows = process_rows_non_in_situ(render_manager, execute_results, render_info);
        break;
      }
      case RenderQuerySpecialtyType::kMesh2d: {
        RUNTIME_EX_ASSERT(
            raster_mesh_metadata,
            "RasterMeshMetadata is null, Unable to process Mesh2d query results");
        num_rows = process_raster_mesh_non_in_situ(render_manager,
                                                   execute_results,
                                                   *raster_mesh_metadata,
                                                   *data_loc,
                                                   render_info);
        break;
      }
    }
  }

  return num_rows;
}

}  // namespace QueryRenderer
