/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExecuteRenderInterface/RenderQueryUtils/RelScanTree.h"
#include "QueryEngine/Rendering/RenderInfo.h"

namespace QueryRenderer {

struct NonInsituQueryClassifier {
  /**
   * Scans the RelAlg DAG of a render query to determine if it's associated render info
   * object should be set or forced to run non-insitu. This needs to be called after
   * parsing the query and generating the RelAlg DAG, but before executing the query.
   *
   * Queries should only be forced non-insitu for the purposes of hit-testing (i.e. when
   * rebuilding the query for hit-testing would not work or would otherwise be slow).
   * There are a handful of cases where queries should run non-insitu regardless of
   * whether hit-testing is enabled or not (i.e. aggregate queries or queries w/ window
   * functions or cursor-less table functions). All that logic is handled here in once
   * place.
   *
   * The logic for forcing a render query non-insitu for the purposes of hit-testing is
   * the following:
   *
   *   If the outermost node of the RelAlg DAG that handles the final column
   *     outputs/projections is an aggregate query
   *   OR
   *   If one of the final column outputs/projections has a dependency on an
   *     aggregate subquery
   *   THEN Mark the RenderInfo object forced-non-insitu.
   *
   * @param render_info The RenderInfo instance associated with the RelAlg DAG. This will
   * possibly be modified if the RelAlg DAG is force to run non-insitu.
   * @param rel_scan_tree The RelScanTree representation of a RelAlgDag, used to quickly
   * determine if the query should be classified as non-insitu
   */
  static void classify(RenderInfo& render_info,
                       const RelAlgDag& rel_alg_dag,
                       const RelScanTree* rel_scan_tree);
};

}  // namespace QueryRenderer
