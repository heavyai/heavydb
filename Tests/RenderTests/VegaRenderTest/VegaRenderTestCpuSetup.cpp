/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "VegaRenderTestCpuSetup.h"

#ifndef HAVE_CUDA

#include <QueryRenderer/QueryDataLayout.h>

void get_random_simple_point_row_data(UniformRandomNumberGenerator<double>& rand,
                                      SimplePointRow* row,
                                      int idx,
                                      double xmin,
                                      double xmax,
                                      double ymin,
                                      double ymax,
                                      double valmin,
                                      double valmax,
                                      int64_t partymin,
                                      int64_t partymax) {
  row[idx].key = idx;
  row[idx].x = xmin + rand() * (xmax - xmin);
  row[idx].y = ymin + rand() * (ymax - ymin);
  row[idx].val = valmin + rand() * (valmax - valmin);
  ;
  row[idx].party =
      partymin + int64_t(floor(rand() * float(partymax - partymin + 1) - 0.0000001));
  row[idx].rowid = idx;
}

std::shared_ptr<::QueryRenderer::QueryDataLayout> SimplePointRow::getQueryDataLayout() {
  std::shared_ptr<::QueryRenderer::QueryDataLayout> rtn(
      new ::QueryRenderer::QueryDataLayout(
          {::QueryRenderer::QueryDataLayout::AttrAliasInfo("key",
                                                           SQLTypeInfo(kBIGINT, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("x",
                                                           SQLTypeInfo(kDOUBLE, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("y",
                                                           SQLTypeInfo(kDOUBLE, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("val",
                                                           SQLTypeInfo(kDOUBLE, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("party",
                                                           SQLTypeInfo(kBIGINT, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("rowid",
                                                           SQLTypeInfo(kBIGINT, true))}));
  return rtn;
}

void SimplePointRow::get_random_data(UniformRandomNumberGenerator<double>& rand,
                                     SimplePointRow* row,
                                     int numPts,
                                     const std::array<double, 2>& xExtents,
                                     const std::array<double, 2>& yExtents,
                                     const std::array<double, 2>& valExtents,
                                     const std::array<int64_t, 2>& partyExtents) {
  for (int i = 0; i < numPts; i++) {
    get_random_simple_point_row_data(rand,
                                     row,
                                     i,
                                     xExtents[0],
                                     xExtents[1],
                                     yExtents[0],
                                     yExtents[1],
                                     valExtents[0],
                                     valExtents[1],
                                     partyExtents[0],
                                     partyExtents[1]);
  }
}

#endif  // HAVE_CUDA
