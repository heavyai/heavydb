/*
 * Copyright 2021 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "Analyzer/Analyzer.h"
#include "Geospatial/Compression.h"

// routines for converting from WKB/OGRGeometry to Analyzer Expressions

namespace Geospatial {

std::shared_ptr<Analyzer::Constant> convert_coords(const std::vector<double>& coords,
                                                   const SQLTypeInfo& ti) {
  std::vector<uint8_t> compressed_coords = Geospatial::compress_coords(coords, ti);
  std::list<std::shared_ptr<Analyzer::Expr>> compressed_coords_exprs;
  for (auto cc : compressed_coords) {
    Datum d;
    d.tinyintval = cc;
    auto e = makeExpr<Analyzer::Constant>(kTINYINT, false, d);
    compressed_coords_exprs.push_back(e);
  }
  SQLTypeInfo arr_ti = SQLTypeInfo(kARRAY, true);
  arr_ti.set_subtype(kTINYINT);
  arr_ti.set_size(compressed_coords.size() * sizeof(int8_t));
  arr_ti.set_compression(ti.get_compression());
  arr_ti.set_comp_param((ti.get_compression() == kENCODING_GEOINT) ? 32 : 64);
  return makeExpr<Analyzer::Constant>(arr_ti, false, compressed_coords_exprs);
}

std::shared_ptr<Analyzer::Constant> convert_rings(const std::vector<int>& rings) {
  std::list<std::shared_ptr<Analyzer::Expr>> ring_size_exprs;
  for (auto c : rings) {
    Datum d;
    d.intval = c;
    auto e = makeExpr<Analyzer::Constant>(kINT, false, d);
    ring_size_exprs.push_back(e);
  }
  SQLTypeInfo arr_ti = SQLTypeInfo(kARRAY, true);
  arr_ti.set_subtype(kINT);
  arr_ti.set_size(rings.size() * sizeof(int32_t));
  return makeExpr<Analyzer::Constant>(arr_ti, false, ring_size_exprs);
}

}  // namespace Geospatial
