/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "RelAlgTranslator.h"

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoColumn(const RexInput* rex_input,
                                                                                  SQLTypeInfo& ti) const {
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  const auto source = rex_input->getSourceNode();
  const auto it_rte_idx = input_to_nest_level_.find(source);
  CHECK(it_rte_idx != input_to_nest_level_.end());
  const int rte_idx = it_rte_idx->second;
  const auto scan_source = dynamic_cast<const RelScan*>(source);
  const auto& in_metainfo = source->getOutputMetainfo();
  CHECK(scan_source);
  // We're at leaf (scan) level and not supposed to have input metadata,
  // the name and type information come directly from the catalog.
  CHECK(in_metainfo.empty());
  const auto table_desc = scan_source->getTableDescriptor();
  const auto gcd = cat_.getMetadataForColumn(table_desc->tableId, rex_input->getIndex() + 1);
  CHECK(gcd);
  ti = gcd->columnType;
  CHECK(IS_GEO(ti.get_type()));
  // Translate geo column reference to a list of physical column refs
  for (auto i = 0; i < ti.get_physical_coord_cols(); i++) {
    const auto pcd = cat_.getMetadataForColumn(table_desc->tableId, rex_input->getIndex() + 1 + i + 1);
    auto pcol_ti = pcd->columnType;
    args.push_back(std::make_shared<Analyzer::ColumnVar>(pcol_ti, table_desc->tableId, pcd->columnId, rte_idx));
  }
  return args;
}

namespace Importer_NS {

bool importGeoFromWkt(std::string& wkt,
                      SQLTypeInfo& ti,
                      std::vector<double>& coords,
                      std::vector<int>& ring_sizes,
                      std::vector<int>& polygon_sizes);

std::vector<uint8_t> compress_coords(std::vector<double>& coords, const SQLTypeInfo& ti);

}  // Importer_NS

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoLiteral(const RexLiteral* rex_literal,
                                                                                   SQLTypeInfo& ti) const {
  CHECK(rex_literal);
  const auto e = translateLiteral(rex_literal);
  auto wkt = std::dynamic_pointer_cast<Analyzer::Constant>(e);
  CHECK(wkt);
  std::vector<double> coords;
  std::vector<int> ring_sizes;
  std::vector<int> poly_rings;
  int32_t srid = ti.get_output_srid();
  if (!Importer_NS::importGeoFromWkt(*wkt->get_constval().stringval, ti, coords, ring_sizes, poly_rings)) {
    throw QueryNotSupported("Could not read geometry from text");
  }
  ti.set_input_srid(srid);
  ti.set_output_srid(srid);
  // Compress geo literals by default
  if (srid == 4326) {
    ti.set_compression(kENCODING_GEOINT);
    ti.set_comp_param(32);
  }
  std::vector<uint8_t> compressed_coords = Importer_NS::compress_coords(coords, ti);
  std::list<std::shared_ptr<Analyzer::Expr>> compressed_coord_exprs;
  for (auto cc : compressed_coords) {
    Datum d;
    d.tinyintval = cc;
    auto e = makeExpr<Analyzer::Constant>(kTINYINT, false, d);
    compressed_coord_exprs.push_back(e);
  }
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  SQLTypeInfo arr_ti = SQLTypeInfo(kARRAY, true);
  arr_ti.set_subtype(kTINYINT);
  arr_ti.set_size(compressed_coords.size() * sizeof(int8_t));
  args.push_back(makeExpr<Analyzer::Constant>(arr_ti, false, compressed_coord_exprs));

  auto lit_type = ti.get_type();
  if (lit_type == kPOLYGON || lit_type == kMULTIPOLYGON) {
    std::list<std::shared_ptr<Analyzer::Expr>> ring_size_exprs;
    for (auto c : ring_sizes) {
      Datum d;
      d.intval = c;
      auto e = makeExpr<Analyzer::Constant>(kINT, false, d);
      ring_size_exprs.push_back(e);
    }
    SQLTypeInfo arr_ti = SQLTypeInfo(kARRAY, true);
    arr_ti.set_subtype(kINT);
    arr_ti.set_size(ring_sizes.size() * sizeof(int32_t));
    args.push_back(makeExpr<Analyzer::Constant>(arr_ti, false, ring_size_exprs));

    if (lit_type == kMULTIPOLYGON) {
      std::list<std::shared_ptr<Analyzer::Expr>> poly_rings_exprs;
      for (auto c : poly_rings) {
        Datum d;
        d.intval = c;
        auto e = makeExpr<Analyzer::Constant>(kINT, false, d);
        poly_rings_exprs.push_back(e);
      }
      SQLTypeInfo arr_ti = SQLTypeInfo(kARRAY, true);
      arr_ti.set_subtype(kINT);
      arr_ti.set_size(poly_rings.size() * sizeof(int32_t));
      args.push_back(makeExpr<Analyzer::Constant>(arr_ti, false, poly_rings_exprs));
    }
  }

  return args;
}

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoFunctionArg(const RexScalar* rex_scalar,
                                                                                       SQLTypeInfo& arg_ti,
                                                                                       int32_t& lindex) const {
  std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;

  const auto rex_input = dynamic_cast<const RexInput*>(rex_scalar);
  if (rex_input) {
    const auto input = translateInput(rex_input);
    const auto column = dynamic_cast<const Analyzer::ColumnVar*>(input.get());
    if (!column || !column->get_type_info().is_geometry()) {
      throw QueryNotSupported("Geo function is expecting a geo column argument");
    }
    return translateGeoColumn(rex_input, arg_ti);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex_scalar);
  if (rex_function) {
    if (rex_function->getName() == std::string("ST_Transform")) {
      CHECK_EQ(size_t(2), rex_function->size());
      const auto rex_scalar0 = dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() + ": unexpected first argument");
      }
      auto arg0 = translateGeoFunctionArg(rex_scalar0, arg_ti, lindex);

      const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_function->getOperand(1));
      if (!rex_literal) {
        throw QueryNotSupported(rex_function->getName() + ": second argument is expected to be a literal");
      }
      const auto e = translateLiteral(rex_literal);
      auto ce = std::dynamic_pointer_cast<Analyzer::Constant>(e);
      if (!ce || !e->get_type_info().is_integer()) {
        throw QueryNotSupported(rex_function->getName() + ": expecting integer SRID");
      }
      int32_t srid = 0;
      if (e->get_type_info().get_type() == kSMALLINT) {
        srid = static_cast<int32_t>(ce->get_constval().smallintval);
      } else if (e->get_type_info().get_type() == kTINYINT) {
        srid = static_cast<int32_t>(ce->get_constval().tinyintval);
      } else if (e->get_type_info().get_type() == kINT) {
        srid = static_cast<int32_t>(ce->get_constval().intval);
      } else {
        throw QueryNotSupported(rex_function->getName() + ": expecting integer SRID");
      }
      if (srid != 900913) {
        throw QueryNotSupported(rex_function->getName() + ": unsupported output SRID " + std::to_string(srid));
      }
      if (arg_ti.get_input_srid() > 0) {
        if (arg_ti.get_input_srid() != 4326) {
          throw QueryNotSupported(rex_function->getName() + ": unsupported input SRID " +
                                  std::to_string(arg_ti.get_input_srid()));
        }
        arg_ti.set_output_srid(srid);  // We have a valid input SRID, register the output SRID for transform
      } else {
        throw QueryNotSupported(rex_function->getName() + ": unexpected input SRID, unable to transform");
      }
      return arg0;
    } else if (rex_function->getName() == std::string("ST_GeomFromText")) {
      CHECK(rex_function->size() == size_t(1) || rex_function->size() == size_t(2));
      // First - register srid, then send it to geo literal translation
      int32_t srid = 0;
      if (rex_function->size() == 2) {
        const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_function->getOperand(1));
        if (!rex_literal) {
          throw QueryNotSupported(rex_function->getName() + ": second argument is expected to be a literal");
        }
        const auto e = translateLiteral(rex_literal);
        auto ce = std::dynamic_pointer_cast<Analyzer::Constant>(e);
        if (!ce || !e->get_type_info().is_integer()) {
          throw QueryNotSupported(rex_function->getName() + ": expecting integer SRID");
        }
        if (e->get_type_info().get_type() == kSMALLINT) {
          srid = static_cast<int32_t>(ce->get_constval().smallintval);
        } else if (e->get_type_info().get_type() == kTINYINT) {
          srid = static_cast<int32_t>(ce->get_constval().tinyintval);
        } else if (e->get_type_info().get_type() == kINT) {
          srid = static_cast<int32_t>(ce->get_constval().intval);
        } else {
          throw QueryNotSupported(rex_function->getName() + " expecting integer SRID");
        }
        if (srid != 0 && srid != 4326 && srid != 900913) {
          throw QueryNotSupported(rex_function->getName() + ": unsupported SRID " + std::to_string(srid));
        }
      }
      arg_ti.set_input_srid(srid);   // Input SRID
      arg_ti.set_output_srid(srid);  // Output SRID is the same - no transform

      const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_function->getOperand(0));
      if (!rex_literal) {
        throw QueryNotSupported(rex_function->getName() + " expects a string literal as first argument");
      }
      auto arg0 = translateGeoLiteral(rex_literal, arg_ti);
      return arg0;
    } else if (rex_function->getName() == std::string("ST_PointN")) {
      CHECK_EQ(size_t(2), rex_function->size());
      const auto rex_scalar0 = dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() + ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(rex_scalar0, arg_ti, lindex);
      if (arg_ti.get_type() != kLINESTRING) {
        throw QueryNotSupported(rex_function->getName() + " expects LINESTRING as first argument");
      }
      const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_function->getOperand(1));
      if (!rex_literal) {
        throw QueryNotSupported(rex_function->getName() + ": second argument is expected to be a literal");
      }
      const auto e = translateLiteral(rex_literal);
      auto ce = std::dynamic_pointer_cast<Analyzer::Constant>(e);
      if (!ce || !e->get_type_info().is_integer()) {
        throw QueryNotSupported(rex_function->getName() + ": expecting integer index as second argument");
      }
      int32_t index = 0;
      if (e->get_type_info().get_type() == kSMALLINT) {
        index = static_cast<int32_t>(ce->get_constval().smallintval);
      } else if (e->get_type_info().get_type() == kTINYINT) {
        index = static_cast<int32_t>(ce->get_constval().tinyintval);
      } else if (e->get_type_info().get_type() == kINT) {
        index = static_cast<int32_t>(ce->get_constval().intval);
      } else {
        throw QueryNotSupported(rex_function->getName() + " expecting integer index");
      }
      if (lindex != 0) {
        throw QueryNotSupported(rex_function->getName() + ": LINESTRING is already indexed");
      }
      if (index == 0) {
        throw QueryNotSupported(rex_function->getName() + ": invalid index");
      }
      lindex = index;
      return arg0;
    } else if (rex_function->getName() == std::string("ST_StartPoint")) {
      CHECK_EQ(size_t(1), rex_function->size());
      const auto rex_scalar0 = dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() + ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(rex_scalar0, arg_ti, lindex);
      if (arg_ti.get_type() != kLINESTRING) {
        throw QueryNotSupported(rex_function->getName() + " expects LINESTRING as first argument");
      }
      if (lindex != 0) {
        throw QueryNotSupported(rex_function->getName() + ": LINESTRING is already indexed");
      }
      lindex = 1;
      return arg0;
    } else if (rex_function->getName() == std::string("ST_EndPoint")) {
      CHECK_EQ(size_t(1), rex_function->size());
      const auto rex_scalar0 = dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() + ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(rex_scalar0, arg_ti, lindex);
      if (arg_ti.get_type() != kLINESTRING) {
        throw QueryNotSupported(rex_function->getName() + " expects LINESTRING as first argument");
      }
      if (lindex != 0) {
        throw QueryNotSupported(rex_function->getName() + ": LINESTRING is already indexed");
      }
      lindex = -1;
      return arg0;
    } else if (rex_function->getName() == std::string("ST_SRID")) {
      CHECK_EQ(size_t(1), rex_function->size());
      const auto rex_scalar0 = dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() + ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(rex_scalar0, arg_ti, lindex);
      if (!IS_GEO(arg_ti.get_type())) {
        throw QueryNotSupported(rex_function->getName() + " expects geometry argument");
      }
      return arg0;
    } else if (rex_function->getName() == std::string("ST_SetSRID")) {
      CHECK_EQ(size_t(2), rex_function->size());
      const auto rex_scalar0 = dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() + ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(rex_scalar0, arg_ti, lindex);
      if (!IS_GEO(arg_ti.get_type())) {
        throw QueryNotSupported(rex_function->getName() + " expects geometry argument");
      }
      const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_function->getOperand(1));
      if (!rex_literal) {
        throw QueryNotSupported(rex_function->getName() + ": second argument is expected to be a literal");
      }
      const auto e = translateLiteral(rex_literal);
      auto ce = std::dynamic_pointer_cast<Analyzer::Constant>(e);
      if (!ce || !e->get_type_info().is_integer()) {
        throw QueryNotSupported(rex_function->getName() + ": expecting integer SRID");
      }
      int32_t srid = 0;
      if (e->get_type_info().get_type() == kSMALLINT) {
        srid = static_cast<int32_t>(ce->get_constval().smallintval);
      } else if (e->get_type_info().get_type() == kTINYINT) {
        srid = static_cast<int32_t>(ce->get_constval().tinyintval);
      } else if (e->get_type_info().get_type() == kINT) {
        srid = static_cast<int32_t>(ce->get_constval().intval);
      } else {
        throw QueryNotSupported(rex_function->getName() + ": expecting integer SRID");
      }
      arg_ti.set_input_srid(srid);   // Input SRID
      arg_ti.set_output_srid(srid);  // Output SRID is the same - no transform
      return arg0;
    } else {
      throw QueryNotSupported("Unsupported argument: " + rex_function->getName());
    }
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_scalar);
  if (rex_literal) {
    return translateGeoLiteral(rex_literal, arg_ti);
  }
  throw QueryNotSupported("Geo function argument not supported");
}

namespace {

std::string suffix(SQLTypes type) {
  if (type == kPOINT)
    return std::string("_Point");
  if (type == kLINESTRING)
    return std::string("_LineString");
  if (type == kPOLYGON)
    return std::string("_Polygon");
  if (type == kMULTIPOLYGON)
    return std::string("_MultiPolygon");
  throw QueryNotSupported("Unsupported argument type");
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateUnaryGeoFunction(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());

  SQLTypeInfo arg_ti;
  int32_t lindex = 0;
  auto geoargs = translateGeoFunctionArg(rex_function->getOperand(0), arg_ti, lindex);

  if (rex_function->getName() == std::string("ST_SRID")) {
    Datum output_srid;
    output_srid.intval = arg_ti.get_output_srid();
    return makeExpr<Analyzer::Constant>(kINT, false, output_srid);
  }

  // All geo function calls translated below only need the coords, extras e.g. ring_sizes are dropped.
  // Specialize for other/new functions if needed.
  geoargs.erase(geoargs.begin() + 1, geoargs.end());

  std::string specialized_geofunc{rex_function->getName()};
  if (rex_function->getName() == std::string("ST_X") || rex_function->getName() == std::string("ST_Y")) {
    if (arg_ti.get_type() == kLINESTRING) {
      if (lindex == 0) {
        throw QueryNotSupported(rex_function->getName() + " expects a POINT, use LINESTRING accessor, e.g. ST_POINTN");
      }
      Datum index;
      index.intval = lindex;
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, index));
    } else if (arg_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() + " expects a POINT");
    }
    specialized_geofunc += suffix(arg_ti.get_type());
  }

  // Add input compression mode and SRID args to enable on-the-fly decompression/transforms
  Datum input_compression;
  input_compression.intval = (arg_ti.get_compression() == kENCODING_GEOINT && arg_ti.get_comp_param() == 32) ? 1 : 0;
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression));
  Datum input_srid;
  input_srid.intval = arg_ti.get_input_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid));

  // Add output SRID arg to enable on-the-fly transforms
  Datum output_srid;
  output_srid.intval = arg_ti.get_output_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

  return makeExpr<Analyzer::FunctionOper>(rex_function->getType(), specialized_geofunc, geoargs);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateBinaryGeoFunction(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());

  std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;
  SQLTypeInfo arg0_ti;
  SQLTypeInfo arg1_ti;
  int32_t lindex0 = 0;
  int32_t lindex1 = 0;

  auto geoargs0 = translateGeoFunctionArg(rex_function->getOperand(0), arg0_ti, lindex0);
  if (arg0_ti.get_type() == kLINESTRING) {
    Datum index;
    index.intval = lindex0;
    geoargs0.push_back(makeExpr<Analyzer::Constant>(kINT, false, index));
  }
  geoargs.insert(geoargs.end(), geoargs0.begin(), geoargs0.end());
  auto geoargs1 = translateGeoFunctionArg(rex_function->getOperand(1), arg1_ti, lindex1);
  if (arg1_ti.get_type() == kLINESTRING) {
    Datum index;
    index.intval = lindex1;
    geoargs1.push_back(makeExpr<Analyzer::Constant>(kINT, false, index));
  }
  geoargs.insert(geoargs.end(), geoargs1.begin(), geoargs1.end());

  if (arg0_ti.get_output_srid() > 0 && arg0_ti.get_output_srid() != arg1_ti.get_output_srid()) {
    throw QueryNotSupported(rex_function->getName() + " cannot accept different SRIDs");
  }

  std::string specialized_geofunc{rex_function->getName() + suffix(arg0_ti.get_type()) + suffix(arg1_ti.get_type())};

  if (arg0_ti.get_output_srid() == 4326) {
    // Need to call geodesic runtime functions
    if (rex_function->getName() == std::string("ST_Distance")) {
      if ((arg0_ti.get_type() == kPOINT || (arg0_ti.get_type() == kLINESTRING && lindex0 != 0)) &&
          (arg1_ti.get_type() == kPOINT || (arg1_ti.get_type() == kLINESTRING && lindex1 != 0))) {
        // Geodesic distance between point or indexed linestrings
        specialized_geofunc += std::string("_Geodesic");
      } else {
        throw QueryNotSupported(rex_function->getName() + " currently doesn't accept non-POINT geographies");
      }
    } else {
      throw QueryNotSupported(rex_function->getName() + " currently doesn't accept geographies");
    }
  }

  // Add first input's compression mode and SRID args to enable on-the-fly decompression/transforms
  Datum input_compression0;
  input_compression0.intval = (arg0_ti.get_compression() == kENCODING_GEOINT && arg0_ti.get_comp_param() == 32) ? 1 : 0;
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression0));
  Datum input_srid0;
  input_srid0.intval = arg0_ti.get_input_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid0));

  // Add second input's compression mode and SRID args to enable on-the-fly decompression/transforms
  Datum input_compression1;
  input_compression1.intval = (arg1_ti.get_compression() == kENCODING_GEOINT && arg1_ti.get_comp_param() == 32) ? 1 : 0;
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression1));
  Datum input_srid1;
  input_srid1.intval = arg1_ti.get_input_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid1));

  // Add output SRID arg to enable on-the-fly transforms
  Datum output_srid;
  output_srid.intval = arg0_ti.get_output_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

  return makeExpr<Analyzer::FunctionOper>(rex_function->getType(), specialized_geofunc, geoargs);
}
