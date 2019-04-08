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

#include "../Shared/geo_types.h"
#include "RelAlgTranslator.h"

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoColumn(
    const RexInput* rex_input,
    SQLTypeInfo& ti,
    const bool with_bounds,
    const bool with_render_group,
    const bool expand_geo_col) const {
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  const auto source = rex_input->getSourceNode();
  const auto it_rte_idx = input_to_nest_level_.find(source);
  CHECK(it_rte_idx != input_to_nest_level_.end());
  const int rte_idx = it_rte_idx->second;
  const auto& in_metainfo = source->getOutputMetainfo();

  int32_t table_id{0};
  int column_id{-1};
  const auto scan_source = dynamic_cast<const RelScan*>(source);
  if (scan_source) {
    // We're at leaf (scan) level and not supposed to have input metadata,
    // the name and type information come directly from the catalog.
    CHECK(in_metainfo.empty());

    const auto td = scan_source->getTableDescriptor();
    table_id = td->tableId;

    const auto gcd = cat_.getMetadataForColumnBySpi(table_id, rex_input->getIndex() + 1);
    CHECK(gcd);
    ti = gcd->columnType;
    column_id = gcd->columnId;

  } else {
    // Likely backed by a temp table. Read the table ID from the source node and negate it
    // (see RelAlgTranslator::translateInput)
    table_id = -source->getId();

    CHECK(!expand_geo_col);

    CHECK(!in_metainfo.empty());
    CHECK_GE(rte_idx, 0);
    column_id = rex_input->getIndex();
    CHECK_LT(column_id, in_metainfo.size());
    ti = in_metainfo[column_id].get_type_info();
  }
  CHECK(IS_GEO(ti.get_type()));

  // Return geo column reference. The geo column may be expanded if required for extension
  // function arguments. Otherwise, the geo column reference will be translated into
  // physical columns as required. Bounds column will be added if present and requested.
  if (expand_geo_col) {
    for (auto i = 0; i < ti.get_physical_coord_cols(); i++) {
      const auto pcd = cat_.getMetadataForColumnBySpi(
          table_id, SPIMAP_GEO_PHYSICAL_INPUT(rex_input->getIndex(), i + 1));
      auto pcol_ti = pcd->columnType;
      args.push_back(std::make_shared<Analyzer::ColumnVar>(
          pcol_ti, table_id, pcd->columnId, rte_idx));
    }
  } else {
    args.push_back(
        std::make_shared<Analyzer::ColumnVar>(ti, table_id, column_id, rte_idx));
  }
  if (with_bounds && ti.has_bounds()) {
    const auto bounds_cd = cat_.getMetadataForColumnBySpi(
        table_id,
        SPIMAP_GEO_PHYSICAL_INPUT(rex_input->getIndex(),
                                  ti.get_physical_coord_cols() + 1));
    auto bounds_ti = bounds_cd->columnType;
    args.push_back(std::make_shared<Analyzer::ColumnVar>(
        bounds_ti, table_id, bounds_cd->columnId, rte_idx));
  }
  if (with_render_group && ti.has_render_group()) {
    const auto render_group_cd = cat_.getMetadataForColumnBySpi(
        table_id,
        SPIMAP_GEO_PHYSICAL_INPUT(rex_input->getIndex(),
                                  ti.get_physical_coord_cols() + 2));
    auto render_group_ti = render_group_cd->columnType;
    args.push_back(std::make_shared<Analyzer::ColumnVar>(
        render_group_ti, table_id, render_group_cd->columnId, rte_idx));
  }
  return args;
}

namespace Importer_NS {

std::vector<uint8_t> compress_coords(std::vector<double>& coords, const SQLTypeInfo& ti);

}  // namespace Importer_NS

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoLiteral(
    const RexLiteral* rex_literal,
    SQLTypeInfo& ti,
    bool with_bounds) const {
  CHECK(rex_literal);
  if (rex_literal->getType() != kTEXT) {
    throw std::runtime_error("Geo literals must be strings");
  }
  const auto e = translateLiteral(rex_literal);
  auto wkt = std::dynamic_pointer_cast<Analyzer::Constant>(e);
  CHECK(wkt);
  std::vector<double> coords;
  std::vector<double> bounds;
  std::vector<int> ring_sizes;
  std::vector<int> poly_rings;
  int32_t srid = ti.get_output_srid();
  if (!Geo_namespace::GeoTypesFactory::getGeoColumns(
          *wkt->get_constval().stringval, ti, coords, bounds, ring_sizes, poly_rings)) {
    throw QueryNotSupported("Could not read geometry from text");
  }
  ti.set_subtype(kGEOMETRY);
  ti.set_input_srid(srid);
  ti.set_output_srid(srid);
  // Compress geo literals by default
  if (srid == 4326) {
    ti.set_compression(kENCODING_GEOINT);
    ti.set_comp_param(32);
  }

  std::vector<std::shared_ptr<Analyzer::Expr>> args;

  std::vector<uint8_t> compressed_coords = Importer_NS::compress_coords(coords, ti);
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
  args.push_back(makeExpr<Analyzer::Constant>(arr_ti, false, compressed_coords_exprs));

  auto lit_type = ti.get_type();
  if (lit_type == kPOLYGON || lit_type == kMULTIPOLYGON) {
    // ring sizes
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

    // poly rings
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

  if (with_bounds && ti.has_bounds()) {
    // bounds
    std::list<std::shared_ptr<Analyzer::Expr>> bounds_exprs;
    for (auto b : bounds) {
      Datum d;
      d.doubleval = b;
      auto e = makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
      bounds_exprs.push_back(e);
    }
    SQLTypeInfo arr_ti = SQLTypeInfo(kARRAY, true);
    arr_ti.set_subtype(kDOUBLE);
    arr_ti.set_size(bounds.size() * sizeof(double));
    args.push_back(makeExpr<Analyzer::Constant>(arr_ti, false, bounds_exprs));
  }

  return args;
}

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoFunctionArg(
    const RexScalar* rex_scalar,
    SQLTypeInfo& arg_ti,
    int32_t& lindex,
    const bool with_bounds,
    const bool with_render_group,
    const bool expand_geo_col) const {
  std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;

  const auto rex_input = dynamic_cast<const RexInput*>(rex_scalar);
  if (rex_input) {
    const auto input = translateInput(rex_input);
    const auto column = dynamic_cast<const Analyzer::ColumnVar*>(input.get());
    if (!column || !column->get_type_info().is_geometry()) {
      throw QueryNotSupported("Geo function is expecting a geo column argument");
    }
    return translateGeoColumn(
        rex_input, arg_ti, with_bounds, with_render_group, expand_geo_col);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex_scalar);
  if (rex_function) {
    if (rex_function->getName() == std::string("ST_Transform")) {
      CHECK_EQ(size_t(2), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() + ": unexpected first argument");
      }
      auto arg0 = translateGeoFunctionArg(
          rex_scalar0, arg_ti, lindex, with_bounds, with_render_group, expand_geo_col);

      const auto rex_literal =
          dynamic_cast<const RexLiteral*>(rex_function->getOperand(1));
      if (!rex_literal) {
        throw QueryNotSupported(rex_function->getName() +
                                ": second argument is expected to be a literal");
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
        throw QueryNotSupported(rex_function->getName() + ": unsupported output SRID " +
                                std::to_string(srid));
      }
      if (arg_ti.get_input_srid() > 0) {
        if (arg_ti.get_input_srid() != 4326) {
          throw QueryNotSupported(rex_function->getName() + ": unsupported input SRID " +
                                  std::to_string(arg_ti.get_input_srid()));
        }
        arg_ti.set_output_srid(
            srid);  // We have a valid input SRID, register the output SRID for transform
      } else {
        throw QueryNotSupported(rex_function->getName() +
                                ": unexpected input SRID, unable to transform");
      }
      return arg0;
    } else if (rex_function->getName() == std::string("ST_GeomFromText") ||
               rex_function->getName() == std::string("ST_GeogFromText")) {
      CHECK(rex_function->size() == size_t(1) || rex_function->size() == size_t(2));
      // First - register srid, then send it to geo literal translation
      int32_t srid = 0;
      if (rex_function->size() == 2) {
        const auto rex_literal =
            dynamic_cast<const RexLiteral*>(rex_function->getOperand(1));
        if (!rex_literal) {
          throw QueryNotSupported(rex_function->getName() +
                                  ": second argument is expected to be a literal");
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
          throw QueryNotSupported(rex_function->getName() + ": unsupported SRID " +
                                  std::to_string(srid));
        }
      }
      arg_ti.set_input_srid(srid);   // Input SRID
      arg_ti.set_output_srid(srid);  // Output SRID is the same - no transform

      const auto rex_literal =
          dynamic_cast<const RexLiteral*>(rex_function->getOperand(0));
      if (!rex_literal) {
        throw QueryNotSupported(rex_function->getName() +
                                " expects a string literal as first argument");
      }
      auto arg0 = translateGeoLiteral(rex_literal, arg_ti, with_bounds);
      arg_ti.set_subtype((rex_function->getName() == std::string("ST_GeogFromText"))
                             ? kGEOGRAPHY
                             : kGEOMETRY);
      return arg0;
    } else if (rex_function->getName() == std::string("ST_PointN")) {
      CHECK_EQ(size_t(2), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(
          rex_scalar0, arg_ti, lindex, with_bounds, with_render_group, expand_geo_col);
      if (arg_ti.get_type() != kLINESTRING) {
        throw QueryNotSupported(rex_function->getName() +
                                " expects LINESTRING as first argument");
      }
      const auto rex_literal =
          dynamic_cast<const RexLiteral*>(rex_function->getOperand(1));
      if (!rex_literal) {
        throw QueryNotSupported(rex_function->getName() +
                                ": second argument is expected to be a literal");
      }
      const auto e = translateLiteral(rex_literal);
      auto ce = std::dynamic_pointer_cast<Analyzer::Constant>(e);
      if (!ce || !e->get_type_info().is_integer()) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expecting integer index as second argument");
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
        throw QueryNotSupported(rex_function->getName() +
                                ": LINESTRING is already indexed");
      }
      if (index == 0) {
        throw QueryNotSupported(rex_function->getName() + ": invalid index");
      }
      lindex = index;
      return arg0;
    } else if (rex_function->getName() == std::string("ST_StartPoint")) {
      CHECK_EQ(size_t(1), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(
          rex_scalar0, arg_ti, lindex, with_bounds, with_render_group, expand_geo_col);
      if (arg_ti.get_type() != kLINESTRING) {
        throw QueryNotSupported(rex_function->getName() +
                                " expects LINESTRING as first argument");
      }
      if (lindex != 0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": LINESTRING is already indexed");
      }
      lindex = 1;
      return arg0;
    } else if (rex_function->getName() == std::string("ST_EndPoint")) {
      CHECK_EQ(size_t(1), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(
          rex_scalar0, arg_ti, lindex, with_bounds, with_render_group, expand_geo_col);
      if (arg_ti.get_type() != kLINESTRING) {
        throw QueryNotSupported(rex_function->getName() +
                                " expects LINESTRING as first argument");
      }
      if (lindex != 0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": LINESTRING is already indexed");
      }
      lindex = -1;
      return arg0;
    } else if (rex_function->getName() == std::string("ST_SRID")) {
      CHECK_EQ(size_t(1), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(
          rex_scalar0, arg_ti, lindex, with_bounds, with_render_group, expand_geo_col);
      if (!IS_GEO(arg_ti.get_type())) {
        throw QueryNotSupported(rex_function->getName() + " expects geometry argument");
      }
      return arg0;
    } else if (rex_function->getName() == std::string("ST_SetSRID")) {
      CHECK_EQ(size_t(2), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(
          rex_scalar0, arg_ti, lindex, with_bounds, with_render_group, expand_geo_col);
      if (!IS_GEO(arg_ti.get_type())) {
        throw QueryNotSupported(rex_function->getName() + " expects geometry argument");
      }
      const auto rex_literal =
          dynamic_cast<const RexLiteral*>(rex_function->getOperand(1));
      if (!rex_literal) {
        throw QueryNotSupported(rex_function->getName() +
                                ": second argument is expected to be a literal");
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
    } else if (rex_function->getName() == std::string("CastToGeography")) {
      CHECK_EQ(size_t(1), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(
          rex_scalar0, arg_ti, lindex, with_bounds, with_render_group, expand_geo_col);
      if (!IS_GEO(arg_ti.get_type())) {
        throw QueryNotSupported(rex_function->getName() + " expects geometry argument");
      }
      if (arg_ti.get_output_srid() != 4326) {
        throw QueryNotSupported(rex_function->getName() +
                                " expects geometry with SRID=4326");
      }
      arg_ti.set_subtype(kGEOGRAPHY);
      return arg0;
    } else {
      throw QueryNotSupported("Unsupported argument: " + rex_function->getName());
    }
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_scalar);
  if (rex_literal) {
    return translateGeoLiteral(rex_literal, arg_ti, with_bounds);
  }
  throw QueryNotSupported("Geo function argument not supported");
}

namespace {

std::string suffix(SQLTypes type) {
  if (type == kPOINT) {
    return std::string("_Point");
  }
  if (type == kLINESTRING) {
    return std::string("_LineString");
  }
  if (type == kPOLYGON) {
    return std::string("_Polygon");
  }
  if (type == kMULTIPOLYGON) {
    return std::string("_MultiPolygon");
  }
  throw QueryNotSupported("Unsupported argument type");
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateUnaryGeoFunction(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());

  int32_t lindex = 0;
  std::string specialized_geofunc{rex_function->getName()};

  // Geo function calls which do not need the coords col but do need cols associated with
  // physical coords (e.g. ring_sizes / poly_rings)
  if (rex_function->getName() == std::string("ST_NRings")) {
    SQLTypeInfo arg_ti;
    auto geoargs = translateGeoFunctionArg(
        rex_function->getOperand(0), arg_ti, lindex, false, false, true);
    if (arg_ti.get_type() == kPOLYGON) {
      CHECK_EQ(geoargs.size(), 2);
      geoargs.erase(geoargs.begin(), geoargs.begin() + 1);  // remove the coords
      return makeExpr<Analyzer::FunctionOper>(
          rex_function->getType(), specialized_geofunc, geoargs);
    } else if (arg_ti.get_type() == kMULTIPOLYGON) {
      CHECK_EQ(geoargs.size(), 3);
      geoargs.erase(geoargs.begin(), geoargs.begin() + 1);  // remove the coords
      geoargs.erase(geoargs.begin() + 1, geoargs.end());    // remove the poly_rings
      return makeExpr<Analyzer::FunctionOper>(
          rex_function->getType(), specialized_geofunc, geoargs);
    } else {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a POLYGON or MULTIPOLYGON");
    }
  } else if (rex_function->getName() == std::string("ST_NPoints")) {
    SQLTypeInfo arg_ti;
    auto geoargs = translateGeoFunctionArg(
        rex_function->getOperand(0), arg_ti, lindex, false, false, true);
    geoargs.erase(geoargs.begin() + 1, geoargs.end());  // remove all but coords
    // Add compression information
    Datum input_compression;
    input_compression.intval =
        (arg_ti.get_compression() == kENCODING_GEOINT && arg_ti.get_comp_param() == 32)
            ? 1
            : 0;
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression));
    return makeExpr<Analyzer::FunctionOper>(
        rex_function->getType(), specialized_geofunc, geoargs);
  } else if (rex_function->getName() == std::string("ST_Perimeter") ||
             rex_function->getName() == std::string("ST_Area")) {
    SQLTypeInfo arg_ti;
    auto geoargs = translateGeoFunctionArg(
        rex_function->getOperand(0), arg_ti, lindex, false, false, true);
    if (arg_ti.get_type() != kPOLYGON && arg_ti.get_type() != kMULTIPOLYGON) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a POLYGON or MULTIPOLYGON");
    }
    specialized_geofunc += suffix(arg_ti.get_type());
    if (arg_ti.get_subtype() == kGEOGRAPHY && arg_ti.get_output_srid() == 4326) {
      specialized_geofunc += std::string("_Geodesic");
    }
    // Add compression information
    Datum input_compression;
    input_compression.intval =
        (arg_ti.get_compression() == kENCODING_GEOINT && arg_ti.get_comp_param() == 32)
            ? 1
            : 0;
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression));
    Datum input_srid;
    input_srid.intval = arg_ti.get_input_srid();
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid));
    Datum output_srid;
    output_srid.intval = arg_ti.get_output_srid();
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));
    return makeExpr<Analyzer::FunctionOper>(
        rex_function->getType(), specialized_geofunc, geoargs);
  }

  // Accessors for poly bounds and render group for in-situ poly render queries
  if (rex_function->getName() == std::string("MapD_GeoPolyBoundsPtr") ||  // deprecated
      rex_function->getName() == std::string("OmniSci_Geo_PolyBoundsPtr")) {
    SQLTypeInfo arg_ti;
    // get geo column plus bounds only (not expanded)
    auto geoargs = translateGeoFunctionArg(
        rex_function->getOperand(0), arg_ti, lindex, true, false, false);
    // this function only works on polys
    if (!IS_GEO_POLY(arg_ti.get_type())) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a POLYGON or MULTIPOLYGON");
    }
    // only need the bounds argument (last), discard the rest
    geoargs.erase(geoargs.begin(), geoargs.end() - 1);
    // done
    return makeExpr<Analyzer::FunctionOper>(
        rex_function->getType(), specialized_geofunc, geoargs);
  } else if (rex_function->getName() ==
                 std::string("MapD_GeoPolyRenderGroup") ||  // deprecated
             rex_function->getName() == std::string("OmniSci_Geo_PolyRenderGroup")) {
    SQLTypeInfo arg_ti;
    // get geo column plus render_group only (not expanded)
    auto geoargs = translateGeoFunctionArg(
        rex_function->getOperand(0), arg_ti, lindex, false, true, false);
    // this function only works on polys
    if (!IS_GEO_POLY(arg_ti.get_type())) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a POLYGON or MULTIPOLYGON");
    }
    // only need the render_group argument (last), discard the rest
    geoargs.erase(geoargs.begin(), geoargs.end() - 1);
    // done
    return makeExpr<Analyzer::FunctionOper>(
        rex_function->getType(), specialized_geofunc, geoargs);
  }

  // All functions below use geo col as reference and expand it as necessary
  SQLTypeInfo arg_ti;
  bool with_bounds = true;
  auto geoargs = translateGeoFunctionArg(
      rex_function->getOperand(0), arg_ti, lindex, with_bounds, false, false);

  if (rex_function->getName() == std::string("ST_SRID")) {
    Datum output_srid;
    output_srid.intval = arg_ti.get_output_srid();
    return makeExpr<Analyzer::Constant>(kINT, false, output_srid);
  }

  if (rex_function->getName() == std::string("ST_XMin") ||
      rex_function->getName() == std::string("ST_YMin") ||
      rex_function->getName() == std::string("ST_XMax") ||
      rex_function->getName() == std::string("ST_YMax")) {
    // If type has bounds - use them, otherwise look at coords
    if (arg_ti.has_bounds()) {
      if (lindex != 0) {
        throw QueryNotSupported(rex_function->getName() +
                                " doesn't support indexed LINESTRINGs");
      }
      // Only need the bounds argument, discard the rest
      geoargs.erase(geoargs.begin(), geoargs.end() - 1);

      // Supply srids too - transformed geo would have a transformed bounding box
      Datum input_srid;
      input_srid.intval = arg_ti.get_input_srid();
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid));
      Datum output_srid;
      output_srid.intval = arg_ti.get_output_srid();
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

      specialized_geofunc += std::string("_Bounds");
      return makeExpr<Analyzer::FunctionOper>(
          rex_function->getType(), specialized_geofunc, geoargs);
    }
  }

  // All geo function calls translated below only need the coords, extras e.g. ring_sizes
  // are dropped. Specialize for other/new functions if needed.
  geoargs.erase(geoargs.begin() + 1, geoargs.end());

  if (rex_function->getName() == std::string("ST_X") ||
      rex_function->getName() == std::string("ST_Y")) {
    if (arg_ti.get_type() == kLINESTRING) {
      if (lindex == 0) {
        throw QueryNotSupported(
            rex_function->getName() +
            " expects a POINT, use LINESTRING accessor, e.g. ST_POINTN");
      }
      Datum index;
      index.intval = lindex;
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, index));
    } else if (arg_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() + " expects a POINT");
    }
    specialized_geofunc += suffix(arg_ti.get_type());
  } else if (rex_function->getName() == std::string("ST_Length")) {
    if (arg_ti.get_type() != kLINESTRING || lindex != 0) {
      throw QueryNotSupported(rex_function->getName() + " expects unindexed LINESTRING");
    }
    specialized_geofunc += suffix(arg_ti.get_type());
    if (arg_ti.get_subtype() == kGEOGRAPHY && arg_ti.get_output_srid() == 4326) {
      specialized_geofunc += std::string("_Geodesic");
    }
  }

  // Add input compression mode and SRID args to enable on-the-fly
  // decompression/transforms
  Datum input_compression;
  input_compression.intval =
      (arg_ti.get_compression() == kENCODING_GEOINT && arg_ti.get_comp_param() == 32) ? 1
                                                                                      : 0;
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression));
  Datum input_srid;
  input_srid.intval = arg_ti.get_input_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid));

  // Add output SRID arg to enable on-the-fly transforms
  Datum output_srid;
  output_srid.intval = arg_ti.get_output_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

  return makeExpr<Analyzer::FunctionOper>(
      rex_function->getType(), specialized_geofunc, geoargs);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateBinaryGeoFunction(
    const RexFunctionOperator* rex_function) const {
  auto function_name = rex_function->getName();
  auto return_type = rex_function->getType();
  bool swap_args = false;
  bool with_bounds = false;
  bool negate_result = false;
  if (function_name == "ST_DWithin") {
    CHECK_EQ(size_t(3), rex_function->size());
    function_name = "ST_Distance";
    return_type = SQLTypeInfo(kDOUBLE, false);
  } else if (function_name == "ST_DFullyWithin") {
    CHECK_EQ(size_t(3), rex_function->size());
    function_name = "ST_MaxDistance";
    return_type = SQLTypeInfo(kDOUBLE, false);
  } else {
    CHECK_EQ(size_t(2), rex_function->size());
  }
  if (function_name == std::string("ST_Within")) {
    function_name = "ST_Contains";
    swap_args = true;
  } else if (function_name == std::string("ST_Disjoint")) {
    function_name = "ST_Intersects";
    negate_result = true;
  }
  if (function_name == std::string("ST_Contains") ||
      function_name == std::string("ST_Intersects")) {
    with_bounds = true;
  }

  std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;
  SQLTypeInfo arg0_ti;
  SQLTypeInfo arg1_ti;
  int32_t lindex0 = 0;
  int32_t lindex1 = 0;

  auto geoargs0 = translateGeoFunctionArg(rex_function->getOperand(swap_args ? 1 : 0),
                                          arg0_ti,
                                          lindex0,
                                          with_bounds,
                                          false,
                                          false);
  if (arg0_ti.get_type() == kLINESTRING) {
    Datum index;
    index.intval = lindex0;
    geoargs0.push_back(makeExpr<Analyzer::Constant>(kINT, false, index));
  }
  geoargs.insert(geoargs.end(), geoargs0.begin(), geoargs0.end());
  auto geoargs1 = translateGeoFunctionArg(rex_function->getOperand(swap_args ? 0 : 1),
                                          arg1_ti,
                                          lindex1,
                                          with_bounds,
                                          false,
                                          false);
  if (arg1_ti.get_type() == kLINESTRING) {
    Datum index;
    index.intval = lindex1;
    geoargs1.push_back(makeExpr<Analyzer::Constant>(kINT, false, index));
  }
  geoargs.insert(geoargs.end(), geoargs1.begin(), geoargs1.end());

  if (arg0_ti.get_subtype() != kNULLT && arg0_ti.get_subtype() != arg1_ti.get_subtype()) {
    throw QueryNotSupported(rex_function->getName() +
                            " accepts either two GEOGRAPHY or two GEOMETRY arguments");
  }
  if (arg0_ti.get_output_srid() > 0 &&
      arg0_ti.get_output_srid() != arg1_ti.get_output_srid()) {
    throw QueryNotSupported(rex_function->getName() + " cannot accept different SRIDs");
  }

  std::string specialized_geofunc{function_name + suffix(arg0_ti.get_type()) +
                                  suffix(arg1_ti.get_type())};

  if (arg0_ti.get_subtype() == kGEOGRAPHY && arg0_ti.get_output_srid() == 4326) {
    // Need to call geodesic runtime functions
    if (function_name == std::string("ST_Distance")) {
      if ((arg0_ti.get_type() == kPOINT ||
           (arg0_ti.get_type() == kLINESTRING && lindex0 != 0)) &&
          (arg1_ti.get_type() == kPOINT ||
           (arg1_ti.get_type() == kLINESTRING && lindex1 != 0))) {
        // Geodesic distance between points (or indexed linestrings)
        specialized_geofunc += std::string("_Geodesic");
      } else {
        throw QueryNotSupported(function_name +
                                " currently doesn't accept non-POINT geographies");
      }
    } else if (rex_function->getName() == std::string("ST_Contains")) {
      // We currently don't have a geodesic implementation of ST_Contains,
      // allowing calls to a [less precise] cartesian implementation.
    } else {
      throw QueryNotSupported(function_name + " doesn't accept geographies");
    }
  }

  // Add first input's compression mode and SRID args to enable on-the-fly
  // decompression/transforms
  Datum input_compression0;
  input_compression0.intval =
      (arg0_ti.get_compression() == kENCODING_GEOINT && arg0_ti.get_comp_param() == 32)
          ? 1
          : 0;
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression0));
  Datum input_srid0;
  input_srid0.intval = arg0_ti.get_input_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid0));

  // Add second input's compression mode and SRID args to enable on-the-fly
  // decompression/transforms
  Datum input_compression1;
  input_compression1.intval =
      (arg1_ti.get_compression() == kENCODING_GEOINT && arg1_ti.get_comp_param() == 32)
          ? 1
          : 0;
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression1));
  Datum input_srid1;
  input_srid1.intval = arg1_ti.get_input_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid1));

  // Add output SRID arg to enable on-the-fly transforms
  Datum output_srid;
  output_srid.intval = arg0_ti.get_output_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

  auto result =
      makeExpr<Analyzer::FunctionOper>(return_type, specialized_geofunc, geoargs);
  if (negate_result) {
    return makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  }
  return result;
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateTernaryGeoFunction(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(3), rex_function->size());

  auto distance_expr = translateScalarRex(rex_function->getOperand(2));
  const auto& distance_ti = SQLTypeInfo(kDOUBLE, false);
  if (distance_expr->get_type_info().get_type() != kDOUBLE) {
    distance_expr = makeExpr<Analyzer::UOper>(distance_ti, false, kCAST, distance_expr);
  }

  // Translate the geo distance function call portion
  const auto geo_distance_expr = translateBinaryGeoFunction(rex_function);

  return makeExpr<Analyzer::BinOper>(
      kBOOLEAN, kLE, kONE, geo_distance_expr, distance_expr);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateFunctionWithGeoArg(
    const RexFunctionOperator* rex_function) const {
  int32_t lindex = 0;
  std::string specialized_geofunc{rex_function->getName()};
  if (rex_function->getName() == std::string("convert_meters_to_pixel_width") ||
      rex_function->getName() == std::string("convert_meters_to_pixel_height")) {
    CHECK_EQ(rex_function->size(), 6);
    SQLTypeInfo arg_ti;
    std::vector<std::shared_ptr<Analyzer::Expr>> args;
    args.push_back(translateScalarRex(rex_function->getOperand(0)));
    auto geoargs = translateGeoFunctionArg(
        rex_function->getOperand(1), arg_ti, lindex, false, true, false);
    // only works on points
    if (arg_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a point for the second argument");
    }

    args.insert(args.end(), geoargs.begin(), geoargs.begin() + 1);

    // Add compression information
    Datum input_compression;
    input_compression.intval =
        (arg_ti.get_compression() == kENCODING_GEOINT && arg_ti.get_comp_param() == 32)
            ? 1
            : 0;
    args.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression));
    if (arg_ti.get_input_srid() != 4326) {
      throw QueryNotSupported(
          rex_function->getName() +
          " currently only supports points of with SRID WGS84/EPSG:4326");
    }
    Datum input_srid;
    input_srid.intval = arg_ti.get_input_srid();
    args.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid));
    Datum output_srid;
    // Forcing web-mercator projection for now
    // TODO(croot): check that the input-to-output conversion routines exist?
    output_srid.intval =
        arg_ti.get_output_srid() != 900913 ? 900913 : arg_ti.get_output_srid();
    args.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

    args.push_back(translateScalarRex(rex_function->getOperand(2)));
    args.push_back(translateScalarRex(rex_function->getOperand(3)));
    args.push_back(translateScalarRex(rex_function->getOperand(4)));
    args.push_back(translateScalarRex(rex_function->getOperand(5)));
    return makeExpr<Analyzer::FunctionOper>(
        rex_function->getType(), specialized_geofunc, args);
  } else if (rex_function->getName() == std::string("is_point_in_view")) {
    CHECK_EQ(rex_function->size(), 5);
    SQLTypeInfo arg_ti;
    std::vector<std::shared_ptr<Analyzer::Expr>> args;
    auto geoargs = translateGeoFunctionArg(
        rex_function->getOperand(0), arg_ti, lindex, false, true, false);
    // only works on points
    if (arg_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a point for the second argument");
    }

    args.insert(args.end(), geoargs.begin(), geoargs.begin() + 1);

    // Add compression information
    Datum input_compression;
    input_compression.intval =
        (arg_ti.get_compression() == kENCODING_GEOINT && arg_ti.get_comp_param() == 32)
            ? 1
            : 0;
    args.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression));
    if (arg_ti.get_input_srid() != 4326) {
      throw QueryNotSupported(
          rex_function->getName() +
          " currently only supports points of with SRID WGS84/EPSG:4326");
    }
    args.push_back(translateScalarRex(rex_function->getOperand(1)));
    args.push_back(translateScalarRex(rex_function->getOperand(2)));
    args.push_back(translateScalarRex(rex_function->getOperand(3)));
    args.push_back(translateScalarRex(rex_function->getOperand(4)));
    return makeExpr<Analyzer::FunctionOper>(
        rex_function->getType(), specialized_geofunc, args);
  } else if (rex_function->getName() == std::string("is_point_size_in_view")) {
    CHECK_EQ(rex_function->size(), 6);
    SQLTypeInfo arg_ti;
    std::vector<std::shared_ptr<Analyzer::Expr>> args;
    auto geoargs = translateGeoFunctionArg(
        rex_function->getOperand(0), arg_ti, lindex, false, true, false);
    // only works on points
    if (arg_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a point for the second argument");
    }

    args.insert(args.end(), geoargs.begin(), geoargs.begin() + 1);

    // Add compression information
    Datum input_compression;
    input_compression.intval =
        (arg_ti.get_compression() == kENCODING_GEOINT && arg_ti.get_comp_param() == 32)
            ? 1
            : 0;
    args.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression));
    if (arg_ti.get_input_srid() != 4326) {
      throw QueryNotSupported(
          rex_function->getName() +
          " currently only supports points of with SRID WGS84/EPSG:4326");
    }
    args.push_back(translateScalarRex(rex_function->getOperand(1)));
    args.push_back(translateScalarRex(rex_function->getOperand(2)));
    args.push_back(translateScalarRex(rex_function->getOperand(3)));
    args.push_back(translateScalarRex(rex_function->getOperand(4)));
    args.push_back(translateScalarRex(rex_function->getOperand(5)));
    return makeExpr<Analyzer::FunctionOper>(
        rex_function->getType(), specialized_geofunc, args);
  }
  CHECK(false);
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateGeoOverlapsOper(
    const RexOperator* rex_operator) const {
  CHECK_EQ(rex_operator->size(), 2);

  auto translate_input =
      [&](const RexScalar* operand) -> std::shared_ptr<Analyzer::Expr> {
    const auto input = dynamic_cast<const RexInput*>(operand);
    CHECK(input);

    SQLTypeInfo ti;
    const auto exprs = translateGeoColumn(input, ti, true, false, false);
    CHECK_GT(exprs.size(), 0);
    if (ti.get_type() == kPOINT) {
      return exprs.front();
    } else {
      return exprs.back();
    }
  };

  SQLQualifier sql_qual{kONE};
  SQLOps sql_op{kOVERLAPS};
  return makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                     false,
                                     sql_op,
                                     sql_qual,
                                     translate_input(rex_operator->getOperand(1)),
                                     translate_input(rex_operator->getOperand(0)));
}
