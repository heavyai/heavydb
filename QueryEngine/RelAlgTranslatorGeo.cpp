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
  for (auto i = 0; i < ti.get_physical_cols(); i++) {
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
  if (!Importer_NS::importGeoFromWkt(*wkt->get_constval().stringval, ti, coords, ring_sizes, poly_rings)) {
    throw QueryNotSupported("Could not read geometry from text");
  }

  std::vector<std::shared_ptr<Analyzer::Expr>> args;

  std::list<std::shared_ptr<Analyzer::Expr>> coord_exprs;
  for (auto c : coords) {
    Datum d;
    d.doubleval = c;
    auto e = makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
    coord_exprs.push_back(e);
  }
  SQLTypeInfo arr_ti = SQLTypeInfo(kARRAY, true);
  arr_ti.set_subtype(kDOUBLE);
  arr_ti.set_size(coords.size() * sizeof(double));
  args.push_back(makeExpr<Analyzer::Constant>(arr_ti, false, coord_exprs));

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
                                                                                       SQLTypeInfo& arg_ti) const {
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
      if (rex_function->size() != 2) {
        throw QueryNotSupported(rex_function->getName() + " expects two arguments");
      }
      const auto rex_scalar0 = dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() + ": unexpected first argument");
      }
      auto arg0 = translateGeoFunctionArg(rex_scalar0, arg_ti);

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
      } else if (e->get_type_info().get_type() == kINT) {
        srid = static_cast<int32_t>(ce->get_constval().intval);
      } else {
        throw QueryNotSupported(rex_function->getName() + ": expecting integer SRID");
      }
      if (srid != 900913) {
        throw QueryNotSupported(rex_function->getName() + ": unsupported output SRID " + std::to_string(srid));
      }
      if (arg_ti.get_dimension() > 0) {
        if (arg_ti.get_dimension() != 4326) {
          throw QueryNotSupported(rex_function->getName() + ": unsupported input SRID " +
                                  std::to_string(arg_ti.get_dimension()));
        }
        arg_ti.set_scale(srid);  // We have a valid input SRID, register the output SRID for transform
      } else {
        throw QueryNotSupported(rex_function->getName() + ": unexpected input SRID, unable to transform");
      }
      return arg0;
    } else if (rex_function->getName() == std::string("ST_GeomFromText")) {
      if (rex_function->size() != 1 && rex_function->size() != 2) {
        throw QueryNotSupported(rex_function->getName() + " expects one or two arguments");
      }
      const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_function->getOperand(0));
      if (!rex_literal) {
        throw QueryNotSupported(rex_function->getName() + " expects a string literal as first argument");
      }
      auto arg0 = translateGeoLiteral(rex_literal, arg_ti);
      if (arg_ti.get_dimension() > 0) {
        throw QueryNotSupported(rex_function->getName() + ": parsed geometry literal has unexpected SRID");
      }
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
        int32_t srid = 0;
        if (e->get_type_info().get_type() == kSMALLINT) {
          srid = static_cast<int32_t>(ce->get_constval().smallintval);
        } else if (e->get_type_info().get_type() == kINT) {
          srid = static_cast<int32_t>(ce->get_constval().intval);
        } else {
          throw QueryNotSupported(rex_function->getName() + " expecting integer SRID");
        }
        if (srid != 4326) {
          throw QueryNotSupported(rex_function->getName() + ": unsupported SRID " + std::to_string(srid));
        }
        arg_ti.set_dimension(srid);  // Input SRID
        arg_ti.set_scale(srid);      // Output SRID is the same - no transform
      }
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
  auto geoargs = translateGeoFunctionArg(rex_function->getOperand(0), arg_ti);

  if (rex_function->getName() == std::string("ST_X") || rex_function->getName() == std::string("ST_Y")) {
    if (arg_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() + " expects a POINT");
    }
  }

  // All geo function calls translated here only need the coords.
  // Specialize for other/new functions if needed.
  geoargs.erase(geoargs.begin() + 1, geoargs.end());

  Datum input_srid;
  input_srid.intval = arg_ti.get_dimension();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid));
  Datum output_srid;
  output_srid.intval = arg_ti.get_scale();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

  return makeExpr<Analyzer::FunctionOper>(rex_function->getType(), rex_function->getName(), geoargs);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateBinaryGeoFunction(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());

  std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;
  SQLTypeInfo arg0_ti;
  SQLTypeInfo arg1_ti;

  auto geoargs0 = translateGeoFunctionArg(rex_function->getOperand(0), arg0_ti);
  geoargs.insert(geoargs.end(), geoargs0.begin(), geoargs0.end());
  auto geoargs1 = translateGeoFunctionArg(rex_function->getOperand(1), arg1_ti);
  geoargs.insert(geoargs.end(), geoargs1.begin(), geoargs1.end());

  if (arg0_ti.get_scale() > 0 && arg0_ti.get_scale() != arg1_ti.get_scale()) {
    throw QueryNotSupported(rex_function->getName() + " cannot accept different SRIDs");
  }
  if (arg0_ti.get_dimension() > 0 && arg0_ti.get_dimension() != arg1_ti.get_dimension()) {
    throw QueryNotSupported(rex_function->getName() + " currently doesn't handle transforms from different SRIDs");
  }

  std::string specialized_geofunc{rex_function->getName() + suffix(arg0_ti.get_type()) + suffix(arg1_ti.get_type())};

  if (arg0_ti.get_scale() == 4326) {
    // Need to call geodesic runtime functions
    if (rex_function->getName() == std::string("ST_Distance")) {
      if (arg0_ti.get_type() == kPOINT && arg1_ti.get_type() == kPOINT) {
        specialized_geofunc += std::string("_Geodesic");
      } else {
        throw QueryNotSupported(rex_function->getName() + " currently doesn't accept non-POINT geographies");
      }
    } else {
      throw QueryNotSupported(rex_function->getName() + " currently doesn't accept geographies");
    }
  } else {
    // Need to call geometric/cartesian runtime functions
    // Add input/output SRID args to enable on-the-fly transforms
    Datum input_srid;
    input_srid.intval = arg0_ti.get_dimension();
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid));
    Datum output_srid;
    output_srid.intval = arg0_ti.get_scale();
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));
  }
  return makeExpr<Analyzer::FunctionOper>(rex_function->getType(), specialized_geofunc, geoargs);
}
