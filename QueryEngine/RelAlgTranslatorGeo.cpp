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
                                                                                  SQLTypes& arg_type) const {
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
  const auto cd = cat_.getMetadataForColumn(table_desc->tableId, rex_input->getIndex() + 1);
  CHECK(cd);
  CHECK(!cd->isPhysicalCol);
  CHECK_GT(cd->numPhysicalColumns, 0);
  auto col_ti = cd->columnType;
  CHECK(IS_GEO(col_ti.get_type()));
  // Translate geo column reference to a list of physical column refs
  for (auto i = 0; i < cd->numPhysicalColumns; i++) {
    const auto cd0 = cat_.getMetadataForColumn(table_desc->tableId, rex_input->getIndex() + 1 + i + 1);
    auto col0_ti = cd0->columnType;
    CHECK(cd0->isPhysicalCol);
    CHECK(!cd0->isVirtualCol);
    args.push_back(std::make_shared<Analyzer::ColumnVar>(col0_ti, table_desc->tableId, cd0->columnId, rte_idx));
  }
  arg_type = col_ti.get_type();
  return args;
}

namespace Importer_NS {

bool importGeoFromWkt(std::string& wkt,
                      SQLTypes& type,
                      std::vector<double>& coords,
                      std::vector<int>& ring_sizes,
                      std::vector<int>& polygon_sizes);

}  // Importer_NS

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoLiteral(const RexLiteral* rex_literal,
                                                                                   SQLTypes& arg_type) const {
  CHECK(rex_literal);
  const auto e = translateLiteral(rex_literal);
  auto wkt = std::dynamic_pointer_cast<Analyzer::Constant>(e);
  CHECK(wkt);
  std::vector<double> coords;
  std::vector<int> ring_sizes;
  std::vector<int> polygon_sizes;
  if (!Importer_NS::importGeoFromWkt(*wkt->get_constval().stringval, arg_type, coords, ring_sizes, polygon_sizes)) {
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
  SQLTypeInfo ti = SQLTypeInfo(kARRAY, true);
  ti.set_subtype(kDOUBLE);
  ti.set_size(coords.size() * sizeof(double));
  args.push_back(makeExpr<Analyzer::Constant>(ti, false, coord_exprs));

  if (arg_type == kPOLYGON) {
    std::list<std::shared_ptr<Analyzer::Expr>> ring_size_exprs;
    for (auto c : ring_sizes) {
      Datum d;
      d.intval = c;
      auto e = makeExpr<Analyzer::Constant>(kINT, false, d);
      ring_size_exprs.push_back(e);
    }
    SQLTypeInfo ti = SQLTypeInfo(kARRAY, true);
    ti.set_subtype(kINT);
    ti.set_size(ring_sizes.size() * sizeof(int32_t));
    args.push_back(makeExpr<Analyzer::Constant>(ti, false, ring_size_exprs));
  }

  return args;
}

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoFunctionArg(const RexScalar* rex_scalar,
                                                                                       SQLTypes& arg_type) const {
  std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;

  const auto rex_input = dynamic_cast<const RexInput*>(rex_scalar);
  if (rex_input) {
    const auto input = translateInput(rex_input);
    const auto column = dynamic_cast<const Analyzer::ColumnVar*>(input.get());
    CHECK(column);
    return translateGeoColumn(rex_input, arg_type);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex_scalar);
  if (rex_function) {
    if (rex_function->getName() != std::string("ST_GeomFromText")) {
      throw QueryNotSupported("Unsupported argument: " + rex_function->getName());
    }
    if (rex_function->size() != 1) {
      throw QueryNotSupported("Expecting one argument: " + rex_function->getName());
    }
    const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_function->getOperand(0));
    if (!rex_literal) {
      throw QueryNotSupported("Expecting a string literal: " + rex_function->getName());
    }
    return translateGeoLiteral(rex_literal, arg_type);
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_scalar);
  if (rex_literal) {
    return translateGeoLiteral(rex_literal, arg_type);
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
  throw QueryNotSupported("Unsupported argument type");
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateGeoFunction(const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());

  if (rex_function->getName() != std::string("ST_Distance") && rex_function->getName() != std::string("ST_Contains")) {
    throw QueryNotSupported("Function " + rex_function->getName() + " not supported");
  }

  std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;
  SQLTypes arg0_type;
  SQLTypes arg1_type;

  auto geoargs0 = translateGeoFunctionArg(rex_function->getOperand(0), arg0_type);
  geoargs.insert(geoargs.end(), geoargs0.begin(), geoargs0.end());
  auto geoargs1 = translateGeoFunctionArg(rex_function->getOperand(1), arg1_type);
  geoargs.insert(geoargs.end(), geoargs1.begin(), geoargs1.end());

  std::string specialized_geofunc{rex_function->getName() + suffix(arg0_type) + suffix(arg1_type)};
  return makeExpr<Analyzer::FunctionOper>(rex_function->getType(), specialized_geofunc, geoargs);
}
