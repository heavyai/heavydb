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

#include "QueryEngine/RelAlgTranslator.h"

#include <memory>
#include <vector>

#include "Geospatial/Compression.h"
#include "Geospatial/Types.h"
#include "QueryEngine/ExpressionRewrite.h"

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
    if (with_bounds || with_render_group) {
      throw QueryNotSupported(
          "Geospatial columns not yet supported in intermediate results.");
    }

    CHECK(!in_metainfo.empty());
    CHECK_GE(rte_idx, 0);
    column_id = rex_input->getIndex();
    CHECK_LT(static_cast<size_t>(column_id), in_metainfo.size());
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

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoLiteral(
    const RexLiteral* rex_literal,
    SQLTypeInfo& ti,
    bool with_bounds) const {
  CHECK(rex_literal);
  if (rex_literal->getType() != kTEXT) {
    throw std::runtime_error("Geo literals must be strings");
  }

  // TODO: use geo conversion here
  const auto e = translateLiteral(rex_literal);
  auto wkt = std::dynamic_pointer_cast<Analyzer::Constant>(e);
  CHECK(wkt);
  std::vector<double> coords;
  std::vector<double> bounds;
  std::vector<int> ring_sizes;
  std::vector<int> poly_rings;
  int32_t srid = ti.get_output_srid();
  if (!Geospatial::GeoTypesFactory::getGeoColumns(
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

SQLTypes get_ti_from_geo(const Geospatial::GeoBase* geo) {
  CHECK(geo);
  switch (geo->getType()) {
    case Geospatial::GeoBase::GeoType::kPOINT: {
      return kPOINT;
    }
    case Geospatial::GeoBase::GeoType::kLINESTRING: {
      return kLINESTRING;
    }
    case Geospatial::GeoBase::GeoType::kPOLYGON: {
      return kPOLYGON;
    }
    case Geospatial::GeoBase::GeoType::kMULTIPOLYGON: {
      return kMULTIPOLYGON;
    }
    default:
      UNREACHABLE();
      return kNULLT;
  }
}

}  // namespace

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoFunctionArg(
    const RexScalar* rex_scalar,
    SQLTypeInfo& arg_ti,
    const bool with_bounds,
    const bool with_render_group,
    const bool expand_geo_col,
    const bool is_projection,
    const bool use_geo_expressions,
    const bool try_to_compress) const {
  std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;

  const auto rex_input = dynamic_cast<const RexInput*>(rex_scalar);
  if (rex_input) {
    const auto input = translateInput(rex_input);
    const auto column = dynamic_cast<const Analyzer::ColumnVar*>(input.get());
    if (!column || !column->get_type_info().is_geometry()) {
      throw QueryNotSupported("Geo function is expecting a geo column argument");
    }
    if (use_geo_expressions) {
      return {makeExpr<Analyzer::GeoColumnVar>(column, with_bounds, with_render_group)};
    }
    return translateGeoColumn(
        rex_input, arg_ti, with_bounds, with_render_group, expand_geo_col);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex_scalar);
  if (rex_function) {
    if (rex_function->getName() == "ST_Transform"sv) {
      CHECK_EQ(size_t(2), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() + ": unexpected first argument");
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
      if (srid != 900913 && ((use_geo_expressions || is_projection) && srid != 4326)) {
        throw QueryNotSupported(rex_function->getName() + ": unsupported output SRID " +
                                std::to_string(srid));
      }
      arg_ti.set_output_srid(srid);  // Forward output srid down to argument translation
      auto arg0 = translateGeoFunctionArg(
          rex_scalar0,
          arg_ti,
          with_bounds,
          with_render_group,
          expand_geo_col,
          is_projection,
          is_projection ? /*use_geo_expressions=*/true : use_geo_expressions);

      if (use_geo_expressions || is_projection) {
        CHECK_EQ(arg0.size(), size_t(1));
        auto arg0_ti = arg0.front()->get_type_info();  // make a copy so we can override
        arg0_ti.set_output_srid(srid);
        if (arg0_ti.get_type() == kPOINT) {
          // geo transforms projections leave the result decompressed in a register
          arg0_ti.set_compression(kENCODING_NONE);
          arg0_ti.set_comp_param(0);
          return {makeExpr<Analyzer::GeoTransformOperator>(
              arg0_ti, rex_function->getName(), arg0, arg0_ti.get_input_srid(), srid)};
        } else {
          if (auto geo_constant =
                  std::dynamic_pointer_cast<Analyzer::GeoConstant>(arg0.front())) {
            // fold transform
            return {geo_constant->add_cast(arg0_ti)};
          } else {
            throw std::runtime_error(
                "Transform on non-POINT geospatial types not yet supported in this "
                "context.");
          }
        }
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
    } else if (func_resolve(
                   rex_function->getName(), "ST_GeomFromText"sv, "ST_GeogFromText"sv)) {
      CHECK(rex_function->size() == size_t(1) || rex_function->size() == size_t(2));
      if (use_geo_expressions) {
        int32_t srid = 0;
        if (rex_function->size() == 2) {
          // user supplied srid
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
        arg_ti.set_input_srid(srid);  // Input SRID
        // leave the output srid unset in case a transform was above us

        if (rex_function->getName() == "ST_GeogFromText"sv) {
          arg_ti.set_subtype(kGEOGRAPHY);
        }

        auto func_args = translateGeoFunctionArg(rex_function->getOperand(0),
                                                 arg_ti,
                                                 with_bounds,
                                                 with_render_group,
                                                 expand_geo_col,
                                                 is_projection,
                                                 use_geo_expressions);
        CHECK_GE(func_args.size(), size_t(1));
        return func_args;
      }

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
      arg_ti.set_subtype((rex_function->getName() == "ST_GeogFromText"sv) ? kGEOGRAPHY
                                                                          : kGEOMETRY);
      return arg0;
    } else if (rex_function->getName() == "ST_PointN"sv) {
      // uses geo expressions
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(rex_scalar0,
                                          arg_ti,
                                          with_bounds,
                                          with_render_group,
                                          expand_geo_col,
                                          /*is_projection=*/false,
                                          /*use_geo_expressions=*/true);
      CHECK_EQ(arg0.size(), size_t(1));
      CHECK(arg0.front());
      if (arg0.front()->get_type_info().get_type() != kLINESTRING) {
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
      if (index == 0) {
        // maybe we will just return NULL here?
        throw QueryNotSupported(rex_function->getName() + ": invalid index");
      }
      arg0.push_back(e);
      auto oper_ti =
          arg0.front()->get_type_info();  // make a copy so we can reset nullness and type
      oper_ti.set_type(kPOINT);
      oper_ti.set_notnull(false);

      arg_ti = oper_ti;  // TODO: remove

      return {makeExpr<Analyzer::GeoOperator>(oper_ti, rex_function->getName(), arg0)};

    } else if (rex_function->getName() == "ST_StartPoint"sv ||
               rex_function->getName() == "ST_EndPoint"sv) {
      std::vector<std::shared_ptr<Analyzer::Expr>> args;
      CHECK_EQ(size_t(1), rex_function->size());
      const auto arg_exprs = translateGeoFunctionArg(rex_function->getOperand(0),
                                                     arg_ti,
                                                     with_bounds,
                                                     with_render_group,
                                                     expand_geo_col,
                                                     is_projection,
                                                     /*use_geo_expressions=*/true);
      CHECK_EQ(arg_exprs.size(), size_t(1));
      CHECK(arg_exprs.front());
      const auto arg_expr_ti = arg_exprs.front()->get_type_info();
      if (arg_expr_ti.get_type() != kLINESTRING) {
        throw QueryNotSupported(rex_function->getName() +
                                " expected LINESTRING argument. Received " +
                                arg_expr_ti.toString());
      }
      args.push_back(arg_exprs.front());

      auto oper_ti = args.back()->get_type_info();  // make a copy so we can override type
      oper_ti.set_type(kPOINT);

      arg_ti = oper_ti;  // TODO: remove

      return {makeExpr<Analyzer::GeoOperator>(oper_ti, rex_function->getName(), args)};
    } else if (rex_function->getName() == "ST_SRID"sv) {
      CHECK_EQ(size_t(1), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(
          rex_scalar0, arg_ti, with_bounds, with_render_group, expand_geo_col);
      if (!IS_GEO(arg_ti.get_type())) {
        throw QueryNotSupported(rex_function->getName() + " expects geometry argument");
      }
      return arg0;
    } else if (rex_function->getName() == "ST_SetSRID"sv) {
      CHECK_EQ(size_t(2), rex_function->size());
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

      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }

      // Only convey the request to compress if dealing with 4326 geo
      auto arg0 = translateGeoFunctionArg(rex_scalar0,
                                          arg_ti,
                                          with_bounds,
                                          with_render_group,
                                          expand_geo_col,
                                          is_projection,
                                          use_geo_expressions,
                                          (try_to_compress && (srid == 4326)));

      CHECK(!arg0.empty() && arg0.front());
      if (!IS_GEO(arg_ti.get_type()) && !use_geo_expressions) {
        throw QueryNotSupported(rex_function->getName() + " expects geometry argument");
      }
      arg_ti.set_input_srid(srid);   // Input SRID
      arg_ti.set_output_srid(srid);  // Output SRID is the same - no transform
      if (auto geo_expr = std::dynamic_pointer_cast<Analyzer::GeoExpr>(arg0.front())) {
        CHECK_EQ(arg0.size(), size_t(1));
        auto ti = geo_expr->get_type_info();
        ti.set_input_srid(srid);
        ti.set_output_srid(srid);
        return {geo_expr->add_cast(ti)};
      }
      return arg0;
    } else if (rex_function->getName() == "CastToGeography"sv) {
      CHECK_EQ(size_t(1), rex_function->size());
      const auto rex_scalar0 =
          dynamic_cast<const RexScalar*>(rex_function->getOperand(0));
      if (!rex_scalar0) {
        throw QueryNotSupported(rex_function->getName() +
                                ": expects scalar as first argument");
      }
      auto arg0 = translateGeoFunctionArg(rex_scalar0,
                                          arg_ti,
                                          with_bounds,
                                          with_render_group,
                                          expand_geo_col,
                                          /*is_projection=*/false,
                                          use_geo_expressions);
      CHECK(!arg0.empty());
      if (auto geo_expr = std::dynamic_pointer_cast<Analyzer::GeoExpr>(arg0.front())) {
        auto arg_ti = geo_expr->get_type_info();  // make a copy
        arg_ti.set_subtype(kGEOGRAPHY);
        return {geo_expr->add_cast(arg_ti)};
      }
      if (use_geo_expressions) {
        arg_ti = arg0.front()->get_type_info();
        arg_ti.set_subtype(kGEOGRAPHY);
        arg0.front()->set_type_info(arg_ti);
      }
      if (!IS_GEO(arg_ti.get_type())) {
        throw QueryNotSupported(rex_function->getName() + " expects geometry argument");
      }
      if (arg_ti.get_output_srid() != 4326) {
        throw QueryNotSupported(rex_function->getName() +
                                " expects geometry with SRID=4326");
      }
      arg_ti.set_subtype(kGEOGRAPHY);
      return arg0;
    } else if (rex_function->getName() == "ST_Point"sv) {
      CHECK_EQ(size_t(2), rex_function->size());
      arg_ti.set_type(kPOINT);
      arg_ti.set_subtype(kGEOMETRY);
      arg_ti.set_input_srid(0);
      arg_ti.set_output_srid(0);
      arg_ti.set_compression(kENCODING_NONE);

      auto coord1 = translateScalarRex(rex_function->getOperand(0));
      auto coord2 = translateScalarRex(rex_function->getOperand(1));
      auto d_ti = SQLTypeInfo(kDOUBLE, false);
      auto cast_coord1 = coord1->add_cast(d_ti);
      auto cast_coord2 = coord2->add_cast(d_ti);
      // First try to fold to geo literal
      auto fold1 = fold_expr(cast_coord1.get());
      auto fold2 = fold_expr(cast_coord2.get());
      auto const_coord1 = std::dynamic_pointer_cast<Analyzer::Constant>(fold1);
      auto const_coord2 = std::dynamic_pointer_cast<Analyzer::Constant>(fold2);
      if (const_coord1 && const_coord2 && !use_geo_expressions) {
        CHECK(const_coord1->get_type_info().get_type() == kDOUBLE);
        CHECK(const_coord2->get_type_info().get_type() == kDOUBLE);
        std::string wkt = "POINT(" +
                          std::to_string(const_coord1->get_constval().doubleval) + " " +
                          std::to_string(const_coord2->get_constval().doubleval) + ")";
        RexLiteral rex_literal{wkt, kTEXT, kNULLT, 0, 0, 0, 0};
        auto args = translateGeoLiteral(&rex_literal, arg_ti, false);
        CHECK(arg_ti.get_type() == kPOINT);
        return args;
      }
      const auto is_local_alloca = !is_projection;
      if (!is_local_alloca || use_geo_expressions) {
        if (try_to_compress) {
          arg_ti.set_input_srid(4326);
          arg_ti.set_output_srid(4326);
        }
        return {makeExpr<Analyzer::GeoOperator>(
            arg_ti,
            rex_function->getName(),
            std::vector<std::shared_ptr<Analyzer::Expr>>{cast_coord1, cast_coord2})};
      }
      // Couldn't fold to geo literal, construct [and compress] on the fly
      auto da_ti = SQLTypeInfo(kARRAY, true);
      da_ti.set_subtype(kDOUBLE);
      da_ti.set_size(16);
      if (try_to_compress) {
        // Switch to compressed coord array
        da_ti.set_subtype(kINT);
        da_ti.set_size(8);
        da_ti.set_input_srid(4326);
        da_ti.set_output_srid(4326);
        da_ti.set_compression(kENCODING_GEOINT);
        da_ti.set_comp_param(32);
        // Register point compression
        arg_ti.set_input_srid(4326);
        arg_ti.set_output_srid(4326);
        arg_ti.set_compression(kENCODING_GEOINT);
        arg_ti.set_comp_param(32);
      }
      auto cast_coords = {cast_coord1, cast_coord2};
      auto ae = makeExpr<Analyzer::ArrayExpr>(da_ti, cast_coords, false, is_local_alloca);
      SQLTypeInfo tia_ti = da_ti;
      tia_ti.set_subtype(kTINYINT);
      return {makeExpr<Analyzer::UOper>(tia_ti, false, kCAST, ae)};
    } else if (rex_function->getName() == "ST_Centroid"sv) {
      CHECK_EQ(size_t(1), rex_function->size());
      arg_ti.set_type(kPOINT);
      arg_ti.set_subtype(kGEOMETRY);
      arg_ti.set_input_srid(0);
      arg_ti.set_output_srid(0);
      arg_ti.set_compression(kENCODING_NONE);

      SQLTypeInfo geo_ti;
      auto geoargs = translateGeoFunctionArg(rex_function->getOperand(0),
                                             geo_ti,
                                             /*with_bounds=*/false,
                                             /*with_render_group=*/false,
                                             /*expand_geo_col=*/true,
                                             /*is_projection=*/false,
                                             /*use_geo_expressions=*/true);
      CHECK_EQ(geoargs.size(), size_t(1));
      if (try_to_compress) {
        // Request point compression
        arg_ti.set_input_srid(4326);
        arg_ti.set_output_srid(4326);
        arg_ti.set_compression(kENCODING_GEOINT);
        arg_ti.set_comp_param(32);
      }
      return {makeExpr<Analyzer::GeoOperator>(
          arg_ti,
          rex_function->getName(),
          std::vector<std::shared_ptr<Analyzer::Expr>>{geoargs.front()})};
    } else if (func_resolve(rex_function->getName(),
                            "ST_Intersection"sv,
                            "ST_Difference"sv,
                            "ST_Union"sv,
                            "ST_Buffer"sv)) {
      CHECK_EQ(size_t(2), rex_function->size());
      // What geo type will the constructor return? Could be anything.
      return {translateGeoBinaryConstructor(rex_function, arg_ti, with_bounds)};
    } else if (func_resolve(rex_function->getName(), "ST_IsEmpty"sv, "ST_IsValid"sv)) {
      CHECK_EQ(size_t(1), rex_function->size());
      return {translateGeoPredicate(rex_function, arg_ti, with_bounds)};
    } else {
      throw QueryNotSupported("Unsupported argument: " + rex_function->getName());
    }
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_scalar);
  if (rex_literal) {
    if (use_geo_expressions) {
      const auto translated_literal = translateLiteral(rex_literal);
      const auto constant_expr =
          dynamic_cast<const Analyzer::Constant*>(translated_literal.get());
      CHECK(constant_expr);
      if (constant_expr->get_is_null()) {
        // TODO: we could lift this limitation by assuming a minimum type per function
        throw QueryNotSupported("Geospatial functions require typed nulls.");
      }
      const auto& datum = constant_expr->get_constval();
      CHECK(datum.stringval);
      auto geospatial_base = Geospatial::GeoTypesFactory::createGeoType(*datum.stringval);
      CHECK(geospatial_base);
      SQLTypeInfo ti;
      ti.set_type(get_ti_from_geo(geospatial_base.get()));
      if (arg_ti.get_subtype() == kGEOGRAPHY) {
        ti.set_subtype(kGEOGRAPHY);
      } else {
        ti.set_subtype(kGEOMETRY);
      }
      ti.set_input_srid(arg_ti.get_input_srid());
      ti.set_output_srid(arg_ti.get_output_srid() == 0 ? arg_ti.get_input_srid()
                                                       : arg_ti.get_output_srid());
      // TODO: remove dependence on arg_ti
      if (ti.get_output_srid() == 4326 || arg_ti.get_compression() == kENCODING_GEOINT) {
        ti.set_compression(kENCODING_GEOINT);
        ti.set_comp_param(32);
      }
      ti.set_notnull(true);
      return {makeExpr<Analyzer::GeoConstant>(std::move(geospatial_base), ti)};
    }
    return translateGeoLiteral(rex_literal, arg_ti, with_bounds);
  }
  throw QueryNotSupported("Geo function argument not supported");
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateGeoProjection(
    const RexFunctionOperator* rex_function,
    SQLTypeInfo& ti,
    const bool with_bounds) const {
  auto geoargs = translateGeoFunctionArg(rex_function, ti, false, false, true, true);
  CHECK(!geoargs.empty());
  if (std::dynamic_pointer_cast<const Analyzer::GeoExpr>(geoargs.front()) &&
      !geoargs.front()->get_type_info().is_array()) {
    // GeoExpression
    return geoargs.front();
  }
  return makeExpr<Analyzer::GeoUOper>(
      Geospatial::GeoBase::GeoOp::kPROJECTION, ti, ti, geoargs);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateGeoBinaryConstructor(
    const RexFunctionOperator* rex_function,
    SQLTypeInfo& ti,
    const bool with_bounds) const {
#ifndef ENABLE_GEOS
  throw QueryNotSupported(rex_function->getName() +
                          " geo constructor requires enabled GEOS support");
#endif
  Geospatial::GeoBase::GeoOp op = Geospatial::GeoBase::GeoOp::kINTERSECTION;
  if (rex_function->getName() == "ST_Difference"sv) {
    op = Geospatial::GeoBase::GeoOp::kDIFFERENCE;
  } else if (rex_function->getName() == "ST_Union"sv) {
    op = Geospatial::GeoBase::GeoOp::kUNION;
  } else if (rex_function->getName() == "ST_Buffer"sv) {
    op = Geospatial::GeoBase::GeoOp::kBUFFER;
  }

  Analyzer::ExpressionPtrVector geoargs0{};
  Analyzer::ExpressionPtrVector geoargs1{};
  SQLTypeInfo arg0_ti;
  SQLTypeInfo arg1_ti;
  if (func_resolve(rex_function->getName(),
                   "ST_Intersection"sv,
                   "ST_Difference"sv,
                   "ST_Union"sv,
                   "ST_Buffer"sv)) {
    // First arg: geometry
    geoargs0 = translateGeoFunctionArg(
        rex_function->getOperand(0), arg0_ti, false, false, true, true);
  }
  if (func_resolve(rex_function->getName(),
                   "ST_Intersection"sv,
                   "ST_Difference"sv,
                   "ST_Union"sv)) {
    // Second arg: geometry
    geoargs1 = translateGeoFunctionArg(
        rex_function->getOperand(1), arg1_ti, false, false, true, true);
  } else if (func_resolve(rex_function->getName(), "ST_Buffer"sv)) {
    // Second arg: double scalar
    auto param_expr = translateScalarRex(rex_function->getOperand(1));
    arg1_ti = SQLTypeInfo(kDOUBLE, false);
    if (param_expr->get_type_info().get_type() != kDOUBLE) {
      param_expr = param_expr->add_cast(arg1_ti);
    }
    geoargs1 = {param_expr};
  }

  auto srid = ti.get_output_srid();
  ti = arg0_ti;
  ti.set_type(kMULTIPOLYGON);
  ti.set_compression(kENCODING_NONE);  // Constructed geometries are not compressed
  ti.set_comp_param(0);
  if (srid > 0) {
    ti.set_output_srid(srid);  // Requested SRID sent from above
  }
  return makeExpr<Analyzer::GeoBinOper>(op, ti, arg0_ti, arg1_ti, geoargs0, geoargs1);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateGeoPredicate(
    const RexFunctionOperator* rex_function,
    SQLTypeInfo& ti,
    const bool with_bounds) const {
#ifndef ENABLE_GEOS
  throw QueryNotSupported(rex_function->getName() +
                          " geo predicate requires enabled GEOS support");
#endif
  SQLTypeInfo arg_ti;
  auto geoargs = translateGeoFunctionArg(
      rex_function->getOperand(0), arg_ti, false, false, true, true);
  ti = SQLTypeInfo(kBOOLEAN, false);
  auto op = (rex_function->getName() == "ST_IsEmpty"sv)
                ? Geospatial::GeoBase::GeoOp::kISEMPTY
                : Geospatial::GeoBase::GeoOp::kISVALID;
  return makeExpr<Analyzer::GeoUOper>(op, ti, arg_ti, geoargs);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateUnaryGeoFunction(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());

  std::string specialized_geofunc{rex_function->getName()};

  // Geo function calls which do not need the coords col but do need cols associated
  // with physical coords (e.g. ring_sizes / poly_rings)
  if (rex_function->getName() == "ST_NRings"sv) {
    SQLTypeInfo arg_ti;
    auto geoargs = translateGeoFunctionArg(rex_function->getOperand(0),
                                           arg_ti,
                                           /*with_bounds=*/false,
                                           /*with_render_group=*/false,
                                           /*expand_geo_col=*/true,
                                           /*is_projection=*/false,
                                           /*use_geo_expressions=*/true);
    CHECK_EQ(geoargs.size(), size_t(1));
    arg_ti = rex_function->getType();  // TODO: remove
    return makeExpr<Analyzer::GeoOperator>(
        rex_function->getType(),
        rex_function->getName(),
        std::vector<std::shared_ptr<Analyzer::Expr>>{geoargs.front()});
  } else if (rex_function->getName() == "ST_NPoints"sv) {
    SQLTypeInfo arg_ti;
    auto geoargs = translateGeoFunctionArg(rex_function->getOperand(0),
                                           arg_ti,
                                           /*with_bounds=*/false,
                                           /*with_render_group=*/false,
                                           /*expand_geo_col=*/true,
                                           /*is_projection=*/false,
                                           /*use_geo_expressions=*/true);
    CHECK_EQ(geoargs.size(), size_t(1));
    return makeExpr<Analyzer::GeoOperator>(
        rex_function->getType(),
        rex_function->getName(),
        std::vector<std::shared_ptr<Analyzer::Expr>>{geoargs.front()});
  } else if (func_resolve(rex_function->getName(), "ST_Perimeter"sv, "ST_Area"sv)) {
    SQLTypeInfo arg_ti;
    auto geoargs = translateGeoFunctionArg(rex_function->getOperand(0),
                                           arg_ti,
                                           /*with_bounds=*/false,
                                           /*with_render_group=*/false,
                                           /*expand_geo_col=*/true,
                                           /*is_projection=*/false,
                                           /*use_geo_expressions=*/true);
    CHECK_EQ(geoargs.size(), size_t(1));
    arg_ti = geoargs.front()->get_type_info();
    if (arg_ti.get_type() != kPOLYGON && arg_ti.get_type() != kMULTIPOLYGON) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a POLYGON or MULTIPOLYGON");
    }

    return makeExpr<Analyzer::GeoOperator>(
        rex_function->getType(),
        rex_function->getName(),
        std::vector<std::shared_ptr<Analyzer::Expr>>{geoargs.front()});
  }

  // Accessors for poly bounds and render group for in-situ poly render queries
  if (func_resolve(rex_function->getName(),
                   "MapD_GeoPolyBoundsPtr"sv /* deprecated */,
                   "OmniSci_Geo_PolyBoundsPtr"sv)) {
    SQLTypeInfo arg_ti;
    // get geo column plus bounds only (not expanded)
    auto geoargs =
        translateGeoFunctionArg(rex_function->getOperand(0), arg_ti, true, false, false);
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
  } else if (func_resolve(rex_function->getName(),
                          "MapD_GeoPolyRenderGroup"sv /* deprecated */,
                          "OmniSci_Geo_PolyRenderGroup"sv)) {
    SQLTypeInfo arg_ti;
    // get geo column plus render_group only (not expanded)
    auto geoargs =
        translateGeoFunctionArg(rex_function->getOperand(0), arg_ti, false, true, false);
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

  // start to move geo expressions above the generic translation call, as geo expression
  // error handling can differ
  if (func_resolve(rex_function->getName(), "ST_X"sv, "ST_Y"sv)) {
    SQLTypeInfo arg_ti;
    auto new_geoargs = translateGeoFunctionArg(rex_function->getOperand(0),
                                               arg_ti,
                                               /*with_bounds=*/false,
                                               /*with_render_group=*/false,
                                               /*expand_geo_col=*/true,
                                               /*is_projection=*/true,
                                               /*use_geo_expressions=*/true);
    CHECK_EQ(new_geoargs.size(), size_t(1));
    CHECK(new_geoargs.front());
    const auto& arg_expr_ti = new_geoargs.front()->get_type_info();
    if (arg_expr_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() + " expects a POINT");
    }
    auto function_ti = rex_function->getType();
    if (std::dynamic_pointer_cast<Analyzer::GeoOperator>(new_geoargs.front())) {
      function_ti.set_notnull(false);
    }
    if (std::dynamic_pointer_cast<Analyzer::GeoConstant>(new_geoargs.front())) {
      // TODO(adb): fixup null handling
      function_ti.set_notnull(true);
    }
    return makeExpr<Analyzer::GeoOperator>(
        function_ti,
        rex_function->getName(),
        std::vector<std::shared_ptr<Analyzer::Expr>>{new_geoargs.front()});
  }

  // All functions below use geo col as reference and expand it as necessary
  SQLTypeInfo arg_ti;
  bool with_bounds = true;
  auto geoargs = translateGeoFunctionArg(
      rex_function->getOperand(0), arg_ti, with_bounds, false, false);

  if (rex_function->getName() == "ST_SRID"sv) {
    Datum output_srid;
    output_srid.intval = arg_ti.get_output_srid();
    return makeExpr<Analyzer::Constant>(kINT, false, output_srid);
  }

  if (func_resolve(
          rex_function->getName(), "ST_XMin"sv, "ST_YMin"sv, "ST_XMax"sv, "ST_YMax"sv)) {
    // If type has bounds - use them, otherwise look at coords
    if (arg_ti.has_bounds()) {
      // Only need the bounds argument, discard the rest
      geoargs.erase(geoargs.begin(), geoargs.end() - 1);

      // Supply srids too - transformed geo would have a transformed bounding box
      Datum input_srid;
      input_srid.intval = arg_ti.get_input_srid();
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid));
      Datum output_srid;
      output_srid.intval = arg_ti.get_output_srid();
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

      specialized_geofunc += "_Bounds"s;
      return makeExpr<Analyzer::FunctionOper>(
          rex_function->getType(), specialized_geofunc, geoargs);
    }
  }

  // All geo function calls translated below only need the coords, extras e.g.
  // ring_sizes are dropped. Specialize for other/new functions if needed.
  geoargs.erase(geoargs.begin() + 1, geoargs.end());

  if (rex_function->getName() == "ST_Length"sv) {
    if (arg_ti.get_type() != kLINESTRING) {
      throw QueryNotSupported(rex_function->getName() + " expects LINESTRING");
    }
    specialized_geofunc += suffix(arg_ti.get_type());
    if (arg_ti.get_subtype() == kGEOGRAPHY && arg_ti.get_output_srid() == 4326) {
      specialized_geofunc += "_Geodesic"s;
    }
  }

  // Add input compression mode and SRID args to enable on-the-fly
  // decompression/transforms
  Datum input_compression;
  input_compression.intval = Geospatial::get_compression_scheme(arg_ti);
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

  if (function_name == "ST_Overlaps"sv) {
    // Overlaps join is the only implementation supported for now, only translate bounds
    CHECK_EQ(size_t(2), rex_function->size());
    auto extract_geo_bounds_from_input =
        [this, &rex_function](const size_t index) -> std::shared_ptr<Analyzer::Expr> {
      const auto rex_input =
          dynamic_cast<const RexInput*>(rex_function->getOperand(index));
      if (rex_input) {
        SQLTypeInfo ti;
        const auto exprs = translateGeoColumn(rex_input, ti, true, false, false);
        CHECK_GT(exprs.size(), size_t(0));
        if (ti.get_type() == kPOINT) {
          throw std::runtime_error("ST_Overlaps is not supported for point arguments.");
        } else {
          return exprs.back();
        }
      } else {
        throw std::runtime_error(
            "Only inputs are supported as arguments to ST_Overlaps for now.");
      }
    };
    std::vector<std::shared_ptr<Analyzer::Expr>> geo_args;
    geo_args.push_back(extract_geo_bounds_from_input(0));
    geo_args.push_back(extract_geo_bounds_from_input(1));

    return makeExpr<Analyzer::FunctionOper>(return_type, function_name, geo_args);
  }

  if (function_name == "ST_Distance"sv || function_name == "ST_MaxDistance"sv) {
    CHECK_EQ(size_t(2), rex_function->size());
    std::vector<std::shared_ptr<Analyzer::Expr>> args;
    for (size_t i = 0; i < rex_function->size(); i++) {
      SQLTypeInfo arg0_ti;  // discard
      auto geoargs = translateGeoFunctionArg(rex_function->getOperand(i),
                                             arg0_ti,
                                             /*with_bounds=*/false,  // TODO
                                             /*with_render_group=*/false,
                                             /*expand_geo_col=*/false,
                                             /*is_projection = */ false,
                                             /*use_geo_expressions=*/true);
      args.insert(args.end(), geoargs.begin(), geoargs.end());
    }
    return makeExpr<Analyzer::GeoOperator>(
        SQLTypeInfo(kDOUBLE, /*not_null=*/false), function_name, args);
  }

  bool swap_args = false;
  bool with_bounds = false;
  bool negate_result = false;
  Analyzer::ExpressionPtr threshold_expr = nullptr;
  if (function_name == "ST_DWithin"sv) {
    CHECK_EQ(size_t(3), rex_function->size());
    function_name = "ST_Distance";
    return_type = SQLTypeInfo(kDOUBLE, false);
    // Inject ST_DWithin's short-circuiting threshold into ST_MaxDistance
    threshold_expr = translateScalarRex(rex_function->getOperand(2));
  } else if (function_name == "ST_DFullyWithin"sv) {
    CHECK_EQ(size_t(3), rex_function->size());
    function_name = "ST_MaxDistance";
    return_type = SQLTypeInfo(kDOUBLE, false);
    // TODO: inject ST_DFullyWithin's short-circuiting threshold into ST_MaxDistance
    threshold_expr = nullptr;
  } else if (function_name == "ST_Distance"sv) {
    // TODO: pick up an outside short-circuiting threshold and inject into ST_Distance
    threshold_expr = nullptr;
  } else if (function_name == "ST_MaxDistance"sv) {
    // TODO: pick up an outside short-circuiting threshold and inject into
    // ST_MaxDistance
    threshold_expr = nullptr;
  } else {
    CHECK_EQ(size_t(2), rex_function->size());
  }
  if (function_name == "ST_Within"sv) {
    function_name = "ST_Contains";
    swap_args = true;
  } else if (function_name == "ST_Disjoint"sv) {
    function_name = "ST_Intersects";
    negate_result = true;
  }
  if (func_resolve(
          function_name, "ST_Contains"sv, "ST_Intersects"sv, "ST_Approx_Overlaps"sv)) {
    with_bounds = true;
  }

  std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;
  SQLTypeInfo arg0_ti;
  SQLTypeInfo arg1_ti;

  auto geoargs0 = translateGeoFunctionArg(
      rex_function->getOperand(swap_args ? 1 : 0), arg0_ti, with_bounds, false, false);
  geoargs.insert(geoargs.end(), geoargs0.begin(), geoargs0.end());

  // If first arg of ST_Contains is compressed, try to compress the second one
  // to be able to switch to ST_cContains
  bool try_to_compress_arg1 = (function_name == "ST_Contains"sv &&
                               arg0_ti.get_compression() == kENCODING_GEOINT &&
                               arg0_ti.get_output_srid() == 4326);

  auto geoargs1 = translateGeoFunctionArg(rex_function->getOperand(swap_args ? 0 : 1),
                                          arg1_ti,
                                          with_bounds,
                                          false,
                                          false,
                                          false,
                                          false,
                                          try_to_compress_arg1);
  geoargs.insert(geoargs.end(), geoargs1.begin(), geoargs1.end());

  if (arg0_ti.get_subtype() != kNULLT && arg0_ti.get_subtype() != arg1_ti.get_subtype()) {
    throw QueryNotSupported(rex_function->getName() +
                            " accepts either two GEOGRAPHY or two GEOMETRY arguments");
  }
  // Check SRID match if at least one is set/valid
  if ((arg0_ti.get_output_srid() > 0 || arg1_ti.get_output_srid() > 0) &&
      arg0_ti.get_output_srid() != arg1_ti.get_output_srid()) {
    throw QueryNotSupported(rex_function->getName() + " cannot accept different SRIDs");
  }

  if (function_name == "ST_Contains"sv) {
    const bool lhs_is_literal =
        geoargs1.size() == 1 &&
        std::dynamic_pointer_cast<const Analyzer::Constant>(geoargs1.front()) != nullptr;
    const bool lhs_is_point = arg1_ti.get_type() == kPOINT;
    if (!lhs_is_literal && lhs_is_point &&
        arg0_ti.get_compression() == kENCODING_GEOINT &&
        arg0_ti.get_input_srid() == arg0_ti.get_output_srid() &&
        arg0_ti.get_compression() == arg1_ti.get_compression() &&
        arg1_ti.get_input_srid() == arg1_ti.get_output_srid()) {
      // use the compressed version of ST_Contains
      function_name = "ST_cContains";
    }
  }

  std::string specialized_geofunc{function_name + suffix(arg0_ti.get_type()) +
                                  suffix(arg1_ti.get_type())};

  if (arg0_ti.get_subtype() == kGEOGRAPHY && arg0_ti.get_output_srid() == 4326) {
    // Need to call geodesic runtime functions
    if (function_name == "ST_Distance"sv) {
      if ((arg0_ti.get_type() == kPOINT && arg1_ti.get_type() == kPOINT) ||
          (arg0_ti.get_type() == kLINESTRING && arg1_ti.get_type() == kPOINT) ||
          (arg0_ti.get_type() == kPOINT && arg1_ti.get_type() == kLINESTRING)) {
        // Geodesic distance between points
        specialized_geofunc += "_Geodesic"s;
      } else {
        throw QueryNotSupported(function_name +
                                " currently doesn't accept non-POINT geographies");
      }
    } else if (rex_function->getName() == "ST_Contains"sv) {
      // We currently don't have a geodesic implementation of ST_Contains,
      // allowing calls to a [less precise] cartesian implementation.
    } else {
      throw QueryNotSupported(function_name + " doesn't accept geographies");
    }
  } else if (function_name == "ST_Distance"sv && rex_function->size() == 3) {
    if (arg0_ti.get_type() == kPOINT && arg1_ti.get_type() == kPOINT) {
      // Cartesian distance between points used by ST_DWithin - switch to faster Squared
      specialized_geofunc += "_Squared"s;
    }
  }

  // Add first input's compression mode and SRID args to enable on-the-fly
  // decompression/transforms
  Datum input_compression0;
  input_compression0.intval = Geospatial::get_compression_scheme(arg0_ti);
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression0));
  Datum input_srid0;
  input_srid0.intval = arg0_ti.get_input_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid0));

  // Add second input's compression mode and SRID args to enable on-the-fly
  // decompression/transforms
  Datum input_compression1;
  input_compression1.intval = Geospatial::get_compression_scheme(arg1_ti);
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression1));
  Datum input_srid1;
  input_srid1.intval = arg1_ti.get_input_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid1));

  // Add output SRID arg to enable on-the-fly transforms
  Datum output_srid;
  output_srid.intval = arg0_ti.get_output_srid();
  geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

  // Some geo distance functions will be injected with a short-circuit threshold.
  // Threshold value would come from Geo comparison operations or from other outer
  // geo operations, e.g. ST_DWithin
  // At this point, only ST_Distance_LineString_LineString requires a threshold arg.
  // TODO: Other combinations that involve LINESTRING, POLYGON and MULTIPOLYGON args
  // TODO: Inject threshold into ST_MaxDistance
  if (function_name == "ST_Distance"sv && arg0_ti.get_subtype() != kGEOGRAPHY &&
      (arg0_ti.get_type() != kPOINT || arg1_ti.get_type() != kPOINT)) {
    if (threshold_expr) {
      if (threshold_expr->get_type_info().get_type() != kDOUBLE) {
        const auto& threshold_ti = SQLTypeInfo(kDOUBLE, false);
        threshold_expr = threshold_expr->add_cast(threshold_ti);
      }
      threshold_expr = fold_expr(threshold_expr.get());
    } else {
      Datum d;
      d.doubleval = 0.0;
      threshold_expr = makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
    }
    geoargs.push_back(threshold_expr);
  }

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
    distance_expr = distance_expr->add_cast(distance_ti);
  }

  auto function_name = rex_function->getName();
  if (function_name == "ST_DWithin"sv) {
    auto return_type = rex_function->getType();
    bool swap_args = false;
    bool with_bounds = true;
    SQLTypeInfo arg0_ti;
    SQLTypeInfo arg1_ti;

    auto geoargs0 = translateGeoFunctionArg(
        rex_function->getOperand(0), arg0_ti, with_bounds, false, false);
    auto geoargs1 = translateGeoFunctionArg(
        rex_function->getOperand(1), arg1_ti, with_bounds, false, false);
    if (arg0_ti.get_subtype() != arg1_ti.get_subtype()) {
      throw QueryNotSupported(rex_function->getName() +
                              " cannot accept mixed GEOMETRY/GEOGRAPHY arguments");
    }
    auto is_geodesic = false;
    if (arg0_ti.get_subtype() == kGEOGRAPHY) {
      if (arg0_ti.get_type() == kPOINT && arg1_ti.get_type() == kPOINT) {
        is_geodesic = true;
      } else {
        throw QueryNotSupported(
            rex_function->getName() +
            " in geodesic form can only accept POINT GEOGRAPHY arguments");
      }
    }
    // Check SRID match if at least one is set/valid
    if ((arg0_ti.get_output_srid() > 0 || arg1_ti.get_output_srid() > 0) &&
        arg0_ti.get_output_srid() != arg1_ti.get_output_srid()) {
      throw QueryNotSupported(rex_function->getName() + " cannot accept different SRIDs");
    }

    if ((arg1_ti.get_type() == kPOINT && arg0_ti.get_type() != kPOINT) ||
        (arg1_ti.get_type() == kLINESTRING && arg0_ti.get_type() == kPOLYGON) ||
        (arg1_ti.get_type() == kPOLYGON && arg0_ti.get_type() == kMULTIPOLYGON)) {
      // Swap arguments and use single implementation per arg pair
      swap_args = true;
    }

    // First input's compression mode and SRID args to enable on-the-fly
    // decompression/transforms
    Datum input_compression0;
    input_compression0.intval = Geospatial::get_compression_scheme(arg0_ti);
    Datum input_srid0;
    input_srid0.intval = arg0_ti.get_input_srid();

    // Second input's compression mode and SRID args to enable on-the-fly
    // decompression/transforms
    Datum input_compression1;
    input_compression1.intval = Geospatial::get_compression_scheme(arg1_ti);
    Datum input_srid1;
    input_srid1.intval = arg1_ti.get_input_srid();

    // Output SRID arg to enable on-the-fly transforms
    Datum output_srid;
    output_srid.intval = arg0_ti.get_output_srid();

    std::string specialized_geofunc{function_name};
    std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;
    if (swap_args) {
      specialized_geofunc += suffix(arg1_ti.get_type()) + suffix(arg0_ti.get_type());
      geoargs.insert(geoargs.end(), geoargs1.begin(), geoargs1.end());
      geoargs.insert(geoargs.end(), geoargs0.begin(), geoargs0.end());
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression1));
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid1));
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression0));
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid0));
    } else {
      specialized_geofunc += suffix(arg0_ti.get_type()) + suffix(arg1_ti.get_type());
      if (is_geodesic) {
        specialized_geofunc += "_Geodesic"s;
      }
      geoargs.insert(geoargs.end(), geoargs0.begin(), geoargs0.end());
      geoargs.insert(geoargs.end(), geoargs1.begin(), geoargs1.end());
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression0));
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid0));
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression1));
      geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid1));
    }
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));
    // Also add the within distance
    geoargs.push_back(distance_expr);

    auto result =
        makeExpr<Analyzer::FunctionOper>(return_type, specialized_geofunc, geoargs);
    return result;
  }

  // Otherwise translate function as binary geo to get distance,
  // with optional short-circuiting threshold held in the third operand
  const auto geo_distance = translateBinaryGeoFunction(rex_function);
  // and generate the comparison
  return makeExpr<Analyzer::BinOper>(kBOOLEAN, kLE, kONE, geo_distance, distance_expr);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateGeoComparison(
    const RexOperator* rex_operator) const {
  if (rex_operator->size() != size_t(2)) {
    return nullptr;
  }

  const auto rex_operand = rex_operator->getOperand(0);
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex_operand);
  if (!rex_function) {
    return nullptr;
  }
  if (rex_function->getName() == "ST_Distance"sv && rex_operator->getOperator() == kLE) {
    // TODO: fixup
    return nullptr;
    /*
    auto ti = rex_operator->getType();
    std::vector<std::unique_ptr<const RexScalar>> st_dwithin_operands;
    st_dwithin_operands.emplace_back(rex_function->getOperandAndRelease(0));
    st_dwithin_operands.emplace_back(rex_function->getOperandAndRelease(1));
    st_dwithin_operands.emplace_back(rex_operator->getOperandAndRelease(1));
    std::unique_ptr<RexFunctionOperator> st_dwithin(
        new RexFunctionOperator("ST_DWithin", st_dwithin_operands, ti));
    return translateTernaryGeoFunction(st_dwithin.get());
    */
  }

  return nullptr;
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateFunctionWithGeoArg(
    const RexFunctionOperator* rex_function) const {
  std::string specialized_geofunc{rex_function->getName()};
  if (func_resolve(rex_function->getName(),
                   "convert_meters_to_pixel_width"sv,
                   "convert_meters_to_pixel_height"sv)) {
    CHECK_EQ(rex_function->size(), 6u);
    SQLTypeInfo arg_ti;
    std::vector<std::shared_ptr<Analyzer::Expr>> args;
    args.push_back(translateScalarRex(rex_function->getOperand(0)));
    auto geoargs =
        translateGeoFunctionArg(rex_function->getOperand(1), arg_ti, false, true, false);
    // only works on points
    if (arg_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a point for the second argument");
    }

    args.insert(args.end(), geoargs.begin(), geoargs.begin() + 1);

    // Add compression information
    Datum input_compression;
    input_compression.intval = Geospatial::get_compression_scheme(arg_ti);
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
  } else if (rex_function->getName() == "is_point_in_view"sv) {
    CHECK_EQ(rex_function->size(), 5u);
    SQLTypeInfo arg_ti;
    std::vector<std::shared_ptr<Analyzer::Expr>> args;
    auto geoargs =
        translateGeoFunctionArg(rex_function->getOperand(0), arg_ti, false, true, false);
    // only works on points
    if (arg_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a point for the second argument");
    }

    args.insert(args.end(), geoargs.begin(), geoargs.begin() + 1);

    // Add compression information
    Datum input_compression;
    input_compression.intval = Geospatial::get_compression_scheme(arg_ti);
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
  } else if (rex_function->getName() == "is_point_size_in_view"sv) {
    CHECK_EQ(rex_function->size(), 6u);
    SQLTypeInfo arg_ti;
    std::vector<std::shared_ptr<Analyzer::Expr>> args;
    auto geoargs =
        translateGeoFunctionArg(rex_function->getOperand(0), arg_ti, false, true, false);
    // only works on points
    if (arg_ti.get_type() != kPOINT) {
      throw QueryNotSupported(rex_function->getName() +
                              " expects a point for the second argument");
    }

    args.insert(args.end(), geoargs.begin(), geoargs.begin() + 1);

    // Add compression information
    Datum input_compression;
    input_compression.intval = Geospatial::get_compression_scheme(arg_ti);
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
  CHECK_EQ(rex_operator->size(), 2u);

  auto translate_input =
      [&](const RexScalar* operand) -> std::shared_ptr<Analyzer::Expr> {
    const auto input = dynamic_cast<const RexInput*>(operand);
    CHECK(input);

    SQLTypeInfo ti;
    const auto exprs = translateGeoColumn(input, ti, true, false, false);
    CHECK_GT(exprs.size(), 0u);
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
