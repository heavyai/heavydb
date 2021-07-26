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

#include "QueryEngine/GeoOperators/Codegen.h"
#include "QueryEngine/GeoOperators/API.h"

namespace spatial_type {

std::unique_ptr<Codegen> Codegen::init(const Analyzer::GeoOperator* geo_operator,
                                       const Catalog_Namespace::Catalog* catalog) {
  const auto operator_name = geo_operator->getName();
  if (operator_name == "ST_NRings") {
    return std::make_unique<NRings>(geo_operator, catalog);
  } else if (operator_name == "ST_NPoints") {
    return std::make_unique<NPoints>(geo_operator, catalog);
  } else if (operator_name == "ST_PointN") {
    return std::make_unique<PointN>(geo_operator, catalog);
  } else if (operator_name == "ST_StartPoint" || operator_name == "ST_EndPoint") {
    return std::make_unique<StartEndPoint>(geo_operator, catalog);
  } else if (operator_name == "ST_X" || operator_name == "ST_Y") {
    return std::make_unique<PointAccessors>(geo_operator, catalog);
  } else if (operator_name == "ST_Point") {
    return std::make_unique<PointConstructor>(geo_operator, catalog);
  } else if (operator_name == "ST_Transform") {
    return std::make_unique<Transform>(geo_operator, catalog);
  } else if (operator_name == "ST_Perimeter" || operator_name == "ST_Area") {
    return std::make_unique<AreaPerimeter>(geo_operator, catalog);
  } else if (operator_name == "ST_Centroid") {
    return std::make_unique<Centroid>(geo_operator, catalog);
  }
  UNREACHABLE();
  return nullptr;
}

std::unique_ptr<CodeGenerator::NullCheckCodegen> Codegen::getNullCheckCodegen(
    llvm::Value* null_lv,
    CgenState* cgen_state,
    Executor* executor) {
  if (isNullable()) {
    CHECK(null_lv);
    return std::make_unique<CodeGenerator::NullCheckCodegen>(
        cgen_state, executor, null_lv, getNullType(), getName() + "_nullcheck");
  } else {
    return nullptr;
  }
}

const Analyzer::Expr* Codegen::getOperand(const size_t index) {
  CHECK_LT(index, operator_->size());
  return operator_->getOperand(index);
}

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
  throw std::runtime_error("Unsupported argument type");
}

}  // namespace spatial_type
