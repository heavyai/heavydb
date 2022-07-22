/*
 * Copyright 2022 HEAVY.AI, Inc.
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

// we do not reorder tables when they are used as input of ST_CONTAINS function
// to prevent incorrect query result
#pragma once

#include <vector>
#include "../Analyzer/Analyzer.h"

class GeospatialFunctionFinder : public ScalarExprVisitor<void*> {
 public:
  const std::vector<const Analyzer::ColumnVar*>& getGeoArgCvs() const;
  const std::pair<int, int> getTableIdsOfGeoExpr() const;
  const std::string& getGeoFunctionName() const;

 protected:
  void* visitGeoExpr(const Analyzer::GeoExpr* geo_expr) const override;
  void* visitFunctionOper(const Analyzer::FunctionOper* func_oper) const override;

 private:
  mutable std::vector<const Analyzer::ColumnVar*> geo_arg_cvs_;
  mutable std::string geo_func_name_{""};
};
