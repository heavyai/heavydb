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

#include "GeospatialFunctionFinder.h"
#include <regex>

const std::vector<const Analyzer::ColumnVar*>& GeospatialFunctionFinder::getGeoArgCvs()
    const {
  return geo_arg_cvs_;
}

const std::pair<int, int> GeospatialFunctionFinder::getTableIdsOfGeoExpr() const {
  // we assume we do not use self geo join and n-ary geo join
  if (!geo_arg_cvs_.empty()) {
    std::unordered_set<int> tableIds;
    std::for_each(geo_arg_cvs_.cbegin(),
                  geo_arg_cvs_.cend(),
                  [&tableIds](const Analyzer::ColumnVar* cv) {
                    tableIds.emplace(cv->get_table_id());
                  });
    if (tableIds.size() == 2) {
      const int inner_table_id = geo_arg_cvs_.front()->get_table_id();
      int outer_table_id = -1;
      std::for_each(tableIds.cbegin(),
                    tableIds.cend(),
                    [&inner_table_id, &outer_table_id](const int table_id) {
                      if (table_id != inner_table_id) {
                        outer_table_id = table_id;
                      }
                    });
      return std::make_pair(inner_table_id, outer_table_id);
    }
  }
  return std::make_pair(-1, -1);
}

const std::string& GeospatialFunctionFinder::getGeoFunctionName() const {
  return geo_func_name_;
}

void* GeospatialFunctionFinder::visitFunctionOper(
    const Analyzer::FunctionOper* func_oper) const {
  geo_func_name_ = func_oper->getName();
  for (size_t i = 0; i < func_oper->getArity(); i++) {
    if (auto cv = dynamic_cast<const Analyzer::ColumnVar*>(func_oper->getArg(i))) {
      geo_arg_cvs_.push_back(cv);
    }
  }
  return nullptr;
}

void* GeospatialFunctionFinder::visitGeoExpr(Analyzer::GeoExpr const* geo_expr) const {
  for (auto arg : geo_expr->getChildExprs()) {
    if (auto cv = dynamic_cast<const Analyzer::ColumnVar*>(arg)) {
      geo_arg_cvs_.push_back(cv);
    }
  }
  return nullptr;
}
