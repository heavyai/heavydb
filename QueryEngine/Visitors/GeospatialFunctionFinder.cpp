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

const std::pair<shared::TableKey, shared::TableKey>
GeospatialFunctionFinder::getTableIdsOfGeoExpr() const {
  // we assume we do not use self geo join and n-ary geo join
  if (!geo_arg_cvs_.empty()) {
    std::unordered_set<shared::TableKey> table_keys;
    std::for_each(geo_arg_cvs_.cbegin(),
                  geo_arg_cvs_.cend(),
                  [&table_keys](const Analyzer::ColumnVar* cv) {
                    table_keys.emplace(cv->getTableKey());
                  });
    if (table_keys.size() == 2) {
      const auto inner_table_key = geo_arg_cvs_.front()->getTableKey();
      shared::TableKey outer_table_key{-1, -1};
      std::for_each(
          table_keys.cbegin(),
          table_keys.cend(),
          [&inner_table_key, &outer_table_key](const shared::TableKey& table_key) {
            if (table_key != inner_table_key) {
              outer_table_key = table_key;
            }
          });
      return std::make_pair(inner_table_key, outer_table_key);
    }
  }
  return std::make_pair(shared::TableKey{-1, -1}, shared::TableKey{-1, -1});
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
