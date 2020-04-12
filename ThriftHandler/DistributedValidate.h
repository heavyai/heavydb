/*
 * Copyright 2019 OmniSci, Inc.
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

#include <Shared/measure.h>
#include "Catalog/Catalog.h"
#include "Shared/Logger.h"

#include "LeafAggregator.h"
#include "MapDServer.h"
#include "QueryState.h"

class DBHandler;

/**
 * @brief Driver for running distributed validation on metadata across cluster.
 * DistributedValidate provides functions for running a validation on system metadata
 * across a cluster, with options to remove broken objects that have been identified
 */
class DistributedValidate {
 public:
  DistributedValidate(const std::string type,
                      const bool is_repair_type_remove,
                      Catalog_Namespace::Catalog& cat,
                      LeafAggregator& leaf_aggregator,
                      const Catalog_Namespace::SessionInfo session_info,
                      DBHandler& mapd_handler)
      : cat_(cat)
      , type_(type)
      , is_repair_type_remove_(is_repair_type_remove)
      , leaf_aggregator_(leaf_aggregator)
      , session_info_(session_info)
      , mapd_handler_(mapd_handler) {}
  /**
   * @brief Compares Aggregators and Leaves metatdata reporting what is different.
   */
  std::string validate(query_state::QueryStateProxy query_state_proxy) const {
    return nullptr;
  };

 private:
  Catalog_Namespace::Catalog& cat_;
  const std::string type_;
  const bool is_repair_type_remove_;
  LeafAggregator& leaf_aggregator_;
  const Catalog_Namespace::SessionInfo session_info_;
  DBHandler& mapd_handler_;
};
