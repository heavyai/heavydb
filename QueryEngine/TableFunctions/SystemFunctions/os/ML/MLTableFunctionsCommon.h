/*
 * Copyright 2022 HEAVY.AI, Inc., Inc.
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

#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsCommon.h"

#include <map>

enum class MLFramework { DEFAULT, ONEDAL, MLPACK, INVALID };

MLFramework get_ml_framework(const std::string& ml_framework_str) {
  const auto upper_ml_framework_str = to_upper(ml_framework_str);
  const static std::map<std::string, MLFramework> ml_framework_map = {
      {"DEFAULT", MLFramework::DEFAULT},
      {"ONEDAL", MLFramework::ONEDAL},
      {"MLPACK", MLFramework::MLPACK}};
  const auto itr = ml_framework_map.find(upper_ml_framework_str);
  if (itr == ml_framework_map.end()) {
    return MLFramework::INVALID;
  }
  return itr->second;
}

enum class KMeansInitStrategy { DEFAULT, DETERMINISTIC, RANDOM, PLUS_PLUS, INVALID };

KMeansInitStrategy get_kmeans_init_type(const std::string& init_type_str) {
  const auto upper_init_type_str = to_upper(init_type_str);
  const static std::map<std::string, KMeansInitStrategy> kmeans_init_type_map = {
      {"DEFAULT", KMeansInitStrategy::DEFAULT},
      {"DETERMINISTIC", KMeansInitStrategy::DETERMINISTIC},
      {"RANDOM", KMeansInitStrategy::RANDOM},
      {"PLUS_PLUS", KMeansInitStrategy::PLUS_PLUS}};
  const auto itr = kmeans_init_type_map.find(upper_init_type_str);
  if (itr == kmeans_init_type_map.end()) {
    return KMeansInitStrategy::INVALID;
  }
  return itr->second;
}