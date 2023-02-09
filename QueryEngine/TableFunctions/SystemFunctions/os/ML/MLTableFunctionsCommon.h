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

#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsCommon.hpp"

#include <map>

enum ModelType { LINEAR_REG, RANDOM_FOREST_REG };

template <typename M>
class ModelMap {
 public:
  void addModel(const std::string& model_name, M& model) {
    std::lock_guard<std::shared_mutex> model_map_write_lock(model_map_mutex_);
    model_map_[model_name] = model;
  }

  // Todo: consider making this a reference to save the copy
  // of a potentially large model, but will need to enhance locking
  // to ensure the model is not changed while being used by a consumer
  M getModel(const std::string& model_name) const {
    std::shared_lock<std::shared_mutex> model_map_read_lock(model_map_mutex_);
    auto model_map_itr = model_map_.find(model_name);
    if (model_map_itr != model_map_.end()) {
      return model_map_itr->second;
    }
    throw std::runtime_error("Model does not exist.");
  }

  std::vector<std::string> getModelNames() const {
    std::shared_lock<std::shared_mutex> model_map_read_lock(model_map_mutex_);
    std::vector<std::string> model_names;
    model_names.reserve(model_map_.size());
    for (auto const& model : model_map_) {
      model_names.emplace_back(model.first);
    }
    return model_names;
  }

 private:
  std::map<std::string, M> model_map_;
  mutable std::shared_mutex model_map_mutex_;
};

enum class MLFramework { DEFAULT, ONEDAL, MLPACK, INVALID };

inline MLFramework get_ml_framework(const std::string& ml_framework_str) {
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

inline KMeansInitStrategy get_kmeans_init_type(const std::string& init_type_str) {
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
