/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include <map>
#include <vector>

#ifndef __CUDACC__

#ifdef HAVE_ONEDAL
#include "daal.h"
#endif

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
    const std::string error_str = "Model '" + model_name + "' does not exist.";
    throw std::runtime_error(error_str);
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

struct LinearRegressionModel {
  std::vector<double> coefs;

  LinearRegressionModel() {}

  LinearRegressionModel(const std::vector<double>& coefs) : coefs(coefs) {}
};

inline ModelMap<ModelType> model_types_;
inline ModelMap<LinearRegressionModel> linear_reg_models_;
inline int64_t temp_model_idx{0};
inline std::string temp_model_prefix{"__temp_model__"};

#ifdef HAVE_ONEDAL

using namespace daal::algorithms;
using namespace daal::data_management;

struct RandomForestRegressionModel {
  decision_forest::regression::interface1::ModelPtr model_ptr;
  std::vector<double> variable_importance;
  double out_of_bag_error;

  RandomForestRegressionModel() {}

  RandomForestRegressionModel(
      decision_forest::regression::interface1::ModelPtr& model_ptr,
      const std::vector<double>& variable_importance,
      const double out_of_bag_error)
      : model_ptr(model_ptr)
      , variable_importance(variable_importance)
      , out_of_bag_error(out_of_bag_error) {}
};

inline ModelMap<RandomForestRegressionModel> random_forest_models_;

#endif  // #ifdef HAVE_ONEDAL

#endif  // #ifndef __CUDACC__