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

#include <string>

enum MLModelType { LINEAR_REG, DECISION_TREE_REG, GBT_REG, RANDOM_FOREST_REG };

inline std::string get_ml_model_type_str(const MLModelType model_type) {
  switch (model_type) {
    case MLModelType::LINEAR_REG: {
      return "LINEAR_REG";
    }
    case MLModelType::DECISION_TREE_REG: {
      return "DECISION_TREE_REG";
    }
    case MLModelType::GBT_REG: {
      return "GBT_REG";
    }
    case MLModelType::RANDOM_FOREST_REG: {
      return "RANDOM_FOREST_REG";
    }
    default: {
      CHECK(false) << "Unknown model type.";
      // Satisfy compiler
      return "LINEAR_REG";
    }
  }
}

class AbstractMLModel {
 public:
  AbstractMLModel() {}
  AbstractMLModel(const std::vector<std::vector<std::string>>& cat_feature_keys)
      : cat_feature_keys_(cat_feature_keys) {}
  virtual MLModelType getModelType() const = 0;
  virtual std::string getModelTypeString() const = 0;
  virtual int64_t getNumFeatures() const = 0;
  virtual ~AbstractMLModel() = default;
  const std::vector<std::vector<std::string>>& getCatFeatureKeys() const {
    return cat_feature_keys_;
  }
  const int64_t getNumCatFeatures() const { return cat_feature_keys_.size(); }

  const int64_t getNumOneHotFeatures() const {
    int64_t num_one_hot_features{0};
    for (const auto& cat_feature_key : cat_feature_keys_) {
      num_one_hot_features += static_cast<int64_t>(cat_feature_key.size());
    }
    return num_one_hot_features;
  }

  const int64_t getNumLogicalFeatures() const {
    return getNumFeatures() - getNumOneHotFeatures() + getNumCatFeatures();
  }

 protected:
  std::vector<std::vector<std::string>> cat_feature_keys_;
};
