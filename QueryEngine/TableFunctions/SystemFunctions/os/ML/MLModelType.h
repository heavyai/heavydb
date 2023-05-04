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

#include "Shared/misc.h"

#include <string>

enum MLModelType { LINEAR_REG, DECISION_TREE_REG, GBT_REG, RANDOM_FOREST_REG, PCA };

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
    case MLModelType::PCA: {
      return "PCA";
    }
    default: {
      CHECK(false) << "Unknown model type.";
      // Satisfy compiler
      return "LINEAR_REG";
    }
  }
}

inline MLModelType get_ml_model_type_from_str(const std::string& model_type_str) {
  const auto upper_model_type_str = to_upper(model_type_str);
  if (upper_model_type_str == "LINEAR_REG") {
    return MLModelType::LINEAR_REG;
  } else if (upper_model_type_str == "DECISION_TREE_REG") {
    return MLModelType::DECISION_TREE_REG;
  } else if (upper_model_type_str == "GBT_REG") {
    return MLModelType::GBT_REG;
  } else if (upper_model_type_str == "RANDOM_FOREST_REG") {
    return MLModelType::RANDOM_FOREST_REG;
  } else if (upper_model_type_str == "PCA") {
    return MLModelType::PCA;
  } else {
    throw std::invalid_argument("Unknown model type: " + upper_model_type_str);
  }
}

inline bool is_regression_model(const MLModelType model_type) {
  return shared::is_any<MLModelType::LINEAR_REG,
                        MLModelType::DECISION_TREE_REG,
                        MLModelType::GBT_REG,
                        MLModelType::RANDOM_FOREST_REG>(model_type);
}
