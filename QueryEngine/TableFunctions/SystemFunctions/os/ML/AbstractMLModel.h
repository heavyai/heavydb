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

#include "MLModelMetadata.h"
#include "MLModelType.h"

#include <string>
#include "Shared/base64.h"

class AbstractMLModel {
 public:
  AbstractMLModel(const std::string& model_metadata)
      : model_metadata_(shared::decode_base64(model_metadata)) {}

  AbstractMLModel(const std::string& model_metadata,
                  const std::vector<std::vector<std::string>>& cat_feature_keys)
      : model_metadata_(shared::decode_base64(model_metadata))
      , cat_feature_keys_(cat_feature_keys) {}
  virtual MLModelType getModelType() const = 0;
  virtual std::string getModelTypeString() const = 0;
  virtual int64_t getNumFeatures() const = 0;
  virtual ~AbstractMLModel() = default;
  const std::string& getModelMetadataStr() const { return model_metadata_; }
  MLModelMetadata getModelMetadata() const {
    return MLModelMetadata("",
                           getModelType(),
                           getModelTypeString(),
                           getNumLogicalFeatures(),
                           getNumFeatures(),
                           getNumCatFeatures(),
                           getNumLogicalFeatures() - getNumCatFeatures(),
                           getModelMetadataStr());
  }
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
  std::string model_metadata_;
  std::vector<std::vector<std::string>> cat_feature_keys_;
};
