/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "MLModelMetadata.h"
#include "MLModelType.h"

#include <string>
#include "Shared/base64.h"

namespace {
std::string default_metadata(const std::string& metadata) {
  if (metadata == "" || metadata == "DEFAULT") {
    return "{}";
  }
  return shared::decode_base64(metadata);
}
}  // namespace

class AbstractMLModel {
 public:
  AbstractMLModel(const std::string& model_metadata)
      : model_metadata_(default_metadata(model_metadata)) {}

  AbstractMLModel(const std::string& model_metadata,
                  const std::vector<std::vector<std::string>>& cat_feature_keys)
      : model_metadata_(default_metadata(model_metadata))
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
