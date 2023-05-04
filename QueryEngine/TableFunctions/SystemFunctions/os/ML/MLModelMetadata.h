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

#include "MLModelType.h"

#include <string>
#include <vector>

class MLModelMetadata {
 public:
  MLModelMetadata(const std::string& model_name,
                  const MLModelType model_type,
                  const std::string& model_type_str,
                  const int64_t num_logical_features,
                  const int64_t num_features,
                  const int64_t num_categorical_features,
                  const int64_t num_numeric_features,
                  const std::string& model_metadata_json)
      : model_name_(model_name)
      , model_type_(model_type)
      , model_type_str_(model_type_str)
      , num_logical_features_(num_logical_features)
      , num_features_(num_features)
      , num_categorical_features_(num_categorical_features)
      , num_numeric_features_(num_numeric_features) {
    extractModelMetadata(model_metadata_json, num_logical_features);
  }

  void extractModelMetadata(const std::string& model_metadata_json,
                            const int64_t num_logical_features);

  const std::string& getModelName() const { return model_name_; }
  const MLModelType getModelType() const { return model_type_; }
  const std::string& getModelTypeStr() const { return model_type_str_; }
  int64_t getNumLogicalFeatures() const { return num_logical_features_; }
  int64_t getNumFeatures() const { return num_features_; }
  int64_t getNumCategoricalFeatures() const { return num_categorical_features_; }
  int64_t getNumNumericFeatures() const { return num_numeric_features_; }
  const std::string& getPredicted() const { return predicted_; }
  const std::vector<std::string>& getFeatures() const { return features_; }
  const std::string& getTrainingQuery() const { return training_query_; }
  double getDataSplitTrainFraction() const { return data_split_train_fraction_; }
  double getDataSplitEvalFraction() const { return data_split_eval_fraction_; }
  const std::vector<int64_t>& getFeaturePermutations() const {
    return feature_permutations_;
  }

 private:
  const std::string model_name_;
  const MLModelType model_type_;
  const std::string model_type_str_;
  const int64_t num_logical_features_;
  const int64_t num_features_;
  const int64_t num_categorical_features_;
  const int64_t num_numeric_features_;
  std::string predicted_;
  std::vector<std::string> features_;
  std::string training_query_;
  double data_split_train_fraction_{1.0};
  double data_split_eval_fraction_{0.0};
  std::vector<int64_t> feature_permutations_;
};
