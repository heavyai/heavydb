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

#include "MLModelMetadata.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

void MLModelMetadata::extractModelMetadata(const std::string& model_metadata_json,
                                           const int64_t num_logical_features) {
  rapidjson::Document model_metadata_doc;
  model_metadata_doc.Parse(model_metadata_json.c_str());
  if (model_metadata_doc.HasMember("predicted") &&
      model_metadata_doc["predicted"].IsString()) {
    predicted_ = model_metadata_doc["predicted"].GetString();
  }
  if (model_metadata_doc.HasMember("training_query") &&
      model_metadata_doc["training_query"].IsString()) {
    training_query_ = model_metadata_doc["training_query"].GetString();
  }
  if (model_metadata_doc.HasMember("features") &&
      model_metadata_doc["features"].IsArray()) {
    const rapidjson::Value& features_array = model_metadata_doc["features"];
    for (const auto& feature : features_array.GetArray()) {
      features_.emplace_back(feature.GetString());
    }
  } else {
    features_.resize(num_logical_features, "");
  }
  if (model_metadata_doc.HasMember("data_split_eval_fraction") &&
      model_metadata_doc["data_split_eval_fraction"].IsDouble()) {
    // Extract the double value
    data_split_eval_fraction_ =
        model_metadata_doc["data_split_eval_fraction"].GetDouble();
  }
}
