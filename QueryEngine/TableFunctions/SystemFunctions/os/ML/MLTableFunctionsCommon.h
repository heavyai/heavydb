/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsCommon.hpp"

#include <map>

enum class MLFramework { DEFAULT, ONEDAL, ONEAPI, MLPACK, INVALID };

inline MLFramework get_ml_framework(const std::string& ml_framework_str) {
  const auto upper_ml_framework_str = to_upper(ml_framework_str);
  const static std::map<std::string, MLFramework> ml_framework_map = {
      {"DEFAULT", MLFramework::DEFAULT},
      {"ONEDAL", MLFramework::ONEDAL},
      {"ONEAPI", MLFramework::ONEAPI},
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

enum class VarImportanceMetric { DEFAULT, NONE, MDI, MDA, MDA_SCALED, INVALID };

inline VarImportanceMetric get_var_importance_metric(
    const std::string& var_importance_metric_str) {
  const auto upper_var_importance_metric_str = to_upper(var_importance_metric_str);
  const static std::map<std::string, VarImportanceMetric> var_importance_metric_map = {
      {"DEFAULT", VarImportanceMetric::DEFAULT},
      {"NONE", VarImportanceMetric::NONE},
      {"MDI", VarImportanceMetric::MDI},
      {"MDA", VarImportanceMetric::MDA},
      {"MDA_SCALED", VarImportanceMetric::MDA_SCALED}};
  const auto itr = var_importance_metric_map.find(upper_var_importance_metric_str);
  if (itr == var_importance_metric_map.end()) {
    return VarImportanceMetric::INVALID;
  }
  return itr->second;
}
