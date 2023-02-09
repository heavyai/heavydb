#include "MLTableFunctions.hpp"

using namespace TableFunctions_Namespace;

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST
int32_t supported_ml_frameworks__cpu_(TableFunctionManager& mgr,
                                      Column<TextEncodingDict>& output_ml_frameworks,
                                      Column<bool>& output_availability,
                                      Column<bool>& output_default) {
  const std::vector<std::string> ml_frameworks = {"onedal", "mlpack"};
  const int32_t num_frameworks = ml_frameworks.size();
  mgr.set_output_row_size(num_frameworks);
  const std::vector<int32_t> ml_framework_string_ids =
      output_ml_frameworks.string_dict_proxy_->getOrAddTransientBulk(ml_frameworks);

#if defined(HAVE_ONEDAL) || defined(HAVE_MLPACK)
  bool found_available_framework = false;
  auto framework_found_actions = [&output_availability,
                                  &output_default,
                                  &found_available_framework](const int64_t out_row_idx) {
    output_availability[out_row_idx] = true;
    if (!found_available_framework) {
      output_default[out_row_idx] = true;
      found_available_framework = true;
    } else {
      output_default[out_row_idx] = false;
    }
  };
#endif

#if !defined(HAVE_ONEDAL) || !defined(HAVE_MLPACK)
  auto framework_not_found_actions = [&output_availability,
                                      &output_default](const int64_t out_row_idx) {
    output_availability[out_row_idx] = false;
    output_default[out_row_idx] = false;
  };
#endif

  for (int32_t out_row_idx = 0; out_row_idx < num_frameworks; ++out_row_idx) {
    output_ml_frameworks[out_row_idx] = ml_framework_string_ids[out_row_idx];
    if (ml_frameworks[out_row_idx] == "onedal") {
#ifdef HAVE_ONEDAL
      framework_found_actions(out_row_idx);
#else
      framework_not_found_actions(out_row_idx);
#endif
    } else if (ml_frameworks[out_row_idx] == "mlpack") {
#ifdef HAVE_MLPACK
      framework_found_actions(out_row_idx);
#else
      framework_not_found_actions(out_row_idx);
#endif
    }
  }
  return num_frameworks;
}

EXTENSION_NOINLINE_HOST int32_t
linear_reg_coefs__cpu_1(TableFunctionManager& mgr,
                        const TextEncodingNone& model_name,
                        Column<int64_t>& output_coef_idx,
                        Column<double>& output_coef) {
  try {
    auto model = linear_reg_models_.getModel(model_name);
    const auto& coefs = model.coefs;
    const auto num_coefs = static_cast<int64_t>(coefs.size());
    mgr.set_output_row_size(num_coefs);
    for (int64_t coef_idx = 0; coef_idx < num_coefs; ++coef_idx) {
      output_coef_idx[coef_idx] = coef_idx;
      output_coef[coef_idx] = coefs[coef_idx];
    }
    return num_coefs;
  } catch (std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
}

EXTENSION_NOINLINE_HOST int32_t
linear_reg_coefs__cpu_2(TableFunctionManager& mgr,
                        const Column<TextEncodingDict>& model_name,
                        Column<int64_t>& output_coef_idx,
                        Column<double>& output_coef) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model name CURSOR.");
  }
  const std::string model_name_str{model_name.getString(0)};
  return linear_reg_coefs__cpu_1(mgr, model_name_str, output_coef_idx, output_coef);
}

EXTENSION_NOINLINE_HOST int32_t
random_forest_reg_var_importance__cpu_1(TableFunctionManager& mgr,
                                        const TextEncodingNone& model_name,
                                        Column<int64_t>& feature_id,
                                        Column<double>& importance_score) {
#ifndef HAVE_ONEDAL
  return mgr.ERROR_MESSAGE(
      "Only OneDAL framework supported for random forest regression.");
#endif
  try {
#ifdef HAVE_ONEDAL
    const auto variable_importance_scores =
        onedal_random_forest_reg_var_importance_impl(model_name);
    const int64_t num_features = variable_importance_scores.size();
    mgr.set_output_row_size(num_features);
    for (int64_t feature_idx = 0; feature_idx < num_features; ++feature_idx) {
      // Make feature ids start at 1, not 0
      feature_id[feature_idx] = feature_idx + 1;
      importance_score[feature_idx] = variable_importance_scores[feature_idx];
    }
    return num_features;
#endif
  } catch (std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
}

EXTENSION_NOINLINE_HOST int32_t
random_forest_reg_var_importance__cpu_2(TableFunctionManager& mgr,
                                        const Column<TextEncodingDict>& model_name,
                                        Column<int64_t>& feature_id,
                                        Column<double>& importance_score) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model name CURSOR.");
  }
  const std::string model_name_str{model_name.getString(0)};
  return random_forest_reg_var_importance__cpu_1(
      mgr, model_name_str, feature_id, importance_score);
}

#endif  // #ifndef __CUDACC__
