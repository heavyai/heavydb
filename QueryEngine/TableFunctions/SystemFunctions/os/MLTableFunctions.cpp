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

std::vector<std::string> get_model_features(
    const std::string& model_name,
    const std::shared_ptr<AbstractMLModel>& model) {
  return model->getModelMetadata().getFeatures();
}

EXTENSION_NOINLINE_HOST int32_t
pca_fit__cpu_1(TableFunctionManager& mgr,
               const TextEncodingNone& model_name,
               const ColumnList<TextEncodingDict>& input_cat_features,
               const int32_t cat_top_k,
               const float cat_min_fraction,
               const TextEncodingNone& preferred_ml_framework_str,
               const TextEncodingNone& model_metadata,
               Column<TextEncodingDict>& output_model_name) {
  CategoricalFeaturesBuilder<double> cat_features_builder(
      input_cat_features, cat_top_k, cat_min_fraction, false /* cat_include_others */);
  return pca_fit_impl(mgr,
                      model_name,
                      cat_features_builder.getFeatures(),
                      cat_features_builder.getCatFeatureKeys(),
                      preferred_ml_framework_str,
                      model_metadata,
                      output_model_name);
}

EXTENSION_NOINLINE_HOST int32_t
linear_reg_coefs__cpu_1(TableFunctionManager& mgr,
                        const TextEncodingNone& model_name,
                        Column<int64_t>& output_coef_idx,
                        Column<TextEncodingDict>& output_feature,
                        Column<int64_t>& output_sub_coef_idx,
                        Column<TextEncodingDict>& output_sub_feature,
                        Column<double>& output_coef) {
  try {
    const auto linear_reg_model = std::dynamic_pointer_cast<LinearRegressionModel>(
        g_ml_models.getModel(model_name));
    if (!linear_reg_model) {
      throw std::runtime_error("Model is not of type linear regression.");
    }

    const auto& coefs = linear_reg_model->getCoefs();
    const auto& cat_feature_keys = linear_reg_model->getCatFeatureKeys();
    const int64_t num_sub_coefs = static_cast<int64_t>(coefs.size());
    const int64_t num_cat_features = static_cast<int64_t>(cat_feature_keys.size());
    mgr.set_output_row_size(num_sub_coefs);

    std::vector<std::string> feature_names =
        get_model_features(model_name, linear_reg_model);
    feature_names.insert(feature_names.begin(), "intercept");

    for (int64_t sub_coef_idx = 0, coef_idx = 0; sub_coef_idx < num_sub_coefs;
         ++coef_idx) {
      if (num_cat_features >= coef_idx && coef_idx > 0) {
        const auto& col_cat_feature_keys = cat_feature_keys[coef_idx - 1];
        int64_t col_cat_feature_idx = 1;
        for (const auto& col_cat_feature_key : col_cat_feature_keys) {
          output_coef_idx[sub_coef_idx] = coef_idx;
          if (feature_names[coef_idx].empty()) {
            output_feature[sub_coef_idx] = inline_null_value<TextEncodingDict>();
          } else {
            output_feature[sub_coef_idx] =
                output_feature.getOrAddTransient(feature_names[coef_idx]);
          }
          output_sub_coef_idx[sub_coef_idx] = col_cat_feature_idx++;
          output_sub_feature[sub_coef_idx] =
              output_sub_feature.getOrAddTransient(col_cat_feature_key);
          output_coef[sub_coef_idx] = coefs[sub_coef_idx];
          ++sub_coef_idx;
        }
      } else {
        output_coef_idx[sub_coef_idx] = coef_idx;
        if (feature_names[coef_idx].empty()) {
          output_feature[sub_coef_idx] = inline_null_value<TextEncodingDict>();
        } else {
          output_feature[sub_coef_idx] =
              output_feature.getOrAddTransient(feature_names[coef_idx]);
        }
        output_sub_coef_idx[sub_coef_idx] = 1;
        output_sub_feature[sub_coef_idx] = inline_null_value<TextEncodingDict>();
        output_coef[sub_coef_idx] = coefs[sub_coef_idx];
        ++sub_coef_idx;
      }
    }

    return num_sub_coefs;
  } catch (std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
}

EXTENSION_NOINLINE_HOST int32_t
linear_reg_coefs__cpu_2(TableFunctionManager& mgr,
                        const Column<TextEncodingDict>& model_name,
                        Column<int64_t>& output_coef_idx,
                        Column<TextEncodingDict>& output_feature,
                        Column<int64_t>& output_sub_coef_idx,
                        Column<TextEncodingDict>& output_sub_feature,
                        Column<double>& output_coef) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model name CURSOR.");
  }
  TextEncodingNone model_name_text_enc_none(mgr, model_name.getString(0));
  return linear_reg_coefs__cpu_1(mgr,
                                 model_name_text_enc_none,
                                 output_coef_idx,
                                 output_feature,
                                 output_sub_coef_idx,
                                 output_sub_feature,
                                 output_coef);
}

EXTENSION_NOINLINE_HOST int32_t
random_forest_reg_var_importance__cpu_1(TableFunctionManager& mgr,
                                        const TextEncodingNone& model_name,
                                        Column<int64_t>& feature_id,
                                        Column<TextEncodingDict>& feature,
                                        Column<int64_t>& sub_feature_id,
                                        Column<TextEncodingDict>& sub_feature,
                                        Column<double>& importance_score) {
#ifndef HAVE_ONEDAL
  return mgr.ERROR_MESSAGE(
      "Only OneDAL framework supported for random forest regression.");
#endif
  try {
#ifdef HAVE_ONEDAL
    const auto base_model = g_ml_models.getModel(model_name);
    const auto rand_forest_model =
        std::dynamic_pointer_cast<RandomForestRegressionModel>(base_model);
    if (!rand_forest_model) {
      throw std::runtime_error("Model is not of type random forest.");
    }
    const auto& variable_importance_scores =
        onedal_random_forest_reg_var_importance_impl(rand_forest_model);
    const int64_t num_features = variable_importance_scores.size();
    mgr.set_output_row_size(num_features);
    if (num_features == 0) {
      return mgr.ERROR_MESSAGE("Variable importance not computed for this model.");
    }
    if (num_features != rand_forest_model->getNumFeatures()) {
      return mgr.ERROR_MESSAGE(
          "Mismatch in number of features and number of variable importance metrics.");
    }
    const auto num_logical_features = rand_forest_model->getNumLogicalFeatures();
    std::vector<std::string> feature_names =
        get_model_features(model_name, rand_forest_model);

    int64_t physical_feature_idx = 0;
    const auto& cat_feature_keys = rand_forest_model->getCatFeatureKeys();
    const auto num_cat_features = rand_forest_model->getNumCatFeatures();
    for (int64_t feature_idx = 0; feature_idx < num_logical_features; ++feature_idx) {
      // Make feature ids start at 1, not 0
      if (feature_idx < num_cat_features) {
        const auto& col_cat_feature_keys = cat_feature_keys[feature_idx];
        int64_t sub_feature_idx = 1;
        for (const auto& col_cat_feature_key : col_cat_feature_keys) {
          feature_id[physical_feature_idx] = feature_idx + 1;
          if (feature_names[feature_idx].empty()) {
            feature[physical_feature_idx] = inline_null_value<TextEncodingDict>();
          } else {
            feature[physical_feature_idx] =
                feature.getOrAddTransient(feature_names[feature_idx]);
          }
          sub_feature_id[physical_feature_idx] = sub_feature_idx++;
          const TextEncodingDict feature_sub_key =
              sub_feature.getOrAddTransient(col_cat_feature_key);
          sub_feature[physical_feature_idx] = feature_sub_key;
          importance_score[physical_feature_idx] =
              variable_importance_scores[physical_feature_idx];
          physical_feature_idx++;
        }
      } else {
        feature_id[physical_feature_idx] = feature_idx + 1;
        if (feature_names[feature_idx].empty()) {
          feature[physical_feature_idx] = inline_null_value<TextEncodingDict>();
        } else {
          feature[physical_feature_idx] =
              feature.getOrAddTransient(feature_names[feature_idx]);
        }
        sub_feature_id[physical_feature_idx] = 1;
        sub_feature[physical_feature_idx] = inline_null_value<TextEncodingDict>();
        importance_score[physical_feature_idx] =
            variable_importance_scores[physical_feature_idx];
        physical_feature_idx++;
      }
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
                                        Column<TextEncodingDict>& feature,
                                        Column<int64_t>& sub_feature_id,
                                        Column<TextEncodingDict>& sub_feature,
                                        Column<double>& importance_score) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model name CURSOR.");
  }
  TextEncodingNone model_name_text_enc_none(mgr, model_name.getString(0));
  return random_forest_reg_var_importance__cpu_1(mgr,
                                                 model_name_text_enc_none,
                                                 feature_id,
                                                 feature,
                                                 sub_feature_id,
                                                 sub_feature,
                                                 importance_score);
}

EXTENSION_NOINLINE_HOST
int32_t get_decision_trees__cpu_1(TableFunctionManager& mgr,
                                  const TextEncodingNone& model_name,
                                  Column<int64_t>& tree_id,
                                  Column<int64_t>& entry_id,
                                  Column<bool>& is_split_node,
                                  Column<int64_t>& feature_id,
                                  Column<int64_t>& left_child,
                                  Column<int64_t>& right_child,
                                  Column<double>& value) {
#ifdef HAVE_ONEDAL
  try {
    const auto model = g_ml_models.getModel(model_name);
    const auto tree_model = std::dynamic_pointer_cast<AbstractTreeModel>(model);
    if (!tree_model) {
      throw std::runtime_error("Model not a tree-type model.");
    }
    const auto num_trees = tree_model->getNumTrees();
    std::vector<std::vector<DecisionTreeEntry>> decision_trees(num_trees);
    for (int64_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
      TreeModelVisitor tree_visitor(decision_trees[tree_idx]);
      tree_model->traverseDF(tree_idx, tree_visitor);
    }
    std::vector<int64_t> decision_tree_offsets(num_trees + 1);
    decision_tree_offsets[0] = 0;
    for (int64_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
      decision_tree_offsets[tree_idx + 1] =
          decision_tree_offsets[tree_idx] +
          static_cast<int64_t>(decision_trees[tree_idx].size());
    }
    const auto num_entries = decision_tree_offsets[num_trees];
    mgr.set_output_row_size(num_entries);
    for (int64_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
      const auto& decision_tree = decision_trees[tree_idx];
      const auto output_offset = decision_tree_offsets[tree_idx];
      const int64_t num_tree_entries = decision_tree.size();
      for (int64_t entry_idx = 0; entry_idx < num_tree_entries; ++entry_idx) {
        const int64_t output_idx = output_offset + entry_idx;
        const auto& tree_entry = decision_tree[entry_idx];
        const bool entry_is_split_node = tree_entry.isSplitNode();
        tree_id[output_idx] = tree_idx;
        entry_id[output_idx] = entry_idx;
        is_split_node[output_idx] = entry_is_split_node;
        feature_id[output_idx] = !entry_is_split_node ? inline_null_value<int64_t>()
                                                      : tree_entry.feature_index;
        left_child[output_idx] = !entry_is_split_node ? inline_null_value<int64_t>()
                                                      : tree_entry.left_child_row_idx;
        right_child[output_idx] = !entry_is_split_node ? inline_null_value<int64_t>()
                                                       : tree_entry.right_child_row_idx;
        value[output_idx] = tree_entry.value;
      }
    }
    return num_entries;
  } catch (std::runtime_error& e) {
    const std::string error_str(e.what());
    return mgr.ERROR_MESSAGE(error_str);
  }
#else  // Not HAVE_ONEDAL
  return mgr.ERROR_MESSAGE("OneDAL library must be available for get_decision_trees.");
#endif
}

EXTENSION_NOINLINE_HOST
int32_t get_decision_trees__cpu_2(TableFunctionManager& mgr,
                                  const Column<TextEncodingDict>& model_name,
                                  Column<int64_t>& tree_id,
                                  Column<int64_t>& entry_id,
                                  Column<bool>& is_split_node,
                                  Column<int64_t>& feature_id,
                                  Column<int64_t>& left_child,
                                  Column<int64_t>& right_child,
                                  Column<double>& value) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model name CURSOR.");
  }
  TextEncodingNone model_name_text_enc_none(mgr, model_name.getString(0));
  return get_decision_trees__cpu_1(mgr,
                                   model_name_text_enc_none,
                                   tree_id,
                                   entry_id,
                                   is_split_node,
                                   feature_id,
                                   left_child,
                                   right_child,
                                   value);
}

EXTENSION_NOINLINE_HOST
void check_model_params(const std::shared_ptr<AbstractMLModel>& model,
                        const int64_t num_cat_features,
                        const int64_t num_numeric_features) {
  if (model->getNumLogicalFeatures() != num_cat_features + num_numeric_features) {
    std::ostringstream error_oss;
    error_oss << "Model expects " << model->getNumLogicalFeatures() << " features but "
              << num_cat_features + num_numeric_features << " were provided.";
    throw std::runtime_error(error_oss.str());
  }
  if (model->getNumCatFeatures() != num_cat_features) {
    std::ostringstream error_oss;
    error_oss << "Model expects " << model->getNumCatFeatures()
              << " categorical features but " << num_cat_features << " were provided.";
    throw std::runtime_error(error_oss.str());
  }
}

#endif  // #ifndef __CUDACC__
