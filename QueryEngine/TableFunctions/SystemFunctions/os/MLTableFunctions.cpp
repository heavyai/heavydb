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
    const auto model = ml_models_.getModel(model_name);
    const auto linear_reg_model = std::dynamic_pointer_cast<LinearRegressionModel>(model);
    if (!linear_reg_model) {
      throw std::runtime_error("Model is not of type linear regression.");
    }
    const auto& coefs = linear_reg_model->getCoefs();
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
    const auto& variable_importance_scores =
        onedal_random_forest_reg_var_importance_impl(model_name);
    const int64_t num_features = variable_importance_scores.size();
    mgr.set_output_row_size(num_features);
    if (num_features == 0) {
      return mgr.ERROR_MESSAGE("Variable importance not computed for this model.");
    }
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
    const auto model = ml_models_.getModel(model_name);
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
  const std::string model_name_str{model_name.getString(0)};
  return get_decision_trees__cpu_1(mgr,
                                   model_name_str,
                                   tree_id,
                                   entry_id,
                                   is_split_node,
                                   feature_id,
                                   left_child,
                                   right_child,
                                   value);
}

#endif  // #ifndef __CUDACC__
