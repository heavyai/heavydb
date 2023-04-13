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

#include "CodeGenerator.h"
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLModel.h"
#include "TreeModelPredictionMgr.h"

#ifdef HAVE_CUDA
#include "DataMgr/Allocators/CudaAllocator.h"
#include "GpuMemUtils.h"
#endif  // HAVE_CUDA

#include <tbb/parallel_for.h>
#include <stack>
#include <vector>

std::vector<std::shared_ptr<Analyzer::Expr>> generated_encoded_and_casted_regressors(
    const std::vector<std::shared_ptr<Analyzer::Expr>>& regressor_exprs,
    const std::vector<std::vector<std::string>>& cat_feature_keys,
    Executor* executor) {
  std::vector<std::shared_ptr<Analyzer::Expr>> casted_regressor_exprs;
  const size_t num_regressor_exprs = regressor_exprs.size();
  const size_t num_cat_regressors = cat_feature_keys.size();

  if (num_cat_regressors > num_regressor_exprs) {
    throw std::runtime_error("More categorical keys than regressors.");
  }

  auto get_int_constant_expr = [](int32_t const_val) {
    Datum d;
    d.intval = const_val;
    return makeExpr<Analyzer::Constant>(SQLTypeInfo(kINT, false), false, d);
  };

  for (size_t regressor_idx = 0; regressor_idx < num_regressor_exprs; ++regressor_idx) {
    auto& regressor_expr = regressor_exprs[regressor_idx];
    const auto& regressor_ti = regressor_expr->get_type_info();
    if (regressor_ti.is_number()) {
      // Don't conditionally cast to double iff type is not double
      // as this was causing issues for the random forest function with
      // mixed types. Need to troubleshoot more but always casting to double
      // regardless of the underlying type always seems to be safe
      casted_regressor_exprs.emplace_back(makeExpr<Analyzer::UOper>(
          SQLTypeInfo(kDOUBLE, false), false, kCAST, regressor_expr));
    } else {
      CHECK(regressor_ti.is_string()) << "Expected text type";
      if (!regressor_ti.is_text_encoding_dict()) {
        throw std::runtime_error("Expected dictionary-encoded text column.");
      }
      if (regressor_idx >= num_cat_regressors) {
        throw std::runtime_error("Model not trained on text type for column.");
      }
      const auto& str_dict_key = regressor_ti.getStringDictKey();
      const auto str_dict_proxy = executor->getStringDictionaryProxy(str_dict_key, true);
      for (const auto& cat_feature_key : cat_feature_keys[regressor_idx]) {
        // For one-hot encoded columns, null values will translate as a 0.0 and not a null
        // We are computing the following:
        // CASE WHEN str_val is NULL then 0.0 ELSE
        // CAST(str_id = one_hot_encoded_str_id AS DOUBLE) END

        // Check if the expression is null
        auto is_null_expr = makeExpr<Analyzer::UOper>(
            SQLTypeInfo(kBOOLEAN, false), false, kISNULL, regressor_expr);
        Datum zero_datum;
        zero_datum.doubleval = 0.0;
        // If null then emit a 0.0 double constant as the THEN expr
        auto is_null_then_expr =
            makeExpr<Analyzer::Constant>(SQLTypeInfo(kDOUBLE, false), false, zero_datum);
        std::list<
            std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
            when_then_exprs;
        when_then_exprs.emplace_back(std::make_pair(is_null_expr, is_null_then_expr));
        // The rest of/core string test logic goes in the ELSE statement
        // Get the string id of the one-hot feature
        const auto str_id = str_dict_proxy->getIdOfString(cat_feature_key);
        auto str_id_expr = get_int_constant_expr(str_id);
        // Get integer id for this row's string
        auto key_for_string_expr = makeExpr<Analyzer::KeyForStringExpr>(regressor_expr);

        // Check if this row's string id is equal to the search one-hot encoded id
        std::shared_ptr<Analyzer::Expr> str_equality_expr =
            makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                        false,
                                        kEQ,
                                        kONE,
                                        key_for_string_expr,
                                        str_id_expr);
        // Cast the above boolean results to a double, 0.0 or 1.0
        auto cast_expr = makeExpr<Analyzer::UOper>(
            SQLTypeInfo(kDOUBLE, false), false, kCAST, str_equality_expr);

        // Generate the full CASE statement and add to the casted regressor exprssions
        casted_regressor_exprs.emplace_back(makeExpr<Analyzer::CaseExpr>(
            SQLTypeInfo(kDOUBLE, false), false, when_then_exprs, cast_expr));
      }
    }
  }
  return casted_regressor_exprs;
}

llvm::Value* CodeGenerator::codegenLinRegPredict(
    const Analyzer::MLPredictExpr* expr,
    const std::string& model_name,
    const std::shared_ptr<AbstractMLModel>& abstract_model,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto linear_reg_model =
      std::dynamic_pointer_cast<LinearRegressionModel>(abstract_model);
  // The parent codegen function called this function `codegenLinRegPredict`
  // iff we had MLModelType::LINEAR_REG_PREDICT, so below is just a sanity
  // check
  CHECK(linear_reg_model);
  const auto& model_coefs = linear_reg_model->getCoefs();
  const auto& cat_feature_keys = linear_reg_model->getCatFeatureKeys();

  const auto& regressor_exprs = expr->get_regressor_values();

  const auto casted_regressor_exprs = generated_encoded_and_casted_regressors(
      regressor_exprs, cat_feature_keys, executor());

  auto get_double_constant_expr = [](double const_val) {
    Datum d;
    d.doubleval = const_val;
    return makeExpr<Analyzer::Constant>(SQLTypeInfo(kDOUBLE, false), false, d);
  };

  std::shared_ptr<Analyzer::Expr> result;

  // Linear regression models are of the form
  // y = b0 + b1*x1 + b2*x2 + ... + bn*xn
  // Where b0 is the constant y-intercept, x1..xn are the dependent
  // varabiles (aka regressors or predictors), and b1..bn are the
  // regression coefficients

  for (size_t model_coef_idx = 0; model_coef_idx < model_coefs.size(); ++model_coef_idx) {
    auto coef_value_expr = get_double_constant_expr(model_coefs[model_coef_idx]);
    if (model_coef_idx == size_t(0)) {
      // We have the y-intercept b0, this is not multiplied by any regressor
      result = coef_value_expr;
    } else {
      // We have a term with a regressor (xi) and regression coefficient (bi)
      const auto& casted_regressor_expr = casted_regressor_exprs[model_coef_idx - 1];
      // Multiply regressor by coefficient
      auto mul_expr = makeExpr<Analyzer::BinOper>(SQLTypeInfo(kDOUBLE, false),
                                                  false,
                                                  kMULTIPLY,
                                                  kONE,
                                                  coef_value_expr,
                                                  casted_regressor_expr);
      // Add term to result
      result = makeExpr<Analyzer::BinOper>(
          SQLTypeInfo(kDOUBLE, false), false, kPLUS, kONE, result, mul_expr);
    }
  }

  // The following will codegen the expression tree we just created modeling
  // the linear regression formula
  return codegenArith(dynamic_cast<Analyzer::BinOper*>(result.get()), co);
}

llvm::Value* CodeGenerator::codegenTreeRegPredict(
    const Analyzer::MLPredictExpr* expr,
    const std::string& model_name,
    const std::shared_ptr<AbstractMLModel>& model,
    const CompilationOptions& co) {
#ifdef HAVE_ONEDAL
  const auto tree_model = std::dynamic_pointer_cast<AbstractTreeModel>(model);
  // The parent codegen function called this function `codegenTreeRegPredict`
  // iff we a tree reg MLModelType, so below is just a sanity
  // check
  CHECK(tree_model);
  const int64_t num_trees = static_cast<int64_t>(tree_model->getNumTrees());
  const auto& regressor_exprs = expr->get_regressor_values();
  const auto& cat_feature_keys = tree_model->getCatFeatureKeys();
  const auto casted_regressor_exprs = generated_encoded_and_casted_regressors(
      regressor_exprs, cat_feature_keys, executor());
  // We cast all regressors to double for simplicity and to match
  // how feature filters are stored in the tree model.
  // Null checks are handled further down in the generated kernel
  // in the runtime function itself

  std::vector<llvm::Value*> regressor_values;
  for (const auto& casted_regressor_expr : casted_regressor_exprs) {
    regressor_values.emplace_back(codegen(casted_regressor_expr.get(), false, co)[0]);
  }

  // First build tables, i.e. vectors of DecisionTreeEntry, for each tree
  std::vector<std::vector<DecisionTreeEntry>> decision_trees(num_trees);
  {
    auto tree_build_timer = DEBUG_TIMER("Tree Visitors Dispatched");
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_trees),
                      [&](const tbb::blocked_range<int64_t>& r) {
                        const auto start_tree_idx = r.begin();
                        const auto end_tree_idx = r.end();
                        for (int64_t tree_idx = start_tree_idx; tree_idx < end_tree_idx;
                             ++tree_idx) {
                          TreeModelVisitor tree_visitor(decision_trees[tree_idx]);
                          tree_model->traverseDF(tree_idx, tree_visitor);
                        }
                      });
  }

  // Next, compute prefix-sum offset such that decision_tree_offsets[k]
  // specifies the starting offset of tree k relative to tree 0, and
  // decision_tree_offsets[k+1] specifies the last entry + 1 of tree
  // k relative to tree 0
  std::vector<int64_t> decision_tree_offsets(num_trees + 1);
  decision_tree_offsets[0] = 0;
  for (int64_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
    decision_tree_offsets[tree_idx + 1] =
        decision_tree_offsets[tree_idx] +
        static_cast<int64_t>(decision_trees[tree_idx].size());
  }

  VLOG(1) << tree_model->getModelTypeString() << " model has " << num_trees
          << " trees and " << decision_tree_offsets[num_trees] << " total entries.";

  // Finally, go back through each tree and adjust all left and right child idx entries
  // such that such values are global relative to the start of tree 0. This will allow
  // the downstream code-generated kernel to be able treat these child idx entries as
  // as absolute offsets from the base pointer for all trees, rather than computing such
  // an offset on the fly
  {
    auto tree_offset_correction_timer = DEBUG_TIMER("Tree Offsets Corrected");
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(1, num_trees),
        [&](const tbb::blocked_range<int64_t>& r) {
          const auto start_tree_idx = r.begin();
          const auto end_tree_idx = r.end();
          for (int64_t tree_idx = start_tree_idx; tree_idx < end_tree_idx; ++tree_idx) {
            const int64_t start_offset = decision_tree_offsets[tree_idx];
            auto& decision_tree = decision_trees[tree_idx];
            const int64_t num_tree_entries = static_cast<int64_t>(decision_tree.size());
            CHECK_EQ(num_tree_entries,
                     decision_tree_offsets[tree_idx + 1] - start_offset);
            for (int64_t decision_entry_idx = 0; decision_entry_idx < num_tree_entries;
                 ++decision_entry_idx) {
              if (decision_tree[decision_entry_idx].isSplitNode()) {
                decision_tree[decision_entry_idx].left_child_row_idx += start_offset;
                decision_tree[decision_entry_idx].right_child_row_idx += start_offset;
              }
            }
          }
        });
  }

  {
    auto tree_model_prediction_mgr_timer =
        DEBUG_TIMER("TreeModelPredictionMgr generation and codegen");
    // TreeModelPredictionMgr copies the decision trees and offsets to host
    // buffers in RowSetMemoryOwner and onto each GPU if the query is running
    // on GPU, and takes care of the tree traversal codegen itself

    const bool compute_avg = tree_model->getModelType() == MLModelType::RANDOM_FOREST_REG;
    auto tree_model_prediction_mgr = std::make_unique<TreeModelPredictionMgr>(
        co.device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL
                                                  : Data_Namespace::CPU_LEVEL,
        executor(),
        decision_trees,
        decision_tree_offsets,
        compute_avg);

    return cgen_state_->moveTreeModelPredictionMgr(std::move(tree_model_prediction_mgr))
        ->codegen(regressor_values, co);
  }
#else
  throw std::runtime_error("OneDAL not available.");
#endif
}

llvm::Value* CodeGenerator::codegen(const Analyzer::MLPredictExpr* expr,
                                    const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  const auto& model_expr = expr->get_model_value();
  CHECK(model_expr);
  auto model_constant_expr = dynamic_cast<const Analyzer::Constant*>(model_expr);
  CHECK(model_constant_expr);
  const auto model_datum = model_constant_expr->get_constval();
  const auto model_name_ptr = model_datum.stringval;
  CHECK(model_name_ptr);
  const auto model_name = *model_name_ptr;
  const auto abstract_model = g_ml_models.getModel(model_name);
  const auto model_type = abstract_model->getModelType();
  const auto& regressor_exprs = expr->get_regressor_values();
  if (abstract_model->getNumLogicalFeatures() !=
      static_cast<int64_t>(regressor_exprs.size())) {
    std::ostringstream error_oss;
    error_oss << "ML_PREDICT: Model '" << model_name
              << "' expects different number of predictor variables ("
              << abstract_model->getNumLogicalFeatures() << ") than provided ("
              << regressor_exprs.size() << ").";
    throw std::runtime_error(error_oss.str());
  }

  switch (model_type) {
    case MLModelType::LINEAR_REG: {
      return codegenLinRegPredict(expr, model_name, abstract_model, co);
    }
    case MLModelType::DECISION_TREE_REG:
    case MLModelType::GBT_REG:
    case MLModelType::RANDOM_FOREST_REG: {
      return codegenTreeRegPredict(expr, model_name, abstract_model, co);
    }
    default: {
      throw std::runtime_error("Unsupported model type.");
    }
  }
}
