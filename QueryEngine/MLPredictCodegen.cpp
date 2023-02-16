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

#ifdef HAVE_ONEDAL

class TreeModelVisitor : public daal::algorithms::regression::TreeNodeVisitor {
 public:
  TreeModelVisitor(std::vector<DecisionTreeEntry>& decision_table)
      : decision_table_(decision_table) {}

  const std::vector<DecisionTreeEntry>& getDecisionTable() const {
    return decision_table_;
  }

  bool onLeafNode(size_t level, double response) override {
    decision_table_.emplace_back(DecisionTreeEntry(response));
    if (last_node_leaf_) {
      decision_table_[parent_nodes_.top()].right_child_row_idx =
          static_cast<int64_t>(decision_table_.size() - 1);
      parent_nodes_.pop();
    }
    last_node_leaf_ = true;
    return true;
  }

  bool onSplitNode(size_t level, size_t featureIndex, double featureValue) override {
    decision_table_.emplace_back(
        DecisionTreeEntry(featureValue,
                          static_cast<int64_t>(featureIndex),
                          static_cast<int64_t>(decision_table_.size() + 1)));
    if (last_node_leaf_) {
      decision_table_[parent_nodes_.top()].right_child_row_idx =
          static_cast<int64_t>(decision_table_.size() - 1);
      parent_nodes_.pop();
    }
    last_node_leaf_ = false;
    parent_nodes_.emplace(decision_table_.size() - 1);
    return true;
  }

 private:
  std::vector<DecisionTreeEntry>& decision_table_;
  std::stack<size_t> parent_nodes_;
  bool last_node_leaf_{false};
};

#endif

llvm::Value* CodeGenerator::codegenLinRegPredict(const Analyzer::MLPredictExpr* expr,
                                                 const std::string& model_name,
                                                 const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto model = linear_reg_models_.getModel(model_name);
  const auto& model_coefs = model.coefs;

  auto get_double_constant_expr = [](double const_val) {
    Datum d;
    d.doubleval = const_val;
    return makeExpr<Analyzer::Constant>(SQLTypeInfo(kDOUBLE, false), false, d);
  };

  const auto& regressor_exprs = expr->get_regressor_values();
  if (model_coefs.size() != regressor_exprs.size() + 1) {
    std::ostringstream error_oss;
    error_oss << "ML_PREDICT: Linear regression model '" << model_name
              << "' expects different number of predictor variables ("
              << model_coefs.size() - 1 << ") than provided (" << regressor_exprs.size()
              << ").";
    throw std::runtime_error(error_oss.str());
  }
  std::shared_ptr<Analyzer::Expr> result;

  for (size_t model_coef_idx = 0; model_coef_idx < model_coefs.size(); ++model_coef_idx) {
    auto coef_value_expr = get_double_constant_expr(model_coefs[model_coef_idx]);
    if (model_coef_idx == size_t(0)) {
      result = coef_value_expr;
    } else {
      auto& regressor_expr = regressor_exprs[model_coef_idx - 1];
      const auto& regressor_ti = regressor_expr->get_type_info();
      std::shared_ptr<Analyzer::Expr> casted_regressor_expr;
      if (regressor_ti.get_type() != kDOUBLE) {
        casted_regressor_expr = makeExpr<Analyzer::UOper>(
            SQLTypeInfo(kDOUBLE, false), false, kCAST, regressor_expr);
      } else {
        casted_regressor_expr = regressor_expr;
      }
      auto mul_expr = makeExpr<Analyzer::BinOper>(SQLTypeInfo(kDOUBLE, false),
                                                  false,
                                                  kMULTIPLY,
                                                  kONE,
                                                  coef_value_expr,
                                                  casted_regressor_expr);
      result = makeExpr<Analyzer::BinOper>(
          SQLTypeInfo(kDOUBLE, false), false, kPLUS, kONE, result, mul_expr);
    }
  }
  return codegenArith(dynamic_cast<Analyzer::BinOper*>(result.get()), co);
}

llvm::Value* CodeGenerator::codegenRandForestRegPredict(
    const Analyzer::MLPredictExpr* expr,
    const std::string& model_name,
    const CompilationOptions& co) {
#ifdef HAVE_ONEDAL
  const auto model = random_forest_models_.getModel(model_name);
  const int64_t num_trees = static_cast<int64_t>(model.model_ptr->getNumberOfTrees());
  const auto& regressor_exprs = expr->get_regressor_values();
  if (regressor_exprs.size() != model.model_ptr->getNumberOfFeatures()) {
    std::ostringstream error_oss;
    error_oss << "ML_PREDICT: Random forest regression model '" << model_name
              << "' expects different number of predictor variables ("
              << model.model_ptr->getNumberOfFeatures() << ") than provided ("
              << regressor_exprs.size() << ").";
    throw std::runtime_error(error_oss.str());
  }
  std::vector<std::shared_ptr<Analyzer::Expr>> casted_regressor_exprs;
  // We cast all regressors to double for simplicity and to match
  // how feature filters are stored in the tree model.
  // Null checks are handled further down in the generated kernel
  // in the runtime function itself
  for (const auto& regressor_expr : regressor_exprs) {
    const auto& regressor_ti = regressor_expr->get_type_info();
    if (regressor_ti.get_type() != kDOUBLE) {
      casted_regressor_exprs.emplace_back(makeExpr<Analyzer::UOper>(
          SQLTypeInfo(kDOUBLE, false), false, kCAST, regressor_expr));
    } else {
      casted_regressor_exprs.emplace_back(regressor_expr);
    }
  }
  std::vector<llvm::Value*> regressor_values;
  for (const auto& casted_regressor_expr : casted_regressor_exprs) {
    regressor_values.emplace_back(codegen(casted_regressor_expr.get(), false, co)[0]);
  }

  // First build tables, i.e. vectors of DecisionTreeEntry, for each tree
  std::vector<std::vector<DecisionTreeEntry>> decision_trees(num_trees);
  tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_trees),
                    [&](const tbb::blocked_range<int64_t>& r) {
                      const auto start_tree_idx = r.begin();
                      const auto end_tree_idx = r.end();
                      for (int64_t tree_idx = start_tree_idx; tree_idx < end_tree_idx;
                           ++tree_idx) {
                        TreeModelVisitor tree_visitor(decision_trees[tree_idx]);
                        model.model_ptr->traverseDF(tree_idx, tree_visitor);
                      }
                    });

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

  VLOG(1) << "Random Forest Model has " << num_trees << " trees and "
          << decision_tree_offsets[num_trees] << " total entries.";

  // Finally, go back through each tree and adjust all left and right child idx entries
  // such that such values are global relative to the start of tree 0. This will allow
  // the downstream code-generated kernel to be able treat these child idx entries as
  // as absolute offsets from the base pointer for all trees, rather than computing such
  // an offset on the fly
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(1, num_trees),
      [&](const tbb::blocked_range<int64_t>& r) {
        const auto start_tree_idx = r.begin();
        const auto end_tree_idx = r.end();
        for (int64_t tree_idx = start_tree_idx; tree_idx < end_tree_idx; ++tree_idx) {
          const int64_t start_offset = decision_tree_offsets[tree_idx];
          auto& decision_tree = decision_trees[tree_idx];
          const int64_t num_tree_entries = static_cast<int64_t>(decision_tree.size());
          CHECK_EQ(num_tree_entries, decision_tree_offsets[tree_idx + 1] - start_offset);
          for (int64_t decision_entry_idx = 0; decision_entry_idx < num_tree_entries;
               ++decision_entry_idx) {
            if (decision_tree[decision_entry_idx].isSplitNode()) {
              decision_tree[decision_entry_idx].left_child_row_idx += start_offset;
              decision_tree[decision_entry_idx].right_child_row_idx += start_offset;
            }
          }
        }
      });

  // TreeModelPredictionMgr copies the decision trees and offsets to host
  // buffers in RowSetMemoryOwner and onto each GPU if the query is running
  // on GPU, and takes care of the tree traversal codegen itself
  auto tree_model_prediction_mgr = std::make_unique<TreeModelPredictionMgr>(
      co.device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL
                                                : Data_Namespace::CPU_LEVEL,
      executor()->deviceCount(co.device_type),
      executor(),
      executor()->getDataMgr(),
      decision_trees,
      decision_tree_offsets);

  return cgen_state_->moveTreeModelPredictionMgr(std::move(tree_model_prediction_mgr))
      ->codegen(regressor_values, co);
#else
  CHECK(false) << "OneDAL not available.";
  return nullptr;
#endif
}

llvm::Value* CodeGenerator::codegen(const Analyzer::MLPredictExpr* expr,
                                    const CompilationOptions& co) {
  const auto& model_expr = expr->get_model_value();
  CHECK(model_expr);
  auto model_constant_expr = dynamic_cast<const Analyzer::Constant*>(model_expr);
  CHECK(model_constant_expr);
  const auto model_datum = model_constant_expr->get_constval();
  const auto model_name_ptr = model_datum.stringval;
  CHECK(model_name_ptr);
  const auto model_name = *model_name_ptr;
  const auto model_type = model_types_.getModel(model_name);

  switch (model_type) {
    case ModelType::LINEAR_REG: {
      return codegenLinRegPredict(expr, model_name, co);
    }
    case ModelType::RANDOM_FOREST_REG: {
      return codegenRandForestRegPredict(expr, model_name, co);
    }
    default: {
      throw std::runtime_error("Unsupported model type.");
    }
  }
}
