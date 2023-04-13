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

#include "AbstractMLModel.h"
#include "MLModelMetadata.h"
#include "QueryEngine/DecisionTreeEntry.h"

#include <map>
#include <memory>
#include <stack>
#include <vector>

#ifndef __CUDACC__

#ifdef HAVE_ONEDAL
#include "daal.h"
#endif

class MLModelMap {
 public:
  void addModel(const std::string& model_name, std::shared_ptr<AbstractMLModel> model) {
    const auto upper_model_name = to_upper(model_name);
    std::lock_guard<std::shared_mutex> model_map_write_lock(model_map_mutex_);
    model_map_[upper_model_name] = model;
  }

  bool modelExists(const std::string& model_name) const {
    const auto upper_model_name = to_upper(model_name);
    std::shared_lock<std::shared_mutex> model_map_read_lock(model_map_mutex_);
    auto model_map_itr = model_map_.find(upper_model_name);
    return model_map_itr != model_map_.end();
  }

  std::shared_ptr<AbstractMLModel> getModel(const std::string& model_name) const {
    const auto upper_model_name = to_upper(model_name);
    std::shared_lock<std::shared_mutex> model_map_read_lock(model_map_mutex_);
    auto model_map_itr = model_map_.find(upper_model_name);
    if (model_map_itr != model_map_.end()) {
      return model_map_itr->second;
    }
    const std::string error_str = "Model '" + upper_model_name + "' does not exist.";
    throw std::runtime_error(error_str);
  }

  void deleteModel(const std::string& model_name) {
    const auto upper_model_name = to_upper(model_name);
    std::lock_guard<std::shared_mutex> model_map_write_lock(model_map_mutex_);
    auto const model_it = model_map_.find(upper_model_name);
    if (model_it == model_map_.end()) {
      std::ostringstream error_oss;
      error_oss << "Cannot erase model " << upper_model_name
                << ". No model by that name was found.";
      throw std::runtime_error(error_oss.str());
    }
    model_map_.erase(model_it);
  }

  std::vector<std::string> getModelNames() const {
    std::shared_lock<std::shared_mutex> model_map_read_lock(model_map_mutex_);
    std::vector<std::string> model_names;
    model_names.reserve(model_map_.size());
    for (auto const& model : model_map_) {
      model_names.emplace_back(model.first);
    }
    return model_names;
  }
  std::vector<MLModelMetadata> getModelMetadata() const {
    std::shared_lock<std::shared_mutex> model_map_read_lock(model_map_mutex_);
    std::vector<MLModelMetadata> model_metadata;
    model_metadata.reserve(model_map_.size());
    for (auto const& model : model_map_) {
      model_metadata.emplace_back(MLModelMetadata(
          model.first,
          model.second->getModelTypeString(),
          model.second->getNumLogicalFeatures(),
          model.second->getNumFeatures(),
          model.second->getNumCatFeatures(),
          model.second->getNumLogicalFeatures() - model.second->getNumCatFeatures(),
          model.second->getModelMetadata()));
    }
    return model_metadata;
  }

 private:
  std::map<std::string, std::shared_ptr<AbstractMLModel>> model_map_;
  mutable std::shared_mutex model_map_mutex_;
};

inline MLModelMap g_ml_models;

class LinearRegressionModel : public AbstractMLModel {
 public:
  LinearRegressionModel(const std::vector<double>& coefs,
                        const std::string& model_metadata)
      : AbstractMLModel(model_metadata), coefs_(coefs) {}

  LinearRegressionModel(const std::vector<double>& coefs,
                        const std::string& model_metadata,
                        const std::vector<std::vector<std::string>>& cat_feature_keys)
      : AbstractMLModel(model_metadata, cat_feature_keys), coefs_(coefs) {}

  virtual MLModelType getModelType() const override { return MLModelType::LINEAR_REG; }

  virtual std::string getModelTypeString() const override { return "Linear Regression"; }

  virtual int64_t getNumFeatures() const override {
    return static_cast<int64_t>(coefs_.size()) - 1;
  }

  const std::vector<double>& getCoefs() const { return coefs_; }

 private:
  std::vector<double> coefs_;
};

#ifdef HAVE_ONEDAL

using namespace daal::algorithms;
using namespace daal::data_management;

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

class AbstractTreeModel : public virtual AbstractMLModel {
 public:
  virtual MLModelType getModelType() const = 0;
  virtual std::string getModelTypeString() const = 0;
  virtual int64_t getNumFeatures() const = 0;
  virtual int64_t getNumTrees() const = 0;
  virtual void traverseDF(const int64_t tree_idx,
                          TreeModelVisitor& tree_node_visitor) const = 0;
  virtual ~AbstractTreeModel() = default;
};

class DecisionTreeRegressionModel : public virtual AbstractTreeModel {
 public:
  DecisionTreeRegressionModel(decision_tree::regression::interface1::ModelPtr& model_ptr,
                              const std::string& model_metadata)
      : AbstractMLModel(model_metadata), model_ptr_(model_ptr) {}
  DecisionTreeRegressionModel(
      decision_tree::regression::interface1::ModelPtr& model_ptr,
      const std::string& model_metadata,
      const std::vector<std::vector<std::string>>& cat_feature_keys)
      : AbstractMLModel(model_metadata, cat_feature_keys), model_ptr_(model_ptr) {}

  virtual MLModelType getModelType() const override {
    return MLModelType::DECISION_TREE_REG;
  }

  virtual std::string getModelTypeString() const override {
    return "Decision Tree Regression";
  }

  virtual int64_t getNumFeatures() const override {
    return model_ptr_->getNumberOfFeatures();
  }
  virtual int64_t getNumTrees() const override { return 1; }
  virtual void traverseDF(const int64_t tree_idx,
                          TreeModelVisitor& tree_node_visitor) const override {
    CHECK_EQ(tree_idx, 0);
    model_ptr_->traverseDF(tree_node_visitor);
  }
  const decision_tree::regression::interface1::ModelPtr getModelPtr() const {
    return model_ptr_;
  }

 private:
  decision_tree::regression::interface1::ModelPtr model_ptr_;
};

class GbtRegressionModel : public virtual AbstractTreeModel {
 public:
  GbtRegressionModel(gbt::regression::interface1::ModelPtr& model_ptr,
                     const std::string& model_metadata)
      : AbstractMLModel(model_metadata), model_ptr_(model_ptr) {}

  GbtRegressionModel(gbt::regression::interface1::ModelPtr& model_ptr,
                     const std::string& model_metadata,
                     const std::vector<std::vector<std::string>>& cat_feature_keys)
      : AbstractMLModel(model_metadata, cat_feature_keys), model_ptr_(model_ptr) {}

  virtual MLModelType getModelType() const override { return MLModelType::GBT_REG; }

  virtual std::string getModelTypeString() const override {
    return "Gradient Boosted Trees Regression";
  }

  virtual int64_t getNumFeatures() const override {
    return model_ptr_->getNumberOfFeatures();
  }
  virtual int64_t getNumTrees() const override { return model_ptr_->getNumberOfTrees(); }
  virtual void traverseDF(const int64_t tree_idx,
                          TreeModelVisitor& tree_node_visitor) const override {
    model_ptr_->traverseDF(tree_idx, tree_node_visitor);
  }
  const gbt::regression::interface1::ModelPtr getModelPtr() const { return model_ptr_; }

 private:
  gbt::regression::interface1::ModelPtr model_ptr_;
};

class RandomForestRegressionModel : public virtual AbstractTreeModel {
 public:
  RandomForestRegressionModel(
      decision_forest::regression::interface1::ModelPtr& model_ptr,
      const std::string& model_metadata,
      const std::vector<double>& variable_importance,
      const double out_of_bag_error)
      : AbstractMLModel(model_metadata)
      , model_ptr_(model_ptr)
      , variable_importance_(variable_importance)
      , out_of_bag_error_(out_of_bag_error) {}

  RandomForestRegressionModel(
      decision_forest::regression::interface1::ModelPtr& model_ptr,
      const std::string& model_metadata,
      const std::vector<std::vector<std::string>>& cat_feature_keys,
      const std::vector<double>& variable_importance,
      const double out_of_bag_error)
      : AbstractMLModel(model_metadata, cat_feature_keys)
      , model_ptr_(model_ptr)
      , variable_importance_(variable_importance)
      , out_of_bag_error_(out_of_bag_error) {}

  virtual MLModelType getModelType() const override {
    return MLModelType::RANDOM_FOREST_REG;
  }

  virtual std::string getModelTypeString() const override {
    return "Random Forest Regression";
  }
  virtual int64_t getNumFeatures() const override {
    return model_ptr_->getNumberOfFeatures();
  }
  virtual int64_t getNumTrees() const override { return model_ptr_->getNumberOfTrees(); }
  virtual void traverseDF(const int64_t tree_idx,
                          TreeModelVisitor& tree_node_visitor) const override {
    model_ptr_->traverseDF(tree_idx, tree_node_visitor);
  }

  const decision_forest::regression::interface1::ModelPtr getModelPtr() const {
    return model_ptr_;
  }

  const std::vector<double>& getVariableImportanceScores() const {
    return variable_importance_;
  }

  const double getOutOfBagError() const { return out_of_bag_error_; }

 private:
  decision_forest::regression::interface1::ModelPtr model_ptr_;
  std::vector<double> variable_importance_;
  double out_of_bag_error_;
};

#endif  // #ifdef HAVE_ONEDAL

#endif  // #ifndef __CUDACC__