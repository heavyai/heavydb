/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef QUERYENGINE_RELALGABSTRACTINTERPRETER_H
#define QUERYENGINE_RELALGABSTRACTINTERPRETER_H

#include "TargetMetaInfo.h"
#include "TypePunning.h"
#include "../Catalog/Catalog.h"

#include <boost/variant.hpp>
#include <boost/make_unique.hpp>
#include <rapidjson/document.h>

#include <memory>
#include <unordered_map>

class Rex {
 public:
  virtual std::string toString() const = 0;

  virtual ~Rex() {}
};

class RexScalar : public Rex {};

// For internal use of the abstract interpreter only. The result after abstract
// interpretation will not have any references to 'RexAbstractInput' objects.
class RexAbstractInput : public RexScalar {
 public:
  RexAbstractInput(const unsigned in_index) : in_index_(in_index) {}

  unsigned getIndex() const { return in_index_; }

  void setIndex(const unsigned in_index) const { in_index_ = in_index; }

  std::string toString() const override { return "(RexAbstractInput " + std::to_string(in_index_) + ")"; }

 private:
  mutable unsigned in_index_;
};

class RexLiteral : public RexScalar {
 public:
  RexLiteral(const int64_t val,
             const SQLTypes type,
             const SQLTypes target_type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        target_type_(target_type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK(type == kDECIMAL || type == kINTERVAL_DAY_TIME || type == kINTERVAL_YEAR_MONTH || IS_TIME(type));
  }

  RexLiteral(const double val,
             const SQLTypes type,
             const SQLTypes target_type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        target_type_(target_type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kDOUBLE, type);
  }

  RexLiteral(const std::string& val,
             const SQLTypes type,
             const SQLTypes target_type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        target_type_(target_type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kTEXT, type);
  }

  RexLiteral(const bool val,
             const SQLTypes type,
             const SQLTypes target_type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        target_type_(target_type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kBOOLEAN, type);
  }

  RexLiteral(const SQLTypes target_type)
      : literal_(nullptr),
        type_(kNULLT),
        target_type_(target_type),
        scale_(0),
        precision_(0),
        type_scale_(0),
        type_precision_(0) {}

  template <class T>
  T getVal() const {
    const auto ptr = boost::get<T>(&literal_);
    CHECK(ptr);
    return *ptr;
  }

  SQLTypes getType() const { return type_; }

  SQLTypes getTargetType() const { return target_type_; }

  unsigned getScale() const { return scale_; }

  unsigned getPrecision() const { return precision_; }

  unsigned getTypeScale() const { return type_scale_; }

  unsigned getTypePrecision() const { return type_precision_; }

  std::string toString() const override { return "(RexLiteral " + boost::lexical_cast<std::string>(literal_) + ")"; }

  std::unique_ptr<RexLiteral> deepCopy() const {
    switch (literal_.which()) {
      case 0: {
        int64_t val = getVal<int64_t>();
        return boost::make_unique<RexLiteral>(
            val, type_, target_type_, scale_, precision_, type_scale_, type_precision_);
      }
      case 1: {
        double val = getVal<double>();
        return boost::make_unique<RexLiteral>(
            val, type_, target_type_, scale_, precision_, type_scale_, type_precision_);
      }
      case 2: {
        auto val = getVal<std::string>();
        return boost::make_unique<RexLiteral>(
            val, type_, target_type_, scale_, precision_, type_scale_, type_precision_);
      }
      case 3: {
        bool val = getVal<bool>();
        return boost::make_unique<RexLiteral>(
            val, type_, target_type_, scale_, precision_, type_scale_, type_precision_);
      }
      case 4: {
        return boost::make_unique<RexLiteral>(target_type_);
      }
      default:
        CHECK(false);
    }
    return nullptr;
  }

 private:
  const boost::variant<int64_t, double, std::string, bool, void*> literal_;
  const SQLTypes type_;
  const SQLTypes target_type_;
  const unsigned scale_;
  const unsigned precision_;
  const unsigned type_scale_;
  const unsigned type_precision_;
};

class RexOperator : public RexScalar {
 public:
  RexOperator(const SQLOps op, std::vector<std::unique_ptr<const RexScalar>>& operands, const SQLTypeInfo& type)
      : op_(op), operands_(std::move(operands)), type_(type) {}

  virtual std::unique_ptr<const RexOperator> getDisambiguated(
      std::vector<std::unique_ptr<const RexScalar>>& operands) const {
    return std::unique_ptr<const RexOperator>(new RexOperator(op_, operands, type_));
  }

  size_t size() const { return operands_.size(); }

  const RexScalar* getOperand(const size_t idx) const {
    CHECK(idx < operands_.size());
    return operands_[idx].get();
  }

  const RexScalar* getOperandAndRelease(const size_t idx) const {
    CHECK(idx < operands_.size());
    return operands_[idx].release();
  }

  SQLOps getOperator() const { return op_; }

  const SQLTypeInfo& getType() const { return type_; }

  std::string toString() const override {
    std::string result = "(RexOperator " + std::to_string(op_);
    for (const auto& operand : operands_) {
      result += " " + operand->toString();
    }
    return result + ")";
  };

 protected:
  const SQLOps op_;
  mutable std::vector<std::unique_ptr<const RexScalar>> operands_;
  const SQLTypeInfo type_;
};

class RelAlgNode;

class ExecutionResult;

class RexSubQuery : public RexScalar {
 public:
  RexSubQuery(const std::shared_ptr<const RelAlgNode> ra) : type_(SQLTypeInfo(kNULLT, false)), ra_(ra) {}

  RexSubQuery(const RexSubQuery&) = delete;

  RexSubQuery& operator=(const RexSubQuery&) = delete;

  RexSubQuery(RexSubQuery&&) = delete;

  RexSubQuery& operator=(RexSubQuery&&) = delete;

  const SQLTypeInfo& getType() const {
    CHECK_NE(kNULLT, type_.get_type());
    return type_;
  }

  std::shared_ptr<const ExecutionResult> getExecutionResult() const {
    CHECK(result_);
    return result_;
  }

  const RelAlgNode* getRelAlg() const { return ra_.get(); }

  std::string toString() const override {
    return "(RexSubQuery " + std::to_string(reinterpret_cast<const uint64_t>(this)) + ")";
  }

  std::unique_ptr<RexSubQuery> deepCopy() const { throw std::runtime_error("Sub-query not supported in this context"); }

  void setExecutionResult(const std::shared_ptr<const ExecutionResult> result);

 private:
  SQLTypeInfo type_;
  std::shared_ptr<const ExecutionResult> result_;
  const std::shared_ptr<const RelAlgNode> ra_;
};

// The actual input node understood by the Executor.
// The in_index_ is relative to the output of node_.
class RexInput : public RexAbstractInput {
 public:
  RexInput(const RelAlgNode* node, const unsigned in_index) : RexAbstractInput(in_index), node_(node) {}

  const RelAlgNode* getSourceNode() const { return node_; }

  // This isn't great, but we need it for coalescing nodes to Compound since
  // RexInput in descendents need to be rebound to the newly created Compound.
  // Maybe create a fresh RA tree with the required changes after each coalescing?
  void setSourceNode(const RelAlgNode* node) const { node_ = node; }

  bool operator==(const RexInput& that) const {
    return getSourceNode() == that.getSourceNode() && getIndex() == that.getIndex();
  }

  std::string toString() const override {
    return "(RexInput " + std::to_string(getIndex()) + " " + std::to_string(reinterpret_cast<const uint64_t>(node_)) +
           ")";
  }

  std::unique_ptr<RexInput> deepCopy() const { return boost::make_unique<RexInput>(node_, getIndex()); }

 private:
  mutable const RelAlgNode* node_;
};

namespace std {

template <>
struct hash<RexInput> {
  size_t operator()(const RexInput& rex_in) const {
    auto addr = rex_in.getSourceNode();
    return *reinterpret_cast<const size_t*>(may_alias_ptr(&addr)) ^ rex_in.getIndex();
  }
};

}  // std

// Not a real node created by Calcite. Created by us because CaseExpr is a node in our Analyzer.
class RexCase : public RexScalar {
 public:
  RexCase(std::vector<std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>>& expr_pair_list,
          std::unique_ptr<const RexScalar>& else_expr)
      : expr_pair_list_(std::move(expr_pair_list)), else_expr_(std::move(else_expr)) {}

  size_t branchCount() const { return expr_pair_list_.size(); }

  const RexScalar* getWhen(const size_t idx) const {
    CHECK(idx < expr_pair_list_.size());
    return expr_pair_list_[idx].first.get();
  }

  const RexScalar* getThen(const size_t idx) const {
    CHECK(idx < expr_pair_list_.size());
    return expr_pair_list_[idx].second.get();
  }

  const RexScalar* getElse() const { return else_expr_.get(); }

  std::string toString() const override {
    std::string ret = "(RexCase";
    for (const auto& expr_pair : expr_pair_list_) {
      ret += " " + expr_pair.first->toString() + " -> " + expr_pair.second->toString();
    }
    if (else_expr_) {
      ret += " else " + else_expr_->toString();
    }
    ret += ")";
    return ret;
  }

 private:
  std::vector<std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>> expr_pair_list_;
  std::unique_ptr<const RexScalar> else_expr_;
};

class RexFunctionOperator : public RexOperator {
 public:
  RexFunctionOperator(const std::string& name,
                      std::vector<std::unique_ptr<const RexScalar>>& operands,
                      const SQLTypeInfo& ti)
      : RexOperator(kFUNCTION, operands, ti), name_(name) {}

  std::unique_ptr<const RexOperator> getDisambiguated(
      std::vector<std::unique_ptr<const RexScalar>>& operands) const override {
    return std::unique_ptr<const RexOperator>(new RexFunctionOperator(name_, operands, getType()));
  }

  const std::string& getName() const { return name_; }

  std::string toString() const override {
    auto result = "(RexFunctionOperator " + name_;
    for (const auto& operand : operands_) {
      result += (" " + operand->toString());
    }
    return result + ")";
  }

 private:
  const std::string name_;
};

// Not a real node created by Calcite. Created by us because targets of a query
// should reference the group by expressions instead of creating completely new one.
class RexRef : public RexScalar {
 public:
  RexRef(const size_t index) : index_(index) {}

  size_t getIndex() const { return index_; }

  virtual std::string toString() const { return "(RexRef " + std::to_string(index_) + ")"; }

  std::unique_ptr<RexRef> deepCopy() const { return boost::make_unique<RexRef>(index_); }

 private:
  const size_t index_;
};

class RexAgg : public Rex {
 public:
  RexAgg(const SQLAgg agg, const bool distinct, const SQLTypeInfo& type, const std::vector<size_t>& operands)
      : agg_(agg), distinct_(distinct), type_(type), operands_(operands) {}

  std::string toString() const override {
    auto result = "(RexAgg " + std::to_string(agg_) + " " + std::to_string(distinct_) + " " + type_.get_type_name() +
                  " " + type_.get_compression_name();
    for (auto operand : operands_) {
      result += " " + std::to_string(operand);
    }
    return result + ")";
  }

  SQLAgg getKind() const { return agg_; }

  bool isDistinct() const { return distinct_; }

  size_t size() const { return operands_.size(); }

  size_t getOperand(size_t idx) const { return operands_[idx]; }

  const SQLTypeInfo& getType() const { return type_; }

  std::unique_ptr<RexAgg> deepCopy() const { return boost::make_unique<RexAgg>(agg_, distinct_, type_, operands_); }

 private:
  const SQLAgg agg_;
  const bool distinct_;
  const SQLTypeInfo type_;
  const std::vector<size_t> operands_;
};

class RelAlgNode {
 public:
  RelAlgNode() : id_(crt_id_++), context_data_(nullptr), is_nop_(false) {}

  virtual ~RelAlgNode() {}

  void setContextData(const void* context_data) const {
    CHECK(!context_data_);
    context_data_ = context_data;
  }

  void setOutputMetainfo(const std::vector<TargetMetaInfo>& targets_metainfo) const {
    targets_metainfo_ = targets_metainfo;
  }

  const std::vector<TargetMetaInfo>& getOutputMetainfo() const { return targets_metainfo_; }

  unsigned getId() const { return id_; }

  const void* getContextData() const {
    CHECK(context_data_);
    return context_data_;
  }

  const size_t inputCount() const { return inputs_.size(); }

  const RelAlgNode* getInput(const size_t idx) const {
    CHECK(idx < inputs_.size());
    return inputs_[idx].get();
  }

  std::shared_ptr<const RelAlgNode> getAndOwnInput(const size_t idx) const {
    CHECK(idx < inputs_.size());
    return inputs_[idx];
  }

  void addManagedInput(std::shared_ptr<const RelAlgNode> input) { inputs_.push_back(input); }

  bool hasInput(const RelAlgNode* needle) const {
    for (auto& input_ptr : inputs_) {
      if (input_ptr.get() == needle) {
        return true;
      }
    }
    return false;
  }

  virtual void replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) {
    for (auto& input_ptr : inputs_) {
      if (input_ptr == old_input) {
        input_ptr = input;
        break;
      }
    }
  }

  bool isNop() const { return is_nop_; }

  void markAsNop() { is_nop_ = true; }

  virtual std::string toString() const = 0;

  virtual size_t size() const = 0;

  virtual std::shared_ptr<RelAlgNode> deepCopy() const = 0;

  static void resetRelAlgFirstId() noexcept;

 protected:
  std::vector<std::shared_ptr<const RelAlgNode>> inputs_;
  const unsigned id_;

 private:
  mutable const void* context_data_;
  bool is_nop_;
  mutable std::vector<TargetMetaInfo> targets_metainfo_;
  static thread_local unsigned crt_id_;
};

class RelScan : public RelAlgNode {
 public:
  RelScan(const TableDescriptor* td, const std::vector<std::string>& field_names)
      : td_(td), field_names_(field_names) {}

  size_t size() const override { return field_names_.size(); }

  const TableDescriptor* getTableDescriptor() const { return td_; }

  const std::vector<std::string>& getFieldNames() const { return field_names_; }

  const std::string getFieldName(const size_t i) const { return field_names_[i]; }

  std::string toString() const override {
    return "(RelScan<" + std::to_string(reinterpret_cast<uint64_t>(this)) + "> " + td_->tableName + ")";
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    CHECK(false);
    return nullptr;
  };

 private:
  const TableDescriptor* td_;
  const std::vector<std::string> field_names_;
};

class RelProject : public RelAlgNode {
 public:
  // Takes memory ownership of the expressions.
  RelProject(std::vector<std::unique_ptr<const RexScalar>>& scalar_exprs,
             const std::vector<std::string>& fields,
             std::shared_ptr<const RelAlgNode> input)
      : scalar_exprs_(std::move(scalar_exprs)), fields_(fields) {
    inputs_.push_back(input);
  }

  void setExpressions(std::vector<std::unique_ptr<const RexScalar>>& exprs) { scalar_exprs_ = std::move(exprs); }

  // True iff all the projected expressions are inputs. If true,
  // this node can be elided and merged into the previous node
  // since it's just a subset and / or permutation of its outputs.
  bool isSimple() const {
    for (const auto& expr : scalar_exprs_) {
      if (!dynamic_cast<const RexInput*>(expr.get())) {
        return false;
      }
    }
    return true;
  }

  bool isIdentity() const;

  bool isRenaming() const;

  size_t size() const override { return scalar_exprs_.size(); }

  const RexScalar* getProjectAt(const size_t idx) const {
    CHECK(idx < scalar_exprs_.size());
    return scalar_exprs_[idx].get();
  }

  const RexScalar* getProjectAtAndRelease(const size_t idx) const {
    CHECK(idx < scalar_exprs_.size());
    return scalar_exprs_[idx].release();
  }

  std::vector<std::unique_ptr<const RexScalar>> getExpressionsAndRelease() { return std::move(scalar_exprs_); }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string>& fields) { fields_ = std::move(fields); }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) override;

  std::string toString() const override {
    std::string result = "(RelProject<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">";
    for (const auto& scalar_expr : scalar_exprs_) {
      result += " " + scalar_expr->toString();
    }
    return result + ")";
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override;

 private:
  mutable std::vector<std::unique_ptr<const RexScalar>> scalar_exprs_;
  std::vector<std::string> fields_;
};

class RelAggregate : public RelAlgNode {
 public:
  // Takes ownership of the aggregate expressions.
  RelAggregate(const size_t groupby_count,
               std::vector<std::unique_ptr<const RexAgg>>& agg_exprs,
               const std::vector<std::string>& fields,
               std::shared_ptr<const RelAlgNode> input)
      : groupby_count_(groupby_count), agg_exprs_(std::move(agg_exprs)), fields_(fields) {
    inputs_.push_back(input);
  }

  size_t size() const override { return groupby_count_ + agg_exprs_.size(); }

  const size_t getGroupByCount() const { return groupby_count_; }

  const size_t getAggExprsCount() const { return agg_exprs_.size(); }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string>& new_fields) { fields_ = std::move(new_fields); }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  std::vector<const RexAgg*> getAggregatesAndRelease() {
    std::vector<const RexAgg*> result;
    for (auto& agg_expr : agg_exprs_) {
      result.push_back(agg_expr.release());
    }
    return result;
  }

  std::vector<std::unique_ptr<const RexAgg>> getAggExprsAndRelease() { return std::move(agg_exprs_); }

  const std::vector<std::unique_ptr<const RexAgg>>& getAggExprs() const { return agg_exprs_; }

  void setAggExprs(std::vector<std::unique_ptr<const RexAgg>>& agg_exprs) { agg_exprs_ = std::move(agg_exprs); }

  std::string toString() const override {
    std::string result = "(RelAggregate<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(groups: [";
    for (size_t group_index = 0; group_index < groupby_count_; ++group_index) {
      result += " " + std::to_string(group_index);
    }
    result += " ] aggs: [";
    for (const auto& agg_expr : agg_exprs_) {
      result += " " + agg_expr->toString();
    }
    return result + " ])";
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override;

 private:
  const size_t groupby_count_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  std::vector<std::string> fields_;
};

class RelJoin : public RelAlgNode {
 public:
  RelJoin(std::shared_ptr<const RelAlgNode> lhs,
          std::shared_ptr<const RelAlgNode> rhs,
          std::unique_ptr<const RexScalar>& condition,
          const JoinType join_type)
      : condition_(std::move(condition)), join_type_(join_type) {
    inputs_.push_back(lhs);
    inputs_.push_back(rhs);
  }

  JoinType getJoinType() const { return join_type_; }

  const RexScalar* getCondition() const { return condition_.get(); }

  const RexScalar* getAndReleaseCondition() const { return condition_.release(); }

  void setCondition(std::unique_ptr<const RexScalar>& condition) {
    CHECK(condition);
    condition_ = std::move(condition);
  }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) override;

  std::string toString() const override {
    std::string result = "(RelJoin<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
    result += condition_ ? condition_->toString() : "null";
    result += " " + std::to_string(static_cast<int>(join_type_));
    return result + ")";
  }

  size_t size() const override { return inputs_[0]->size() + inputs_[1]->size(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override;

 private:
  mutable std::unique_ptr<const RexScalar> condition_;
  const JoinType join_type_;
};

class RelMultiJoin : public RelAlgNode {
 public:
  RelMultiJoin(const std::vector<std::shared_ptr<RelJoin>>& seq) {
    for (auto join : seq) {
      append(join);
    }
  }

  JoinType getJoinType() const {
    return sequence_.empty() || sequence_.back() == nullptr ? JoinType::INVALID : sequence_.back()->getJoinType();
  }

  std::string toString() const override {
    std::string result = "RelMultiJoin<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
    for (auto join : sequence_) {
      result += "(RelJoin<" + std::to_string(reinterpret_cast<uint64_t>(join.get())) + ">(";
      result += join->getCondition() ? join->getCondition()->toString() : "null";
      result += " " + std::to_string(static_cast<int>(join->getJoinType()));
      result += ")";
      if (join != sequence_.back()) {
        result += ", ";
      }
    }
    result += ")";
    return result;
  }

  size_t size() const override {
    size_t sz = 0;
    for (auto input : inputs_) {
      sz += input->size();
    }
    return sz;
  }

  size_t joinCount() const { return sequence_.size(); }

  const RelJoin* getJoinAt(const size_t i) const {
    CHECK_LT(i, sequence_.size());
    return sequence_[i].get();
  }

  std::shared_ptr<RelJoin> getAndOwnJoinAt(const size_t i) const {
    CHECK_LT(i, sequence_.size());
    return sequence_[i];
  }

  const std::vector<std::unique_ptr<const RexScalar>>& getConditions() const { return conditions_; }

  void setConditions(std::vector<std::unique_ptr<const RexScalar>>& conditions) {
    CHECK(!conditions.empty());
    conditions_ = std::move(conditions);
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    CHECK(false);
    return nullptr;
  }

 private:
  void append(std::shared_ptr<RelJoin> join);

  mutable std::vector<std::shared_ptr<RelJoin>> sequence_;
  std::vector<std::unique_ptr<const RexScalar>> conditions_;
};

class RelFilter : public RelAlgNode {
 public:
  RelFilter(std::unique_ptr<const RexScalar>& filter, std::shared_ptr<const RelAlgNode> input)
      : filter_(std::move(filter)) {
    CHECK(filter_);
    inputs_.push_back(input);
  }

  const RexScalar* getCondition() const { return filter_.get(); }

  const RexScalar* getAndReleaseCondition() { return filter_.release(); }

  void setCondition(std::unique_ptr<const RexScalar>& condition) {
    CHECK(condition);
    filter_ = std::move(condition);
  }

  size_t size() const override { return inputs_[0]->size(); }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) override;

  std::string toString() const override {
    std::string result = "(RelFilter<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
    result += filter_->toString();
    return result + ")";
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override;

 private:
  std::unique_ptr<const RexScalar> filter_;
};

// Synthetic node to assist execution of left-deep join relational algebra.
class RelLeftDeepInnerJoin : public RelAlgNode {
 public:
  RelLeftDeepInnerJoin(const std::shared_ptr<RelFilter>& filter,
                       std::vector<std::shared_ptr<const RelAlgNode>> inputs,
                       std::vector<std::shared_ptr<const RelJoin>>& original_joins);

  const RexScalar* getInnerCondition() const;

  const RexScalar* getOuterCondition(const size_t nesting_level) const;

  std::string toString() const override;

  size_t size() const override;

  std::shared_ptr<RelAlgNode> deepCopy() const override;

  bool coversOriginalNode(const RelAlgNode* node) const;

 private:
  std::unique_ptr<const RexScalar> condition_;
  std::vector<std::unique_ptr<const RexScalar>> outer_conditions_per_level_;
  const std::shared_ptr<RelFilter> original_filter_;
  const std::vector<std::shared_ptr<const RelJoin>> original_joins_;
};

// The 'RelCompound' node combines filter and on the fly aggregate computation.
// It's the result of combining a sequence of 'RelFilter' (optional), 'RelProject',
// 'RelAggregate' (optional) and a simple 'RelProject' (optional) into a single node
// which can be efficiently executed with no intermediate buffers.
class RelCompound : public RelAlgNode {
 public:
  // 'target_exprs_' are either scalar expressions owned by 'scalar_sources_'
  // or aggregate expressions owned by 'agg_exprs_', with the arguments
  // owned by 'scalar_sources_'.
  RelCompound(std::unique_ptr<const RexScalar>& filter_expr,
              const std::vector<const Rex*>& target_exprs,
              const size_t groupby_count,
              const std::vector<const RexAgg*>& agg_exprs,
              const std::vector<std::string>& fields,
              std::vector<std::unique_ptr<const RexScalar>>& scalar_sources,
              const bool is_agg)
      : filter_expr_(std::move(filter_expr)),
        target_exprs_(target_exprs),
        groupby_count_(groupby_count),
        fields_(fields),
        is_agg_(is_agg),
        scalar_sources_(std::move(scalar_sources)) {
    CHECK_EQ(fields.size(), target_exprs.size());
    for (auto agg_expr : agg_exprs) {
      agg_exprs_.emplace_back(agg_expr);
    }
  }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) override;

  size_t size() const override { return target_exprs_.size(); }

  const RexScalar* getFilterExpr() const { return filter_expr_.get(); }

  void setFilterExpr(std::unique_ptr<const RexScalar>& new_expr) { filter_expr_ = std::move(new_expr); }

  const Rex* getTargetExpr(const size_t i) const { return target_exprs_[i]; }

  const std::vector<std::string>& getFields() const { return fields_; }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  const size_t getScalarSourcesSize() const { return scalar_sources_.size(); }

  const RexScalar* getScalarSource(const size_t i) const { return scalar_sources_[i].get(); }

  void setScalarSources(std::vector<std::unique_ptr<const RexScalar>>& new_sources) {
    CHECK_EQ(new_sources.size(), scalar_sources_.size());
    scalar_sources_ = std::move(new_sources);
  }

  const size_t getGroupByCount() const { return groupby_count_; }

  bool isAggregate() const { return is_agg_; }

  std::string toString() const override {
    std::string result = "(RelCompound<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
    result += (filter_expr_ ? filter_expr_->toString() : "null") + " ";
    for (const auto target_expr : target_exprs_) {
      result += target_expr->toString() + " ";
    }
    result += "groups: [";
    for (size_t group_index = 0; group_index < groupby_count_; ++group_index) {
      result += " " + std::to_string(group_index);
    }
    result += " ] sources: [";
    for (const auto& scalar_source : scalar_sources_) {
      result += " " + scalar_source->toString();
    }
    return result + " ])";
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override;

 private:
  std::unique_ptr<const RexScalar> filter_expr_;
  const std::vector<const Rex*> target_exprs_;
  const size_t groupby_count_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  const std::vector<std::string> fields_;
  const bool is_agg_;
  std::vector<std::unique_ptr<const RexScalar>>
      scalar_sources_;  // building blocks for group_indices_ and agg_exprs_; not actually projected, just owned
};

enum class SortDirection { Ascending, Descending };

enum class NullSortedPosition { First, Last };

class SortField {
 public:
  SortField(const size_t field, const SortDirection sort_dir, const NullSortedPosition nulls_pos)
      : field_(field), sort_dir_(sort_dir), nulls_pos_(nulls_pos) {}

  bool operator==(const SortField& that) const {
    return field_ == that.field_ && sort_dir_ == that.sort_dir_ && nulls_pos_ == that.nulls_pos_;
  }

  size_t getField() const { return field_; }

  SortDirection getSortDir() const { return sort_dir_; }

  NullSortedPosition getNullsPosition() const { return nulls_pos_; }

  std::string toString() const {
    return "(" + std::to_string(field_) + " " + (sort_dir_ == SortDirection::Ascending ? "asc" : "desc") + " " +
           (nulls_pos_ == NullSortedPosition::First ? "nulls_first" : "nulls_last") + ")";
  }

 private:
  const size_t field_;
  const SortDirection sort_dir_;
  const NullSortedPosition nulls_pos_;
};

class RelSort : public RelAlgNode {
 public:
  RelSort(const std::vector<SortField>& collation,
          const size_t limit,
          const size_t offset,
          std::shared_ptr<const RelAlgNode> input)
      : collation_(collation), limit_(limit), offset_(offset) {
    inputs_.push_back(input);
  }

  bool operator==(const RelSort& that) const {
    return limit_ == that.limit_ && offset_ == that.offset_ && hasEquivCollationOf(that);
  }

  size_t collationCount() const { return collation_.size(); }

  SortField getCollation(const size_t i) const {
    CHECK_LT(i, collation_.size());
    return collation_[i];
  }

  void setCollation(std::vector<SortField>&& collation) { collation_ = std::move(collation); }

  size_t getLimit() const { return limit_; }

  size_t getOffset() const { return offset_; }

  std::string toString() const override {
    std::string result = "(RelSort<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
    result += "limit: " + std::to_string(limit_) + " ";
    result += "offset: " + std::to_string(offset_) + " ";
    result += "collation: [ ";
    for (const auto& sort_field : collation_) {
      result += sort_field.toString() + " ";
    }
    result += "]";
    return result + ")";
  }

  size_t size() const override { return inputs_[0]->size(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override;

 private:
  std::vector<SortField> collation_;
  const size_t limit_;
  const size_t offset_;

  bool hasEquivCollationOf(const RelSort& that) const;
};

class RelLogicalValues : public RelAlgNode {
 public:
  RelLogicalValues(const std::vector<TargetMetaInfo>& tuple_type) : tuple_type_(tuple_type) {}

  const std::vector<TargetMetaInfo> getTupleType() const { return tuple_type_; }

  std::string toString() const override {
    // TODO
    return "(RelLogicalValues)";
  }

  size_t size() const override { return tuple_type_.size(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override { return std::make_shared<RelLogicalValues>(tuple_type_); }

 private:
  const std::vector<TargetMetaInfo> tuple_type_;
};

class QueryNotSupported : public std::runtime_error {
 public:
  QueryNotSupported(const std::string& reason) : std::runtime_error(reason) {}
};

class RelAlgExecutor;

std::shared_ptr<const RelAlgNode> deserialize_ra_dag(const std::string& query_ra,
                                                     const Catalog_Namespace::Catalog& cat,
                                                     RelAlgExecutor* ra_executor);

std::string tree_string(const RelAlgNode*, const size_t indent = 0);

typedef std::vector<RexInput> RANodeOutput;

RANodeOutput get_node_output(const RelAlgNode* ra_node);

#endif  // QUERYENGINE_RELALGABSTRACTINTERPRETER_H
