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

/** Notes:
 *  * All copy constuctors of child classes of RelAlgNode are deep copies,
 *    and are invoked by the the RelAlgNode::deepCopy() overloads.
 */

#pragma once

#include <atomic>
#include <iterator>
#include <memory>
#include <optional>
#include <unordered_map>

#include <rapidjson/document.h>
#include <boost/core/noncopyable.hpp>
#include <boost/functional/hash.hpp>

#include "Catalog/Catalog.h"
#include "QueryEngine/QueryHint.h"
#include "QueryEngine/Rendering/RenderInfo.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "QueryEngine/TypePunning.h"
#include "QueryHint.h"
#include "Shared/toString.h"
#include "Utils/FsiUtils.h"

class RelAggregate;
class RelAlgNode;
class RelCompound;
class RelFilter;
class RelJoin;
class RelLeftDeepInnerJoin;
class RelLogicalUnion;
class RelLogicalValues;
class RelModify;
class RelProject;
class RelScan;
class RelSort;
class RelTableFunction;
class RelTranslatedJoin;
class Rex;
class RexAbstractInput;
class RexAgg;
class RexCase;
class RexFunctionOperator;
class RexInput;
class RexLiteral;
class RexOperator;
class RexRef;
class RexScalar;
class RexSubQuery;
class RexWindowFunctionOperator;
class SortField;

using RelAlgInputs = std::vector<std::shared_ptr<const RelAlgNode>>;
using ColumnNameList = std::vector<std::string>;

struct RelRexToStringConfig {
  bool skip_input_nodes{true};
  bool attributes_only{false};

  static RelRexToStringConfig defaults() { return RelRexToStringConfig(); }
};

class RelAlgDagNode {
 public:
  class Visitor {
   public:
    virtual ~Visitor() {}
    virtual bool visit(RelAggregate const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelCompound const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelFilter const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelJoin const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelLeftDeepInnerJoin const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelLogicalUnion const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelLogicalValues const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelModify const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelProject const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelScan const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelSort const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelTableFunction const* n, std::string s) { return v(n, s); }
    virtual bool visit(RelTranslatedJoin const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexAbstractInput const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexCase const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexFunctionOperator const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexInput const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexLiteral const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexOperator const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexRef const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexAgg const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexSubQuery const* n, std::string s) { return v(n, s); }
    virtual bool visit(RexWindowFunctionOperator const* n, std::string s) {
      return v(n, s);
    }

   protected:
    virtual bool visitAny(RelAlgDagNode const* n, std::string s) {
      return /*recurse=*/true;
    }

   private:
    template <typename T>
    bool v(T const* n, std::string s) {
      return visitAny(static_cast<RelAlgDagNode const*>(n), s);
    }
  };

  RelAlgDagNode() : id_in_plan_tree_(std::nullopt) {}

  virtual void accept(Visitor& v, std::string name) const = 0;
  virtual void acceptChildren(Visitor& v) const = 0;
  virtual std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const = 0;

  virtual size_t getStepNumber() const { return step_; }
  virtual void setStepNumber(size_t step) const { step_ = step; }

  std::optional<size_t> getIdInPlanTree() const { return id_in_plan_tree_; }
  void setIdInPlanTree(size_t id) const { id_in_plan_tree_ = id; }

 protected:
  mutable size_t step_{0};
  mutable std::optional<size_t> id_in_plan_tree_;
};

class Rex : public RelAlgDagNode {
 public:
  virtual ~Rex() {}

  virtual size_t getStepNumber() const { return 0; }

  virtual size_t toHash() const = 0;

 protected:
  mutable std::optional<size_t> hash_;

  friend struct RelAlgDagSerializer;
};

class RexScalar : public Rex {};

// For internal use of the abstract interpreter only. The result after abstract
// interpretation will not have any references to 'RexAbstractInput' objects.
class RexAbstractInput : public RexScalar {
 public:
  // default constructor used for deserialization only
  RexAbstractInput() : in_index_{0} {};

  RexAbstractInput(const unsigned in_index) : in_index_(in_index) {}

  virtual void acceptChildren(Visitor& v) const override {}
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  unsigned getIndex() const { return in_index_; }

  void setIndex(const unsigned in_index) const { in_index_ = in_index; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    return cat(::typeName(this), "(", std::to_string(in_index_), ")");
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RexAbstractInput const&);

  mutable unsigned in_index_;

  friend struct RelAlgDagSerializer;
};

class RexLiteral : public RexScalar {
 public:
  // default constructor used for deserialization only
  RexLiteral()
      : literal_(), scale_{0}, precision_{0}, target_scale_{0}, target_precision_{0} {}

  RexLiteral(const int64_t val,
             const SQLTypes type,
             const SQLTypes target_type,
             const unsigned scale,
             const unsigned precision,
             const unsigned target_scale,
             const unsigned target_precision)
      : literal_(val)
      , type_(type)
      , target_type_(target_type)
      , scale_(scale)
      , precision_(precision)
      , target_scale_(target_scale)
      , target_precision_(target_precision) {
    CHECK(type == kDECIMAL || type == kINTERVAL_DAY_TIME ||
          type == kINTERVAL_YEAR_MONTH || is_datetime(type) || type == kBIGINT ||
          type == kINT);
  }

  RexLiteral(const double val,
             const SQLTypes type,
             const SQLTypes target_type,
             const unsigned scale,
             const unsigned precision,
             const unsigned target_scale,
             const unsigned target_precision)
      : literal_(val)
      , type_(type)
      , target_type_(target_type)
      , scale_(scale)
      , precision_(precision)
      , target_scale_(target_scale)
      , target_precision_(target_precision) {
    CHECK_EQ(kDOUBLE, type);
  }

  RexLiteral(const std::string& val,
             const SQLTypes type,
             const SQLTypes target_type,
             const unsigned scale,
             const unsigned precision,
             const unsigned target_scale,
             const unsigned target_precision)
      : literal_(val)
      , type_(type)
      , target_type_(target_type)
      , scale_(scale)
      , precision_(precision)
      , target_scale_(target_scale)
      , target_precision_(target_precision) {
    CHECK_EQ(kTEXT, type);
  }

  RexLiteral(const bool val,
             const SQLTypes type,
             const SQLTypes target_type,
             const unsigned scale,
             const unsigned precision,
             const unsigned target_scale,
             const unsigned target_precision)
      : literal_(val)
      , type_(type)
      , target_type_(target_type)
      , scale_(scale)
      , precision_(precision)
      , target_scale_(target_scale)
      , target_precision_(target_precision) {
    CHECK_EQ(kBOOLEAN, type);
  }

  RexLiteral(const SQLTypes target_type)
      : literal_()
      , type_(kNULLT)
      , target_type_(target_type)
      , scale_(0)
      , precision_(0)
      , target_scale_(0)
      , target_precision_(0) {}

  virtual void acceptChildren(Visitor& v) const override {}
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

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

  unsigned getTargetScale() const { return target_scale_; }

  unsigned getTargetPrecision() const { return target_precision_; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    std::ostringstream oss;
    oss << "RexLiteral(" << literal_ << " type=" << type_ << '(' << precision_ << ','
        << scale_ << ") target_type=" << target_type_ << '(' << target_precision_ << ','
        << target_scale_ << "))";
    return oss.str();
  }

  std::unique_ptr<RexLiteral> deepCopy() const {
    return std::make_unique<RexLiteral>(*this);
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RexLiteral const&);

  boost::variant<boost::blank, int64_t, double, std::string, bool> literal_;
  SQLTypes type_;
  SQLTypes target_type_;
  unsigned scale_;
  unsigned precision_;
  unsigned target_scale_;
  unsigned target_precision_;

  friend struct RelAlgDagSerializer;
};

using RexLiteralArray = std::vector<RexLiteral>;
using TupleContentsArray = std::vector<RexLiteralArray>;

class RexOperator : public RexScalar {
 public:
  // default constructor for deserialization only
  RexOperator() : op_{SQLOps::kINVALID_OP} {}

  RexOperator(const SQLOps op,
              std::vector<std::unique_ptr<const RexScalar>>& operands,
              const SQLTypeInfo& type)
      : op_(op), operands_(std::move(operands)), type_(type) {}

  virtual void acceptChildren(Visitor& v) const override {
    for (size_t i = 0; i < size(); ++i) {
      if (getOperand(i)) {
        getOperand(i)->accept(v, "operand");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

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

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      auto ret = cat(::typeName(this), "(", std::to_string(op_), ", operands=");
      for (auto& operand : operands_) {
        ret += operand->toString(config) + " ";
      }
      return cat(ret, ", type=", type_.to_string(), ")");
    } else {
      return cat(
          ::typeName(this), "(", ::toString(op_), ", type=", type_.get_type(), ")");
    }
  };

  virtual size_t toHash() const override { return hash_value(*this); }

 protected:
  friend std::size_t hash_value(RexOperator const&);

  SQLOps op_;
  mutable std::vector<std::unique_ptr<const RexScalar>> operands_;
  SQLTypeInfo type_;

  friend struct RelAlgDagSerializer;
};

// Not a real node created by Calcite. Created by us because CaseExpr is a node in our
// Analyzer.
class RexCase : public RexScalar {
 public:
  // default constructor used for deserialization only
  RexCase() = default;

  RexCase(std::vector<std::pair<std::unique_ptr<const RexScalar>,
                                std::unique_ptr<const RexScalar>>>& expr_pair_list,
          std::unique_ptr<const RexScalar>& else_expr)
      : expr_pair_list_(std::move(expr_pair_list)), else_expr_(std::move(else_expr)) {}

  virtual void acceptChildren(Visitor& v) const override {
    for (size_t i = 0; i < branchCount(); ++i) {
      getWhen(i)->accept(v, "when");
      getThen(i)->accept(v, "then");
    }
    if (getElse()) {
      getElse()->accept(v, "else");
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

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

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      auto ret = cat(::typeName(this), "(expr_pair_list=");
      for (auto& expr : expr_pair_list_) {
        ret += expr.first->toString(config) + " " + expr.second->toString(config) + " ";
      }
      return cat(
          ret, ", else_expr=", (else_expr_ ? else_expr_->toString(config) : "null"), ")");
    } else {
      return ::typeName(this);
    }
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RexCase const&);

  std::vector<
      std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>>
      expr_pair_list_;
  std::unique_ptr<const RexScalar> else_expr_;

  friend struct RelAlgDagSerializer;
};

class RexFunctionOperator : public RexOperator {
 public:
  using ConstRexScalarPtr = std::unique_ptr<const RexScalar>;
  using ConstRexScalarPtrVector = std::vector<ConstRexScalarPtr>;

  // default constructor used for deserialization only
  RexFunctionOperator() = default;

  RexFunctionOperator(const std::string& name,
                      ConstRexScalarPtrVector& operands,
                      const SQLTypeInfo& ti)
      : RexOperator(kFUNCTION, operands, ti), name_(name) {}

  std::unique_ptr<const RexOperator> getDisambiguated(
      std::vector<std::unique_ptr<const RexScalar>>& operands) const override {
    return std::unique_ptr<const RexOperator>(
        new RexFunctionOperator(name_, operands, getType()));
  }

  virtual void acceptChildren(Visitor& v) const override {
    for (size_t i = 0; i < size(); ++i) {
      if (getOperand(i)) {
        getOperand(i)->accept(v, "operand");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  const std::string& getName() const { return name_; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      auto ret = cat(::typeName(this), "(", name_, ", operands=");
      for (auto& operand : operands_) {
        ret += operand->toString(config) + " ";
      }
      return cat(ret, ")");
    } else {
      return cat(::typeName(this), "(", name_, ")");
    }
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RexFunctionOperator const&);

  std::string name_;

  friend struct RelAlgDagSerializer;
};

enum class SortDirection { Ascending, Descending };

enum class NullSortedPosition { First, Last };

class SortField {
 public:
  SortField(const size_t field,
            const SortDirection sort_dir,
            const NullSortedPosition nulls_pos)
      : field_(field), sort_dir_(sort_dir), nulls_pos_(nulls_pos) {}

  bool operator==(const SortField& that) const {
    return field_ == that.field_ && sort_dir_ == that.sort_dir_ &&
           nulls_pos_ == that.nulls_pos_;
  }

  size_t getField() const { return field_; }

  SortDirection getSortDir() const { return sort_dir_; }

  NullSortedPosition getNullsPosition() const { return nulls_pos_; }

  std::string toString() const {
    return cat(::typeName(this),
               "(",
               std::to_string(field_),
               ", sort_dir=",
               (sort_dir_ == SortDirection::Ascending ? "asc" : "desc"),
               ", null_pos=",
               (nulls_pos_ == NullSortedPosition::First ? "nulls_first" : "nulls_last"),
               ")");
  }

  virtual size_t toHash() const { return hash_value(*this); }

 private:
  friend std::size_t hash_value(SortField const&);

  size_t field_;
  SortDirection sort_dir_;
  NullSortedPosition nulls_pos_;
};

class RexWindowFunctionOperator : public RexFunctionOperator {
 public:
  struct RexWindowBound {
    bool unbounded{false};
    bool preceding{false};
    bool following{false};
    bool is_current_row{false};
    std::shared_ptr<const RexScalar> bound_expr;
    int order_key{0};

    bool isUnboundedPreceding() const { return unbounded && preceding && !bound_expr; }

    bool isUnboundedFollowing() const { return unbounded && following && !bound_expr; }

    bool isRowOffsetPreceding() const { return !unbounded && preceding && bound_expr; }

    bool isRowOffsetFollowing() const { return !unbounded && following && bound_expr; }

    bool isCurrentRow() const {
      return !unbounded && is_current_row && !preceding && !following && !bound_expr;
    }

    bool hasNoFraming() const {
      return !unbounded && !preceding && !following && !is_current_row && !bound_expr;
    }
  };

  // default constructor used for deserialization only
  RexWindowFunctionOperator()
      : RexFunctionOperator(), kind_{SqlWindowFunctionKind::UNKNOWN}, is_rows_{false} {}

  RexWindowFunctionOperator(const SqlWindowFunctionKind kind,
                            ConstRexScalarPtrVector& operands,
                            ConstRexScalarPtrVector& partition_keys,
                            ConstRexScalarPtrVector& order_keys,
                            const std::vector<SortField> collation,
                            const RexWindowBound& frame_start_bound,
                            const RexWindowBound& frame_end_bound,
                            const bool is_rows,
                            const SQLTypeInfo& ti)
      : RexFunctionOperator(::toString(kind), operands, ti)
      , kind_(kind)
      , partition_keys_(std::move(partition_keys))
      , order_keys_(std::move(order_keys))
      , collation_(collation)
      , frame_start_bound_(frame_start_bound)
      , frame_end_bound_(frame_end_bound)
      , is_rows_(is_rows) {}

  virtual void acceptChildren(Visitor& v) const override {
    for (auto const& partition_key : getPartitionKeys()) {
      if (partition_key) {
        partition_key->accept(v, "partition");
      }
    }
    for (auto const& order_key : getOrderKeys()) {
      if (order_key) {
        order_key->accept(v, "order");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  SqlWindowFunctionKind getKind() const { return kind_; }

  const ConstRexScalarPtrVector& getPartitionKeys() const { return partition_keys_; }

  ConstRexScalarPtrVector getPartitionKeysAndRelease() const {
    return std::move(partition_keys_);
  }

  ConstRexScalarPtrVector getOrderKeysAndRelease() const {
    return std::move(order_keys_);
  }

  const ConstRexScalarPtrVector& getOrderKeys() const { return order_keys_; }

  void replacePartitionKey(size_t offset,
                           std::unique_ptr<const RexScalar>&& new_partition_key) {
    CHECK_LT(offset, partition_keys_.size());
    partition_keys_[offset] = std::move(new_partition_key);
  }

  void replaceOrderKey(size_t offset, std::unique_ptr<const RexScalar>&& new_order_key) {
    CHECK_LT(offset, order_keys_.size());
    order_keys_[offset] = std::move(new_order_key);
  }

  void replaceOperands(std::vector<std::unique_ptr<const RexScalar>>&& new_operands) {
    operands_ = std::move(new_operands);
  }

  const std::vector<SortField>& getCollation() const { return collation_; }

  const RexWindowBound& getFrameStartBound() const { return frame_start_bound_; }

  const RexWindowBound& getFrameEndBound() const { return frame_end_bound_; }

  bool isRows() const { return is_rows_; }

  std::unique_ptr<const RexOperator> disambiguatedOperands(
      ConstRexScalarPtrVector& operands,
      ConstRexScalarPtrVector& partition_keys,
      ConstRexScalarPtrVector& order_keys,
      const std::vector<SortField>& collation) const {
    return std::unique_ptr<const RexOperator>(
        new RexWindowFunctionOperator(kind_,
                                      operands,
                                      partition_keys,
                                      order_keys,
                                      collation,
                                      getFrameStartBound(),
                                      getFrameEndBound(),
                                      isRows(),
                                      getType()));
  }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      auto ret = cat(::typeName(this), "(", getName(), ", operands=");
      for (auto& operand : operands_) {
        ret += operand->toString(config) + " ";
      }
      ret += ", partition_keys=";
      for (auto& key : partition_keys_) {
        ret += key->toString(config) + " ";
      }
      ret += ", order_keys=";
      for (auto& key : order_keys_) {
        ret += key->toString(config) + " ";
      }
      return cat(ret, ")");
    } else {
      return cat(::typeName(this), "(", getName(), ")");
    }
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RexWindowFunctionOperator const&);

  SqlWindowFunctionKind kind_;
  mutable ConstRexScalarPtrVector partition_keys_;
  mutable ConstRexScalarPtrVector order_keys_;
  std::vector<SortField> collation_;
  RexWindowBound frame_start_bound_;
  RexWindowBound frame_end_bound_;
  bool is_rows_;

  friend struct RelAlgDagSerializer;
};

// Not a real node created by Calcite. Created by us because targets of a query
// should reference the group by expressions instead of creating completely new one.
class RexRef : public RexScalar {
 public:
  // default constructor used for deserialization only
  RexRef() : index_{0} {}

  RexRef(const size_t index) : index_(index) {}

  size_t getIndex() const { return index_; }

  virtual void acceptChildren(Visitor& v) const override {}
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    return cat(::typeName(this), "(", std::to_string(index_), ")");
  }

  std::unique_ptr<RexRef> deepCopy() const { return std::make_unique<RexRef>(index_); }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RexRef const&);

  size_t index_;

  friend struct RelAlgDagSerializer;
};

class RexAgg : public Rex {
 public:
  // default constructor, used for deserialization only
  RexAgg() : agg_(SQLAgg::kINVALID_AGG), distinct_(false) {}

  RexAgg(const SQLAgg agg,
         const bool distinct,
         const SQLTypeInfo& type,
         const std::vector<size_t>& operands)
      : agg_(agg), distinct_(distinct), type_(type), operands_(operands) {}

  virtual void acceptChildren(Visitor& v) const override {}
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    return cat(::typeName(this),
               "(agg=",
               std::to_string(agg_),
               ", distinct=",
               std::to_string(distinct_),
               ", type=",
               type_.get_type_name(),
               ", operands=",
               ::toString(operands_),
               ")");
  }

  SQLAgg getKind() const { return agg_; }

  bool isDistinct() const { return distinct_; }

  size_t size() const { return operands_.size(); }

  size_t getOperand(size_t idx) const { return operands_[idx]; }

  const SQLTypeInfo& getType() const { return type_; }

  std::unique_ptr<RexAgg> deepCopy() const {
    return std::make_unique<RexAgg>(agg_, distinct_, type_, operands_);
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RexAgg const&);

  SQLAgg agg_;
  bool distinct_;
  SQLTypeInfo type_;
  std::vector<size_t> operands_;

  friend struct RelAlgDagSerializer;
};

class RaExecutionDesc;

class RelAlgNode : public RelAlgDagNode {
 public:
  RelAlgNode(RelAlgInputs inputs = {})
      : inputs_(std::move(inputs))
      , id_(crt_id_++)
      , context_data_(nullptr)
      , is_nop_(false)
      , query_plan_dag_("")
      , query_plan_dag_hash_(0) {}

  virtual ~RelAlgNode() {}

  void resetQueryExecutionState() {
    context_data_ = nullptr;
    targets_metainfo_ = {};
  }

  void setContextData(const RaExecutionDesc* context_data) const {
    CHECK(!context_data_);
    context_data_ = context_data;
  }

  void setOutputMetainfo(std::vector<TargetMetaInfo> targets_metainfo) const {
    targets_metainfo_ = std::move(targets_metainfo);
  }

  void setQueryPlanDag(const std::string& extracted_query_plan_dag) const {
    if (!extracted_query_plan_dag.empty()) {
      query_plan_dag_ = extracted_query_plan_dag;
      query_plan_dag_hash_ = boost::hash_value(extracted_query_plan_dag);
    }
  }

  std::string getQueryPlanDag() const { return query_plan_dag_; }

  size_t getQueryPlanDagHash() const { return query_plan_dag_hash_; }

  const std::vector<TargetMetaInfo>& getOutputMetainfo() const {
    return targets_metainfo_;
  }

  unsigned getId() const { return id_; }

  bool hasContextData() const { return !(context_data_ == nullptr); }

  const RaExecutionDesc* getContextData() const { return context_data_; }

  const size_t inputCount() const { return inputs_.size(); }

  const RelAlgNode* getInput(const size_t idx) const {
    CHECK_LT(idx, inputs_.size());
    return inputs_[idx].get();
  }

  const std::vector<RelAlgNode const*> getInputs() const {
    std::vector<RelAlgNode const*> ret;
    for (auto& n : inputs_) {
      ret.push_back(n.get());
    }
    return ret;
  }

  std::shared_ptr<const RelAlgNode> getAndOwnInput(const size_t idx) const {
    CHECK_LT(idx, inputs_.size());
    return inputs_[idx];
  }

  void addManagedInput(std::shared_ptr<const RelAlgNode> input) {
    inputs_.push_back(input);
  }

  bool hasInput(const RelAlgNode* needle) const {
    for (auto& input_ptr : inputs_) {
      if (input_ptr.get() == needle) {
        return true;
      }
    }
    return false;
  }

  virtual void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                            std::shared_ptr<const RelAlgNode> input) {
    for (auto& input_ptr : inputs_) {
      if (input_ptr == old_input) {
        input_ptr = input;
        break;
      }
    }
  }

  // to keep an assigned DAG node id for data recycler
  void setRelNodeDagId(const size_t id) const { dag_node_id_ = id; }

  size_t getRelNodeDagId() const { return dag_node_id_; }

  bool isNop() const { return is_nop_; }

  void markAsNop() { is_nop_ = true; }

  virtual std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const = 0;

  virtual size_t size() const = 0;

  virtual std::shared_ptr<RelAlgNode> deepCopy() const = 0;

  static void resetRelAlgFirstId() noexcept;

  /**
   * Clears the ptr to the result for this descriptor. Is only used for overriding step
   * results in distributed mode.
   */
  void clearContextData() const { context_data_ = nullptr; }

  virtual size_t toHash() const = 0;

 protected:
  RelAlgInputs inputs_;
  unsigned id_;
  mutable std::optional<size_t> hash_;

 private:
  mutable const RaExecutionDesc* context_data_;
  bool is_nop_;
  mutable std::vector<TargetMetaInfo> targets_metainfo_;
  static thread_local unsigned crt_id_;
  mutable size_t dag_node_id_;
  mutable std::string query_plan_dag_;
  mutable size_t query_plan_dag_hash_;

  friend struct RelAlgDagSerializer;
};

class ExecutionResult;

class RexSubQuery : public RexScalar {
 public:
  using ExecutionResultShPtr = std::shared_ptr<const ExecutionResult>;

  // default constructor used for deserialization only
  // NOTE: the result_ member needs to be pointing to a shared_ptr
  RexSubQuery() : result_(new ExecutionResultShPtr(nullptr)) {}

  RexSubQuery(const std::shared_ptr<const RelAlgNode> ra)
      : type_(new SQLTypeInfo(kNULLT, false))
      , result_(new ExecutionResultShPtr(nullptr))
      , ra_(ra) {}

  // for deep copy
  RexSubQuery(std::shared_ptr<SQLTypeInfo> type,
              std::shared_ptr<ExecutionResultShPtr> result,
              const std::shared_ptr<const RelAlgNode> ra)
      : type_(type), result_(result), ra_(ra) {}

  RexSubQuery(const RexSubQuery&) = delete;

  RexSubQuery& operator=(const RexSubQuery&) = delete;

  RexSubQuery(RexSubQuery&&) = delete;

  RexSubQuery& operator=(RexSubQuery&&) = delete;

  virtual void acceptChildren(Visitor& v) const override {
    if (getRelAlg()) {
      getRelAlg()->accept(v, "node");
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  const SQLTypeInfo& getType() const {
    CHECK_NE(kNULLT, type_->get_type());
    return *(type_.get());
  }

  ExecutionResultShPtr getExecutionResult() const {
    CHECK(result_);
    CHECK(result_.get());
    return *(result_.get());
  }

  unsigned getId() const;

  const RelAlgNode* getRelAlg() const { return ra_.get(); }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override;

  std::unique_ptr<RexSubQuery> deepCopy() const;

  void setExecutionResult(const ExecutionResultShPtr result);

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RexSubQuery const&);

  std::shared_ptr<SQLTypeInfo> type_;
  std::shared_ptr<ExecutionResultShPtr> result_;
  std::shared_ptr<const RelAlgNode> ra_;

  friend struct RelAlgDagSerializer;
};

// The actual input node understood by the Executor.
// The in_index_ is relative to the output of node_.
class RexInput : public RexAbstractInput {
 public:
  // default constructor used for deserialization only
  RexInput() : node_{nullptr} {}

  RexInput(const RelAlgNode* node, const unsigned in_index)
      : RexAbstractInput(in_index), node_(node) {}

  virtual void acceptChildren(Visitor& v) const override {
    if (getSourceNode()) {
      getSourceNode()->accept(v, "source");
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  const RelAlgNode* getSourceNode() const { return node_; }

  // This isn't great, but we need it for coalescing nodes to Compound since
  // RexInput in descendents need to be rebound to the newly created Compound.
  // Maybe create a fresh RA tree with the required changes after each coalescing?
  void setSourceNode(const RelAlgNode* node) const { node_ = node; }

  bool operator==(const RexInput& that) const {
    return getSourceNode() == that.getSourceNode() && getIndex() == that.getIndex();
  }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override;

  std::unique_ptr<RexInput> deepCopy() const {
    return std::make_unique<RexInput>(node_, getIndex());
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RexInput const&);

  mutable const RelAlgNode* node_;

  friend struct RelAlgDagSerializer;
};

namespace std {

template <>
struct hash<RexInput> {
  size_t operator()(const RexInput& rex_in) const { return rex_in.toHash(); }
};

}  // namespace std

class RelScan : public RelAlgNode {
 public:
  // constructor used for deserialization only
  RelScan(const TableDescriptor* td, const Catalog_Namespace::Catalog& catalog)
      : td_{td}, hint_applied_{false}, catalog_(catalog) {}

  RelScan(const TableDescriptor* td,
          const std::vector<std::string>& field_names,
          const Catalog_Namespace::Catalog& catalog)
      : td_(td)
      , field_names_(field_names)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>())
      , catalog_(catalog) {}

  virtual void acceptChildren(Visitor& v) const override {}
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  size_t size() const override { return field_names_.size(); }

  const TableDescriptor* getTableDescriptor() const { return td_; }

  const Catalog_Namespace::Catalog& getCatalog() const { return catalog_; }

  const size_t getNumFragments() const { return td_->fragmenter->getNumFragments(); }

  const size_t getNumShards() const { return td_->nShards; }

  const std::vector<std::string>& getFieldNames() const { return field_names_; }

  const std::string getFieldName(const size_t i) const { return field_names_[i]; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      return cat(
          ::typeName(this), "(", td_->tableName, ", ", ::toString(field_names_), ")");
    } else {
      return cat(::typeName(this), "(", td_->tableName, ")");
    }
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    CHECK(false);
    return nullptr;
  };

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(const QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelScan const&);

  const TableDescriptor* td_;
  std::vector<std::string> field_names_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
  const Catalog_Namespace::Catalog& catalog_;

  friend struct RelAlgDagSerializer;
};

class ModifyManipulationTarget {
 public:
  ModifyManipulationTarget(bool const update_via_select = false,
                           bool const delete_via_select = false,
                           bool const varlen_update_required = false,
                           TableDescriptor const* table_descriptor = nullptr,
                           ColumnNameList target_columns = ColumnNameList(),
                           const Catalog_Namespace::Catalog* catalog = nullptr)
      : is_update_via_select_(update_via_select)
      , is_delete_via_select_(delete_via_select)
      , varlen_update_required_(varlen_update_required)
      , table_descriptor_(table_descriptor)
      , target_columns_(target_columns)
      , catalog_(catalog) {}

  void setUpdateViaSelectFlag(bool required) const { is_update_via_select_ = required; }
  void setDeleteViaSelectFlag(bool required) const { is_delete_via_select_ = required; }
  void setVarlenUpdateRequired(bool required) const {
    varlen_update_required_ = required;
  }
  void forceRowwiseOutput() const { force_rowwise_output_ = true; }

  TableDescriptor const* getModifiedTableDescriptor() const { return table_descriptor_; }
  TableDescriptor const* getTableDescriptor() const { return table_descriptor_; }
  void setModifiedTableDescriptor(TableDescriptor const* td) const {
    table_descriptor_ = td;
  }

  const Catalog_Namespace::Catalog* getModifiedTableCatalog() const { return catalog_; }

  void setModifiedTableCatalog(const Catalog_Namespace::Catalog* catalog) const {
    catalog_ = catalog;
  }

  auto const isUpdateViaSelect() const { return is_update_via_select_; }
  auto const isDeleteViaSelect() const { return is_delete_via_select_; }
  auto const isVarlenUpdateRequired() const { return varlen_update_required_; }
  auto const isProjectForUpdate() const {
    return is_update_via_select_ || is_delete_via_select_ || varlen_update_required_;
  }
  auto const isRowwiseOutputForced() const { return force_rowwise_output_; }

  void setTargetColumns(ColumnNameList const& target_columns) const {
    target_columns_ = target_columns;
  }
  ColumnNameList const& getTargetColumns() const { return target_columns_; }

  void invalidateTargetColumns() const { target_columns_.clear(); }

  template <typename VALIDATION_FUNCTOR>
  bool validateTargetColumns(VALIDATION_FUNCTOR validator) const {
    for (auto const& column_name : target_columns_) {
      if (validator(column_name) == false) {
        return false;
      }
    }
    return true;
  }

 private:
  mutable bool is_update_via_select_ = false;
  mutable bool is_delete_via_select_ = false;
  mutable bool varlen_update_required_ = false;
  mutable TableDescriptor const* table_descriptor_ = nullptr;
  mutable ColumnNameList target_columns_;
  mutable bool force_rowwise_output_ = false;
  mutable const Catalog_Namespace::Catalog* catalog_{nullptr};

  friend struct RelAlgDagSerializer;
};

class RelProject : public RelAlgNode, public ModifyManipulationTarget {
 public:
  friend class RelModify;
  using ConstRexScalarPtr = std::unique_ptr<const RexScalar>;
  using ConstRexScalarPtrVector = std::vector<ConstRexScalarPtr>;

  // constructor used for deserialization only
  RelProject(const TableDescriptor* td, const Catalog_Namespace::Catalog* catalog)
      : ModifyManipulationTarget(false, false, false, td, {}, catalog)
      , hint_applied_{false}
      , has_pushed_down_window_expr_{false} {}

  // Takes memory ownership of the expressions.
  RelProject(std::vector<std::unique_ptr<const RexScalar>>& scalar_exprs,
             const std::vector<std::string>& fields,
             std::shared_ptr<const RelAlgNode> input)
      : ModifyManipulationTarget(false, false, false, nullptr)
      , scalar_exprs_(std::move(scalar_exprs))
      , fields_(fields)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>())
      , has_pushed_down_window_expr_(false) {
    CHECK_EQ(scalar_exprs_.size(), fields_.size());
    inputs_.push_back(input);
  }

  RelProject(RelProject const&);

  virtual void acceptChildren(Visitor& v) const override {
    for (size_t i = 0; i < size(); ++i) {
      if (getProjectAt(i)) {
        getProjectAt(i)->accept(v, "\"" + getFieldName(i) + "\"");
      }
    }
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  void setExpressions(std::vector<std::unique_ptr<const RexScalar>>& exprs) const {
    scalar_exprs_ = std::move(exprs);
  }

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

  void propagateModifyManipulationTarget(
      std::shared_ptr<RelProject> new_project_node) const {
    if (isUpdateViaSelect()) {
      new_project_node->setUpdateViaSelectFlag(true);
    }
    if (isDeleteViaSelect()) {
      new_project_node->setDeleteViaSelectFlag(true);
    }
    if (isVarlenUpdateRequired()) {
      new_project_node->setVarlenUpdateRequired(true);
    }
    new_project_node->setModifiedTableDescriptor(getModifiedTableDescriptor());
    new_project_node->setTargetColumns(getTargetColumns());
    new_project_node->setModifiedTableCatalog(getModifiedTableCatalog());
    resetModifyManipulationTarget();
  }

  void resetModifyManipulationTarget() const {
    setModifiedTableDescriptor(nullptr);
    setModifiedTableCatalog(nullptr);
    setUpdateViaSelectFlag(false);
    setDeleteViaSelectFlag(false);
    setVarlenUpdateRequired(false);
    invalidateTargetColumns();
  }

  bool hasPushedDownWindowExpr() const { return has_pushed_down_window_expr_; }

  void setPushedDownWindowExpr() { has_pushed_down_window_expr_ = true; }

  const RexScalar* getProjectAt(const size_t idx) const {
    CHECK(idx < scalar_exprs_.size());
    return scalar_exprs_[idx].get();
  }

  const RexScalar* getProjectAtAndRelease(const size_t idx) const {
    CHECK(idx < scalar_exprs_.size());
    return scalar_exprs_[idx].release();
  }

  std::vector<std::unique_ptr<const RexScalar>> getExpressionsAndRelease() {
    return std::move(scalar_exprs_);
  }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string>&& fields) { fields_ = std::move(fields); }

  const std::string getFieldName(const size_t idx) const {
    CHECK(idx < fields_.size());
    return fields_[idx];
  }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override {
    replaceInput(old_input, input, std::nullopt);
  }

  void replaceInput(
      std::shared_ptr<const RelAlgNode> old_input,
      std::shared_ptr<const RelAlgNode> input,
      std::optional<std::unordered_map<unsigned, unsigned>> old_to_new_index_map);

  void appendInput(std::string new_field_name,
                   std::unique_ptr<const RexScalar> new_input);

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      auto ret = cat(::typeName(this), "(");
      for (auto& expr : scalar_exprs_) {
        ret += expr->toString(config) + " ";
      }
      return cat(ret, ", ", ::toString(fields_), ")");
    } else {
      return cat(::typeName(this), "(", ::toString(fields_), ")");
    }
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    auto copied_project_node = std::make_shared<RelProject>(*this);
    if (isProjectForUpdate()) {
      propagateModifyManipulationTarget(copied_project_node);
    }
    if (has_pushed_down_window_expr_) {
      copied_project_node->setPushedDownWindowExpr();
    }
    return copied_project_node;
  }

  bool hasWindowFunctionExpr() const;

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelProject const&);

  template <typename EXPR_VISITOR_FUNCTOR>
  void visitScalarExprs(EXPR_VISITOR_FUNCTOR visitor_functor) const {
    for (int i = 0; i < static_cast<int>(scalar_exprs_.size()); i++) {
      visitor_functor(i);
    }
  }

  void injectOffsetInFragmentExpr() const {
    RexFunctionOperator::ConstRexScalarPtrVector transient_vector;
    scalar_exprs_.emplace_back(
        std::make_unique<RexFunctionOperator const>(std::string("OFFSET_IN_FRAGMENT"),
                                                    transient_vector,
                                                    SQLTypeInfo(kBIGINT, false)));
    fields_.emplace_back("EXPR$DELETE_OFFSET_IN_FRAGMENT");
  }

  mutable std::vector<std::unique_ptr<const RexScalar>> scalar_exprs_;
  mutable std::vector<std::string> fields_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
  bool has_pushed_down_window_expr_;

  friend struct RelAlgDagSerializer;
};

class RelAggregate : public RelAlgNode {
 public:
  // default constructor, for deserialization only
  RelAggregate() : groupby_count_{0}, hint_applied_{false} {}

  // Takes ownership of the aggregate expressions.
  RelAggregate(const size_t groupby_count,
               std::vector<std::unique_ptr<const RexAgg>>& agg_exprs,
               const std::vector<std::string>& fields,
               std::shared_ptr<const RelAlgNode> input)
      : groupby_count_(groupby_count)
      , agg_exprs_(std::move(agg_exprs))
      , fields_(fields)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {
    inputs_.push_back(input);
  }

  RelAggregate(RelAggregate const&);

  virtual void acceptChildren(Visitor& v) const override {
    for (auto const& n : getAggExprs()) {
      if (n) {
        n.get()->accept(v, "aggregate");
      }
    }
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  size_t size() const override { return groupby_count_ + agg_exprs_.size(); }

  const size_t getGroupByCount() const { return groupby_count_; }

  const size_t getAggExprsCount() const { return agg_exprs_.size(); }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string>&& new_fields) {
    fields_ = std::move(new_fields);
  }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  std::vector<const RexAgg*> getAggregatesAndRelease() {
    std::vector<const RexAgg*> result;
    for (auto& agg_expr : agg_exprs_) {
      result.push_back(agg_expr.release());
    }
    return result;
  }

  std::vector<std::unique_ptr<const RexAgg>> getAggExprsAndRelease() {
    return std::move(agg_exprs_);
  }

  const std::vector<std::unique_ptr<const RexAgg>>& getAggExprs() const {
    return agg_exprs_;
  }

  void setAggExprs(std::vector<std::unique_ptr<const RexAgg>>& agg_exprs) {
    agg_exprs_ = std::move(agg_exprs);
  }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      auto ret = cat(::typeName(this),
                     "(",
                     std::to_string(groupby_count_),
                     ", agg_exprs=",
                     ::toString(agg_exprs_),
                     ", fields=",
                     ::toString(fields_));
      if (!config.skip_input_nodes) {
        ret += ::toString(inputs_);
      } else {
        ret += ", input node id={";
        for (auto& input : inputs_) {
          auto node_id_in_plan = input->getIdInPlanTree();
          auto node_id_str = node_id_in_plan ? std::to_string(*node_id_in_plan)
                                             : std::to_string(input->getId());
          ret += node_id_str + " ";
        }
        ret += "}";
      }
      return cat(ret, ")");
    } else {
      return cat(::typeName(this),
                 "(",
                 std::to_string(groupby_count_),
                 ", fields=",
                 ::toString(fields_),
                 ")");
    }
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelAggregate>(*this);
  }

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelAggregate const&);

  size_t groupby_count_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  std::vector<std::string> fields_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;

  friend struct RelAlgDagSerializer;
};

class RelJoin : public RelAlgNode {
 public:
  // default constructor used for deserialization only
  RelJoin() : join_type_(JoinType::INVALID), hint_applied_(false) {}

  RelJoin(std::shared_ptr<const RelAlgNode> lhs,
          std::shared_ptr<const RelAlgNode> rhs,
          std::unique_ptr<const RexScalar>& condition,
          const JoinType join_type)
      : condition_(std::move(condition))
      , join_type_(join_type)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {
    inputs_.push_back(lhs);
    inputs_.push_back(rhs);
  }

  RelJoin(RelJoin const&);

  virtual void acceptChildren(Visitor& v) const override {
    if (getCondition()) {
      getCondition()->accept(v, "condition");
    }
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  JoinType getJoinType() const { return join_type_; }

  const RexScalar* getCondition() const { return condition_.get(); }

  const RexScalar* getAndReleaseCondition() const { return condition_.release(); }

  void setCondition(std::unique_ptr<const RexScalar>& condition) {
    CHECK(condition);
    condition_ = std::move(condition);
  }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override;

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      auto ret = cat(::typeName(this), "(");
      if (!config.skip_input_nodes) {
        ret += ::toString(inputs_);
      } else {
        ret += ", input node id={";
        for (auto& input : inputs_) {
          auto node_id_in_plan = input->getIdInPlanTree();
          auto node_id_str = node_id_in_plan ? std::to_string(*node_id_in_plan)
                                             : std::to_string(input->getId());
          ret += node_id_str + " ";
        }
        ret += "}";
      }
      return cat(ret,
                 ", condition=",
                 (condition_ ? condition_->toString(config) : "null"),
                 ", join_type=",
                 ::toString(join_type_));
    } else {
      return cat(::typeName(this),
                 "(condition=",
                 (condition_ ? condition_->toString(config) : "null"),
                 ", join_type=",
                 ::toString(join_type_),
                 ")");
    }
  }

  size_t size() const override { return inputs_[0]->size() + inputs_[1]->size(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelJoin>(*this);
  }

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelJoin const&);

  mutable std::unique_ptr<const RexScalar> condition_;
  JoinType join_type_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;

  friend struct RelAlgDagSerializer;
};

// a helper node that contains detailed information of each level of join qual
// which is used when extracting query plan DAG
class RelTranslatedJoin : public RelAlgNode {
 public:
  RelTranslatedJoin(const RelAlgNode* lhs,
                    const RelAlgNode* rhs,
                    const std::vector<const Analyzer::ColumnVar*> lhs_join_cols,
                    const std::vector<const Analyzer::ColumnVar*> rhs_join_cols,
                    const std::vector<std::shared_ptr<const Analyzer::Expr>> filter_ops,
                    const RexScalar* outer_join_cond,
                    const bool nested_loop,
                    const JoinType join_type,
                    const std::string& op_type,
                    const std::string& qualifier,
                    const std::string& op_typeinfo)
      : lhs_(lhs)
      , rhs_(rhs)
      , lhs_join_cols_(lhs_join_cols)
      , rhs_join_cols_(rhs_join_cols)
      , filter_ops_(filter_ops)
      , outer_join_cond_(outer_join_cond)
      , nested_loop_(nested_loop)
      , join_type_(join_type)
      , op_type_(op_type)
      , qualifier_(qualifier)
      , op_typeinfo_(op_typeinfo) {}

  virtual void acceptChildren(Visitor& v) const override {
    if (getLHS()) {
      getLHS()->accept(v, "left");
    }
    if (getRHS()) {
      getRHS()->accept(v, "right");
    }
    if (getOuterJoinCond()) {
      getOuterJoinCond()->accept(v, "outer condition");
    }
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    std::ostringstream oss;
    oss << ::typeName(this) << "( join_quals { lhs: " << ::toString(lhs_join_cols_);
    oss << "rhs: " << ::toString(rhs_join_cols_);
    oss << " }, filter_quals: { " << ::toString(filter_ops_);
    oss << " }, outer_join_cond: { ";
    if (outer_join_cond_) {
      oss << outer_join_cond_->toString(config);
    } else {
      oss << "N/A";
    }
    oss << " }, loop_join: " << ::toString(nested_loop_);
    oss << ", join_type: " << ::toString(join_type_);
    oss << ", op_type: " << ::toString(op_type_);
    oss << ", qualifier: " << ::toString(qualifier_);
    oss << ", op_type_info: " << ::toString(op_typeinfo_) << ")";
    return oss.str();
  }
  const RelAlgNode* getLHS() const { return lhs_; }
  const RelAlgNode* getRHS() const { return rhs_; }
  size_t getFilterCondSize() const { return filter_ops_.size(); }
  const std::vector<std::shared_ptr<const Analyzer::Expr>> getFilterCond() const {
    return filter_ops_;
  }
  const RexScalar* getOuterJoinCond() const { return outer_join_cond_; }
  std::string getOpType() const { return op_type_; }
  std::string getQualifier() const { return qualifier_; }
  std::string getOpTypeInfo() const { return op_typeinfo_; }
  size_t size() const override { return 0; }
  JoinType getJoinType() const { return join_type_; }
  const RexScalar* getCondition() const {
    CHECK(false);
    return nullptr;
  }
  const RexScalar* getAndReleaseCondition() const {
    CHECK(false);
    return nullptr;
  }
  void setCondition(std::unique_ptr<const RexScalar>& condition) { CHECK(false); }
  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override {
    CHECK(false);
  }
  std::shared_ptr<RelAlgNode> deepCopy() const override {
    CHECK(false);
    return nullptr;
  }
  std::string getFieldName(const size_t i) const;
  std::vector<const Analyzer::ColumnVar*> getJoinCols(bool lhs) const {
    if (lhs) {
      return lhs_join_cols_;
    }
    return rhs_join_cols_;
  }
  bool isNestedLoopQual() const { return nested_loop_; }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelTranslatedJoin const&);

  const RelAlgNode* lhs_;
  const RelAlgNode* rhs_;
  const std::vector<const Analyzer::ColumnVar*> lhs_join_cols_;
  const std::vector<const Analyzer::ColumnVar*> rhs_join_cols_;
  const std::vector<std::shared_ptr<const Analyzer::Expr>> filter_ops_;
  const RexScalar* outer_join_cond_;
  const bool nested_loop_;
  const JoinType join_type_;
  const std::string op_type_;
  const std::string qualifier_;
  const std::string op_typeinfo_;
};

class RelFilter : public RelAlgNode {
 public:
  // default constructor, used for deserialization only
  RelFilter() = default;

  RelFilter(std::unique_ptr<const RexScalar>& filter,
            std::shared_ptr<const RelAlgNode> input)
      : filter_(std::move(filter)) {
    CHECK(filter_);
    inputs_.push_back(input);
  }

  // for dummy filter node for data recycler
  RelFilter(std::unique_ptr<const RexScalar>& filter) : filter_(std::move(filter)) {
    CHECK(filter_);
  }

  RelFilter(RelFilter const&);

  virtual void acceptChildren(Visitor& v) const override {
    if (getCondition()) {
      getCondition()->accept(v, "condition");
    }
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  const RexScalar* getCondition() const { return filter_.get(); }

  const RexScalar* getAndReleaseCondition() { return filter_.release(); }

  void setCondition(std::unique_ptr<const RexScalar>& condition) {
    CHECK(condition);
    filter_ = std::move(condition);
  }

  size_t size() const override { return inputs_[0]->size(); }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override;

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      auto ret = cat(
          ::typeName(this), "(", (filter_ ? filter_->toString(config) : "null"), ", ");
      if (!config.skip_input_nodes) {
        ret += ::toString(inputs_);
      } else {
        ret += ", input node id={";
        for (auto& input : inputs_) {
          auto node_id_in_plan = input->getIdInPlanTree();
          auto node_id_str = node_id_in_plan ? std::to_string(*node_id_in_plan)
                                             : std::to_string(input->getId());
          ret += node_id_str + " ";
        }
        ret += "}";
      }
      return cat(ret, ")");
    } else {
      return cat(
          ::typeName(this), "(", (filter_ ? filter_->toString(config) : "null"), ")");
    }
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelFilter>(*this);
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelFilter const&);

  std::unique_ptr<const RexScalar> filter_;

  friend struct RelAlgDagSerializer;
};

// Synthetic node to assist execution of left-deep join relational algebra.
class RelLeftDeepInnerJoin : public RelAlgNode {
 public:
  // default constructor used for deserialization only
  RelLeftDeepInnerJoin() = default;

  RelLeftDeepInnerJoin(const std::shared_ptr<RelFilter>& filter,
                       RelAlgInputs inputs,
                       std::vector<std::shared_ptr<const RelJoin>>& original_joins);

  virtual void acceptChildren(Visitor& v) const override {
    if (getInnerCondition()) {
      getInnerCondition()->accept(v, "inner condition");
    }
    for (size_t level = 1; level <= getOuterConditionsSize(); ++level) {
      if (getOuterCondition(level)) {
        getOuterCondition(level)->accept(v, "outer condition");
      }
    }
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  const RexScalar* getInnerCondition() const;

  const RexScalar* getOuterCondition(const size_t nesting_level) const;

  const JoinType getJoinType(const size_t nesting_level) const;

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override;

  size_t size() const override;
  virtual size_t getOuterConditionsSize() const;

  std::shared_ptr<RelAlgNode> deepCopy() const override;

  bool coversOriginalNode(const RelAlgNode* node) const;

  const RelFilter* getOriginalFilter() const;

  std::vector<std::shared_ptr<const RelJoin>> getOriginalJoins() const;

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelLeftDeepInnerJoin const&);

  std::unique_ptr<const RexScalar> condition_;
  std::vector<std::unique_ptr<const RexScalar>> outer_conditions_per_level_;
  std::shared_ptr<RelFilter> original_filter_;
  std::vector<std::shared_ptr<const RelJoin>> original_joins_;

  friend struct RelAlgDagSerializer;
};

// The 'RelCompound' node combines filter and on the fly aggregate computation.
// It's the result of combining a sequence of 'RelFilter' (optional), 'RelProject',
// 'RelAggregate' (optional) and a simple 'RelProject' (optional) into a single node
// which can be efficiently executed with no intermediate buffers.
class RelCompound : public RelAlgNode, public ModifyManipulationTarget {
 public:
  // constructor used for deserialization only
  RelCompound(const TableDescriptor* td, const Catalog_Namespace::Catalog* catalog)
      : ModifyManipulationTarget(false, false, false, td, {}, catalog)
      , groupby_count_{0}
      , is_agg_{false}
      , hint_applied_{false} {}

  // 'target_exprs_' are either scalar expressions owned by 'scalar_sources_'
  // or aggregate expressions owned by 'agg_exprs_', with the arguments
  // owned by 'scalar_sources_'.
  RelCompound(std::unique_ptr<const RexScalar>& filter_expr,
              const std::vector<const Rex*>& target_exprs,
              const size_t groupby_count,
              const std::vector<const RexAgg*>& agg_exprs,
              const std::vector<std::string>& fields,
              std::vector<std::unique_ptr<const RexScalar>>& scalar_sources,
              const bool is_agg,
              bool update_disguised_as_select = false,
              bool delete_disguised_as_select = false,
              bool varlen_update_required = false,
              TableDescriptor const* manipulation_target_table = nullptr,
              ColumnNameList target_columns = ColumnNameList(),
              const Catalog_Namespace::Catalog* catalog = nullptr)
      : ModifyManipulationTarget(update_disguised_as_select,
                                 delete_disguised_as_select,
                                 varlen_update_required,
                                 manipulation_target_table,
                                 target_columns,
                                 catalog)
      , filter_expr_(std::move(filter_expr))
      , groupby_count_(groupby_count)
      , fields_(fields)
      , is_agg_(is_agg)
      , scalar_sources_(std::move(scalar_sources))
      , target_exprs_(target_exprs)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {
    CHECK_EQ(fields.size(), target_exprs.size());
    for (auto agg_expr : agg_exprs) {
      agg_exprs_.emplace_back(agg_expr);
    }
  }

  RelCompound(RelCompound const&);

  virtual void acceptChildren(Visitor& v) const override {
    if (getFilterExpr()) {
      getFilterExpr()->accept(v, "filter");
    }
    for (size_t i = 0; i < getScalarSourcesSize(); ++i) {
      if (getScalarSource(i)) {
        getScalarSource(i)->accept(v, "scalar");
      }
    }
    for (size_t i = 0; i < getAggExprSize(); ++i) {
      if (getAggExpr(i)) {
        getAggExpr(i)->accept(v, "aggregate");
      }
    }
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override;

  size_t size() const override { return target_exprs_.size(); }

  const RexScalar* getFilterExpr() const { return filter_expr_.get(); }

  void setFilterExpr(std::unique_ptr<const RexScalar>& new_expr) {
    filter_expr_ = std::move(new_expr);
  }

  const Rex* getTargetExpr(const size_t i) const { return target_exprs_[i]; }

  const std::vector<std::string>& getFields() const { return fields_; }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  void setFields(std::vector<std::string>&& fields) { fields_ = std::move(fields); }

  const size_t getScalarSourcesSize() const { return scalar_sources_.size(); }

  const RexScalar* getScalarSource(const size_t i) const {
    return scalar_sources_[i].get();
  }

  void setScalarSources(std::vector<std::unique_ptr<const RexScalar>>& new_sources) {
    CHECK_EQ(new_sources.size(), scalar_sources_.size());
    scalar_sources_ = std::move(new_sources);
  }

  const size_t getGroupByCount() const { return groupby_count_; }

  bool isAggregate() const { return is_agg_; }

  size_t getAggExprSize() const { return agg_exprs_.size(); }

  const RexAgg* getAggExpr(size_t i) const { return agg_exprs_[i].get(); }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override;

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelCompound>(*this);
  }

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelCompound const&);

  std::unique_ptr<const RexScalar> filter_expr_;
  size_t groupby_count_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  std::vector<std::string> fields_;
  bool is_agg_;
  std::vector<std::unique_ptr<const RexScalar>>
      scalar_sources_;  // building blocks for group_indices_ and agg_exprs_; not
                        // actually projected, just owned
  std::vector<const Rex*> target_exprs_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;

  friend struct RelAlgDagSerializer;
};

class RelSort : public RelAlgNode {
 public:
  // default constructor used for deserialization only
  RelSort() : limit_(std::nullopt), offset_(0) {}

  RelSort(const std::vector<SortField>& collation,
          std::optional<size_t> limit,
          const size_t offset,
          std::shared_ptr<const RelAlgNode> input)
      : collation_(collation), limit_(limit), offset_(offset) {
    inputs_.push_back(input);
  }

  virtual void acceptChildren(Visitor& v) const override {
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  bool operator==(const RelSort& that) const {
    return limit_ == that.limit_ && offset_ == that.offset_ && hasEquivCollationOf(that);
  }

  size_t collationCount() const { return collation_.size(); }

  SortField getCollation(const size_t i) const {
    CHECK_LT(i, collation_.size());
    return collation_[i];
  }

  void setCollation(std::vector<SortField>&& collation) {
    collation_ = std::move(collation);
  }

  bool isEmptyResult() const { return limit_.value_or(-1) == 0; }

  bool isLimitDelivered() const { return limit_.has_value(); }

  std::optional<size_t> getLimit() const { return limit_; }

  size_t getOffset() const { return offset_; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    const std::string limit_info = limit_ ? std::to_string(*limit_) : "N/A";
    auto ret = cat(::typeName(this),
                   "(",
                   "collation=",
                   ::toString(collation_),
                   ", limit=",
                   limit_info,
                   ", offset",
                   std::to_string(offset_));
    if (!config.skip_input_nodes) {
      ret += ", inputs=", ::toString(inputs_);
    } else {
      ret += ", input node id={";
      for (auto& input : inputs_) {
        auto node_id_in_plan = input->getIdInPlanTree();
        auto node_id_str = node_id_in_plan ? std::to_string(*node_id_in_plan)
                                           : std::to_string(input->getId());
        ret += node_id_str + " ";
      }
      ret += "}";
    }
    return cat(ret, ")");
  }

  size_t size() const override { return inputs_[0]->size(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelSort>(*this);
  }

  virtual size_t toHash() const override { return hash_value(*this); }

  std::list<Analyzer::OrderEntry> getOrderEntries() const {
    std::list<Analyzer::OrderEntry> result;
    for (size_t i = 0; i < collation_.size(); ++i) {
      const auto sort_field = collation_[i];
      result.emplace_back(sort_field.getField() + 1,
                          sort_field.getSortDir() == SortDirection::Descending,
                          sort_field.getNullsPosition() == NullSortedPosition::First);
    }
    return result;
  }

 private:
  friend std::size_t hash_value(RelSort const&);

  std::vector<SortField> collation_;
  std::optional<size_t> limit_;
  size_t offset_;

  bool hasEquivCollationOf(const RelSort& that) const;

  friend struct RelAlgDagSerializer;
};

class RelModify : public RelAlgNode {
 public:
  enum class ModifyOperation { Insert, Delete, Update };
  using RelAlgNodeInputPtr = std::shared_ptr<const RelAlgNode>;
  using TargetColumnList = std::vector<std::string>;

  static std::string yieldModifyOperationString(ModifyOperation const op) {
    switch (op) {
      case ModifyOperation::Delete:
        return "DELETE";
      case ModifyOperation::Insert:
        return "INSERT";
      case ModifyOperation::Update:
        return "UPDATE";
      default:
        break;
    }
    throw std::runtime_error("Unexpected ModifyOperation enum encountered.");
  }

  static ModifyOperation yieldModifyOperationEnum(std::string const& op_string) {
    if (op_string == "INSERT") {
      return ModifyOperation::Insert;
    } else if (op_string == "DELETE") {
      return ModifyOperation::Delete;
    } else if (op_string == "UPDATE") {
      return ModifyOperation::Update;
    }

    throw std::runtime_error(
        std::string("Unsupported logical modify operation encountered " + op_string));
  }

  // constructor used for deserialization only
  RelModify(Catalog_Namespace::Catalog const& cat, TableDescriptor const* const td)
      : catalog_{cat}
      , table_descriptor_{td}
      , flattened_{false}
      , operation_{ModifyOperation::Insert} {}

  RelModify(Catalog_Namespace::Catalog const& cat,
            TableDescriptor const* const td,
            bool flattened,
            std::string const& op_string,
            TargetColumnList const& target_column_list,
            RelAlgNodeInputPtr input)
      : catalog_(cat)
      , table_descriptor_(td)
      , flattened_(flattened)
      , operation_(yieldModifyOperationEnum(op_string))
      , target_column_list_(target_column_list) {
    foreign_storage::validate_non_foreign_table_write(table_descriptor_);
    inputs_.push_back(input);
  }

  RelModify(Catalog_Namespace::Catalog const& cat,
            TableDescriptor const* const td,
            bool flattened,
            ModifyOperation op,
            TargetColumnList const& target_column_list,
            RelAlgNodeInputPtr input)
      : catalog_(cat)
      , table_descriptor_(td)
      , flattened_(flattened)
      , operation_(op)
      , target_column_list_(target_column_list) {
    foreign_storage::validate_non_foreign_table_write(table_descriptor_);
    inputs_.push_back(input);
  }

  virtual void acceptChildren(Visitor& v) const override {
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  TableDescriptor const* const getTableDescriptor() const { return table_descriptor_; }

  const Catalog_Namespace::Catalog& getCatalog() const { return catalog_; }

  bool const isFlattened() const { return flattened_; }
  ModifyOperation getOperation() const { return operation_; }
  TargetColumnList const& getUpdateColumnNames() const { return target_column_list_; }
  int getUpdateColumnCount() const { return target_column_list_.size(); }

  size_t size() const override { return 0; }
  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelModify>(*this);
  }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    auto ret = cat(::typeName(this),
                   "(",
                   table_descriptor_->tableName,
                   ", flattened=",
                   std::to_string(flattened_),
                   ", op=",
                   yieldModifyOperationString(operation_),
                   ", target_column_list=",
                   ::toString(target_column_list_));

    if (!config.skip_input_nodes) {
      ret += ", inputs=", ::toString(inputs_);
    } else {
      ret += ", input node id={";
      for (auto& input : inputs_) {
        auto node_id_in_plan = input->getIdInPlanTree();
        auto node_id_str = node_id_in_plan ? std::to_string(*node_id_in_plan)
                                           : std::to_string(input->getId());
        ret += node_id_str + " ";
      }
      ret += "}";
    }
    return cat(ret, ")");
  }

  void applyUpdateModificationsToInputNode() {
    auto previous_node = dynamic_cast<RelProject const*>(inputs_[0].get());
    CHECK(previous_node != nullptr);
    auto previous_project_node = const_cast<RelProject*>(previous_node);

    if (previous_project_node->hasWindowFunctionExpr()) {
      if (table_descriptor_->fragmenter->getNumFragments() > 1) {
        throw std::runtime_error(
            "UPDATE of a column of multi-fragmented table using window function not "
            "currently supported.");
      }
      if (table_descriptor_->nShards > 0) {
        throw std::runtime_error(
            "UPDATE of a column of sharded table using window function not "
            "currently supported.");
      }
    }

    previous_project_node->setUpdateViaSelectFlag(true);
    // remove the offset column in the projection for update handling
    target_column_list_.pop_back();

    previous_project_node->setModifiedTableDescriptor(table_descriptor_);
    previous_project_node->setTargetColumns(target_column_list_);
    previous_project_node->setModifiedTableCatalog(&catalog_);

    int target_update_column_expr_start = 0;
    int target_update_column_expr_end = (int)(target_column_list_.size() - 1);
    CHECK(target_update_column_expr_start >= 0);
    CHECK(target_update_column_expr_end >= 0);

    bool varlen_update_required = false;

    auto varlen_scan_visitor = [this,
                                &varlen_update_required,
                                target_update_column_expr_start,
                                target_update_column_expr_end](int index) {
      if (index >= target_update_column_expr_start &&
          index <= target_update_column_expr_end) {
        auto target_index = index - target_update_column_expr_start;

        auto* column_desc = catalog_.getMetadataForColumn(
            table_descriptor_->tableId, target_column_list_[target_index]);
        CHECK(column_desc);

        if (table_descriptor_->nShards) {
          const auto shard_cd =
              catalog_.getShardColumnMetadataForTable(table_descriptor_);
          CHECK(shard_cd);
          if ((column_desc->columnName == shard_cd->columnName)) {
            throw std::runtime_error("UPDATE of a shard key is currently unsupported.");
          }
        }

        // Check for valid types
        if (column_desc->columnType.is_varlen()) {
          varlen_update_required = true;
        }
        if (column_desc->columnType.is_geometry()) {
          throw std::runtime_error("UPDATE of a geo column is unsupported.");
        }
      }
    };

    previous_project_node->visitScalarExprs(varlen_scan_visitor);
    previous_project_node->setVarlenUpdateRequired(varlen_update_required);
  }

  void applyDeleteModificationsToInputNode() {
    auto previous_node = dynamic_cast<RelProject const*>(inputs_[0].get());
    CHECK(previous_node != nullptr);
    auto previous_project_node = const_cast<RelProject*>(previous_node);
    previous_project_node->setDeleteViaSelectFlag(true);
    previous_project_node->setModifiedTableDescriptor(table_descriptor_);
    previous_project_node->setModifiedTableCatalog(&catalog_);
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelModify const&);

  Catalog_Namespace::Catalog const& catalog_;
  const TableDescriptor* table_descriptor_;
  bool flattened_;
  ModifyOperation operation_;
  TargetColumnList target_column_list_;

  friend struct RelAlgDagSerializer;
};

class RelTableFunction : public RelAlgNode {
 public:
  // default constructor used for deserialization only
  RelTableFunction() = default;

  RelTableFunction(const std::string& function_name,
                   RelAlgInputs inputs,
                   std::vector<std::string>& fields,
                   std::vector<const Rex*> col_inputs,
                   std::vector<std::unique_ptr<const RexScalar>>& table_func_inputs,
                   std::vector<std::unique_ptr<const RexScalar>>& target_exprs)
      : function_name_(function_name)
      , fields_(fields)
      , col_inputs_(col_inputs)
      , table_func_inputs_(std::move(table_func_inputs))
      , target_exprs_(std::move(target_exprs)) {
    for (const auto& input : inputs) {
      inputs_.emplace_back(input);
    }
  }

  RelTableFunction(RelTableFunction const&);

  virtual void acceptChildren(Visitor& v) const override {
    for (size_t i = 0; i < getTableFuncInputsSize(); ++i) {
      if (getTableFuncInputAt(i)) {
        getTableFuncInputAt(i)->accept(v, "argument");
      }
    }
    for (size_t i = 0; i < size(); ++i) {
      if (getTargetExpr(i)) {
        getTargetExpr(i)->accept(v, "target");
      }
    }
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override;

  std::string getFunctionName() const { return function_name_; }

  size_t size() const override { return target_exprs_.size(); }

  const RexScalar* getTargetExpr(size_t idx) const {
    CHECK_LT(idx, target_exprs_.size());
    return target_exprs_[idx].get();
  }

  size_t getTableFuncInputsSize() const { return table_func_inputs_.size(); }

  size_t getColInputsSize() const { return col_inputs_.size(); }

  int32_t countRexLiteralArgs() const;

  const RexScalar* getTableFuncInputAt(const size_t idx) const {
    CHECK_LT(idx, table_func_inputs_.size());
    return table_func_inputs_[idx].get();
  }

  const RexScalar* getTableFuncInputAtAndRelease(const size_t idx) {
    // NOTE: as of 08/06/22, this appears to only be called by the bind_inputs free
    // function in RelAlgDag.cpp to disambituate inputs. If you follow the bind_inputs
    // path, it eventually could call this method in a bind_table_func_to_input lambda
    // According to that lambda, the released pointer released here is be immediately be
    // placed in a new vector of RexScalar unique_ptrs which is then ultimatey used as an
    // argument for the setTableFuncInputs method below. As such, if a table_func_input_
    // is released here that is being pointed to by one of the col_inputs_, then we can
    // keep that col_inputs_ pointer as that table_func_input_ should be returned back.
    CHECK_LT(idx, table_func_inputs_.size());
    return table_func_inputs_[idx].release();
  }

  void setTableFuncInputs(std::vector<std::unique_ptr<const RexScalar>>&& exprs);

  std::string getFieldName(const size_t idx) const {
    CHECK_LT(idx, fields_.size());
    return fields_[idx];
  }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string>&& fields) { fields_ = std::move(fields); }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelTableFunction>(*this);
  }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    if (!config.attributes_only) {
      auto ret = cat(::typeName(this), "(", function_name_);
      if (!config.skip_input_nodes) {
        ret += ", inputs=", ::toString(inputs_);
      } else {
        ret += ", input node id={";
        for (auto& input : inputs_) {
          auto node_id_in_plan = input->getIdInPlanTree();
          auto node_id_str = node_id_in_plan ? std::to_string(*node_id_in_plan)
                                             : std::to_string(input->getId());
          ret += node_id_str + " ";
        }
        ret += "}";
      }
      ret += cat(
          ", fields=", ::toString(fields_), ", col_inputs=...", ", table_func_inputs=");
      if (!config.skip_input_nodes) {
        ret += ::toString(table_func_inputs_);
      } else {
        for (auto& expr : table_func_inputs_) {
          ret += expr->toString(config) + " ";
        }
      }
      ret += ", target_exprs=";
      for (auto& expr : target_exprs_) {
        ret += expr->toString(config) + " ";
      }
      return cat(ret, ")");
    } else {
      return cat(
          ::typeName(this), "(", function_name_, ", fields=", ::toString(fields_), ")");
    }
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelTableFunction const&);

  std::string function_name_;
  std::vector<std::string> fields_;

  std::vector<const Rex*> col_inputs_;  // owned by `table_func_inputs_`, but allows
                                        // picking out the specific input columns vs
                                        // other table function inputs (e.g. literals)
  std::vector<std::unique_ptr<const RexScalar>> table_func_inputs_;

  std::vector<std::unique_ptr<const RexScalar>>
      target_exprs_;  // Note: these should all be RexRef but are stored as RexScalar
                      // for consistency

  friend struct RelAlgDagSerializer;
};

class RelLogicalValues : public RelAlgNode {
 public:
  using RowValues = std::vector<std::unique_ptr<const RexScalar>>;

  // default constructor used for deserialization only
  RelLogicalValues() = default;

  RelLogicalValues(const std::vector<TargetMetaInfo>& tuple_type,
                   std::vector<RowValues>& values)
      : tuple_type_(tuple_type), values_(std::move(values)) {}

  RelLogicalValues(RelLogicalValues const&);

  virtual void acceptChildren(Visitor& v) const override {
    for (size_t row_idx = 0; row_idx < getNumRows(); ++row_idx) {
      for (size_t col_idx = 0; col_idx < getRowsSize(); ++col_idx) {
        if (getValueAt(row_idx, col_idx)) {
          getValueAt(row_idx, col_idx)->accept(v, "value");
        }
      }
    }
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  const std::vector<TargetMetaInfo> getTupleType() const { return tuple_type_; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    std::string ret = ::typeName(this) + "(";
    for (const auto& target_meta_info : tuple_type_) {
      ret += "(" + target_meta_info.get_resname() + " " +
             target_meta_info.get_type_info().get_type_name() + ")";
    }
    ret += ")";
    return ret;
  }

  std::string getFieldName(const size_t col_idx) const {
    CHECK_LT(col_idx, size());
    return tuple_type_[col_idx].get_resname();
  }

  const RexScalar* getValueAt(const size_t row_idx, const size_t col_idx) const {
    CHECK_LT(row_idx, values_.size());
    const auto& row = values_[row_idx];
    CHECK_LT(col_idx, row.size());
    return row[col_idx].get();
  }

  size_t getRowsSize() const {
    if (values_.empty()) {
      return 0;
    } else {
      return values_.front().size();
    }
  }

  size_t getNumRows() const { return values_.size(); }

  size_t size() const override { return tuple_type_.size(); }

  bool hasRows() const { return !values_.empty(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelLogicalValues>(*this);
  }

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelLogicalValues const&);

  std::vector<TargetMetaInfo> tuple_type_;
  std::vector<RowValues> values_;

  friend struct RelAlgDagSerializer;
};

class RelLogicalUnion : public RelAlgNode {
 public:
  // default constructor used for deserialization only
  RelLogicalUnion() : is_all_{false} {}

  RelLogicalUnion(RelAlgInputs, bool is_all);

  virtual void acceptChildren(Visitor& v) const override {
    for (auto& n : getInputs()) {
      if (n) {
        n->accept(v, "input");
      }
    }
  }
  virtual void accept(Visitor& v, std::string name) const override {
    if (v.visit(this, std::move(name))) {
      acceptChildren(v);
    }
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelLogicalUnion>(*this);
  }
  size_t size() const override;
  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override;

  std::string getFieldName(const size_t i) const;

  inline bool isAll() const { return is_all_; }
  // Will throw a std::runtime_error if MetaInfo types don't match.
  std::vector<TargetMetaInfo> getCompatibleMetainfoTypes() const;
  RexScalar const* copyAndRedirectSource(RexScalar const*, size_t input_idx) const;

  // Not unique_ptr to allow for an easy deepCopy() implementation.
  mutable std::vector<std::shared_ptr<const RexScalar>> scalar_exprs_;

  virtual size_t toHash() const override { return hash_value(*this); }

 private:
  friend std::size_t hash_value(RelLogicalUnion const&);
  bool allStringCastsAreToDictionaryEncodedStrings() const;

  bool is_all_;

  friend struct RelAlgDagSerializer;
};

class QueryNotSupported : public std::runtime_error {
 public:
  QueryNotSupported(const std::string& reason) : std::runtime_error(reason) {}
};

/**
 * Class defining an in-memory, easy-to-navigate internal representation of a relational
 * algebra DAG interpreted from a JSON provided by Calcite. Must be built through the
 * RelAlgDagBuilder interface.
 */
class RelAlgDag : public boost::noncopyable {
 public:
  enum class BuildState { kNotBuilt, kBuiltNotOptimized, kBuiltOptimized };

  RelAlgDag() : build_state_{BuildState::kNotBuilt} {}

  BuildState getBuildState() const { return build_state_; }

  void eachNode(std::function<void(RelAlgNode const*)> const&) const;

  /**
   * Returns the root node of the DAG.
   */
  const RelAlgNode& getRootNode() const {
    CHECK(nodes_.size());
    const auto& last_ptr = nodes_.back();
    CHECK(last_ptr);
    return *last_ptr;
  }

  std::shared_ptr<const RelAlgNode> getRootNodeShPtr() const {
    CHECK(nodes_.size());
    return nodes_.back();
  }

  std::vector<std::shared_ptr<RelAlgNode>>& getNodes() { return nodes_; }

  /**
   * Registers a subquery with a root DAG builder. Should only be called during DAG
   * building and registration should only occur on the root.
   */
  void registerSubquery(std::shared_ptr<RexSubQuery> subquery) {
    subqueries_.push_back(subquery);
  }

  /**
   * Gets all registered subqueries. Only the root DAG can contain subqueries.
   */
  const std::vector<std::shared_ptr<RexSubQuery>>& getSubqueries() const {
    return subqueries_;
  }

  // todo(yoonmin): simplify and improve query register logic
  void registerQueryHints(std::shared_ptr<RelAlgNode> node,
                          Hints* hints_delivered,
                          RegisteredQueryHint& global_query_hint) {
    std::optional<bool> has_global_columnar_output_hint = std::nullopt;
    std::optional<bool> has_global_rowwise_output_hint = std::nullopt;
    RegisteredQueryHint query_hint;
    for (auto it = hints_delivered->begin(); it != hints_delivered->end(); it++) {
      auto target = it->second;
      auto hint_type = it->first;
      switch (hint_type) {
        case QueryHint::kCpuMode: {
          query_hint.registerHint(QueryHint::kCpuMode);
          query_hint.cpu_mode = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kCpuMode);
            global_query_hint.cpu_mode = true;
          }
          break;
        }
        case QueryHint::kColumnarOutput: {
          has_global_columnar_output_hint = target.isGlobalHint();
          break;
        }
        case QueryHint::kRowwiseOutput: {
          has_global_rowwise_output_hint = target.isGlobalHint();
          break;
        }
        case QueryHint::kBBoxIntersectBucketThreshold: {
          if (target.getListOptions().size() != 1) {
            VLOG(1) << "Skip the given query hint \"bbox_intersect_bucket_threshold\" ("
                    << target.getListOptions()[0]
                    << ") : invalid # hint options are given";
            break;
          }
          double bbox_intersect_bucket_threshold = std::stod(target.getListOptions()[0]);
          if (bbox_intersect_bucket_threshold >= 0.0 &&
              bbox_intersect_bucket_threshold <= 90.0) {
            query_hint.registerHint(QueryHint::kBBoxIntersectBucketThreshold);
            query_hint.bbox_intersect_bucket_threshold = bbox_intersect_bucket_threshold;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kBBoxIntersectBucketThreshold);
              global_query_hint.bbox_intersect_bucket_threshold =
                  bbox_intersect_bucket_threshold;
            }
          } else {
            VLOG(1) << "Skip the given query hint \"bbox_intersect_bucket_threshold\" ("
                    << bbox_intersect_bucket_threshold
                    << ") : the hint value should be within 0.0 ~ 90.0";
          }
          break;
        }
        case QueryHint::kBBoxIntersectMaxSize: {
          if (target.getListOptions().size() != 1) {
            VLOG(1) << "Skip the given query hint \"bbox_intersect_max_size\" ("
                    << target.getListOptions()[0]
                    << ") : invalid # hint options are given";
            break;
          }
          std::stringstream ss(target.getListOptions()[0]);
          int bbox_intersect_max_size;
          ss >> bbox_intersect_max_size;
          if (bbox_intersect_max_size >= 0) {
            query_hint.registerHint(QueryHint::kBBoxIntersectMaxSize);
            query_hint.bbox_intersect_max_size = (size_t)bbox_intersect_max_size;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kBBoxIntersectMaxSize);
              global_query_hint.bbox_intersect_max_size = (size_t)bbox_intersect_max_size;
            }
          } else {
            VLOG(1) << "Skip the query hint \"bbox_intersect_max_size\" ("
                    << bbox_intersect_max_size
                    << ") : the hint value should be larger than or equal to zero";
          }
          break;
        }
        case QueryHint::kBBoxIntersectAllowGpuBuild: {
          query_hint.registerHint(QueryHint::kBBoxIntersectAllowGpuBuild);
          query_hint.bbox_intersect_allow_gpu_build = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kBBoxIntersectAllowGpuBuild);
            global_query_hint.bbox_intersect_allow_gpu_build = true;
          }
          break;
        }
        case QueryHint::kBBoxIntersectNoCache: {
          query_hint.registerHint(QueryHint::kBBoxIntersectNoCache);
          query_hint.bbox_intersect_no_cache = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kBBoxIntersectNoCache);
            global_query_hint.bbox_intersect_no_cache = true;
          }
          VLOG(1) << "Skip auto tuner and hashtable caching for bbox_intersect join.";
          break;
        }
        case QueryHint::kBBoxIntersectKeysPerBin: {
          if (target.getListOptions().size() != 1) {
            VLOG(1) << "Skip the given query hint \"bbox_intersect_keys_per_bin\" ("
                    << target.getListOptions()[0]
                    << ") : invalid # hint options are given";
            break;
          }
          double bbox_intersect_keys_per_bin = std::stod(target.getListOptions()[0]);
          if (bbox_intersect_keys_per_bin > 0.0 &&
              bbox_intersect_keys_per_bin < std::numeric_limits<double>::max()) {
            query_hint.registerHint(QueryHint::kBBoxIntersectKeysPerBin);
            query_hint.bbox_intersect_keys_per_bin = bbox_intersect_keys_per_bin;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kBBoxIntersectKeysPerBin);
              global_query_hint.bbox_intersect_keys_per_bin = bbox_intersect_keys_per_bin;
            }
          } else {
            VLOG(1) << "Skip the given query hint \"bbox_intersect_keys_per_bin\" ("
                    << bbox_intersect_keys_per_bin
                    << ") : the hint value should be larger than zero";
          }
          break;
        }
        case QueryHint::kKeepResult: {
          if (!g_enable_data_recycler || !g_use_query_resultset_cache) {
            VLOG(1) << "Skip query hint \'keep_result\' because neither data recycler "
                       "nor query resultset recycler is enabled";
          } else {
            query_hint.registerHint(QueryHint::kKeepResult);
            query_hint.keep_result = true;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kKeepResult);
              global_query_hint.keep_result = true;
            }
          }
          break;
        }
        case QueryHint::kKeepTableFuncResult: {
          if (!g_enable_data_recycler || !g_use_query_resultset_cache) {
            VLOG(1) << "Skip query hint \'keep_table_function_result\' because neither "
                       "data recycler "
                       "nor query resultset recycler is enabled";
          } else {
            // we assume table function's hint is handled as global hint by default
            global_query_hint.registerHint(QueryHint::kKeepTableFuncResult);
            global_query_hint.keep_table_function_result = true;
          }
          break;
        }
        case QueryHint::kAggregateTreeFanout: {
          if (target.getListOptions().size() != 1u) {
            VLOG(1) << "Skip the given query hint \"aggregate_tree_fanout\" ("
                    << target.getListOptions()[0]
                    << ") : invalid # hint options are given";
            break;
          }
          int aggregate_tree_fanout = std::stoi(target.getListOptions()[0]);
          if (aggregate_tree_fanout < 0) {
            VLOG(1) << "A fan-out of an aggregate tree should be larger than zero";
          } else if (aggregate_tree_fanout > 1024) {
            VLOG(1) << "Too large fanout is provided (i.e., fanout < 1024)";
          } else {
            query_hint.registerHint(QueryHint::kAggregateTreeFanout);
            query_hint.aggregate_tree_fanout = aggregate_tree_fanout;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kAggregateTreeFanout);
              global_query_hint.aggregate_tree_fanout = aggregate_tree_fanout;
            }
          }
          break;
        }
        case QueryHint::kCudaBlockSize: {
          CHECK_EQ(1u, target.getListOptions().size());
          int cuda_block_size = std::stoi(target.getListOptions()[0]);
          if (cuda_block_size <= 0) {
            VLOG(1) << "CUDA block size should be larger than zero";
          } else if (cuda_block_size > 1024) {
            VLOG(1) << "CUDA block size should be less or equal to 1024";
          } else {
            query_hint.registerHint(QueryHint::kCudaBlockSize);
            query_hint.cuda_block_size = cuda_block_size;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kCudaBlockSize);
              global_query_hint.cuda_block_size = cuda_block_size;
            }
          }
          break;
        }
        case QueryHint::kCudaGridSize: {
          CHECK_EQ(1u, target.getListOptions().size());
          double cuda_grid_size_multiplier = std::stod(target.getListOptions()[0]);
          double min_grid_size_multiplier{0};
          double max_grid_size_multiplier{1024};
          if (cuda_grid_size_multiplier <= min_grid_size_multiplier) {
            VLOG(1) << "CUDA grid size multiplier should be larger than zero";
          } else if (cuda_grid_size_multiplier > max_grid_size_multiplier) {
            VLOG(1) << "CUDA grid size multiplier should be less than 1024";
          } else {
            query_hint.registerHint(QueryHint::kCudaGridSize);
            query_hint.cuda_grid_size_multiplier = cuda_grid_size_multiplier;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kCudaGridSize);
              global_query_hint.cuda_grid_size_multiplier = cuda_grid_size_multiplier;
            }
          }
          break;
        }
        case QueryHint::kOptCudaBlockAndGridSizes: {
          query_hint.registerHint(QueryHint::kOptCudaBlockAndGridSizes);
          query_hint.opt_cuda_grid_and_block_size = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kOptCudaBlockAndGridSizes);
            global_query_hint.opt_cuda_grid_and_block_size = true;
          }
          break;
        }
        case QueryHint::kWatchdog: {
          if (g_enable_watchdog) {
            VLOG(1) << "Skip the given query hint \"watchdog\": already enabled";
          } else {
            query_hint.registerHint(QueryHint::kWatchdog);
            query_hint.watchdog = true;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kWatchdog);
              global_query_hint.watchdog = true;
            }
          }
          break;
        }
        case QueryHint::kWatchdogOff: {
          if (!g_enable_watchdog) {
            VLOG(1) << "Skip the given query hint \"watchdog_off\": already disabled";
          } else {
            query_hint.registerHint(QueryHint::kWatchdogOff);
            query_hint.watchdog = false;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kWatchdogOff);
              global_query_hint.watchdog = false;
            }
          }

          break;
        }
        case QueryHint::kDynamicWatchdog: {
          if (g_enable_dynamic_watchdog) {
            VLOG(1) << "Skip the given query hint \"dynamic_watchdog\": already enabled";
          } else {
            query_hint.registerHint(QueryHint::kDynamicWatchdog);
            query_hint.dynamic_watchdog = true;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kDynamicWatchdog);
              global_query_hint.dynamic_watchdog = true;
            }
          }
          break;
        }
        case QueryHint::kDynamicWatchdogOff: {
          if (!g_enable_dynamic_watchdog) {
            VLOG(1)
                << "Skip the given query hint \"dynamic_watchdog_off\": already disabled";
          } else {
            query_hint.registerHint(QueryHint::kDynamicWatchdogOff);
            query_hint.dynamic_watchdog = false;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kDynamicWatchdogOff);
              global_query_hint.dynamic_watchdog = false;
            }
          }
          break;
        }
        case QueryHint::kQueryTimeLimit: {
          if (hints_delivered->find(QueryHint::kDynamicWatchdogOff) !=
              hints_delivered->end()) {
            VLOG(1) << "Skip the given query hint \"query_time_limit\" ("
                    << target.getListOptions()[0]
                    << ") : cannot use it with \"dynamic_watchdog_off\" hint";
            break;
          }
          if (target.getListOptions().size() != 1) {
            VLOG(1) << "Skip the given query hint \"query_time_limit\" ("
                    << target.getListOptions()[0]
                    << ") : invalid # hint options are given";
            break;
          }
          double query_time_limit = std::stoi(target.getListOptions()[0]);
          if (query_time_limit <= 0) {
            VLOG(1) << "Skip the given query hint \"query_time_limit\" ("
                    << target.getListOptions()[0]
                    << ") : the hint value should be larger than zero";
            break;
          }
          query_hint.registerHint(QueryHint::kQueryTimeLimit);
          query_hint.query_time_limit = query_time_limit;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kQueryTimeLimit);
            global_query_hint.query_time_limit = query_time_limit;
          }
          break;
        }
        case QueryHint::kAllowLoopJoin: {
          query_hint.registerHint(QueryHint::kAllowLoopJoin);
          query_hint.use_loop_join = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kAllowLoopJoin);
            global_query_hint.use_loop_join = true;
          }
          break;
        }
        case QueryHint::kDisableLoopJoin: {
          query_hint.registerHint(QueryHint::kDisableLoopJoin);
          query_hint.use_loop_join = false;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kDisableLoopJoin);
            global_query_hint.use_loop_join = false;
          }
          break;
        }
        case QueryHint::kLoopJoinInnerTableMaxNumRows: {
          CHECK_EQ(1u, target.getListOptions().size());
          int loop_size_threshold = std::stoi(target.getListOptions()[0]);
          if (loop_size_threshold <= 0) {
            VLOG(1)
                << "Skip the given query hint \"loop_join_inner_table_max_num_rows\" ("
                << target.getListOptions()[0]
                << ") : the hint value should be larger than zero";
          } else {
            query_hint.registerHint(QueryHint::kLoopJoinInnerTableMaxNumRows);
            query_hint.loop_join_inner_table_max_num_rows = loop_size_threshold;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kLoopJoinInnerTableMaxNumRows);
              global_query_hint.loop_join_inner_table_max_num_rows = loop_size_threshold;
            }
          }
          break;
        }
        case QueryHint::kMaxJoinHashTableSize: {
          CHECK_EQ(1u, target.getListOptions().size());
          int max_join_hash_table_size = std::stoi(target.getListOptions()[0]);
          if (max_join_hash_table_size <= 0) {
            VLOG(1) << "Skip the given query hint \"max_join_hashtable_size\" ("
                    << target.getListOptions()[0]
                    << ") : the hint value should be larger than zero";
          } else {
            query_hint.registerHint(QueryHint::kMaxJoinHashTableSize);
            query_hint.max_join_hash_table_size = max_join_hash_table_size;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kMaxJoinHashTableSize);
              global_query_hint.max_join_hash_table_size = max_join_hash_table_size;
            }
          }
          break;
        }
        case QueryHint::kforceBaselineHashJoin: {
          query_hint.registerHint(QueryHint::kforceBaselineHashJoin);
          query_hint.force_baseline_hash_join = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kforceBaselineHashJoin);
            global_query_hint.force_baseline_hash_join = true;
          }
          break;
        }
        case QueryHint::kforceOneToManyHashJoin: {
          query_hint.registerHint(QueryHint::kforceOneToManyHashJoin);
          query_hint.force_one_to_many_hash_join = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kforceOneToManyHashJoin);
            global_query_hint.force_one_to_many_hash_join = true;
          }
          break;
        }
        case QueryHint::kWatchdogMaxProjectedRowsPerDevice: {
          CHECK_EQ(1u, target.getListOptions().size());
          int watchdog_max_projected_rows_per_device =
              std::stoi(target.getListOptions()[0]);
          if (watchdog_max_projected_rows_per_device <= 0) {
            VLOG(1) << "Skip the given query hint "
                       "\"watchdog_max_projected_rows_per_device\" ("
                    << target.getListOptions()[0]
                    << ") : the hint value should be larger than zero";
          } else {
            query_hint.registerHint(QueryHint::kWatchdogMaxProjectedRowsPerDevice);
            query_hint.watchdog_max_projected_rows_per_device =
                watchdog_max_projected_rows_per_device;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(
                  QueryHint::kWatchdogMaxProjectedRowsPerDevice);
              global_query_hint.watchdog_max_projected_rows_per_device =
                  watchdog_max_projected_rows_per_device;
            }
          }
          break;
        }
        case QueryHint::kPreflightCountQueryThreshold: {
          CHECK_EQ(1u, target.getListOptions().size());
          int preflight_count_query_threshold = std::stoi(target.getListOptions()[0]);
          if (preflight_count_query_threshold <= 0) {
            VLOG(1) << "Skip the given query hint \"preflight_count_query_threshold\" ("
                    << target.getListOptions()[0]
                    << ") : the hint value should be larger than zero";
          } else {
            query_hint.registerHint(QueryHint::kPreflightCountQueryThreshold);
            query_hint.preflight_count_query_threshold = preflight_count_query_threshold;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kPreflightCountQueryThreshold);
              global_query_hint.preflight_count_query_threshold =
                  preflight_count_query_threshold;
            }
          }
          break;
        }
        case QueryHint::kTableReorderingOff: {
          query_hint.registerHint(QueryHint::kTableReorderingOff);
          query_hint.table_reordering_off = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kTableReorderingOff);
            global_query_hint.table_reordering_off = true;
          }
          break;
        }
        default:
          break;
      }
    }
    // we have four cases depending on 1) g_enable_columnar_output flag
    // and 2) query hint status: columnar_output and rowwise_output
    // case 1. g_enable_columnar_output = true
    // case 1.a) columnar_output = true (so rowwise_output = false);
    // case 1.b) rowwise_output = true (so columnar_output = false);
    // case 2. g_enable_columnar_output = false
    // case 2.a) columnar_output = true (so rowwise_output = false);
    // case 2.b) rowwise_output = true (so columnar_output = false);
    // case 1.a --> use columnar output
    // case 1.b --> use rowwise output
    // case 2.a --> use columnar output
    // case 2.b --> use rowwise output
    if (has_global_columnar_output_hint.has_value() &&
        has_global_rowwise_output_hint.has_value()) {
      VLOG(1)
          << "Two hints 1) columnar output and 2) rowwise output are enabled together, "
          << "so skip them and use the runtime configuration "
             "\"g_enable_columnar_output\"";
    } else if (has_global_columnar_output_hint.has_value() &&
               !has_global_rowwise_output_hint.has_value()) {
      if (g_enable_columnar_output) {
        VLOG(1) << "We already enable columnar output by default "
                   "(g_enable_columnar_output = true), so skip this columnar output hint";
      } else {
        query_hint.registerHint(QueryHint::kColumnarOutput);
        query_hint.columnar_output = true;
        if (*has_global_columnar_output_hint) {
          global_query_hint.registerHint(QueryHint::kColumnarOutput);
          global_query_hint.columnar_output = true;
        }
      }
    } else if (!has_global_columnar_output_hint.has_value() &&
               has_global_rowwise_output_hint.has_value()) {
      if (!g_enable_columnar_output) {
        VLOG(1) << "We already use the default rowwise output (g_enable_columnar_output "
                   "= false), so skip this rowwise output hint";
      } else {
        query_hint.registerHint(QueryHint::kRowwiseOutput);
        query_hint.rowwise_output = true;
        if (*has_global_rowwise_output_hint) {
          global_query_hint.registerHint(QueryHint::kRowwiseOutput);
          global_query_hint.rowwise_output = true;
        }
      }
    }
    registerQueryHint(node.get(), query_hint);
  }

  void registerQueryHint(const RelAlgNode* node, const RegisteredQueryHint& query_hint) {
    auto node_key = node->toHash();
    auto it = query_hint_.find(node_key);
    if (it == query_hint_.end()) {
      std::unordered_map<unsigned, RegisteredQueryHint> hint_map;
      hint_map.emplace(node->getId(), query_hint);
      query_hint_.emplace(node_key, hint_map);
    } else {
      it->second.emplace(node->getId(), query_hint);
    }
  }

  std::optional<RegisteredQueryHint> getQueryHint(const RelAlgNode* node) const {
    auto node_it = query_hint_.find(node->toHash());
    if (node_it != query_hint_.end()) {
      auto const& registered_query_hint_map = node_it->second;
      auto hint_it = registered_query_hint_map.find(node->getId());
      if (hint_it != registered_query_hint_map.end()) {
        auto const& registered_query_hint = hint_it->second;
        if (global_hints_.isAnyQueryHintDelivered()) {
          // apply global hint to the registered query hint for this query block
          return std::make_optional(registered_query_hint || global_hints_);
        } else {
          return std::make_optional(registered_query_hint);
        }
      }
    }
    if (global_hints_.isAnyQueryHintDelivered()) {
      // if no hint is registered from this query block
      // we return global hint instead
      return std::make_optional(global_hints_);
    }
    return std::nullopt;
  }

  std::unordered_map<size_t, std::unordered_map<unsigned, RegisteredQueryHint>>&
  getQueryHints() {
    return query_hint_;
  }

  const RegisteredQueryHint& getGlobalHints() const { return global_hints_; }

  void setGlobalQueryHints(const RegisteredQueryHint& global_hints) {
    global_hints_ = global_hints;
  }

  /**
   * Gets all registered subqueries. Only the root DAG can contain subqueries.
   */
  void resetQueryExecutionState();

 private:
  BuildState build_state_;

  std::vector<std::shared_ptr<RelAlgNode>> nodes_;
  std::vector<std::shared_ptr<RexSubQuery>> subqueries_;

  // node hash --> {node id --> registered hint}
  // we additionally consider node id to recognize corresponding hint correctly
  // i.e., to recognize the correct hint when two subqueries are identical
  std::unordered_map<size_t, std::unordered_map<unsigned, RegisteredQueryHint>>
      query_hint_;
  RegisteredQueryHint global_hints_;

  friend struct RelAlgDagSerializer;
  friend struct RelAlgDagModifier;
};

/**
 * A middle-layer class that can be inherited from to gain modify rights to a RelAlgDag
 * class. This provides a way to access private members of the RelAlgDag class without
 * dirtying its public interface with non-const accessors that may only apply for very
 * specific classes. Only classes/structs that inherit from this middle-layer class will
 * have modify rights.
 */
struct RelAlgDagModifier {
 protected:
  static std::vector<std::shared_ptr<RelAlgNode>>& getNodes(RelAlgDag& rel_alg_dag) {
    return rel_alg_dag.nodes_;
  }

  static std::vector<std::shared_ptr<RexSubQuery>>& getSubqueries(
      RelAlgDag& rel_alg_dag) {
    return rel_alg_dag.subqueries_;
  }

  static std::unordered_map<size_t, std::unordered_map<unsigned, RegisteredQueryHint>>&
  getQueryHints(RelAlgDag& rel_alg_dag) {
    return rel_alg_dag.query_hint_;
  }

  static void setBuildState(RelAlgDag& rel_alg_dag,
                            const RelAlgDag::BuildState build_state) {
    rel_alg_dag.build_state_ = build_state;
  }
};

/**
 * Builder struct to create a RelAlgDag instance. Can optionally apply high level
 * optimizations which can be expressed through relational algebra extended with
 * RelCompound. The RelCompound node is an equivalent representation for sequences of
 * RelFilter, RelProject and RelAggregate nodes. This coalescing minimizes the amount of
 * intermediate buffers required to evaluate a query. Lower level optimizations are
 * taken care by lower levels, mainly RelAlgTranslator and the IR code generation.
 */
struct RelAlgDagBuilder : public RelAlgDagModifier {
  /**
   * Constructs a RelAlg DAG from a relative-algebra JSON representation.
   * @param query_ra A JSON string representation of an RA tree from Calcite.
   * @param cat DB catalog for the current user.
   */
  static std::unique_ptr<RelAlgDag> buildDag(const std::string& query_ra,
                                             const bool optimize_dag);

  /**
   * Constructs a rel-alg DAG for any subqueries. Should only be called during DAG
   * building.
   * @param root_dag The root DAG builder. The root stores pointers to all
   * subqueries.
   * @param query_ast The current JSON node to build a DAG for.
   * @param cat DB catalog for the current user.
   */
  static std::unique_ptr<RelAlgDag> buildDagForSubquery(
      RelAlgDag& root_dag,
      const rapidjson::Value& query_ast);

  static void optimizeDag(RelAlgDag& rel_alg_dag);

 private:
  static std::unique_ptr<RelAlgDag> build(const rapidjson::Value& query_ast,
                                          RelAlgDag* root_dag,
                                          const bool optimize_dag);
};

using RANodeOutput = std::vector<RexInput>;

RANodeOutput get_node_output(const RelAlgNode* ra_node);

std::string tree_string(const RelAlgNode*, const size_t depth = 0);

namespace boost {
// boost::hash_value(T*) by default will hash just the address.
// Specialize this function template for each type so that the object itself is hashed.
constexpr size_t HASH_NULLPTR{0u};
#define HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(T)    \
  template <>                                          \
  inline std::size_t hash_value(T const* const& ptr) { \
    return ptr ? ptr->toHash() : HASH_NULLPTR;         \
  }                                                    \
  template <>                                          \
  inline std::size_t hash_value(T* const& ptr) {       \
    return ptr ? ptr->toHash() : HASH_NULLPTR;         \
  }

HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelAggregate)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelCompound)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelFilter)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelJoin)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelLeftDeepInnerJoin)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelLogicalUnion)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelLogicalValues)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelModify)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelProject)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelScan)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelSort)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelTableFunction)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelTranslatedJoin)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexAbstractInput)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexCase)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexFunctionOperator)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexInput)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexLiteral)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexOperator)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexRef)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexAgg)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexSubQuery)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexWindowFunctionOperator)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RelAlgNode)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(RexScalar)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(Rex)
HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER(SortField)
#undef HEAVYAI_SPECIALIZE_HASH_VALUE_OF_POINTER
}  // namespace boost
