/*
 * Copyright 2022 HEAVY.AI, Inc.
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

using ColumnNameList = std::vector<std::string>;
static auto const HASH_N = boost::hash_value("n");

struct RelRexToStringConfig {
  bool skip_input_nodes{false};

  static RelRexToStringConfig defaults() { return RelRexToStringConfig{false}; }
};

class Rex {
 public:
  virtual std::string toString(RelRexToStringConfig config) const = 0;

  // return hashed value of string representation of this rex
  virtual size_t toHash() const = 0;

  virtual ~Rex() {}

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

  unsigned getIndex() const { return in_index_; }

  void setIndex(const unsigned in_index) const { in_index_ = in_index; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    return cat(::typeName(this), "(", std::to_string(in_index_), ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexAbstractInput).hash_code();
      boost::hash_combine(*hash_, in_index_);
    }
    return *hash_;
  }

 private:
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

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexLiteral).hash_code();
      boost::apply_visitor(
          [this](auto&& current_val) {
            using T = std::decay_t<decltype(current_val)>;
            if constexpr (!std::is_same_v<boost::blank, T>) {
              static_assert(std::is_same_v<int64_t, T> || std::is_same_v<double, T> ||
                            std::is_same_v<std::string, T> || std::is_same_v<bool, T>);
              boost::hash_combine(*hash_, current_val);
            }
          },
          literal_);
      boost::hash_combine(*hash_, type_);
      boost::hash_combine(*hash_, target_type_);
      boost::hash_combine(*hash_, scale_);
      boost::hash_combine(*hash_, precision_);
      boost::hash_combine(*hash_, target_scale_);
      boost::hash_combine(*hash_, target_precision_);
    }
    return *hash_;
  }

  std::unique_ptr<RexLiteral> deepCopy() const {
    return std::make_unique<RexLiteral>(*this);
  }

 private:
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
    auto ret = cat(::typeName(this), "(", std::to_string(op_), ", operands=");
    for (auto& operand : operands_) {
      ret += operand->toString(config) + " ";
    }
    return cat(ret, ", type=", type_.to_string(), ")");
  };

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexOperator).hash_code();
      boost::hash_combine(*hash_, op_);
      for (auto& operand : operands_) {
        boost::hash_combine(*hash_, operand->toHash());
      }
      boost::hash_combine(*hash_, getType().get_type_name());
    }
    return *hash_;
  }

 protected:
  SQLOps op_;
  mutable std::vector<std::unique_ptr<const RexScalar>> operands_;
  SQLTypeInfo type_;

  friend struct RelAlgDagSerializer;
};

class RelAlgNode;
using RelAlgInputs = std::vector<std::shared_ptr<const RelAlgNode>>;

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

  size_t toHash() const override;

  std::unique_ptr<RexSubQuery> deepCopy() const;

  void setExecutionResult(const ExecutionResultShPtr result);

 private:
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

  size_t toHash() const override;

  std::unique_ptr<RexInput> deepCopy() const {
    return std::make_unique<RexInput>(node_, getIndex());
  }

 private:
  mutable const RelAlgNode* node_;

  friend struct RelAlgDagSerializer;
};

namespace std {

template <>
struct hash<RexInput> {
  size_t operator()(const RexInput& rex_in) const { return rex_in.toHash(); }
};

}  // namespace std

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
    auto ret = cat(::typeName(this), "(expr_pair_list=");
    for (auto& expr : expr_pair_list_) {
      ret += expr.first->toString(config) + " " + expr.second->toString(config) + " ";
    }
    return cat(
        ret, ", else_expr=", (else_expr_ ? else_expr_->toString(config) : "null"), ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexCase).hash_code();
      for (size_t i = 0; i < branchCount(); ++i) {
        boost::hash_combine(*hash_, getWhen(i)->toHash());
        boost::hash_combine(*hash_, getThen(i)->toHash());
      }
      boost::hash_combine(*hash_, getElse() ? getElse()->toHash() : HASH_N);
    }
    return *hash_;
  }

 private:
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

  const std::string& getName() const { return name_; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    auto ret = cat(::typeName(this), "(", name_, ", operands=");
    for (auto& operand : operands_) {
      ret += operand->toString(config) + " ";
    }
    return cat(ret, ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexFunctionOperator).hash_code();
      boost::hash_combine(*hash_, ::toString(op_));
      boost::hash_combine(*hash_, getType().get_type_name());
      for (auto& operand : operands_) {
        boost::hash_combine(*hash_, operand->toHash());
      }
      boost::hash_combine(*hash_, name_);
    }
    return *hash_;
  }

 private:
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

  size_t toHash() const {
    auto hash = boost::hash_value(field_);
    boost::hash_combine(hash, sort_dir_ == SortDirection::Ascending ? "a" : "d");
    boost::hash_combine(hash, nulls_pos_ == NullSortedPosition::First ? "f" : "l");
    return hash;
  }

 private:
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
      : RexFunctionOperator(), kind_{SqlWindowFunctionKind::INVALID}, is_rows_{false} {}

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
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexWindowFunctionOperator).hash_code();
      boost::hash_combine(*hash_, getType().get_type_name());
      boost::hash_combine(*hash_, getName());
      boost::hash_combine(*hash_, is_rows_);
      for (auto& collation : collation_) {
        boost::hash_combine(*hash_, collation.toHash());
      }
      for (auto& operand : operands_) {
        boost::hash_combine(*hash_, operand->toHash());
      }
      for (auto& key : partition_keys_) {
        boost::hash_combine(*hash_, key->toHash());
      }
      for (auto& key : order_keys_) {
        boost::hash_combine(*hash_, key->toHash());
      }
      auto get_window_bound_hash =
          [](const RexWindowFunctionOperator::RexWindowBound& bound) {
            auto h =
                boost::hash_value(bound.bound_expr ? bound.bound_expr->toHash() : HASH_N);
            boost::hash_combine(h, bound.unbounded);
            boost::hash_combine(h, bound.preceding);
            boost::hash_combine(h, bound.following);
            boost::hash_combine(h, bound.is_current_row);
            boost::hash_combine(h, bound.order_key);
            return h;
          };
      boost::hash_combine(*hash_, get_window_bound_hash(frame_start_bound_));
      boost::hash_combine(*hash_, get_window_bound_hash(frame_end_bound_));
    }
    return *hash_;
  }

 private:
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

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    return cat(::typeName(this), "(", std::to_string(index_), ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexRef).hash_code();
      boost::hash_combine(*hash_, index_);
    }
    return *hash_;
  }

  std::unique_ptr<RexRef> deepCopy() const { return std::make_unique<RexRef>(index_); }

 private:
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

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexAgg).hash_code();
      for (auto& operand : operands_) {
        boost::hash_combine(*hash_, operand);
      }
      boost::hash_combine(*hash_, agg_);
      boost::hash_combine(*hash_, distinct_);
      boost::hash_combine(*hash_, type_.get_type_name());
    }
    return *hash_;
  }

  SQLAgg getKind() const { return agg_; }

  bool isDistinct() const { return distinct_; }

  size_t size() const { return operands_.size(); }

  size_t getOperand(size_t idx) const { return operands_[idx]; }

  const SQLTypeInfo& getType() const { return type_; }

  std::unique_ptr<RexAgg> deepCopy() const {
    return std::make_unique<RexAgg>(agg_, distinct_, type_, operands_);
  }

 private:
  SQLAgg agg_;
  bool distinct_;
  SQLTypeInfo type_;
  std::vector<size_t> operands_;

  friend struct RelAlgDagSerializer;
};

class RaExecutionDesc;

class RelAlgNode {
 public:
  RelAlgNode(RelAlgInputs inputs = {})
      : inputs_(std::move(inputs))
      , id_(crt_id_++)
      , id_in_plan_tree_(std::nullopt)
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

  void setIdInPlanTree(size_t id) const { id_in_plan_tree_ = id; }

  std::optional<size_t> getIdInPlanTree() const { return id_in_plan_tree_; }

  bool hasContextData() const { return !(context_data_ == nullptr); }

  const RaExecutionDesc* getContextData() const { return context_data_; }

  const size_t inputCount() const { return inputs_.size(); }

  const RelAlgNode* getInput(const size_t idx) const {
    CHECK_LT(idx, inputs_.size());
    return inputs_[idx].get();
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

  virtual std::string toString(RelRexToStringConfig config) const = 0;

  // return hashed value of a string representation of this rel node
  virtual size_t toHash() const = 0;

  virtual size_t size() const = 0;

  virtual std::shared_ptr<RelAlgNode> deepCopy() const = 0;

  static void resetRelAlgFirstId() noexcept;

  /**
   * Clears the ptr to the result for this descriptor. Is only used for overriding step
   * results in distributed mode.
   */
  void clearContextData() const { context_data_ = nullptr; }

 protected:
  RelAlgInputs inputs_;
  unsigned id_;
  mutable std::optional<size_t> id_in_plan_tree_;
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

class RelScan : public RelAlgNode {
 public:
  // constructor used for deserialization only
  RelScan(const TableDescriptor* td) : td_{td}, hint_applied_{false} {}

  RelScan(const TableDescriptor* td, const std::vector<std::string>& field_names)
      : td_(td)
      , field_names_(field_names)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {}

  size_t size() const override { return field_names_.size(); }

  const TableDescriptor* getTableDescriptor() const { return td_; }

  const size_t getNumFragments() const { return td_->fragmenter->getNumFragments(); }

  const size_t getNumShards() const { return td_->nShards; }

  const std::vector<std::string>& getFieldNames() const { return field_names_; }

  const std::string getFieldName(const size_t i) const { return field_names_[i]; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    return cat(
        ::typeName(this), "(", td_->tableName, ", ", ::toString(field_names_), ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelScan).hash_code();
      boost::hash_combine(*hash_, td_->tableId);
      boost::hash_combine(*hash_, td_->tableName);
      boost::hash_combine(*hash_, ::toString(field_names_));
    }
    return *hash_;
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

 private:
  const TableDescriptor* td_;
  std::vector<std::string> field_names_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;

  friend struct RelAlgDagSerializer;
};

class ModifyManipulationTarget {
 public:
  ModifyManipulationTarget(bool const update_via_select = false,
                           bool const delete_via_select = false,
                           bool const varlen_update_required = false,
                           TableDescriptor const* table_descriptor = nullptr,
                           ColumnNameList target_columns = ColumnNameList())
      : is_update_via_select_(update_via_select)
      , is_delete_via_select_(delete_via_select)
      , varlen_update_required_(varlen_update_required)
      , table_descriptor_(table_descriptor)
      , target_columns_(target_columns) {}

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

  friend struct RelAlgDagSerializer;
};

class RelProject : public RelAlgNode, public ModifyManipulationTarget {
 public:
  friend class RelModify;
  using ConstRexScalarPtr = std::unique_ptr<const RexScalar>;
  using ConstRexScalarPtrVector = std::vector<ConstRexScalarPtr>;

  // constructor used for deserialization only
  RelProject(const TableDescriptor* td)
      : ModifyManipulationTarget(false, false, false, td)
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
    inputs_.push_back(input);
  }

  RelProject(RelProject const&);

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
    resetModifyManipulationTarget();
  }

  void resetModifyManipulationTarget() const {
    setModifiedTableDescriptor(nullptr);
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

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

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
    auto ret = cat(::typeName(this), "(");
    for (auto& expr : scalar_exprs_) {
      ret += expr->toString(config) + " ";
    }
    return cat(ret, ", ", ::toString(fields_), ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelProject).hash_code();
      for (auto& target_expr : scalar_exprs_) {
        boost::hash_combine(*hash_, target_expr->toHash());
      }
      boost::hash_combine(*hash_, ::toString(fields_));
    }
    return *hash_;
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

 private:
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
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelAggregate).hash_code();
      boost::hash_combine(*hash_, groupby_count_);
      for (auto& agg_expr : agg_exprs_) {
        boost::hash_combine(*hash_, agg_expr->toHash());
      }
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
      boost::hash_combine(*hash_, ::toString(fields_));
    }
    return *hash_;
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

 private:
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
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelJoin).hash_code();
      boost::hash_combine(*hash_, condition_ ? condition_->toHash() : HASH_N);
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
      boost::hash_combine(*hash_, ::toString(getJoinType()));
    }
    return *hash_;
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

 private:
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

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    return cat(::typeName(this),
               "( join_quals { lhs: ",
               ::toString(lhs_join_cols_),
               ", rhs: ",
               ::toString(rhs_join_cols_),
               " }, filter_quals: { ",
               ::toString(filter_ops_),
               " }, outer_join_cond: { ",
               outer_join_cond_->toString(config),
               " }, loop_join: ",
               ::toString(nested_loop_),
               ", join_type: ",
               ::toString(join_type_),
               ", op_type: ",
               ::toString(op_type_),
               ", qualifier: ",
               ::toString(qualifier_),
               ", op_type_info: ",
               ::toString(op_typeinfo_),
               ")");
  }
  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelTranslatedJoin).hash_code();
      boost::hash_combine(*hash_, lhs_->toHash());
      boost::hash_combine(*hash_, rhs_->toHash());
      boost::hash_combine(*hash_, outer_join_cond_ ? outer_join_cond_->toHash() : HASH_N);
      boost::hash_combine(*hash_, nested_loop_);
      boost::hash_combine(*hash_, ::toString(join_type_));
      boost::hash_combine(*hash_, op_type_);
      boost::hash_combine(*hash_, qualifier_);
      boost::hash_combine(*hash_, op_typeinfo_);
      for (auto& filter_op : filter_ops_) {
        boost::hash_combine(*hash_, filter_op->toString());
      }
    }
    return *hash_;
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

 private:
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
    auto ret =
        cat(::typeName(this), "(", (filter_ ? filter_->toString(config) : "null"), ", ");
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
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelFilter).hash_code();
      boost::hash_combine(*hash_, filter_ ? filter_->toHash() : HASH_N);
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
    }
    return *hash_;
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelFilter>(*this);
  }

 private:
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

  const RexScalar* getInnerCondition() const;

  const RexScalar* getOuterCondition(const size_t nesting_level) const;

  const JoinType getJoinType(const size_t nesting_level) const;

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override;

  size_t toHash() const override;

  size_t size() const override;

  std::shared_ptr<RelAlgNode> deepCopy() const override;

  bool coversOriginalNode(const RelAlgNode* node) const;

  const RelFilter* getOriginalFilter() const;

  std::vector<std::shared_ptr<const RelJoin>> getOriginalJoins() const;

 private:
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
  RelCompound(const TableDescriptor* td)
      : ModifyManipulationTarget(false, false, false, td)
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
              ColumnNameList target_columns = ColumnNameList())
      : ModifyManipulationTarget(update_disguised_as_select,
                                 delete_disguised_as_select,
                                 varlen_update_required,
                                 manipulation_target_table,
                                 target_columns)
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

  size_t toHash() const override;

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

 private:
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
  RelSort() : limit_{0}, offset_{0}, empty_result_{false}, limit_delivered_{false} {}

  RelSort(const std::vector<SortField>& collation,
          const size_t limit,
          const size_t offset,
          std::shared_ptr<const RelAlgNode> input,
          bool limit_delivered)
      : collation_(collation)
      , limit_(limit)
      , offset_(offset)
      , limit_delivered_(limit_delivered) {
    inputs_.push_back(input);
  }

  bool operator==(const RelSort& that) const {
    return limit_ == that.limit_ && offset_ == that.offset_ &&
           empty_result_ == that.empty_result_ &&
           limit_delivered_ == that.limit_delivered_ && hasEquivCollationOf(that);
  }

  size_t collationCount() const { return collation_.size(); }

  SortField getCollation(const size_t i) const {
    CHECK_LT(i, collation_.size());
    return collation_[i];
  }

  void setCollation(std::vector<SortField>&& collation) {
    collation_ = std::move(collation);
  }

  void setEmptyResult(bool emptyResult) { empty_result_ = emptyResult; }

  bool isEmptyResult() const { return empty_result_; }

  bool isLimitDelivered() const { return limit_delivered_; }

  size_t getLimit() const { return limit_; }

  size_t getOffset() const { return offset_; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    const std::string limit_info = limit_delivered_ ? std::to_string(limit_) : "N/A";
    auto ret = cat(::typeName(this),
                   "(",
                   "empty_result: ",
                   ::toString(empty_result_),
                   ", collation=",
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

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelSort).hash_code();
      for (auto& collation : collation_) {
        boost::hash_combine(*hash_, collation.toHash());
      }
      boost::hash_combine(*hash_, empty_result_);
      boost::hash_combine(*hash_, limit_delivered_);
      if (limit_delivered_) {
        boost::hash_combine(*hash_, limit_);
      }
      boost::hash_combine(*hash_, offset_);
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
    }
    return *hash_;
  }

  size_t size() const override { return inputs_[0]->size(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelSort>(*this);
  }

 private:
  std::vector<SortField> collation_;
  size_t limit_;
  size_t offset_;
  bool empty_result_;
  bool limit_delivered_;

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

  TableDescriptor const* const getTableDescriptor() const { return table_descriptor_; }
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

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelModify).hash_code();
      boost::hash_combine(*hash_, table_descriptor_->tableName);
      boost::hash_combine(*hash_, flattened_);
      boost::hash_combine(*hash_, yieldModifyOperationString(operation_));
      boost::hash_combine(*hash_, ::toString(target_column_list_));
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
    }
    return *hash_;
  }

  void applyUpdateModificationsToInputNode() {
    RelProject const* previous_project_node =
        dynamic_cast<RelProject const*>(inputs_[0].get());
    CHECK(previous_project_node != nullptr);

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
    RelProject const* previous_project_node =
        dynamic_cast<RelProject const*>(inputs_[0].get());
    CHECK(previous_project_node != nullptr);
    previous_project_node->setDeleteViaSelectFlag(true);
    previous_project_node->setModifiedTableDescriptor(table_descriptor_);
  }

 private:
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
    ret +=
        cat(", fields=", ::toString(fields_), ", col_inputs=...", ", table_func_inputs=");
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
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelTableFunction).hash_code();
      for (auto& table_func_input : table_func_inputs_) {
        boost::hash_combine(*hash_, table_func_input->toHash());
      }
      for (auto& target_expr : target_exprs_) {
        boost::hash_combine(*hash_, target_expr->toHash());
      }
      boost::hash_combine(*hash_, function_name_);
      boost::hash_combine(*hash_, ::toString(fields_));
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
    }
    return *hash_;
  }

 private:
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

  const std::vector<TargetMetaInfo> getTupleType() const { return tuple_type_; }

  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override {
    std::string ret = ::typeName(this) + "(";
    for (const auto& target_meta_info : tuple_type_) {
      ret += " (" + target_meta_info.get_resname() + " " +
             target_meta_info.get_type_info().get_type_name() + ")";
    }
    ret += ")";
    return ret;
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelLogicalValues).hash_code();
      for (auto& target_meta_info : tuple_type_) {
        boost::hash_combine(*hash_, target_meta_info.get_resname());
        boost::hash_combine(*hash_, target_meta_info.get_type_info().get_type_name());
      }
    }
    return *hash_;
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

 private:
  std::vector<TargetMetaInfo> tuple_type_;
  std::vector<RowValues> values_;

  friend struct RelAlgDagSerializer;
};

class RelLogicalUnion : public RelAlgNode {
 public:
  // default constructor used for deserialization only
  RelLogicalUnion() : is_all_{false} {}

  RelLogicalUnion(RelAlgInputs, bool is_all);
  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelLogicalUnion>(*this);
  }
  size_t size() const override;
  std::string toString(
      RelRexToStringConfig config = RelRexToStringConfig::defaults()) const override;
  size_t toHash() const override;

  std::string getFieldName(const size_t i) const;

  inline bool isAll() const { return is_all_; }
  // Will throw a std::runtime_error if MetaInfo types don't match.
  std::vector<TargetMetaInfo> getCompatibleMetainfoTypes() const;
  RexScalar const* copyAndRedirectSource(RexScalar const*, size_t input_idx) const;

  // Not unique_ptr to allow for an easy deepCopy() implementation.
  mutable std::vector<std::shared_ptr<const RexScalar>> scalar_exprs_;

 private:
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
        case QueryHint::kOverlapsBucketThreshold: {
          if (target.getListOptions().size() != 1) {
            VLOG(1) << "Skip the given query hint \"overlaps_bucket_threshold\" ("
                    << target.getListOptions()[0]
                    << ") : invalid # hint options are given";
            break;
          }
          double overlaps_bucket_threshold = std::stod(target.getListOptions()[0]);
          if (overlaps_bucket_threshold >= 0.0 && overlaps_bucket_threshold <= 90.0) {
            query_hint.registerHint(QueryHint::kOverlapsBucketThreshold);
            query_hint.overlaps_bucket_threshold = overlaps_bucket_threshold;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kOverlapsBucketThreshold);
              global_query_hint.overlaps_bucket_threshold = overlaps_bucket_threshold;
            }
          } else {
            VLOG(1) << "Skip the given query hint \"overlaps_bucket_threshold\" ("
                    << overlaps_bucket_threshold
                    << ") : the hint value should be within 0.0 ~ 90.0";
          }
          break;
        }
        case QueryHint::kOverlapsMaxSize: {
          if (target.getListOptions().size() != 1) {
            VLOG(1) << "Skip the given query hint \"overlaps_max_size\" ("
                    << target.getListOptions()[0]
                    << ") : invalid # hint options are given";
            break;
          }
          std::stringstream ss(target.getListOptions()[0]);
          int overlaps_max_size;
          ss >> overlaps_max_size;
          if (overlaps_max_size >= 0) {
            query_hint.registerHint(QueryHint::kOverlapsMaxSize);
            query_hint.overlaps_max_size = (size_t)overlaps_max_size;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kOverlapsMaxSize);
              global_query_hint.overlaps_max_size = (size_t)overlaps_max_size;
            }
          } else {
            VLOG(1) << "Skip the query hint \"overlaps_max_size\" (" << overlaps_max_size
                    << ") : the hint value should be larger than or equal to zero";
          }
          break;
        }
        case QueryHint::kOverlapsAllowGpuBuild: {
          query_hint.registerHint(QueryHint::kOverlapsAllowGpuBuild);
          query_hint.overlaps_allow_gpu_build = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kOverlapsAllowGpuBuild);
            global_query_hint.overlaps_allow_gpu_build = true;
          }
          break;
        }
        case QueryHint::kOverlapsNoCache: {
          query_hint.registerHint(QueryHint::kOverlapsNoCache);
          query_hint.overlaps_no_cache = true;
          if (target.isGlobalHint()) {
            global_query_hint.registerHint(QueryHint::kOverlapsNoCache);
            global_query_hint.overlaps_no_cache = true;
          }
          VLOG(1) << "Skip auto tuner and hashtable caching for overlaps join.";
          break;
        }
        case QueryHint::kOverlapsKeysPerBin: {
          if (target.getListOptions().size() != 1) {
            VLOG(1) << "Skip the given query hint \"overlaps_keys_per_bin\" ("
                    << target.getListOptions()[0]
                    << ") : invalid # hint options are given";
            break;
          }
          double overlaps_keys_per_bin = std::stod(target.getListOptions()[0]);
          if (overlaps_keys_per_bin > 0.0 &&
              overlaps_keys_per_bin < std::numeric_limits<double>::max()) {
            query_hint.registerHint(QueryHint::kOverlapsKeysPerBin);
            query_hint.overlaps_keys_per_bin = overlaps_keys_per_bin;
            if (target.isGlobalHint()) {
              global_query_hint.registerHint(QueryHint::kOverlapsKeysPerBin);
              global_query_hint.overlaps_keys_per_bin = overlaps_keys_per_bin;
            }
          } else {
            VLOG(1) << "Skip the given query hint \"overlaps_keys_per_bin\" ("
                    << overlaps_keys_per_bin
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
            VLOG(1) << "The loop size threshold should be larger than zero";
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
            VLOG(1) << "The maximum hash table size should be larger than zero";
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
                                             const Catalog_Namespace::Catalog& cat,
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
      const rapidjson::Value& query_ast,
      const Catalog_Namespace::Catalog& cat);

  static void optimizeDag(RelAlgDag& rel_alg_dag);

 private:
  static std::unique_ptr<RelAlgDag> build(const rapidjson::Value& query_ast,
                                          const Catalog_Namespace::Catalog& cat,
                                          RelAlgDag* root_dag,
                                          const bool optimize_dag);
};

using RANodeOutput = std::vector<RexInput>;

RANodeOutput get_node_output(const RelAlgNode* ra_node);

std::string tree_string(const RelAlgNode*, const size_t depth = 0);
