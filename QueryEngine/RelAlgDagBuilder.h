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

/** Notes:
 *  * All copy constuctors of child classes of RelAlgNode are deep copies,
 *    and are invoked by the the RelAlgNode::deepCopy() overloads.
 */

#pragma once

#include <atomic>
#include <iterator>
#include <memory>
#include <unordered_map>

#include <rapidjson/document.h>
#include <boost/core/noncopyable.hpp>
#include <boost/functional/hash.hpp>

#include "Catalog/Catalog.h"
#include "Descriptors/InputDescriptors.h"
#include "QueryEngine/QueryHint.h"
#include "QueryEngine/Rendering/RenderInfo.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "QueryEngine/TypePunning.h"
#include "QueryHint.h"
#include "SchemaMgr/ColumnInfo.h"
#include "SchemaMgr/SchemaProvider.h"
#include "SchemaMgr/TableInfo.h"
#include "Shared/sqltypes_geo.h"
#include "Shared/toString.h"

class Rex {
 public:
  virtual std::string toString() const = 0;

  // return hashed value of string representation of this rex
  virtual size_t toHash() const = 0;

  virtual ~Rex() {}

 protected:
  mutable std::optional<size_t> hash_;
};

class RexScalar : public Rex {};
// For internal use of the abstract interpreter only. The result after abstract
// interpretation will not have any references to 'RexAbstractInput' objects.
class RexAbstractInput : public RexScalar {
 public:
  RexAbstractInput(const unsigned in_index) : in_index_(in_index) {}

  unsigned getIndex() const { return in_index_; }

  void setIndex(const unsigned in_index) const { in_index_ = in_index; }

  std::string toString() const override {
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
};

class RexLiteral : public RexScalar {
 public:
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
      : literal_(nullptr)
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

  std::string toString() const override {
    std::ostringstream oss;
    oss << "RexLiteral(" << literal_ << " type=" << type_ << '(' << precision_ << ','
        << scale_ << ") target_type=" << target_type_ << '(' << target_precision_ << ','
        << target_scale_ << "))";
    return oss.str();
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexLiteral).hash_code();
      boost::hash_combine(*hash_, literal_);
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
  const boost::variant<int64_t, double, std::string, bool, void*> literal_;
  const SQLTypes type_;
  const SQLTypes target_type_;
  const unsigned scale_;
  const unsigned precision_;
  const unsigned target_scale_;
  const unsigned target_precision_;
};

using RexLiteralArray = std::vector<RexLiteral>;
using TupleContentsArray = std::vector<RexLiteralArray>;

class RexOperator : public RexScalar {
 public:
  RexOperator(const SQLOps op,
              std::vector<std::unique_ptr<const RexScalar>> operands,
              const SQLTypeInfo& type)
      : op_(op), operands_(std::move(operands)), type_(type) {}

  virtual std::unique_ptr<const RexOperator> getDisambiguated(
      std::vector<std::unique_ptr<const RexScalar>>& operands) const {
    return std::unique_ptr<const RexOperator>(
        new RexOperator(op_, std::move(operands), type_));
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
    return cat(::typeName(this),
               "(",
               std::to_string(op_),
               ", operands=",
               ::toString(operands_),
               ", type=",
               type_.to_string(),
               ")");
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
  const SQLOps op_;
  mutable std::vector<std::unique_ptr<const RexScalar>> operands_;
  const SQLTypeInfo type_;
};

class RelAlgNode;
using RelAlgInputs = std::vector<std::shared_ptr<const RelAlgNode>>;

class ExecutionResult;

class RexSubQuery : public RexScalar {
 public:
  RexSubQuery(const std::shared_ptr<const RelAlgNode> ra)
      : type_(new SQLTypeInfo(kNULLT, false))
      , result_(new std::shared_ptr<const ExecutionResult>(nullptr))
      , ra_(ra) {}

  // for deep copy
  RexSubQuery(std::shared_ptr<SQLTypeInfo> type,
              std::shared_ptr<std::shared_ptr<const ExecutionResult>> result,
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

  std::shared_ptr<const ExecutionResult> getExecutionResult() const {
    CHECK(result_);
    CHECK(result_.get());
    return *(result_.get());
  }

  unsigned getId() const;

  const RelAlgNode* getRelAlg() const { return ra_.get(); }

  std::string toString() const override;

  size_t toHash() const override;

  std::unique_ptr<RexSubQuery> deepCopy() const;

  void setExecutionResult(const std::shared_ptr<const ExecutionResult> result);

 private:
  std::shared_ptr<SQLTypeInfo> type_;
  std::shared_ptr<std::shared_ptr<const ExecutionResult>> result_;
  const std::shared_ptr<const RelAlgNode> ra_;
};

// The actual input node understood by the Executor.
// The in_index_ is relative to the output of node_.
class RexInput : public RexAbstractInput {
 public:
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

  std::string toString() const override;

  size_t toHash() const override;

  std::unique_ptr<RexInput> deepCopy() const {
    return std::make_unique<RexInput>(node_, getIndex());
  }

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

}  // namespace std

// Not a real node created by Calcite. Created by us because CaseExpr is a node in our
// Analyzer.
class RexCase : public RexScalar {
 public:
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

  std::string toString() const override {
    return cat(::typeName(this),
               "(expr_pair_list=",
               ::toString(expr_pair_list_),
               ", else_expr=",
               (else_expr_ ? else_expr_->toString() : "null"),
               ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RexCase).hash_code();
      for (size_t i = 0; i < branchCount(); ++i) {
        boost::hash_combine(*hash_, getWhen(i)->toHash());
        boost::hash_combine(*hash_, getThen(i)->toHash());
      }
      boost::hash_combine(*hash_,
                          getElse() ? getElse()->toHash() : boost::hash_value("n"));
    }
    return *hash_;
  }

 private:
  std::vector<
      std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>>
      expr_pair_list_;
  std::unique_ptr<const RexScalar> else_expr_;
};

class RexFunctionOperator : public RexOperator {
 public:
  using ConstRexScalarPtr = std::unique_ptr<const RexScalar>;
  using ConstRexScalarPtrVector = std::vector<ConstRexScalarPtr>;

  RexFunctionOperator(const std::string& name,
                      ConstRexScalarPtrVector& operands,
                      const SQLTypeInfo& ti)
      : RexOperator(kFUNCTION, std::move(operands), ti), name_(name) {}

  std::unique_ptr<const RexOperator> getDisambiguated(
      std::vector<std::unique_ptr<const RexScalar>>& operands) const override {
    return std::unique_ptr<const RexOperator>(
        new RexFunctionOperator(name_, operands, getType()));
  }

  const std::string& getName() const { return name_; }

  std::string toString() const override {
    return cat(::typeName(this), "(", name_, ", operands=", ::toString(operands_), ")");
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
  const std::string name_;
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
  const size_t field_;
  const SortDirection sort_dir_;
  const NullSortedPosition nulls_pos_;
};

class RexWindowFunctionOperator : public RexFunctionOperator {
 public:
  struct RexWindowBound {
    bool unbounded;
    bool preceding;
    bool following;
    bool is_current_row;
    std::shared_ptr<const RexScalar> offset;
    int order_key;
  };

  RexWindowFunctionOperator(const SqlWindowFunctionKind kind,
                            ConstRexScalarPtrVector& operands,
                            ConstRexScalarPtrVector& partition_keys,
                            ConstRexScalarPtrVector& order_keys,
                            const std::vector<SortField> collation,
                            const RexWindowBound& lower_bound,
                            const RexWindowBound& upper_bound,
                            const bool is_rows,
                            const SQLTypeInfo& ti)
      : RexFunctionOperator(::toString(kind), operands, ti)
      , kind_(kind)
      , partition_keys_(std::move(partition_keys))
      , order_keys_(std::move(order_keys))
      , collation_(collation)
      , lower_bound_(lower_bound)
      , upper_bound_(upper_bound)
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

  const std::vector<SortField>& getCollation() const { return collation_; }

  const RexWindowBound& getLowerBound() const { return lower_bound_; }

  const RexWindowBound& getUpperBound() const { return upper_bound_; }

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
                                      getLowerBound(),
                                      getUpperBound(),
                                      isRows(),
                                      getType()));
  }

  std::string toString() const override {
    return cat(::typeName(this),
               "(",
               getName(),
               ", operands=",
               ::toString(operands_),
               ", partition_keys=",
               ::toString(partition_keys_),
               ", order_keys=",
               ::toString(order_keys_),
               ")");
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
            auto h = boost::hash_value(bound.offset ? bound.offset->toHash()
                                                    : boost::hash_value("n"));
            boost::hash_combine(h, bound.unbounded);
            boost::hash_combine(h, bound.preceding);
            boost::hash_combine(h, bound.following);
            boost::hash_combine(h, bound.is_current_row);
            boost::hash_combine(h, bound.order_key);
            return h;
          };
      boost::hash_combine(*hash_, get_window_bound_hash(lower_bound_));
      boost::hash_combine(*hash_, get_window_bound_hash(upper_bound_));
    }
    return *hash_;
  }

 private:
  const SqlWindowFunctionKind kind_;
  mutable ConstRexScalarPtrVector partition_keys_;
  mutable ConstRexScalarPtrVector order_keys_;
  const std::vector<SortField> collation_;
  const RexWindowBound lower_bound_;
  const RexWindowBound upper_bound_;
  const bool is_rows_;
};

// Not a real node created by Calcite. Created by us because targets of a query
// should reference the group by expressions instead of creating completely new one.
class RexRef : public RexScalar {
 public:
  RexRef(const size_t index) : index_(index) {}

  size_t getIndex() const { return index_; }

  std::string toString() const override {
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
  const size_t index_;
};

class RexAgg : public Rex {
 public:
  RexAgg(const SQLAgg agg,
         const bool distinct,
         const SQLTypeInfo& type,
         const std::vector<size_t>& operands)
      : agg_(agg), distinct_(distinct), type_(type), operands_(operands) {}

  std::string toString() const override {
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
  const SQLAgg agg_;
  const bool distinct_;
  const SQLTypeInfo type_;
  const std::vector<size_t> operands_;
};

class RelAlgNode {
 public:
  RelAlgNode(RelAlgInputs inputs = {})
      : inputs_(std::move(inputs))
      , id_(crt_id_++)
      , context_data_(nullptr)
      , is_nop_(false) {}

  virtual ~RelAlgNode() {}

  void resetQueryExecutionState() {
    context_data_ = nullptr;
    targets_metainfo_ = {};
  }

  void setContextData(const void* context_data) const {
    CHECK(!context_data_);
    context_data_ = context_data;
  }

  void setOutputMetainfo(const std::vector<TargetMetaInfo>& targets_metainfo) const {
    targets_metainfo_ = targets_metainfo;
  }

  const std::vector<TargetMetaInfo>& getOutputMetainfo() const {
    return targets_metainfo_;
  }

  unsigned getId() const { return id_; }

  bool hasContextData() const { return !(context_data_ == nullptr); }

  const void* getContextData() const {
    CHECK(context_data_);
    return context_data_;
  }

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

  virtual std::string toString() const = 0;

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
  const unsigned id_;
  mutable std::optional<size_t> hash_;

 private:
  mutable const void* context_data_;
  bool is_nop_;
  mutable std::vector<TargetMetaInfo> targets_metainfo_;
  static thread_local unsigned crt_id_;
  mutable size_t dag_node_id_;
};

using RelAlgNodePtr = std::shared_ptr<RelAlgNode>;

class RelScan : public RelAlgNode {
 public:
  RelScan(TableInfoPtr tinfo, std::vector<ColumnInfoPtr> column_infos)
      : table_info_(std::move(tinfo))
      , column_infos_(std::move(column_infos))
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {}

  size_t size() const override { return column_infos_.size(); }

  TableInfoPtr getTableInfo() const { return table_info_; }

  const size_t getNumFragments() const { return table_info_->fragments; }

  const std::string& getFieldName(const size_t i) const { return column_infos_[i]->name; }

  int32_t getDatabaseId() const { return table_info_->db_id; }

  int32_t getTableId() const { return table_info_->table_id; }

  bool isVirtualColBySpi(int spi) const {
    // GEO column is never virtual.
    if (spi >= SPIMAP_MAGIC1) {
      return false;
    }

    CHECK_LE(spi, column_infos_.size());
    return column_infos_[spi - 1]->is_rowid;
  }

  int getColumnIdBySpi(int spi) const {
    int col_idx;
    int geo_idx = 0;
    if (spi >= SPIMAP_MAGIC1) {
      col_idx = (spi - SPIMAP_MAGIC1) / SPIMAP_MAGIC2 - 1;
      geo_idx = (spi - SPIMAP_MAGIC1) % SPIMAP_MAGIC2;
    } else {
      col_idx = spi - 1;
    }

    CHECK_LT(col_idx, column_infos_.size());
    return column_infos_[col_idx]->column_id + geo_idx;
  }

  std::string getColumnNameBySpi(int spi) const {
    int col_idx;
    int geo_idx = 0;
    if (spi >= SPIMAP_MAGIC1) {
      col_idx = (spi - SPIMAP_MAGIC1) / SPIMAP_MAGIC2 - 1;
      geo_idx = (spi - SPIMAP_MAGIC1) % SPIMAP_MAGIC2;
    } else {
      col_idx = spi - 1;
    }

    // Physical geo column case.
    if (geo_idx > 0) {
      CHECK(column_infos_[col_idx]->type.is_geometry());
      return get_geo_physical_col_name(
          column_infos_[col_idx]->name, column_infos_[col_idx]->type, geo_idx - 1);
    }

    CHECK_LT(col_idx, column_infos_.size());
    return column_infos_[col_idx]->name;
  }

  SQLTypeInfo getColumnTypeBySpi(int spi) const {
    int col_idx;
    int geo_idx = 0;
    if (spi >= SPIMAP_MAGIC1) {
      col_idx = (spi - SPIMAP_MAGIC1) / SPIMAP_MAGIC2 - 1;
      geo_idx = (spi - SPIMAP_MAGIC1) % SPIMAP_MAGIC2;
    } else {
      col_idx = spi - 1;
    }

    // Physical geo column case.
    if (geo_idx > 0) {
      CHECK(column_infos_[col_idx]->type.is_geometry());
      return get_geo_physical_col_type(column_infos_[col_idx]->type, geo_idx - 1);
    }

    CHECK_LT(col_idx, column_infos_.size());
    return column_infos_[col_idx]->type;
  }

  ColumnInfoPtr getColumnInfoBySpi(int spi) const {
    if (spi >= SPIMAP_MAGIC1) {
      return std::make_shared<ColumnInfo>(table_info_->db_id,
                                          table_info_->table_id,
                                          getColumnIdBySpi(spi),
                                          getColumnNameBySpi(spi),
                                          getColumnTypeBySpi(spi),
                                          isVirtualColBySpi(spi));
    }

    return column_infos_[spi - 1];
  }

  ColumnInfoPtr getDeleteColumnInfo() const { return delete_col_info_; }
  void setDeleteColumnInfo(ColumnInfoPtr info) { delete_col_info_ = info; }

  std::string toString() const override {
    std::vector<std::string_view> field_names;
    field_names.reserve(column_infos_.size());
    for (auto& info : column_infos_) {
      field_names.emplace_back(info->name);
    }
    return cat(
        ::typeName(this), "(", table_info_->name, ", ", ::toString(field_names), ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelScan).hash_code();
      boost::hash_combine(*hash_, table_info_->table_id);
      boost::hash_combine(*hash_, table_info_->name);
      for (auto& info : column_infos_) {
        boost::hash_combine(*hash_, info->name);
      }
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
  TableInfoPtr table_info_;
  const std::vector<ColumnInfoPtr> column_infos_;
  ColumnInfoPtr delete_col_info_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
};

class ModifyManipulationTarget {
 public:
  ModifyManipulationTarget(bool const update_via_select = false,
                           bool const delete_via_select = false,
                           bool const varlen_update_required = false,
                           TableDescriptor const* table_descriptor = nullptr,
                           ColumnInfoList target_columns = ColumnInfoList())
      : is_update_via_select_(update_via_select)
      , is_delete_via_select_(delete_via_select)
      , varlen_update_required_(varlen_update_required)
      , table_descriptor_(table_descriptor)
      , target_columns_(target_columns) {}

  void setUpdateViaSelectFlag() const { is_update_via_select_ = true; }
  void setDeleteViaSelectFlag() const { is_delete_via_select_ = true; }
  void setVarlenUpdateRequired(bool required) const {
    varlen_update_required_ = required;
  }

  TableDescriptor const* getModifiedTableDescriptor() const { return table_descriptor_; }
  void setModifiedTableDescriptor(TableDescriptor const* td) const {
    table_descriptor_ = td;
  }

  auto const isUpdateViaSelect() const { return is_update_via_select_; }
  auto const isDeleteViaSelect() const { return is_delete_via_select_; }
  auto const isVarlenUpdateRequired() const { return varlen_update_required_; }

  void setTargetColumns(ColumnInfoList const& target_columns) const {
    target_columns_ = target_columns;
  }
  ColumnInfoList const& getTargetColumns() const { return target_columns_; }

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
  mutable ColumnInfoList target_columns_;
};

class RelProject : public RelAlgNode, public ModifyManipulationTarget {
 public:
  friend class RelModify;
  using ConstRexScalarPtr = std::unique_ptr<const RexScalar>;
  using ConstRexScalarPtrVector = std::vector<ConstRexScalarPtr>;

  // Takes memory ownership of the expressions.
  RelProject(std::vector<std::unique_ptr<const RexScalar>> scalar_exprs,
             const std::vector<std::string>& fields,
             std::shared_ptr<const RelAlgNode> input)
      : ModifyManipulationTarget(false, false, false, nullptr)
      , scalar_exprs_(std::move(scalar_exprs))
      , fields_(fields)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {
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
  void setFields(std::vector<std::string>& fields) { fields_ = std::move(fields); }

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

  std::string toString() const override {
    return cat(
        ::typeName(this), "(", ::toString(scalar_exprs_), ", ", ::toString(fields_), ")");
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
    return std::make_shared<RelProject>(*this);
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
};

class RelAggregate : public RelAlgNode {
 public:
  // Takes ownership of the aggregate expressions.
  RelAggregate(const size_t groupby_count,
               std::vector<std::unique_ptr<const RexAgg>> agg_exprs,
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
  void setFields(std::vector<std::string>& new_fields) {
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

  std::string toString() const override {
    return cat(::typeName(this),
               "(",
               std::to_string(groupby_count_),
               ", agg_exprs=",
               ::toString(agg_exprs_),
               ", fields=",
               ::toString(fields_),
               ", inputs=",
               ::toString(inputs_),
               ")");
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
  const size_t groupby_count_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  std::vector<std::string> fields_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
};

class RelJoin : public RelAlgNode {
 public:
  RelJoin(std::shared_ptr<const RelAlgNode> lhs,
          std::shared_ptr<const RelAlgNode> rhs,
          std::unique_ptr<const RexScalar> condition,
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

  std::string toString() const override {
    return cat(::typeName(this),
               "(",
               ::toString(inputs_),
               ", condition=",
               (condition_ ? condition_->toString() : "null"),
               ", join_type=",
               ::toString(join_type_));
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelJoin).hash_code();
      boost::hash_combine(*hash_,
                          condition_ ? condition_->toHash() : boost::hash_value("n"));
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
  const JoinType join_type_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
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
      , op_typeinfo_(op_typeinfo) {
    CHECK_EQ(lhs_join_cols_.size(), rhs_join_cols_.size());
  }

  std::string toString() const override {
    return cat(::typeName(this),
               "( join_quals { lhs: ",
               ::toString(lhs_join_cols_),
               ", rhs: ",
               ::toString(rhs_join_cols_),
               " }, filter_quals: { ",
               ::toString(filter_ops_),
               " }, outer_join_cond: { ",
               ::toString(outer_join_cond_),
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
      boost::hash_combine(
          *hash_, outer_join_cond_ ? outer_join_cond_->toHash() : boost::hash_value("n"));
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

  std::string toString() const override {
    return cat(::typeName(this),
               "(",
               (filter_ ? filter_->toString() : "null"),
               ", ",
               ::toString(inputs_) + ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelFilter).hash_code();
      boost::hash_combine(*hash_, filter_ ? filter_->toHash() : boost::hash_value("n"));
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
};

// Synthetic node to assist execution of left-deep join relational algebra.
class RelLeftDeepInnerJoin : public RelAlgNode {
 public:
  RelLeftDeepInnerJoin(const std::shared_ptr<RelFilter>& filter,
                       RelAlgInputs inputs,
                       std::vector<std::shared_ptr<const RelJoin>>& original_joins);

  const RexScalar* getInnerCondition() const;

  const RexScalar* getOuterCondition(const size_t nesting_level) const;

  const JoinType getJoinType(const size_t nesting_level) const;

  std::string toString() const override;

  size_t toHash() const override;

  size_t size() const override;

  std::shared_ptr<RelAlgNode> deepCopy() const override;

  bool coversOriginalNode(const RelAlgNode* node) const;

  const RelFilter* getOriginalFilter() const;

  std::vector<std::shared_ptr<const RelJoin>> getOriginalJoins() const;

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
class RelCompound : public RelAlgNode, public ModifyManipulationTarget {
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
              const bool is_agg,
              bool update_disguised_as_select = false,
              bool delete_disguised_as_select = false,
              bool varlen_update_required = false,
              TableDescriptor const* manipulation_target_table = nullptr,
              ColumnInfoList target_columns = ColumnInfoList())
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

  std::string toString() const override;

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
  const size_t groupby_count_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  const std::vector<std::string> fields_;
  const bool is_agg_;
  std::vector<std::unique_ptr<const RexScalar>>
      scalar_sources_;  // building blocks for group_indices_ and agg_exprs_; not actually
                        // projected, just owned
  const std::vector<const Rex*> target_exprs_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
};

class RelSort : public RelAlgNode {
 public:
  RelSort(const std::vector<SortField>& collation,
          const size_t limit,
          const size_t offset,
          std::shared_ptr<const RelAlgNode> input)
      : collation_(collation), limit_(limit), offset_(offset), empty_result_(false) {
    inputs_.push_back(input);
  }

  bool operator==(const RelSort& that) const {
    return limit_ == that.limit_ && offset_ == that.offset_ &&
           empty_result_ == that.empty_result_ && hasEquivCollationOf(that);
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

  size_t getLimit() const { return limit_; }

  size_t getOffset() const { return offset_; }

  std::string toString() const override {
    return cat(::typeName(this),
               "(",
               "empty_result: ",
               ::toString(empty_result_),
               ", collation=",
               ::toString(collation_),
               ", limit=",
               std::to_string(limit_),
               ", offset",
               std::to_string(offset_),
               ", inputs=",
               ::toString(inputs_),
               ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelSort).hash_code();
      for (auto& collation : collation_) {
        boost::hash_combine(*hash_, collation.toHash());
      }
      boost::hash_combine(*hash_, empty_result_);
      boost::hash_combine(*hash_, limit_);
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
  const size_t limit_;
  const size_t offset_;
  bool empty_result_;

  bool hasEquivCollationOf(const RelSort& that) const;
};

class RelModify : public RelAlgNode {
 public:
  enum class ModifyOperation { Insert, Delete, Update };
  using RelAlgNodeInputPtr = std::shared_ptr<const RelAlgNode>;
  using TargetColumnList = std::vector<ColumnInfoPtr>;

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

  // TODO: remove catalog from constructor
  RelModify(Catalog_Namespace::Catalog const& cat,
            TableDescriptor const* const td,
            bool flattened,
            std::string const& op_string,
            TargetColumnList const& target_column_list,
            RelAlgNodeInputPtr input)
      : table_descriptor_(td)
      , flattened_(flattened)
      , operation_(yieldModifyOperationEnum(op_string))
      , target_column_list_(target_column_list) {
    inputs_.push_back(input);
  }

  RelModify(Catalog_Namespace::Catalog const& cat,
            TableDescriptor const* const td,
            bool flattened,
            ModifyOperation op,
            TargetColumnList const& target_column_list,
            RelAlgNodeInputPtr input)
      : table_descriptor_(td)
      , flattened_(flattened)
      , operation_(op)
      , target_column_list_(target_column_list) {
    inputs_.push_back(input);
  }

  TableDescriptor const* const getTableDescriptor() const { return table_descriptor_; }
  bool const isFlattened() const { return flattened_; }
  ModifyOperation getOperation() const { return operation_; }
  TargetColumnList const& getUpdateColumnInfos() const { return target_column_list_; }
  int getUpdateColumnCount() const { return target_column_list_.size(); }

  size_t size() const override { return 0; }
  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelModify>(*this);
  }

  std::string toString() const override {
    return cat(::typeName(this),
               "(",
               table_descriptor_->tableName,
               ", flattened=",
               std::to_string(flattened_),
               ", op=",
               yieldModifyOperationString(operation_),
               ", target_column_list=",
               ::toString(target_column_list_),
               ", inputs=",
               ::toString(inputs_),
               ")");
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
      throw std::runtime_error(
          "UPDATE of a column using a window function is not currently supported.");
    }

    previous_project_node->setUpdateViaSelectFlag();
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
        auto& column_info = target_column_list_[target_index];

        // Check for valid types
        if (column_info->type.is_varlen()) {
          varlen_update_required = true;
        }
        if (column_info->type.is_geometry()) {
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
    previous_project_node->setDeleteViaSelectFlag();
    previous_project_node->setModifiedTableDescriptor(table_descriptor_);
  }

 private:
  const TableDescriptor* table_descriptor_;
  bool flattened_;
  ModifyOperation operation_;
  TargetColumnList target_column_list_;
};

class RelTableFunction : public RelAlgNode {
 public:
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
    CHECK_LT(idx, table_func_inputs_.size());
    return table_func_inputs_[idx].release();
  }

  void setTableFuncInputs(std::vector<std::unique_ptr<const RexScalar>>& exprs) {
    table_func_inputs_ = std::move(exprs);
  }

  std::string getFieldName(const size_t idx) const {
    CHECK_LT(idx, fields_.size());
    return fields_[idx];
  }

  const std::vector<std::string>& getFields() const { return fields_; }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelTableFunction>(*this);
  }

  std::string toString() const override {
    return cat(::typeName(this),
               "(",
               function_name_,
               ", inputs=",
               ::toString(inputs_),
               ", fields=",
               ::toString(fields_),
               ", col_inputs=...",
               ", table_func_inputs=",
               ::toString(table_func_inputs_),
               ", target_exprs=",
               ::toString(target_exprs_),
               ")");
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

  std::vector<const Rex*>
      col_inputs_;  // owned by `table_func_inputs_`, but allows picking out the specific
                    // input columns vs other table function inputs (e.g. literals)
  std::vector<std::unique_ptr<const RexScalar>> table_func_inputs_;

  std::vector<std::unique_ptr<const RexScalar>>
      target_exprs_;  // Note: these should all be RexRef but are stored as RexScalar for
                      // consistency
};

class RelLogicalValues : public RelAlgNode {
 public:
  using RowValues = std::vector<std::unique_ptr<const RexScalar>>;

  RelLogicalValues(const std::vector<TargetMetaInfo>& tuple_type,
                   std::vector<RowValues>& values)
      : tuple_type_(tuple_type), values_(std::move(values)) {}

  RelLogicalValues(RelLogicalValues const&);

  const std::vector<TargetMetaInfo> getTupleType() const { return tuple_type_; }

  std::string toString() const override {
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
  const std::vector<TargetMetaInfo> tuple_type_;
  const std::vector<RowValues> values_;
};

class RelLogicalUnion : public RelAlgNode {
 public:
  RelLogicalUnion(RelAlgInputs, bool is_all);
  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelLogicalUnion>(*this);
  }
  size_t size() const override;
  std::string toString() const override;
  size_t toHash() const override;

  std::string getFieldName(const size_t i) const;

  inline bool isAll() const { return is_all_; }
  // Will throw a std::runtime_error if MetaInfo types don't match.
  void checkForMatchingMetaInfoTypes() const;
  RexScalar const* copyAndRedirectSource(RexScalar const*, size_t input_idx) const;

  // Not unique_ptr to allow for an easy deepCopy() implementation.
  mutable std::vector<std::shared_ptr<const RexScalar>> scalar_exprs_;

 private:
  bool const is_all_;
};

class QueryNotSupported : public std::runtime_error {
 public:
  QueryNotSupported(const std::string& reason) : std::runtime_error(reason) {}
};

class RelAlgDag {
 public:
  RelAlgDag() = default;
  virtual ~RelAlgDag() = default;

  void eachNode(std::function<void(RelAlgNode const*)> const& callback) const;

  /**
   * Returns the root node of the DAG.
   */
  const RelAlgNode& getRootNode() const {
    CHECK(root_);
    return *root_;
  }

  std::shared_ptr<const RelAlgNode> getRootNodeShPtr() const { return root_; }

  /**
   * Registers a subquery with a root DAG builder. Should only be called during DAG
   * building and registration should only occur on the root.
   */
  void registerSubquery(std::shared_ptr<RexSubQuery> subquery) {
    subqueries_.push_back(subquery);
  }

  void registerQueryHints(RelAlgNodePtr node, Hints* hints_delivered);

  /**
   * Gets all registered subqueries. Only the root DAG can contain subqueries.
   */
  const std::vector<std::shared_ptr<RexSubQuery>>& getSubqueries() const {
    return subqueries_;
  }

  std::optional<RegisteredQueryHint> getQueryHint(const RelAlgNode* node) const {
    auto it = query_hint_.find(node->toHash());
    return it != query_hint_.end() ? std::make_optional(it->second) : std::nullopt;
  }

  const std::unordered_map<size_t, RegisteredQueryHint>& getQueryHints() const {
    return query_hint_;
  }

  /**
   * Gets all registered subqueries. Only the root DAG can contain subqueries.
   */
  void resetQueryExecutionState();

 protected:
  // Root node of the query.
  RelAlgNodePtr root_;
  // All nodes including the root one.
  std::vector<RelAlgNodePtr> nodes_;
  std::vector<std::shared_ptr<RexSubQuery>> subqueries_;
  std::unordered_map<size_t, RegisteredQueryHint> query_hint_;
};

/**
 * Builder class to create an in-memory, easy-to-navigate relational algebra DAG
 * interpreted from a JSON representation from Calcite. Also, applies high level
 * optimizations which can be expressed through relational algebra extended with
 * RelCompound. The RelCompound node is an equivalent representation for sequences of
 * RelFilter, RelProject and RelAggregate nodes. This coalescing minimizes the amount of
 * intermediate buffers required to evaluate a query. Lower level optimizations are
 * taken care by lower levels, mainly RelAlgTranslator and the IR code generation.
 */
class RelAlgDagBuilder : public RelAlgDag, public boost::noncopyable {
 public:
  RelAlgDagBuilder() = delete;

  /**
   * Constructs a RelAlg DAG from a JSON representation.
   * @param query_ra A JSON string representation of an RA tree from Calcite.
   * @param cat DB catalog for the current user.
   * @param render_opts Additional build options for render queries.
   */
  RelAlgDagBuilder(const std::string& query_ra,
                   const Catalog_Namespace::Catalog* cat,
                   SchemaProviderPtr schema_provider,
                   const RenderInfo* render_info);

  /**
   * Constructs a sub-DAG for any subqueries. Should only be called during DAG
   * building.
   * @param root_dag_builder The root DAG builder. The root stores pointers to all
   * subqueries.
   * @param query_ast The current JSON node to build a DAG for.
   * @param cat DB catalog for the current user.
   * @param render_opts Additional build options for render queries.
   */
  RelAlgDagBuilder(RelAlgDagBuilder& root_dag_builder,
                   const rapidjson::Value& query_ast,
                   const Catalog_Namespace::Catalog* cat,
                   SchemaProviderPtr schema_provider,
                   const RenderInfo* render_opts);

  RelAlgDagBuilder(const std::string& query_ra,
                   int db_id,
                   SchemaProviderPtr schema_provider,
                   const RenderInfo* render_info);

 private:
  void build(const rapidjson::Value& query_ast, RelAlgDagBuilder& root_dag_builder);

  const Catalog_Namespace::Catalog* cat_;
  int db_id_;
  SchemaProviderPtr schema_provider_;
  const RenderInfo* render_info_;
};

using RANodeOutput = std::vector<RexInput>;

RANodeOutput get_node_output(const RelAlgNode* ra_node);

std::string tree_string(const RelAlgNode*, const size_t depth = 0);

inline InputColDescriptor column_var_to_descriptor(const Analyzer::ColumnVar* var) {
  return InputColDescriptor(var->get_column_info(), var->get_rte_idx());
}
