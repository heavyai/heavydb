#ifndef QUERYENGINE_RELALGABSTRACTINTERPRETER_H
#define QUERYENGINE_RELALGABSTRACTINTERPRETER_H

#include "CalciteDeserializerUtils.h"

#include "../Catalog/Catalog.h"

#include <boost/variant.hpp>
#include <rapidjson/document.h>

#include <memory>

class ScanScope {
  // TODO
};

class ScanBufferDesc {
 public:
  ScanBufferDesc();                           // for results of other queries
  ScanBufferDesc(const TableDescriptor* td);  // for tables

 private:
  const TableDescriptor* td_;
};

class Rex {
 public:
  virtual ~Rex(){};
};

class RexScalar : public Rex {};

// For internal use of the abstract interpreter only. The result after abstract
// interpretation will not have any references to 'RexAbstractInput' objects.
class RexAbstractInput : public RexScalar {
 public:
  RexAbstractInput(const unsigned in_index) : in_index_(in_index) {}

  unsigned getIndex() const { return in_index_; }

 private:
  unsigned in_index_;
};

class RexLiteral : public RexScalar {
 public:
  RexLiteral(const int64_t val,
             const SQLTypes type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kDECIMAL, type);
  }

  RexLiteral(const double val,
             const SQLTypes type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kDOUBLE, type);
  }

  RexLiteral(const std::string& val,
             const SQLTypes type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kTEXT, type);
  }

  RexLiteral(const bool val,
             const SQLTypes type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kBOOLEAN, type);
  }

  RexLiteral() : literal_(nullptr), type_(kNULLT), scale_(0), precision_(0), type_scale_(0), type_precision_(0) {}

 private:
  const boost::variant<int64_t, double, std::string, bool, void*> literal_;
  const SQLTypes type_;
  const unsigned scale_;
  const unsigned precision_;
  const unsigned type_scale_;
  const unsigned type_precision_;
};

class RexOperator : public RexScalar {
 public:
  RexOperator(const SQLOps op, const std::vector<const RexScalar*> operands) : op_(op) {
    for (auto operand : operands) {
      operands_.emplace_back(operand);
    }
  }

  size_t size() const { return operands_.size(); }

  const RexScalar* getOperand(const size_t idx) const {
    CHECK(idx < operands_.size());
    return operands_[idx].get();
  }

  SQLOps getOperator() const { return op_; }

 private:
  const SQLOps op_;
  std::vector<std::unique_ptr<const RexScalar>> operands_;
};

class RelAlgNode;

// The actual input node understood by the Executor.
// The in_index_ is relative to the output of node_.
class RexInput : public RexAbstractInput {
 public:
  RexInput(const RelAlgNode* node, const unsigned in_index) : RexAbstractInput(in_index), node_(node) {}

 private:
  const RelAlgNode* node_;
};

class RexAgg : public Rex {
 public:
  RexAgg(const SQLAgg agg,
         const bool distinct,
         const SQLTypes type,
         const bool nullable,
         const std::vector<size_t> operands)
      : agg_(agg), distinct_(distinct), type_(type), nullable_(nullable), operands_(operands){};

 private:
  const SQLAgg agg_;
  const bool distinct_;
  const SQLTypes type_;
  const bool nullable_;
  const std::vector<size_t> operands_;
};

class RelAlgNode {
 public:
  RelAlgNode() : context_data_(nullptr) {}
  void setContextData(const void* context_data) { context_data_ = context_data; }
  const size_t inputCount() const { return inputs_.size(); }
  const RelAlgNode* getInput(const size_t idx) const {
    CHECK(idx < inputs_.size());
    return inputs_[idx].get();
  }

  const RelAlgNode* getInputAndRelease(const size_t idx) {
    CHECK(idx < inputs_.size());
    return inputs_[idx].release();
  }

  const void addInput(const RelAlgNode* input) { inputs_.emplace_back(input); }

  bool replaceInput(const RelAlgNode* old_input, const RelAlgNode* input) {
    for (auto& input_ptr : inputs_) {
      if (input_ptr.get() == old_input) {
        input_ptr.reset(input);
        return true;
      }
    }
    return false;
  }

  virtual ~RelAlgNode() {}

 protected:
  std::vector<std::unique_ptr<const RelAlgNode>> inputs_;
  const void* context_data_;
};

class RelScan : public RelAlgNode {
 public:
  RelScan(const TableDescriptor* td, const std::vector<std::string>& field_names)
      : td_(td), field_names_(field_names) {}

  size_t size() const { return field_names_.size(); }

 private:
  const TableDescriptor* td_;
  const std::vector<std::string> field_names_;
};

class RelProject : public RelAlgNode {
 public:
  // Takes memory ownership of the expressions.
  RelProject(const std::vector<const RexScalar*>& exprs,
             const std::vector<std::string>& fields,
             const RelAlgNode* input)
      : fields_(fields) {
    CHECK_EQ(exprs.size(), fields.size());
    for (auto expr : exprs) {
      scalar_exprs_.emplace_back(expr);
    }
    inputs_.emplace_back(input);
  }

  void setExpressions(const std::vector<const RexScalar*>& exprs) {
    decltype(scalar_exprs_)().swap(scalar_exprs_);
    for (auto expr : exprs) {
      scalar_exprs_.emplace_back(expr);
    }
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

  size_t size() const { return scalar_exprs_.size(); }

  const RexScalar* getProjectAt(const size_t idx) const {
    CHECK(idx < scalar_exprs_.size());
    return scalar_exprs_[idx].get();
  }

  std::vector<const RexScalar*> getExpressionsAndRelease() {
    std::vector<const RexScalar*> result;
    for (auto& expr : scalar_exprs_) {
      result.push_back(expr.release());
    }
    return result;
  }

  const std::vector<std::string>& getFields() const { return fields_; }

 private:
  std::vector<std::unique_ptr<const RexScalar>> scalar_exprs_;
  const std::vector<std::string> fields_;
};

class RelAggregate : public RelAlgNode {
 public:
  // Takes ownership of the aggregate expressions.
  RelAggregate(const std::vector<size_t>& group_indices,
               const std::vector<const RexAgg*>& agg_exprs,
               const std::vector<std::string>& fields,
               const RelAlgNode* input)
      : group_indices_(group_indices), fields_(fields) {
    for (auto agg_expr : agg_exprs) {
      agg_exprs_.emplace_back(agg_expr);
    }
    inputs_.emplace_back(input);
  }

  size_t size() const { return group_indices_.size() + agg_exprs_.size(); }

  const std::vector<size_t>& getGroupIndices() const { return group_indices_; }

  const std::vector<std::string>& getFields() const { return fields_; }

  std::vector<const RexAgg*> getAggregatesAndRelease() {
    std::vector<const RexAgg*> result;
    for (auto& agg_expr : agg_exprs_) {
      result.push_back(agg_expr.release());
    }
    return result;
  }

 private:
  const std::vector<size_t> group_indices_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  const std::vector<std::string> fields_;
};

enum class RelJoinType { INNER, LEFT };

class RelJoin : public RelAlgNode {
 public:
  RelJoin(const RelAlgNode* lhs, const RelAlgNode* rhs, const RexScalar* condition, const RelJoinType join_type)
      : condition_(condition), join_type_(join_type) {
    inputs_.emplace_back(lhs);
    inputs_.emplace_back(rhs);
  }

 private:
  const std::unique_ptr<const RexScalar> condition_;
  const RelJoinType join_type_;
};

class RelFilter : public RelAlgNode {
 public:
  RelFilter(const RexScalar* filter, const RelAlgNode* input) : filter_(filter) { inputs_.emplace_back(input); }

  const RexScalar* getCondition() const { return filter_.get(); }

  const RexScalar* getAndReleaseCondition() { return filter_.release(); }

  void setCondition(const RexScalar* condition) { filter_.reset(condition); }

 private:
  std::unique_ptr<const RexScalar> filter_;
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
  RelCompound(const RexScalar* filter_expr,
              const std::vector<const Rex*>& target_exprs,
              const std::vector<size_t>& group_indices,
              const std::vector<const RexAgg*>& agg_exprs,
              const std::vector<std::string>& fields,
              const std::vector<const RexScalar*>& scalar_sources)
      : filter_expr_(filter_expr), target_exprs_(target_exprs), group_indices_(group_indices), fields_(fields) {
    CHECK_EQ(fields.size(), target_exprs.size());
    for (auto agg_expr : agg_exprs) {
      agg_exprs_.emplace_back(agg_expr);
    }
    for (auto scalar_source : scalar_sources) {
      scalar_sources_.emplace_back(scalar_source);
    }
  }

  size_t size() const { return target_exprs_.size(); }

 private:
  const std::unique_ptr<const RexScalar> filter_expr_;
  const std::vector<const Rex*> target_exprs_;
  const std::vector<size_t> group_indices_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  const std::vector<std::string> fields_;
  std::vector<std::unique_ptr<const RexScalar>>
      scalar_sources_;  // building blocks for group_indices_ and agg_exprs_; not actually projected, just owned
};

enum class SortDirection { Ascending, Descending };

enum class NullSortedPosition { First, Last };

class SortField {
 public:
  SortField(const size_t field, const SortDirection sort_dir, const NullSortedPosition nulls_pos)
      : field_(field), sort_dir_(sort_dir), nulls_pos_(nulls_pos) {}

 private:
  const size_t field_;
  const SortDirection sort_dir_;
  const NullSortedPosition nulls_pos_;
};

class RelSort : public RelAlgNode {
 public:
  RelSort(const std::vector<SortField>& collation, const RelAlgNode* input) : collation_(collation) {
    inputs_.emplace_back(input);
  }

 private:
  const std::vector<SortField> collation_;
};

std::unique_ptr<const RelAlgNode> ra_interpret(const rapidjson::Value&, const Catalog_Namespace::Catalog&);

#endif  // QUERYENGINE_RELALGABSTRACTINTERPRETER_H
