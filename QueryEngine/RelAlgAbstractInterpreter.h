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

class LoweringInfo {
  // TODO
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

 protected:
  std::vector<std::unique_ptr<const RelAlgNode>> inputs_;  // TODO
  const void* context_data_;
};

class RelScan : public RelAlgNode {
 public:
  RelScan(const TableDescriptor* td, const std::vector<std::string>& field_names)
      : td_(td), field_names_(field_names) {}

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

  // True iff all the projected expressions are inputs. If true,
  // this node can be elided and merged into the previous node
  // since it's just a subset and / or permutation of its outputs.
  bool isSimple() const;

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

 private:
  const std::vector<size_t> group_indices_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  const std::vector<std::string> fields_;
};

class RelJoin : public RelAlgNode {
 public:
  RelJoin(const RelAlgNode* lhs, const RelAlgNode* rhs) {
    inputs_.emplace_back(lhs);
    inputs_.emplace_back(rhs);
  }
};

class RelFilter : public RelAlgNode {
 public:
  RelFilter(const RexScalar* filter, const RelAlgNode* input) : filter_(filter) { inputs_.emplace_back(input); }

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
              const std::vector<const RexScalar*>& group_exprs,
              const std::vector<const RexAgg*>& agg_exprs,
              const std::vector<std::string>& fields,
              const std::vector<const RexScalar*>& scalar_sources)
      : filter_expr_(filter_expr), target_exprs_(target_exprs), group_exprs_(group_exprs), fields_(fields) {
    CHECK_EQ(fields.size(), target_exprs.size());
    for (auto agg_expr : agg_exprs) {
      agg_exprs_.emplace_back(agg_expr);
    }
    for (auto scalar_source : scalar_sources) {
      scalar_sources_.emplace_back(scalar_source);
    }
  }

 private:
  const std::unique_ptr<const RexScalar> filter_expr_;
  const std::vector<const Rex*> target_exprs_;
  const std::vector<const RexScalar*> group_exprs_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  const std::vector<std::string> fields_;
  std::vector<std::unique_ptr<const RexScalar>>
      scalar_sources_;  // building blocks for group_exprs_ and agg_exprs_; not actually projected, just owned
};

// A sequence of nodes to be executed as-is. The node / buffer elision is completed
// at this point and every RelProject / RelAggregate / RelCompound node should output
// a new buffer and trigger a result reduction between multiple devices.
class RelSequence : public RelAlgNode {
 public:
  void addNode(const RelAlgNode* node) { sequence_.push_back(node); }

 private:
  std::vector<const RelAlgNode*> sequence_;
};

class RelNop : public RelAlgNode {};

LoweringInfo ra_interpret(const rapidjson::Value&, const Catalog_Namespace::Catalog&);

#endif  // QUERYENGINE_RELALGABSTRACTINTERPRETER_H
