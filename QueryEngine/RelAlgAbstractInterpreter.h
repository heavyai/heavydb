#ifndef QUERYENGINE_RELALGABSTRACTINTERPRETER_H
#define QUERYENGINE_RELALGABSTRACTINTERPRETER_H

#include "../Catalog/Catalog.h"

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
// interpretation will not have any references to RelAlgInput objects.
class RexAbstractInput : public RexScalar {
 public:
  RexAbstractInput(const unsigned in_index) : in_index_(in_index) {}

 private:
  unsigned in_index_;
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
  // The 'arg' expression is owned by the project node which created it.
  RexAgg(const SQLAgg agg, const Rex* arg) : agg_(agg), arg_(arg){};

 private:
  const SQLAgg agg_;
  const Rex* arg_;
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
  RelProject(const std::vector<const RexScalar*>& exprs, const std::vector<std::string>& fields) : fields_(fields) {
    CHECK_EQ(exprs.size(), fields.size());
    for (auto expr : exprs) {
      scalar_exprs_.emplace_back(expr);
    }
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
  // Takes ownership of the aggregate expressions, the group by expressions
  // are owned by the previous project node. The pointers to group by expressions
  // and to the arguments of aggregate expressions are guaranteed to be the
  // same the project node created so that the codegen can cache them.
  RelAggregate(const std::vector<const RexScalar*>& group_exprs, const std::vector<const RexAgg*>& agg_exprs)
      : group_exprs_(group_exprs) {
    for (auto agg_expr : agg_exprs) {
      agg_exprs_.emplace_back(agg_expr);
    }
  }

 private:
  const std::vector<const RexScalar*> group_exprs_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
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
  RelFilter(const RexScalar* filter) : filter_(filter) {}

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
