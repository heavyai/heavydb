/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/NonInsituQueryClassifier.h"

#include "GfxDriver/RenderLogger.h"
#include "QueryEngine/Visitors/RelRexDagVisitor.h"

namespace QueryRenderer {

namespace {

using ProjectNodePtr = const RelProject*;
using CompoundNodePtr = const RelCompound*;
using FilterNodePtr = const RelFilter*;
using SortNodePtr = const RelSort*;

/**
 * Class that traverses the outputs of the top-level (outermost) RelProject or RelCompound
 * node of a RelAlg DAG to determine if it's associated RenderInfo instance should be
 * set or forced to run non-insitu.
 *
 * NOTE: this class is a derivation of the RelRexDagVisitor class but it's traversal logic
 * is different. We're only concerned with traversing the top-level node that manages the
 * outputs. Not it's inputs. This class only traverses the dependencies of the top-level
 * node's output fields.
 *
 * NOTE: this DAG traversal is not as efficient as it could be. Since the RelRexDagVisitor
 * class does not have a way to stop the DAG traversal, node visitation can carry on even
 * after the RenderInfo instance is forced non-insitu (at which point the traversal could
 * stop). This performance hit should not be at all significant in general usage.
 * Code simplification was prioritized over efficiency here initially.
 */
template <bool is_hittesting_enabled>
class NonInsituClassifier : public RelRexDagVisitor {
 public:
  NonInsituClassifier(RenderInfo& render_info)
      : RelRexDagVisitor()
      , render_info_{render_info}
      , active_output_idx_{-1}
      , current_node_idx_{-1} {}

  /**
   * Primary entry point for classifying a query (via a RelAlg DAG) as non-insitu, either
   * forced, or direct
   */
  void classify(const RelScanTree& rel_scan_tree) {
    RENDER_LOG_SCOPE();
    if (!rel_scan_tree.size()) {
      // no direct table scan nodes are found in the tree, which means all outputs are
      // descendants of aggregates, so we are able to say this is a non-insitu render.
      render_info_.setNonInSitu();
      return;
    }

    auto const& project = rel_scan_tree.getRootProjectNode();
    visitTopLevelNode(project);

    if constexpr (is_hittesting_enabled) {
      auto const total_non_null = static_cast<size_t>(
          std::count_if(output_column_aggregate_deps_.begin(),
                        output_column_aggregate_deps_.end(),
                        [](auto const* node_ptr) { return node_ptr != nullptr; }));

      if (total_non_null > 0) {
        // in this case at least one output column, but not all output columns, depend on
        // an aggregate. In this case we force non-insitu to be able to hit-test the
        // aggregate-dependent queries, but we allow to hit-test extra columns from the
        // primary projected tables.
        render_info_.forceNonInSitu();
      }
    }
  }

 private:
  /**
   * Templated function to handle some of the logic around visiting a top-level RelProject
   * or RelCompound node
   */
  void visitTopLevelNode(const RelProject& top_level_project) {
    // Check for simple projections where the input is a RelSort node
    // These queries will end up in CPU memory so should be treated as non-insitu
    // See [QE-1148] for motivation
    if (top_level_project.isSimple()) {
      CHECK_EQ(size_t(1), top_level_project.inputCount());
      auto const input_ra = top_level_project.getInput(0);
      if (dynamic_cast<const RelSort*>(input_ra)) {
        render_info_.setNonInSitu();
        return;
      }
    }

    CHECK_EQ(output_column_aggregate_deps_.size(), 0u);

    // get the size of the RexScalar outputs of the project/compound
    auto const scalar_size = top_level_project.size();
    CHECK_EQ(top_level_project.getFields().size(), scalar_size);

    // init the vector of dependent aggregate nodes to nullptrs
    output_column_aggregate_deps_.insert(
        output_column_aggregate_deps_.begin(), scalar_size, nullptr);

    // visit the RexScalar outputs
    for (auto i = 0u; i < scalar_size; ++i) {
      active_output_idx_ = i;
      auto const prev_node_idx = current_node_idx_;
      current_node_idx_ = i;
      RelRexDagVisitor::visit(top_level_project.getProjectAt(i));
      current_node_idx_ = prev_node_idx;
    }
  }

  /**
   * Recursively visits a generic RelAlgNode pointer. This method determines the
   * derivation type of the node and calls the appropriate handler.
   */
  void visit(const RelAlgNode* rel_alg_node) final {
    // NOTE: this is an exact copy of RelRexDagVisitor::visit(RelAlgNode const*)
    // except that we do not need to traverse inputs in this case.
    // So only do the cast and visit.
    castAndVisit(rel_alg_node);
  }

  /**
   * Adds a RelAggregate or RelCompound node to the list of aggregate dependencies for an
   * output column.
   */
  void addAggregateNodeAsOutputDependency(const RelAlgNode* node) {
    CHECK(active_output_idx_ >= 0 &&
          active_output_idx_ < static_cast<int32_t>(output_column_aggregate_deps_.size()))
        << active_output_idx_ << ":" << output_column_aggregate_deps_.size();
    if (output_column_aggregate_deps_[active_output_idx_] == nullptr) {
      output_column_aggregate_deps_[active_output_idx_] = node;
    }
  }

  void visit(const RelProject* node) override {
    CHECK_GE(current_node_idx_, 0);
    CHECK_LT(current_node_idx_, static_cast<int>(node->size()));
    RelRexDagVisitor::visit(node->getProjectAt(current_node_idx_));
  }

  void visit(const RelFilter* node) override {
    CHECK_EQ(node->inputCount(), 1u);
    visit(node->getInput(0));
  }

  void visit(const RelSort* node) override {
    CHECK_EQ(node->inputCount(), 1u);
    visit(node->getInput(0));
  }

  void visit(const RelJoin* node) override {
    CHECK_EQ(node->inputCount(), 2u);
    auto const* lhs = node->getInput(0);
    auto const lhs_size = static_cast<int32_t>(lhs->size());
    if (current_node_idx_ < lhs_size) {
      visit(lhs);
    } else {
      auto const prev_node_idx = current_node_idx_;
      current_node_idx_ -= lhs_size;
      visit(node->getInput(1));
      current_node_idx_ = prev_node_idx;
    }
  }

  void visit(const RelAggregate* node) final { addAggregateNodeAsOutputDependency(node); }

  void visit(const RelCompound* node) final {
    if (node->isAggregate() || node->getAggExprSize()) {
      // if the visited compound is an aggregate, then we can stop traversal early.
      addAggregateNodeAsOutputDependency(node);
      return;
    }
    // NOTE: we do not have to worry about the RelCompound::filter_expr_ here since we're
    // only concerned about traversing the tree for the final output columns. NOTE; the
    // default RelRexDagVisitor::visit(RelCompound*) visits the fiter node.
    CHECK_LT(current_node_idx_, static_cast<int32_t>(node->getScalarSourcesSize()));
    RelRexDagVisitor::visit(node->getScalarSource(current_node_idx_));
  }

  void visit(const RelModify*) final {
    // should never get here, but throw an error if these somehow slip in.
    throw std::runtime_error("It is illegal to use modify expressions in render queries");
  }

  void visit(const RelTableFunction* node) final {
    // cursor-less table functions are also going to set non-insitu for now
    // NOTE: we may want to force this non-insitu instead, or perhaps let it run insitu
    // (if the table function portion isn't expensive to repeat - could it be cached?)
    // once we get proper rowid injection working
    if (!node->getColInputsSize()) {
      render_info_.setNonInSitu();
      return;
    }

    CHECK_LT(current_node_idx_, static_cast<int32_t>(node->size()));
    RelRexDagVisitor::visit(node->getTargetExpr(current_node_idx_));
  }

  void visit(const RelTranslatedJoin*) override {
    // As of 05/03/2022, RelTranslatedJoin is only created and used during execution of a
    // query, so we should be safe to ignore it here. Will throw an error tho just in
    // case.
    throw std::runtime_error(
        "RelTranslatedJoin is not currently supported in render queries");
  }

  void visit(const RelLeftDeepInnerJoin*) override {
    // As of 05/03/2022, RelLeftDeepInnerJoin nodes are only created as part of the
    // optimize step, so as long as non-insitu classification is performed before
    // optimization, we should never reach here. Will throw an error just in case.
    throw std::runtime_error(
        "RelLeftDeepInnerJoin is not currently supported in render queries");
  }

  void visit(const RelLogicalUnion* node) override {
    // visit each input of a union. The current_node_idx_ can carry through to all
    for (auto i = 0u; i < node->inputCount(); ++i) {
      visit(node->getInput(i));
    }
  }

  void visit(const RelLogicalValues* node) override {
    // will consider logical values as aggrgates for now as it's unclear how to handle
    // them at this point
    // TODO(croot): as it currently stands, this should only be reached if an output has a
    // direct path to logical values. In that case it might make sense to keep the query
    // going and rebuild the logical values during the hit-test call, if this column is
    // requested that is.
    addAggregateNodeAsOutputDependency(node);
    return;
  }

  void visit(const RexInput* rex_input) override {
    auto const prev_node_idx = current_node_idx_;
    current_node_idx_ = rex_input->getIndex();
    visit(rex_input->getSourceNode());
    current_node_idx_ = prev_node_idx;
  }

  void visit(const RexRef*) final {
    // I beleve a RexRef is only possible when referencing an aggregate expression of some
    // kind, so force non-insitu if we ever hit a RexRef
    // TODO(croot): verify that RexRefs are aggregate-only. It looks that way according to
    // RelAlgDag.cpp:creat_compound(), but I can't be sure.
    if constexpr (is_hittesting_enabled) {
      render_info_.forceNonInSitu();
    }
  }

  void visit(const RexWindowFunctionOperator*) final {
    // window functions are always going to be non-insitu, for now
    // NOTE: we may want to force this non-insitu instead, or perhaps let it run insitu
    // (if the window function portion isn't expensive to repeat - could it be cached?)
    render_info_.setNonInSitu();
  }

  RenderInfo& render_info_;

  // stores a pointer to either a RelAggregate or an aggregate RelCompound node that is a
  // dependency of an output column RexScalar.
  std::vector<const RelAlgNode*> output_column_aggregate_deps_;

  // stores the output column index that is currently being traversed
  int32_t active_output_idx_;
  int32_t current_node_idx_;
};

}  // namespace

void NonInsituQueryClassifier::classify(RenderInfo& render_info,
                                        const RelAlgDag& /*rel_alg_dag*/,
                                        const RelScanTree* rel_scan_tree) {
  if (rel_scan_tree) {
    if (render_info.getRenderQueryOptions().isHitTestingEnabled()) {
      NonInsituClassifier<true>{render_info}.classify(*rel_scan_tree);
    } else {
      NonInsituClassifier<false>{render_info}.classify(*rel_scan_tree);
    }
  }
}

}  // namespace QueryRenderer
