/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/RelScanTree.h"

namespace QueryRenderer {

namespace {
/**
 * This takes std::decay a step further. std::decay surprisingly doesn't decay pointer
 * types.
 * For example, with an type such as "const RelProject *&"
 * std::decay_t results in "const RelProject*"
 * This decay_pointer results in "RelProject"
 */
template <typename T>
struct decay_pointer {
 private:
  using U = typename std::decay_t<T>;

 public:
  using type = typename std::conditional_t<
      std::is_pointer_v<U>,
      typename std::remove_cv_t<typename std::remove_pointer_t<U>>,
      U>;
};

template <typename T>
using decay_pointer_t = typename decay_pointer<T>::type;

/**
 * Recursively finds the top level projection node in an RA tree of a projection query.
 * Traversal is stopped on non-projection-query nodes (i.e. RelAggregate/RelCompound)
 * @param curr_node The current node to inspect in the tree.
 */
const RelProject* find_top_level_project_node(const RelAlgNode* curr_node) {
  const auto project = dynamic_cast<const RelProject*>(curr_node);
  if (project) {
    return project;
  }

  if (dynamic_cast<const RelFilter*>(curr_node) ||
      dynamic_cast<const RelSort*>(curr_node)) {
    // Filter/sort preserves input fields, so visit input
    CHECK_EQ(curr_node->inputCount(), size_t(1))
        << curr_node->toString(RelRexToStringConfig::defaults());
    return find_top_level_project_node(curr_node->getInput(0));
  }

  // Stop node traversal on everything else (RelJoin, RelAggregate, RelCompound,
  // RelDeepLeftInnerJoin, RelLogicalValues, RelModify)
  return nullptr;
}

[[noreturn]] void throw_mismatched_rex_input_source(const RelAlgNode* source,
                                                    const RelAlgNode* expected_lhs) {
  auto cfg = RelRexToStringConfig::defaults();
  cfg.attributes_only = true;
  throw std::runtime_error(
      "RelScanTree::getScanNodeForOutputIndex: RexInput source (" +
      (source ? source->toString(cfg) : std::string("null")) +
      ") does not match project input (" +
      (expected_lhs ? expected_lhs->toString(cfg) : std::string("null")) + ")");
}

// After bind_inputs(), a RelProject over a RelJoin binds RexInputs to the join's
// children (see get_node_output(RelJoin)), not the join itself. Convert that
// child-local index to a join-local index so the RelJoin visitor can pick lhs vs
// rhs. Also accept a RexInput already sourced at the join (injectInputColumn).
bool set_join_local_index_from_rex_input(const RexInput* input_node,
                                         const RelAlgNode& lhs_ra,
                                         uint32_t& current_index) {
  auto const* join = dynamic_cast<const RelJoin*>(&lhs_ra);
  if (!join) {
    return false;
  }
  auto const* src = input_node->getSourceNode();
  if (src == join->getInput(0) || src == join) {
    current_index = input_node->getIndex();
    return true;
  }
  if (src == join->getInput(1)) {
    current_index =
        input_node->getIndex() + static_cast<uint32_t>(join->getInput(0)->size());
    return true;
  }
  throw_mismatched_rex_input_source(src, &lhs_ra);
  return false;
}

}  // namespace

std::unique_ptr<RelScanTree> RelScanTree::create(RelAlgDag& rel_alg_dag) {
  auto const_proj_node = find_top_level_project_node(&rel_alg_dag.getRootNode());
  if (!const_proj_node) {
    // early out
    return nullptr;
  }

  CHECK_EQ(const_proj_node->inputCount(), size_t(1))
      << const_proj_node->toString(RelRexToStringConfig::defaults());

  // gather up all the scanned tables (RelScan) in the query by visiting the inputs of the
  // top-level RelProject node
  return std::make_unique<RelScanTree>(*const_proj_node, getNodes(rel_alg_dag));
}

RelScanTree::RelScanTree(const RelProject& root_project_node,
                         std::vector<std::shared_ptr<RelAlgNode>>& nodes)
    : root_project_node_{root_project_node}, ra_tree_nodes_{nodes} {
  // build the tree starting from the top-level project
  auto& root_node = *(nodes_.emplace_back(std::make_unique<TreeNode>(root_project_node)));
  CHECK_EQ(root_project_node.inputCount(), 1u);
  auto* lhs = buildRelScanTree(root_project_node.getInput(0));
  if (lhs) {
    root_node.lhs = lhs;
    lhs->parent = &root_node;
  }
}

const RelScan& RelScanTree::operator[](const size_t index) {
  CHECK(std::holds_alternative<const RelScan*>(
      scan_node_leaves_[index]->rel_alg_node_ptr_variant))
      << index;
  return static_cast<const RelScan&>(scan_node_leaves_[index]->rel_alg_node);
}

std::pair<const RelScan*, uint32_t> RelScanTree::getScanNodeForOutputIndex(
    const uint32_t output_index) const {
  auto const* current_tree_node = getRootNode();
  auto current_index = output_index;
  const RelScan* rel_scan{nullptr};

  while (current_tree_node) {
    current_tree_node = std::visit(
        [&](auto&& current_node) -> const TreeNode* {
          using T = decay_pointer_t<decltype(current_node)>;
          if constexpr (std::is_same_v<RelProject, T>) {
            // check the input type. We currently continue to traverse RexInput only
            auto const* scalar_input = current_node->getProjectAt(current_index);
            if (auto const* input_node = dynamic_cast<const RexInput*>(scalar_input);
                input_node != nullptr) {
              if (current_tree_node->lhs) {
                auto const& lhs_ra = current_tree_node->lhs->rel_alg_node;
                if (!set_join_local_index_from_rex_input(
                        input_node, lhs_ra, current_index)) {
                  if (input_node->getSourceNode() != &lhs_ra) {
                    throw_mismatched_rex_input_source(input_node->getSourceNode(),
                                                      &lhs_ra);
                  }
                  current_index = input_node->getIndex();
                }
              } else {
                return nullptr;
              }
            } else {
              // Found an output that is created via an operator or expression. This is
              // not currently supported.
              throw std::runtime_error(
                  "Projected output \"" + current_node->getFieldName(current_index) +
                  "\" with index " + std::to_string(current_index) +
                  " in query RA is not a direct reference to a physical table column. "
                  "This is not supported");
            }
            return current_tree_node->lhs;
          } else if constexpr (std::is_same_v<RelJoin, T>) {
            auto const* lhs = current_node->getInput(0);
            if (current_index < lhs->size()) {
              // going down the left branch of the join.
              // current_index can stay the same
              return current_tree_node->lhs;
            }
            // going down the right branch of the join, so negate the size the left
            // branch to make the index right-branch local
            current_index -= lhs->size();
            return current_tree_node->rhs;
          } else if constexpr (std::is_same_v<RelScan, T>) {
            // found the RelScan, return nullptr to stop the loop
            rel_scan = current_node;
            return nullptr;
          } else {
            // RelFilter/RelSort preserves index, so pass thru on the lhs
            static_assert(std::is_same_v<RelFilter, T> || std::is_same_v<RelSort, T>);
            return current_tree_node->lhs;
          }
        },
        current_tree_node->rel_alg_node_ptr_variant);
  }
  return {rel_scan, current_index};
}

void RelScanTree::injectInputColumn(const RelScan& table_scan_node,
                                    const int column_index,
                                    const std::string& output_column_name) {
  const RelAlgNode* input_node = &table_scan_node;
  auto input_index = column_index;

  auto start_node_itr = std::find_if(
      scan_node_leaves_.begin(), scan_node_leaves_.end(), [&](auto const* tree_node) {
        return &tree_node->rel_alg_node == input_node;
      });

  CHECK(start_node_itr != scan_node_leaves_.end())
      << output_column_name << ", " << column_index;

  // unwind the tree, eventually adding a RexInput node to each RelProject node along
  // the way.  We need to keep track of the actual input RelAlgNode and the column index
  // as that's required when creating a new RexInput node. The index needs to be
  // reflective of the actual input column used, so we "unwind" to properly build up the
  // index/input accordingly
  auto* tree_node = *start_node_itr;
  while (tree_node->parent != nullptr) {
    tree_node = tree_node->parent;

    // visit the current RelAlgNode variant. If it's a project node, add a new RexInput.
    // Also properly update the index depending on the RelAlgNode visited.
    std::visit(
        [&](auto&& current_node) {
          using T = decay_pointer_t<decltype(current_node)>;
          if constexpr (std::is_same_v<RelProject, T>) {
            // We need a non-const version of the project node in order to
            // append columns to it. However, during tree traversal, we can only
            // traverse const nodes because some nodes' inputs are stored as const
            // (i.e.  RelFilter). We could just cast constness away, but the 'safer'
            // way would just be to find the equivalent non-const ptr from the
            // original nodes.  So that's what we'll do with a simple find_if.
            auto& non_const_project = getNonConstProject(*current_node);
            non_const_project.appendInput(
                output_column_name, std::make_unique<RexInput>(input_node, input_index));
            input_index = current_node->size() - 1;
          } else if constexpr (std::is_same_v<RelJoin, T>) {
            // unwind the join. Was the original RelScan in the left or right hand side
            // of the join?
            auto const* rhs = current_node->getInput(1);
            if (input_node == rhs) {
              input_index += current_node->getInput(0)->size();
            }
          } else {
            // do nothing to the index. RelFilter/RelSort preserve column index. RelScan
            // should never be hit,
            static_assert(std::is_same_v<RelFilter, T> || std::is_same_v<RelSort, T> ||
                          std::is_same_v<RelScan, T>);
          }
        },
        tree_node->rel_alg_node_ptr_variant);

    input_node = &tree_node->rel_alg_node;
  }
}

const RelScanTree::TreeNode* RelScanTree::getRootNode() const {
  if (nodes_.size()) {
    return nodes_[0].get();
  }
  return nullptr;
}

RelProject& RelScanTree::getNonConstProject(const RelProject& const_proj_to_find) {
  auto itr =
      std::find_if(ra_tree_nodes_.begin(), ra_tree_nodes_.end(), [&](auto& node_ptr) {
        return node_ptr.get() == &const_proj_to_find;
      });
  CHECK(itr != ra_tree_nodes_.end());
  return *static_cast<RelProject*>(itr->get());
};

RelScanTree::TreeNode* RelScanTree::buildRelScanTree(const RelAlgNode* rel_alg_node) {
  // Project/Filter/Sort preserves input fields with a single input, so visit input, but
  // also preserve the order of the node hierarchy to reach the scan node
  if (emplaceTreeNode<RelProject>(rel_alg_node) ||
      emplaceTreeNode<RelSort>(rel_alg_node) ||
      emplaceTreeNode<RelFilter>(rel_alg_node)) {
    auto& new_node = *nodes_.back();
    auto* input_node = buildRelScanTree(rel_alg_node->getInput(0));
    if (input_node) {
      new_node.lhs = input_node;
      input_node->parent = &new_node;
    }
    return &new_node;
  }

  if (emplaceTreeNode<RelJoin>(rel_alg_node)) {
    // Join scan nodes from the two join branches
    CHECK_EQ(rel_alg_node->inputCount(), 2u)
        << rel_alg_node->toString(RelRexToStringConfig::defaults());

    auto& new_node = *nodes_.back();
    auto* lhs_node = buildRelScanTree(rel_alg_node->getInput(0));
    auto* rhs_node = buildRelScanTree(rel_alg_node->getInput(1));
    if (lhs_node) {
      new_node.lhs = lhs_node;
      lhs_node->parent = &new_node;
    }
    if (rhs_node) {
      new_node.rhs = rhs_node;
      rhs_node->parent = &new_node;
    }
    return &new_node;
  }

  if (emplaceTreeNode<RelScan>(rel_alg_node)) {
    auto& new_node = *nodes_.back();
    scan_node_leaves_.emplace_back(&new_node);
    return &new_node;
  }

  // Stop node traversal on everything else (RelAggregate, RelCompound,
  // RelDeepLeftInnerJoin, RelLogicalValues, RelModify)
  return nullptr;
}

}  // namespace QueryRenderer
