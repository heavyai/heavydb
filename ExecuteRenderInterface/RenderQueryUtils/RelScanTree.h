/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <variant>
#include <vector>

#include "QueryEngine/RelAlgDag.h"

namespace QueryRenderer {

/**
 * Class that builds a simple, bi-directional tree that represents the RelAlgNode DAG
 * paths that are valid for auto column injection. It uses a std::variant as a primary
 * member of the tree node that designates the node's internal RelAlgNode type. The
 * variant is used for two primary purposes: 1) using std::visit allows for compile time
 * validation that all possible variants are accounted for. 2) It allows for skipping
 * repeated dynamic_casts (which for a cast can be expensive thanks to RTTI and other
 * possible run-time checks)
 */
class RelScanTree : public RelAlgDagModifier {
 public:
  /**
   * Builds a RelScanTree from a RelAlgDag. You are able to get modifiable RelAlgNode
   * objects from the tree, allowing for ways to inject nodes after calcite parse. Any
   * modification steps need to be performed before the RelAlgDag is optimized.
   * The RelAlgDag needs to outlive the RelScanTree as well.
   */
  static std::unique_ptr<RelScanTree> create(RelAlgDag& rel_alg_dag);

  RelScanTree(const RelProject& root_project_node,
              std::vector<std::shared_ptr<RelAlgNode>>& nodes);

  /**
   * Gets the root/top-level RelProject node of the tree
   */
  const RelProject& getRootProjectNode() const { return root_project_node_; }

  std::vector<std::shared_ptr<RelAlgNode>>& getRelAlgNodes() const {
    return ra_tree_nodes_;
  }

  /**
   * returns the number of valid RelScan nodes in the tree
   */
  size_t size() const { return scan_node_leaves_.size(); }

  /**
   * Get a RelScan node at a specific index
   */
  const RelScan& operator[](const size_t index);

  /**
   * Finds a RelScan node in an RA tree for a specific output index.
   * @param output_index The index of an output field to traverse. The index is relative
   * to the root/top-level project node of the tree.
   */
  std::pair<const RelScan*, uint32_t> getScanNodeForOutputIndex(
      const uint32_t output_index) const;

  /**
   * Injects a simple column project into the tree. The column to project is defined by
   * the RelScan (table), the column index, and the desired output column name
   * @param table_scan_node The RelScan node to project. This should already be a known
   * RelScan in the tree
   * @param column_index The index of the RelScan column to project
   * @param output_column_name The desired output column name
   */
  void injectInputColumn(const RelScan& table_scan_node,
                         const int column_index,
                         const std::string& output_column_name);

 private:
  struct TreeNode {
    template <typename T>
    TreeNode(const T& in_rel_alg_node)
        : rel_alg_node{in_rel_alg_node}
        , rel_alg_node_ptr_variant{&in_rel_alg_node}
        , parent{nullptr}
        , lhs{nullptr}
        , rhs{nullptr} {}

    const RelAlgNode& rel_alg_node;
    std::variant<const RelProject*,
                 const RelFilter*,
                 const RelSort*,
                 const RelJoin*,
                 const RelScan*>
        rel_alg_node_ptr_variant;

    TreeNode* parent{nullptr};
    TreeNode* lhs{nullptr};
    TreeNode* rhs{nullptr};
  };

  // the root/top-level RelProject node of the tree
  const RelProject& root_project_node_;

  // original RelAlgNode DAG nodes, used to find non-const versions of const nodes without
  // doing const_cast
  std::vector<std::shared_ptr<RelAlgNode>>& ra_tree_nodes_;

  // container which actually owns the nodes for this internal RelScanTree. The use of
  // unique_ptr is intentional so that pointers to the TreeNode objects will always stay
  // in tact, even if the size of the vector is modified in some way.
  std::vector<std::unique_ptr<TreeNode>> nodes_;

  // pointers to the leaves of the trees that are RelScan nodes.
  std::vector<TreeNode*> scan_node_leaves_;

  const TreeNode* getRootNode() const;

  /**
   * Gets the non-const version of a RelProject node. This is used to get a non-const
   * version to modify without needing to const_cast
   */
  RelProject& getNonConstProject(const RelProject& const_proj_to_find);

  /**
   * Utility that builds a tree node with the right variant
   */
  template <typename T>
  bool emplaceTreeNode(const RelAlgNode* rel_alg_node) {
    if (auto const* traversable_node = dynamic_cast<const T*>(rel_alg_node);
        traversable_node != nullptr) {
      nodes_.emplace_back(std::make_unique<TreeNode>(*traversable_node));
      return true;
    }
    return false;
  }

  /**
   * Recusive function that builds the RelScan tree. The root node should have already
   * been created
   */
  TreeNode* buildRelScanTree(const RelAlgNode* rel_alg_node);
};

}  // namespace QueryRenderer
