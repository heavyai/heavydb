/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RelAlgOptimizer.h"
#include "Catalog/TableConstraints.h"
#include "Logger/Logger.h"
#include "RelAlgDag.h"
#include "RexVisitor.h"
#include "Visitors/RexSubQueryIdCollector.h"

#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>

bool g_enable_experimental_query_rewrites{false};
bool g_trust_unenforced_table_constraints{false};

namespace {

class RexProjectInputRedirector : public RexDeepCopyVisitor {
 public:
  RexProjectInputRedirector(const std::unordered_set<const RelProject*>& crt_inputs)
      : crt_projects_(crt_inputs) {}

  RetType visitInput(const RexInput* input) const override {
    auto source = dynamic_cast<const RelProject*>(input->getSourceNode());
    if (!source || !crt_projects_.count(source)) {
      return input->deepCopy();
    }
    auto new_source = source->getInput(0);
    auto new_input =
        dynamic_cast<const RexInput*>(source->getProjectAt(input->getIndex()));
    if (!new_input) {
      return input->deepCopy();
    }
    if (auto join = dynamic_cast<const RelJoin*>(new_source)) {
      CHECK(new_input->getSourceNode() == join->getInput(0) ||
            new_input->getSourceNode() == join->getInput(1));
    } else {
      CHECK_EQ(new_input->getSourceNode(), new_source);
    }
    return new_input->deepCopy();
  }

 private:
  const std::unordered_set<const RelProject*>& crt_projects_;
};

class CleanupReachabilityVisitor : public RelAlgDagNode::Visitor {
 public:
  const std::unordered_set<const RelAlgNode*>& rexInputSources() const {
    return rex_input_sources_;
  }

  const std::vector<const RelAlgNode*>& subqueryRoots() const { return subquery_roots_; }

  bool visit(RexInput const* rex_input, std::string) override {
    if (const auto source = rex_input->getSourceNode()) {
      rex_input_sources_.insert(source);
    }
    return false;
  }

  bool visit(RexSubQuery const* subquery, std::string) override {
    if (const auto subquery_root = subquery->getRelAlg()) {
      subquery_roots_.push_back(subquery_root);
    }
    return false;
  }

 protected:
  bool visitAny(RelAlgDagNode const* node, std::string) override {
    return !dynamic_cast<const RelAlgNode*>(node);
  }

 private:
  std::unordered_set<const RelAlgNode*> rex_input_sources_;
  std::vector<const RelAlgNode*> subquery_roots_;
};

void collect_structural_reachable_nodes(
    const RelAlgNode* root,
    std::unordered_set<const RelAlgNode*>& reachable_nodes,
    std::unordered_set<const RelAlgNode*>& rex_input_sources) {
  if (!root) {
    return;
  }

  std::vector<const RelAlgNode*> stack{root};
  while (!stack.empty()) {
    const auto node = stack.back();
    stack.pop_back();
    if (!node || !reachable_nodes.insert(node).second) {
      continue;
    }

    for (size_t input_idx = 0; input_idx < node->inputCount(); ++input_idx) {
      stack.push_back(node->getInput(input_idx));
    }

    CleanupReachabilityVisitor visitor;
    node->acceptChildren(visitor);
    rex_input_sources.insert(visitor.rexInputSources().begin(),
                             visitor.rexInputSources().end());
    for (const auto subquery_root : visitor.subqueryRoots()) {
      stack.push_back(subquery_root);
    }
  }
}

bool is_materializable_rex_input_source(const RelAlgNode* node) {
  return node && !dynamic_cast<const RelJoin*>(node) &&
         !dynamic_cast<const RelLeftDeepInnerJoin*>(node);
}

std::vector<std::string> get_output_field_names(const RelAlgNode* node) {
  if (const auto project = dynamic_cast<const RelProject*>(node)) {
    std::vector<std::string> fields;
    fields.reserve(project->size());
    for (size_t i = 0; i < project->size(); ++i) {
      fields.push_back(project->getFieldName(i));
    }
    return fields;
  }
  if (const auto compound = dynamic_cast<const RelCompound*>(node)) {
    return compound->getFields();
  }
  if (const auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    return aggregate->getFields();
  }
  return {};
}

bool has_same_output_signature(const RelAlgNode* lhs, const RelAlgNode* rhs) {
  if (!lhs || !rhs || typeid(*lhs) != typeid(*rhs) || lhs->size() != rhs->size()) {
    return false;
  }
  const auto lhs_fields = get_output_field_names(lhs);
  const auto rhs_fields = get_output_field_names(rhs);
  return !lhs_fields.empty() && lhs_fields == rhs_fields;
}

bool has_same_structural_signature(const RelAlgNode* lhs, const RelAlgNode* rhs) {
  return has_same_output_signature(lhs, rhs) && lhs->toHash() == rhs->toHash() &&
         lhs->toString(RelRexToStringConfig::defaults()) ==
             rhs->toString(RelRexToStringConfig::defaults());
}

using ReachableNodesByHash = std::unordered_map<size_t, std::vector<const RelAlgNode*>>;

class RebindUnreachableRexInputSourcesVisitor : public RelAlgDagNode::Visitor {
 public:
  RebindUnreachableRexInputSourcesVisitor(
      const std::unordered_set<const RelAlgNode*>& structurally_reachable_nodes,
      const std::unordered_set<const RelAlgNode*>& nodes_owned_by_dag,
      const ReachableNodesByHash& reachable_nodes_by_hash,
      const std::vector<const RelAlgNode*>& materializable_reachable_nodes)
      : structurally_reachable_nodes_(structurally_reachable_nodes)
      , nodes_owned_by_dag_(nodes_owned_by_dag)
      , reachable_nodes_by_hash_(reachable_nodes_by_hash)
      , materializable_reachable_nodes_(materializable_reachable_nodes) {}

  bool visit(RexInput const* rex_input, std::string) override {
    const auto source = rex_input->getSourceNode();
    if (!source || structurally_reachable_nodes_.count(source)) {
      return false;
    }

    // Avoid touching pointers that are not owned by this DAG.  Those are outside the
    // scope of dead-node cleanup and may already have a different lifetime contract.
    if (!nodes_owned_by_dag_.count(source)) {
      return false;
    }

    const auto replacement = findUniqueStructuralReplacement(source);
    if (!replacement || rex_input->getIndex() >= replacement->size()) {
      return false;
    }

    rex_input->setSourceNode(replacement);
    return false;
  }

 protected:
  bool visitAny(RelAlgDagNode const* node, std::string) override {
    return !dynamic_cast<const RelAlgNode*>(node);
  }

 private:
  const RelAlgNode* findUniqueStructuralReplacement(const RelAlgNode* source) const {
    const RelAlgNode* replacement{nullptr};
    const auto hash_it = reachable_nodes_by_hash_.find(source->toHash());
    const auto& candidates = hash_it == reachable_nodes_by_hash_.end()
                                 ? materializable_reachable_nodes_
                                 : hash_it->second;
    for (const auto candidate : candidates) {
      if (!has_same_structural_signature(source, candidate)) {
        continue;
      }
      if (replacement) {
        return nullptr;
      }
      replacement = candidate;
    }
    return replacement;
  }

  const std::unordered_set<const RelAlgNode*>& structurally_reachable_nodes_;
  const std::unordered_set<const RelAlgNode*>& nodes_owned_by_dag_;
  const ReachableNodesByHash& reachable_nodes_by_hash_;
  const std::vector<const RelAlgNode*>& materializable_reachable_nodes_;
};

std::unordered_set<const RelAlgNode*> rebind_rex_input_sources_before_cleanup(
    const std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  std::unordered_set<const RelAlgNode*> rex_input_sources;
  if (nodes.empty()) {
    return rex_input_sources;
  }

  const RelAlgNode* root{nullptr};
  for (auto node_it = nodes.rbegin(); node_it != nodes.rend(); ++node_it) {
    if (*node_it) {
      root = node_it->get();
      break;
    }
  }
  if (!root) {
    return rex_input_sources;
  }

  std::unordered_set<const RelAlgNode*> structurally_reachable_nodes;
  collect_structural_reachable_nodes(
      root, structurally_reachable_nodes, rex_input_sources);

  std::unordered_set<const RelAlgNode*> nodes_owned_by_dag;
  for (const auto& node : nodes) {
    if (node) {
      nodes_owned_by_dag.insert(node.get());
    }
  }

  ReachableNodesByHash reachable_nodes_by_hash;
  std::vector<const RelAlgNode*> materializable_reachable_nodes;
  for (const auto node : structurally_reachable_nodes) {
    if (is_materializable_rex_input_source(node)) {
      reachable_nodes_by_hash[node->toHash()].push_back(node);
      materializable_reachable_nodes.push_back(node);
    }
  }

  RebindUnreachableRexInputSourcesVisitor rebind_visitor(structurally_reachable_nodes,
                                                         nodes_owned_by_dag,
                                                         reachable_nodes_by_hash,
                                                         materializable_reachable_nodes);
  for (const auto node : structurally_reachable_nodes) {
    node->acceptChildren(rebind_visitor);
  }

  rex_input_sources.clear();
  structurally_reachable_nodes.clear();
  collect_structural_reachable_nodes(
      root, structurally_reachable_nodes, rex_input_sources);
  return rex_input_sources;
}

class RexRebindInputsVisitor : public RexVisitor<void*> {
 public:
  RexRebindInputsVisitor(const RelAlgNode* old_input, const RelAlgNode* new_input)
      : old_input_(old_input), new_input_(new_input) {}

  void* visitInput(const RexInput* rex_input) const override {
    const auto old_source = rex_input->getSourceNode();
    if (old_source == old_input_) {
      rex_input->setSourceNode(new_input_);
    }
    return nullptr;
  };

  void visitNode(const RelAlgNode* node) const {
    if (dynamic_cast<const RelAggregate*>(node) || dynamic_cast<const RelSort*>(node)) {
      return;
    }
    if (auto join = dynamic_cast<const RelJoin*>(node)) {
      if (auto condition = join->getCondition()) {
        visit(condition);
      }
      return;
    }
    if (auto project = dynamic_cast<const RelProject*>(node)) {
      for (size_t i = 0; i < project->size(); ++i) {
        visit(project->getProjectAt(i));
      }
      return;
    }
    if (auto filter = dynamic_cast<const RelFilter*>(node)) {
      visit(filter->getCondition());
      return;
    }
    CHECK(false);
  }

 private:
  const RelAlgNode* old_input_;
  const RelAlgNode* new_input_;
};

size_t get_actual_source_size(
    const RelProject* curr_project,
    const std::unordered_set<const RelProject*>& projects_to_remove) {
  auto source = curr_project->getInput(0);
  while (auto filter = dynamic_cast<const RelFilter*>(source)) {
    source = filter->getInput(0);
  }
  if (auto src_project = dynamic_cast<const RelProject*>(source)) {
    if (projects_to_remove.count(src_project)) {
      return get_actual_source_size(src_project, projects_to_remove);
    }
  }
  return curr_project->getInput(0)->size();
}

bool safe_to_redirect(
    const RelProject* project,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>&
        du_web) {
  if (!project->isSimple()) {
    return false;
  }
  auto usrs_it = du_web.find(project);
  CHECK(usrs_it != du_web.end());
  for (auto usr : usrs_it->second) {
    if (!dynamic_cast<const RelProject*>(usr) && !dynamic_cast<const RelFilter*>(usr)) {
      return false;
    }
  }
  return true;
}

bool is_identical_copy(
    const RelProject* project,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>&
        du_web,
    const std::unordered_set<const RelProject*>& projects_to_remove,
    std::unordered_set<const RelProject*>& permutating_projects) {
  auto source_size = get_actual_source_size(project, projects_to_remove);
  if (project->size() > source_size) {
    return false;
  }

  if (project->size() < source_size) {
    auto usrs_it = du_web.find(project);
    CHECK(usrs_it != du_web.end());
    bool guard_found = false;
    while (usrs_it->second.size() == size_t(1)) {
      auto only_usr = *usrs_it->second.begin();
      if (dynamic_cast<const RelProject*>(only_usr)) {
        guard_found = true;
        break;
      }
      if (dynamic_cast<const RelAggregate*>(only_usr) ||
          dynamic_cast<const RelSort*>(only_usr) ||
          dynamic_cast<const RelJoin*>(only_usr) ||
          dynamic_cast<const RelTableFunction*>(only_usr) ||
          dynamic_cast<const RelLogicalUnion*>(only_usr)) {
        return false;
      }
      CHECK(dynamic_cast<const RelFilter*>(only_usr))
          << "only_usr: " << only_usr->toString(RelRexToStringConfig::defaults());
      usrs_it = du_web.find(only_usr);
      CHECK(usrs_it != du_web.end());
    }

    if (!guard_found) {
      return false;
    }
  }

  bool identical = true;
  for (size_t i = 0; i < project->size(); ++i) {
    auto target = dynamic_cast<const RexInput*>(project->getProjectAt(i));
    CHECK(target);
    if (i != target->getIndex()) {
      identical = false;
      break;
    }
  }

  if (identical) {
    return true;
  }

  if (safe_to_redirect(project, du_web)) {
    permutating_projects.insert(project);
    return true;
  }

  return false;
}

bool is_project_for_filtered_left_join(const RelProject* project) {
  if (auto filter_node = dynamic_cast<const RelFilter*>(project->getInput(0))) {
    if (auto join_node = dynamic_cast<const RelJoin*>(filter_node->getInput(0))) {
      if (join_node->getJoinType() == JoinType::LEFT && filter_node->getCondition()) {
        return true;
      }
    }
  }
  return false;
}

unsigned project_output_index_after_redirect(const RelProject* project,
                                             const unsigned project_output_idx) {
  CHECK(project);
  CHECK_LT(static_cast<size_t>(project_output_idx), project->size());
  auto rex_input =
      dynamic_cast<const RexInput*>(project->getProjectAt(project_output_idx));
  CHECK(rex_input);
  CHECK_EQ(rex_input->getSourceNode(), project->getInput(0));
  return rex_input->getIndex();
}

std::unordered_map<unsigned, unsigned> make_join_output_map_after_project_redirect(
    const RelJoin* join,
    const std::unordered_set<const RelProject*>& projects_to_remove) {
  CHECK(join);
  CHECK_EQ(join->inputCount(), size_t(2));
  const auto left_input = join->getInput(0);
  const auto right_input = join->getInput(1);
  const auto left_project = dynamic_cast<const RelProject*>(left_input);
  const auto right_project = dynamic_cast<const RelProject*>(right_input);
  const bool remove_left = left_project && projects_to_remove.count(left_project) > 0;
  const bool remove_right = right_project && projects_to_remove.count(right_project) > 0;
  const auto new_left_size =
      remove_left ? left_project->getInput(0)->size() : left_input->size();

  std::unordered_map<unsigned, unsigned> old_to_new_index_map;
  old_to_new_index_map.reserve(left_input->size() + right_input->size());

  const auto append_side_mapping = [&](const RelAlgNode* old_side,
                                       const RelProject* removed_project,
                                       const bool remove_side,
                                       const unsigned old_base,
                                       const unsigned new_base) {
    for (unsigned old_local_idx = 0; old_local_idx < old_side->size(); ++old_local_idx) {
      const auto new_local_idx = remove_side ? project_output_index_after_redirect(
                                                   removed_project, old_local_idx)
                                             : old_local_idx;
      old_to_new_index_map.emplace(old_base + old_local_idx, new_base + new_local_idx);
    }
  };

  append_side_mapping(left_input, left_project, remove_left, 0, 0);
  append_side_mapping(right_input,
                      right_project,
                      remove_right,
                      static_cast<unsigned>(left_input->size()),
                      static_cast<unsigned>(new_left_size));
  return old_to_new_index_map;
}

bool aggregate_input_can_be_remapped(
    const RelAggregate* aggregate,
    const std::unordered_map<unsigned, unsigned>& old_to_new_index_map) {
  CHECK(aggregate);
  for (size_t group_idx = 0; group_idx < aggregate->getGroupByCount(); ++group_idx) {
    const auto group_it = old_to_new_index_map.find(static_cast<unsigned>(group_idx));
    if (group_it == old_to_new_index_map.end() ||
        group_it->second != static_cast<unsigned>(group_idx)) {
      return false;
    }
  }
  for (const auto& agg_expr : aggregate->getAggExprs()) {
    for (size_t operand_idx = 0; operand_idx < agg_expr->size(); ++operand_idx) {
      if (!old_to_new_index_map.count(
              static_cast<unsigned>(agg_expr->getOperand(operand_idx)))) {
        return false;
      }
    }
  }
  return true;
}

void remap_aggregate_input(
    RelAggregate* aggregate,
    const std::unordered_map<unsigned, unsigned>& old_to_new_index_map) {
  CHECK(aggregate_input_can_be_remapped(aggregate, old_to_new_index_map));
  auto old_exprs = aggregate->getAggExprsAndRelease();
  std::vector<std::unique_ptr<const RexAgg>> new_exprs;
  new_exprs.reserve(old_exprs.size());
  for (auto& agg_expr : old_exprs) {
    std::vector<size_t> operands;
    operands.reserve(agg_expr->size());
    for (size_t operand_idx = 0; operand_idx < agg_expr->size(); ++operand_idx) {
      const auto operand_it = old_to_new_index_map.find(
          static_cast<unsigned>(agg_expr->getOperand(operand_idx)));
      CHECK(operand_it != old_to_new_index_map.end());
      operands.push_back(operand_it->second);
    }
    new_exprs.push_back(std::make_unique<RexAgg>(
        agg_expr->getKind(), agg_expr->isDistinct(), agg_expr->getType(), operands));
  }
  aggregate->setAggExprs(new_exprs);
}

void propagate_rex_input_renumber(
    const RelFilter* excluded_root,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>&
        du_web) {
  CHECK(excluded_root);
  auto src_project = dynamic_cast<const RelProject*>(excluded_root->getInput(0));
  CHECK(src_project && src_project->isSimple());
  const auto indirect_join_src = dynamic_cast<const RelJoin*>(src_project->getInput(0));
  std::unordered_map<size_t, size_t> old_to_new_idx;
  for (size_t i = 0; i < src_project->size(); ++i) {
    auto rex_in = dynamic_cast<const RexInput*>(src_project->getProjectAt(i));
    CHECK(rex_in);
    size_t src_base = 0;
    if (indirect_join_src != nullptr &&
        indirect_join_src->getInput(1) == rex_in->getSourceNode()) {
      src_base = indirect_join_src->getInput(0)->size();
    }
    old_to_new_idx.insert(std::make_pair(i, src_base + rex_in->getIndex()));
    old_to_new_idx.insert(std::make_pair(i, rex_in->getIndex()));
  }
  CHECK(old_to_new_idx.size());
  RexInputRenumber<false> renumber(old_to_new_idx);
  auto usrs_it = du_web.find(excluded_root);
  CHECK(usrs_it != du_web.end());
  std::vector<const RelAlgNode*> work_set(usrs_it->second.begin(), usrs_it->second.end());
  while (!work_set.empty()) {
    auto node = work_set.back();
    work_set.pop_back();
    auto modified_node = const_cast<RelAlgNode*>(node);
    if (auto filter = dynamic_cast<RelFilter*>(modified_node)) {
      auto new_condition = renumber.visit(filter->getCondition());
      filter->setCondition(new_condition);
      auto usrs_it = du_web.find(filter);
      CHECK(usrs_it != du_web.end() && usrs_it->second.size() == 1);
      work_set.push_back(*usrs_it->second.begin());
      continue;
    }
    if (auto project = dynamic_cast<RelProject*>(modified_node)) {
      std::vector<std::unique_ptr<const RexScalar>> new_exprs;
      for (size_t i = 0; i < project->size(); ++i) {
        new_exprs.push_back(renumber.visit(project->getProjectAt(i)));
      }
      project->setExpressions(new_exprs);
      continue;
    }
    CHECK(false);
  }
}

// This function appears to redirect/remove redundant Projection input nodes(?)
void redirect_inputs_of(
    std::shared_ptr<RelAlgNode> node,
    const std::unordered_set<const RelProject*>& projects,
    const std::unordered_set<const RelProject*>& permutating_projects,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>&
        du_web) {
  if (dynamic_cast<RelLogicalUnion*>(node.get())) {
    return;  // UNION keeps all Projection inputs.
  }
  std::shared_ptr<const RelProject> src_project = nullptr;
  for (size_t i = 0; i < node->inputCount(); ++i) {
    if (auto project =
            std::dynamic_pointer_cast<const RelProject>(node->getAndOwnInput(i))) {
      if (projects.count(project.get())) {
        src_project = project;
        break;
      }
    }
  }
  if (!src_project) {
    return;
  }
  if (auto join = std::dynamic_pointer_cast<RelJoin>(node)) {
    auto other_project =
        src_project == node->getAndOwnInput(0)
            ? std::dynamic_pointer_cast<const RelProject>(node->getAndOwnInput(1))
            : std::dynamic_pointer_cast<const RelProject>(node->getAndOwnInput(0));
    const auto join_output_map =
        make_join_output_map_after_project_redirect(join.get(), projects);
    auto usrs_it = du_web.find(join.get());
    CHECK(usrs_it != du_web.end());
    for (auto usr : usrs_it->second) {
      if (const auto aggregate = dynamic_cast<const RelAggregate*>(usr)) {
        if (!aggregate_input_can_be_remapped(aggregate, join_output_map)) {
          return;
        }
      }
    }
    join->replaceInput(src_project, src_project->getAndOwnInput(0));
    RexRebindInputsVisitor rebinder(src_project.get(), src_project->getInput(0));
    for (auto usr : usrs_it->second) {
      rebinder.visitNode(usr);
    }

    if (other_project && projects.count(other_project.get())) {
      join->replaceInput(other_project, other_project->getAndOwnInput(0));
      RexRebindInputsVisitor other_rebinder(other_project.get(),
                                            other_project->getInput(0));
      for (auto usr : usrs_it->second) {
        other_rebinder.visitNode(usr);
      }
    }
    for (auto usr : usrs_it->second) {
      if (auto aggregate =
              const_cast<RelAggregate*>(dynamic_cast<const RelAggregate*>(usr))) {
        remap_aggregate_input(aggregate, join_output_map);
      }
    }
    return;
  }
  if (auto project = std::dynamic_pointer_cast<RelProject>(node)) {
    project->RelAlgNode::replaceInput(src_project, src_project->getAndOwnInput(0));
    RexProjectInputRedirector redirector(projects);
    std::vector<std::unique_ptr<const RexScalar>> new_exprs;
    for (size_t i = 0; i < project->size(); ++i) {
      new_exprs.push_back(redirector.visit(project->getProjectAt(i)));
    }
    project->setExpressions(new_exprs);
    return;
  }
  if (auto filter = std::dynamic_pointer_cast<RelFilter>(node)) {
    const bool is_permutating_proj = permutating_projects.count(src_project.get());
    if (is_permutating_proj || dynamic_cast<const RelJoin*>(src_project->getInput(0))) {
      if (is_permutating_proj) {
        propagate_rex_input_renumber(filter.get(), du_web);
      }
      filter->RelAlgNode::replaceInput(src_project, src_project->getAndOwnInput(0));
      RexProjectInputRedirector redirector(projects);
      auto new_condition = redirector.visit(filter->getCondition());
      filter->setCondition(new_condition);
    } else {
      filter->replaceInput(src_project, src_project->getAndOwnInput(0));
    }
    return;
  }
  if (std::dynamic_pointer_cast<RelSort>(node)) {
    auto const src_project_input = src_project->getInput(0);
    if (dynamic_cast<const RelScan*>(src_project_input) ||
        dynamic_cast<const RelLogicalValues*>(src_project_input) ||
        dynamic_cast<const RelLogicalUnion*>(src_project_input)) {
      return;
    }
  }
  if (std::dynamic_pointer_cast<RelModify>(node)) {
    return;  // NOTE:  Review this.  Not sure about this.
  }
  if (std::dynamic_pointer_cast<RelTableFunction>(node)) {
    return;
  }
  CHECK(std::dynamic_pointer_cast<RelAggregate>(node) ||
        std::dynamic_pointer_cast<RelSort>(node));
  node->replaceInput(src_project, src_project->getAndOwnInput(0));
}

void cleanup_dead_nodes(std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  const auto rex_input_sources = rebind_rex_input_sources_before_cleanup(nodes);
  for (auto nodeIt = nodes.rbegin(); nodeIt != nodes.rend(); ++nodeIt) {
    if (nodeIt->use_count() == 1) {
      if (rex_input_sources.count(nodeIt->get())) {
        continue;
      }
      nodeIt->reset();
    }
  }

  std::vector<std::shared_ptr<RelAlgNode>> new_nodes;
  for (auto node : nodes) {
    if (!node) {
      continue;
    }
    new_nodes.push_back(node);
  }
  nodes.swap(new_nodes);
}

std::unordered_set<const RelProject*> get_visible_projects(const RelAlgNode* root) {
  if (auto project = dynamic_cast<const RelProject*>(root)) {
    return {project};
  }

  if (dynamic_cast<const RelAggregate*>(root) || dynamic_cast<const RelScan*>(root) ||
      dynamic_cast<const RelLogicalValues*>(root) ||
      dynamic_cast<const RelModify*>(root)) {
    return std::unordered_set<const RelProject*>{};
  }

  if (auto join = dynamic_cast<const RelJoin*>(root)) {
    auto lhs_projs = get_visible_projects(join->getInput(0));
    auto rhs_projs = get_visible_projects(join->getInput(1));
    lhs_projs.insert(rhs_projs.begin(), rhs_projs.end());
    return lhs_projs;
  }

  if (auto logical_union = dynamic_cast<const RelLogicalUnion*>(root)) {
    auto projections = get_visible_projects(logical_union->getInput(0));
    for (size_t i = 1; i < logical_union->inputCount(); ++i) {
      auto next = get_visible_projects(logical_union->getInput(i));
      projections.insert(next.begin(), next.end());
    }
    return projections;
  }

  CHECK(dynamic_cast<const RelFilter*>(root) || dynamic_cast<const RelSort*>(root))
      << "root = " << root->toString(RelRexToStringConfig::defaults());
  return get_visible_projects(root->getInput(0));
}

// TODO(miyu): checking this at runtime is more accurate
bool is_distinct(const size_t input_idx, const RelAlgNode* node) {
  if (dynamic_cast<const RelFilter*>(node) || dynamic_cast<const RelSort*>(node)) {
    CHECK_EQ(size_t(1), node->inputCount());
    return is_distinct(input_idx, node->getInput(0));
  }
  if (auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    CHECK_EQ(size_t(1), node->inputCount());
    if (aggregate->getGroupByCount() == 1 && !input_idx) {
      return true;
    }
    if (input_idx < aggregate->getGroupByCount()) {
      return is_distinct(input_idx, node->getInput(0));
    }
    return false;
  }
  if (auto project = dynamic_cast<const RelProject*>(node)) {
    CHECK_LT(input_idx, project->size());
    if (auto input = dynamic_cast<const RexInput*>(project->getProjectAt(input_idx))) {
      CHECK_EQ(size_t(1), node->inputCount());
      return is_distinct(input->getIndex(), project->getInput(0));
    }
    return false;
  }
  CHECK(dynamic_cast<const RelJoin*>(node) || dynamic_cast<const RelScan*>(node));
  return false;
}

}  // namespace

std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>> build_du_web(
    const std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>> web;
  std::unordered_set<const RelAlgNode*> visited;
  std::vector<const RelAlgNode*> work_set;
  for (auto node : nodes) {
    if (std::dynamic_pointer_cast<RelScan>(node) ||
        std::dynamic_pointer_cast<RelModify>(node) || visited.count(node.get())) {
      continue;
    }
    work_set.push_back(node.get());
    while (!work_set.empty()) {
      auto walker = work_set.back();
      work_set.pop_back();
      if (visited.count(walker)) {
        continue;
      }
      CHECK(!web.count(walker));
      auto it_ok =
          web.insert(std::make_pair(walker, std::unordered_set<const RelAlgNode*>{}));
      CHECK(it_ok.second);
      visited.insert(walker);
      CHECK(dynamic_cast<const RelJoin*>(walker) ||
            dynamic_cast<const RelProject*>(walker) ||
            dynamic_cast<const RelAggregate*>(walker) ||
            dynamic_cast<const RelFilter*>(walker) ||
            dynamic_cast<const RelSort*>(walker) ||
            dynamic_cast<const RelLeftDeepInnerJoin*>(walker) ||
            dynamic_cast<const RelLogicalValues*>(walker) ||
            dynamic_cast<const RelTableFunction*>(walker) ||
            dynamic_cast<const RelLogicalUnion*>(walker));
      for (size_t i = 0; i < walker->inputCount(); ++i) {
        auto src = walker->getInput(i);
        if (dynamic_cast<const RelScan*>(src) || dynamic_cast<const RelModify*>(src)) {
          continue;
        }
        if (web.empty() || !web.count(src)) {
          web.insert(std::make_pair(src, std::unordered_set<const RelAlgNode*>{}));
        }
        web[src].insert(walker);
        work_set.push_back(src);
      }
    }
  }
  return web;
}

/**
 * Return true if the input project separates two sort nodes, i.e. Sort -> Project ->
 * Sort. This pattern often occurs in machine generated SQL, e.g. SELECT * FROM (SELECT *
 * FROM t LIMIT 10) t0 LIMIT 1;
 * Use this function to prevent optimizing out the intermediate project, as the project is
 * required to ensure the first sort runs to completion prior to the second sort. Back to
 * back sort nodes are not executable and will throw an error.
 */
bool project_separates_sort(const RelProject* project, const RelAlgNode* next_node) {
  CHECK(project);
  if (!next_node) {
    return false;
  }

  auto sort = dynamic_cast<const RelSort*>(next_node);
  if (!sort) {
    return false;
  }
  if (!(project->inputCount() == 1)) {
    return false;
  }

  if (dynamic_cast<const RelSort*>(project->getInput(0))) {
    return true;
  }
  return false;
}

// For now, the only target to eliminate is restricted to project-aggregate pair between
// scan/sort and join
// TODO(miyu): allow more chance if proved safe
void eliminate_identical_copy(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  std::unordered_set<std::shared_ptr<const RelAlgNode>> copies;
  auto sink = nodes.back();
  for (auto node : nodes) {
    auto aggregate = std::dynamic_pointer_cast<const RelAggregate>(node);
    if (!aggregate || aggregate == sink ||
        !(aggregate->getGroupByCount() == 1 && aggregate->getAggExprsCount() == 0)) {
      continue;
    }
    auto project =
        std::dynamic_pointer_cast<const RelProject>(aggregate->getAndOwnInput(0));
    if (project && project->size() == aggregate->size() &&
        project->getFields() == aggregate->getFields()) {
      CHECK_EQ(size_t(0), copies.count(aggregate));
      copies.insert(aggregate);
    }
  }
  for (auto node : nodes) {
    if (!node->inputCount()) {
      continue;
    }
    auto last_source = node->getAndOwnInput(node->inputCount() - 1);
    if (!copies.count(last_source)) {
      continue;
    }
    auto aggregate = std::dynamic_pointer_cast<const RelAggregate>(last_source);
    CHECK(aggregate);
    if (!std::dynamic_pointer_cast<const RelJoin>(node) || aggregate->size() != 1) {
      continue;
    }
    auto project =
        std::dynamic_pointer_cast<const RelProject>(aggregate->getAndOwnInput(0));
    CHECK(project);
    CHECK_EQ(size_t(1), project->size());
    if (!is_distinct(size_t(0), project.get())) {
      continue;
    }
    auto new_source = project->getAndOwnInput(0);
    if (std::dynamic_pointer_cast<const RelSort>(new_source) ||
        std::dynamic_pointer_cast<const RelScan>(new_source)) {
      node->replaceInput(last_source, new_source);
    }
  }
  decltype(copies)().swap(copies);

  auto web = build_du_web(nodes);

  std::unordered_set<const RelProject*> projects;
  std::unordered_set<const RelProject*> permutating_projects;
  auto const visible_projs = get_visible_projects(nodes.back().get());
  for (auto node_it = nodes.begin(); node_it != nodes.end(); node_it++) {
    auto node = *node_it;
    auto project = std::dynamic_pointer_cast<RelProject>(node);
    auto next_node_it = std::next(node_it);
    if (project && project->isSimple() &&
        (!visible_projs.count(project.get()) || !project->isRenaming()) &&
        is_identical_copy(project.get(), web, projects, permutating_projects) &&
        !project_separates_sort(
            project.get(), next_node_it == nodes.end() ? nullptr : next_node_it->get()) &&
        !is_project_for_filtered_left_join(project.get())) {
      projects.insert(project.get());
    }
  }

  for (auto node : nodes) {
    redirect_inputs_of(node, projects, permutating_projects, web);
  }

  cleanup_dead_nodes(nodes);
}

namespace {

class RexInputCollector : public RexVisitor<std::unordered_set<RexInput>> {
 private:
  const RelAlgNode* node_;

 protected:
  using RetType = std::unordered_set<RexInput>;
  RetType aggregateResult(const RetType& aggregate,
                          const RetType& next_result) const override {
    RetType result(aggregate.begin(), aggregate.end());
    result.insert(next_result.begin(), next_result.end());
    return result;
  }

 public:
  RexInputCollector(const RelAlgNode* node) : node_(node) {}

  RetType visitInput(const RexInput* input) const override {
    RetType result;
    if (node_->inputCount() == 1) {
      auto src = node_->getInput(0);
      if (auto join = dynamic_cast<const RelJoin*>(src)) {
        CHECK_EQ(join->inputCount(), size_t(2));
        const auto src2_in_offset = join->getInput(0)->size();
        if (input->getSourceNode() == join->getInput(1)) {
          result.emplace(src, input->getIndex() + src2_in_offset);
        } else {
          result.emplace(src, input->getIndex());
        }
        return result;
      }
    }
    result.insert(*input);
    return result;
  }
};

std::optional<size_t> pick_always_live_col_idx(const RelAlgNode* node) {
  if (!node || !node->size()) {
    return std::nullopt;
  }
  RexInputCollector collector(node);
  if (auto filter = dynamic_cast<const RelFilter*>(node)) {
    auto rex_ins = collector.visit(filter->getCondition());
    if (!rex_ins.empty()) {
      return static_cast<size_t>(rex_ins.begin()->getIndex());
    }
    return pick_always_live_col_idx(filter->getInput(0));
  } else if (auto join = dynamic_cast<const RelJoin*>(node)) {
    auto rex_ins = collector.visit(join->getCondition());
    if (!rex_ins.empty()) {
      const auto lhs = join->getInput(0);
      const auto rhs = join->getInput(1);
      const auto rhs_idx_base = lhs->size();
      const auto first_input = rex_ins.begin();
      if (first_input->getSourceNode() == rhs) {
        return rhs_idx_base + static_cast<size_t>(first_input->getIndex());
      }
      return static_cast<size_t>(first_input->getIndex());
    }
    if (auto lhs_idx = pick_always_live_col_idx(join->getInput(0))) {
      return *lhs_idx;
    }
    if (auto rhs_idx = pick_always_live_col_idx(join->getInput(1))) {
      return *rhs_idx + join->getInput(0)->size();
    }
  } else if (auto sort = dynamic_cast<const RelSort*>(node)) {
    if (sort->collationCount()) {
      return sort->getCollation(0).getField();
    }
    return pick_always_live_col_idx(sort->getInput(0));
  }
  return size_t(0);
}

void add_row_cardinality_live_input(std::unordered_set<size_t>& live_in,
                                    const RelAlgNode* input) {
  if (auto live_idx = pick_always_live_col_idx(input)) {
    live_in.insert(*live_idx);
  }
}

std::vector<std::unordered_set<size_t>> get_live_ins(
    const RelAlgNode* node,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& live_outs) {
  if (!node || dynamic_cast<const RelScan*>(node)) {
    return {};
  }
  RexInputCollector collector(node);
  auto it = live_outs.find(node);
  CHECK(it != live_outs.end());
  auto live_out = it->second;
  if (auto project = dynamic_cast<const RelProject*>(node)) {
    CHECK_EQ(size_t(1), project->inputCount());
    std::unordered_set<size_t> live_in;
    if (project->size() == 0) {
      add_row_cardinality_live_input(live_in, project->getInput(0));
      return {live_in};
    }
    for (const auto& idx : live_out) {
      CHECK_LT(idx, project->size());
      auto partial_in = collector.visit(project->getProjectAt(idx));
      for (auto rex_in : partial_in) {
        live_in.insert(rex_in.getIndex());
      }
    }
    if (project->size() == 1 &&
        dynamic_cast<const RexLiteral*>(project->getProjectAt(0))) {
      CHECK(live_in.empty());
      add_row_cardinality_live_input(live_in, project->getInput(0));
    }
    return {live_in};
  }
  if (auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    CHECK_EQ(size_t(1), aggregate->inputCount());
    const auto group_key_count = static_cast<size_t>(aggregate->getGroupByCount());
    const auto agg_expr_count = static_cast<size_t>(aggregate->getAggExprsCount());
    std::unordered_set<size_t> live_in;
    for (size_t i = 0; i < group_key_count; ++i) {
      live_in.insert(i);
    }
    bool has_count_star_only{false};
    for (const auto& idx : live_out) {
      if (idx < group_key_count) {
        continue;
      }
      const auto agg_idx = idx - group_key_count;
      CHECK_LT(agg_idx, agg_expr_count);
      const auto& cur_agg_expr = aggregate->getAggExprs()[agg_idx];
      const auto n_operands = cur_agg_expr->size();
      for (size_t i = 0; i < n_operands; ++i) {
        live_in.insert(static_cast<size_t>(cur_agg_expr->getOperand(i)));
      }
      if (n_operands == 0) {
        has_count_star_only = true;
      }
    }
    if (has_count_star_only && !group_key_count && aggregate->getInput(0)->size() > 0) {
      live_in.insert(size_t(0));
    }
    return {live_in};
  }
  if (auto join = dynamic_cast<const RelJoin*>(node)) {
    std::unordered_set<size_t> lhs_live_ins;
    std::unordered_set<size_t> rhs_live_ins;
    CHECK_EQ(size_t(2), join->inputCount());
    auto lhs = join->getInput(0);
    auto rhs = join->getInput(1);
    const auto rhs_idx_base = lhs->size();
    for (const auto idx : live_out) {
      if (idx < rhs_idx_base) {
        lhs_live_ins.insert(idx);
      } else {
        rhs_live_ins.insert(idx - rhs_idx_base);
      }
    }
    auto rex_ins = collector.visit(join->getCondition());
    for (const auto& rex_in : rex_ins) {
      const auto in_idx = static_cast<size_t>(rex_in.getIndex());
      if (rex_in.getSourceNode() == lhs) {
        lhs_live_ins.insert(in_idx);
        continue;
      }
      if (rex_in.getSourceNode() == rhs) {
        rhs_live_ins.insert(in_idx);
        continue;
      }
      CHECK(false);
    }
    return {lhs_live_ins, rhs_live_ins};
  }
  if (auto left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(node)) {
    CHECK_GE(left_deep_join->inputCount(), size_t(2));
    std::vector<std::unordered_set<size_t>> live_ins(left_deep_join->inputCount());

    size_t output_base = 0;
    auto mark_live_output_range = [&](const size_t input_idx) {
      CHECK_LT(input_idx, left_deep_join->inputCount());
      const auto input = left_deep_join->getInput(input_idx);
      const auto input_output_size = get_node_output(input).size();
      for (const auto idx : live_out) {
        if (idx >= output_base && idx < output_base + input_output_size) {
          live_ins[input_idx].insert(idx - output_base);
        }
      }
      output_base += input_output_size;
    };

    mark_live_output_range(0);
    for (size_t nesting_level = 1; nesting_level < left_deep_join->inputCount();
         ++nesting_level) {
      switch (left_deep_join->getJoinType(nesting_level)) {
        case JoinType::SEMI:
        case JoinType::ANTI:
          break;
        default:
          mark_live_output_range(nesting_level);
          break;
      }
    }

    auto mark_condition_inputs = [&](const RexScalar* condition) {
      if (!condition) {
        return;
      }
      auto rex_ins = collector.visit(condition);
      for (const auto& rex_in : rex_ins) {
        for (size_t input_idx = 0; input_idx < left_deep_join->inputCount();
             ++input_idx) {
          if (rex_in.getSourceNode() == left_deep_join->getInput(input_idx)) {
            live_ins[input_idx].insert(static_cast<size_t>(rex_in.getIndex()));
            break;
          }
        }
      }
    };

    mark_condition_inputs(left_deep_join->getInnerCondition());
    for (size_t nesting_level = 1;
         nesting_level <= left_deep_join->getOuterConditionsSize();
         ++nesting_level) {
      mark_condition_inputs(left_deep_join->getOuterCondition(nesting_level));
    }
    return live_ins;
  }
  if (auto sort = dynamic_cast<const RelSort*>(node)) {
    CHECK_EQ(size_t(1), sort->inputCount());
    std::unordered_set<size_t> live_in(live_out.begin(), live_out.end());
    for (size_t i = 0; i < sort->collationCount(); ++i) {
      live_in.insert(sort->getCollation(i).getField());
    }
    return {live_in};
  }
  if (auto filter = dynamic_cast<const RelFilter*>(node)) {
    CHECK_EQ(size_t(1), filter->inputCount());
    std::unordered_set<size_t> live_in(live_out.begin(), live_out.end());
    auto rex_ins = collector.visit(filter->getCondition());
    for (const auto& rex_in : rex_ins) {
      live_in.insert(static_cast<size_t>(rex_in.getIndex()));
    }
    return {live_in};
  }
  if (auto table_func = dynamic_cast<const RelTableFunction*>(node)) {
    const auto input_count = table_func->size();
    std::unordered_set<size_t> live_in;
    for (size_t i = 0; i < input_count; i++) {
      live_in.insert(i);
    }

    std::vector<std::unordered_set<size_t>> result;
    // Is the computed result correct in general?
    for (size_t i = table_func->inputCount(); i > 0; i--) {
      result.push_back(live_in);
    }

    return result;
  }
  if (auto logical_union = dynamic_cast<const RelLogicalUnion*>(node)) {
    return std::vector<std::unordered_set<size_t>>(logical_union->inputCount(), live_out);
  }
  return {};
}

bool any_dead_col_in(const RelAlgNode* node,
                     const std::unordered_set<size_t>& live_outs) {
  CHECK(!dynamic_cast<const RelScan*>(node));
  if (auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    for (size_t i = aggregate->getGroupByCount(); i < aggregate->size(); ++i) {
      if (!live_outs.count(i)) {
        return true;
      }
    }
    return false;
  }

  return node->size() > live_outs.size();
}

bool does_redef_cols(const RelAlgNode* node) {
  return dynamic_cast<const RelAggregate*>(node) || dynamic_cast<const RelProject*>(node);
}

class AvailabilityChecker {
 public:
  AvailabilityChecker(
      const std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>&
          liveouts,
      const std::unordered_set<const RelAlgNode*>& intact_nodes)
      : liveouts_(liveouts), intact_nodes_(intact_nodes) {}

  bool hasAllSrcReady(const RelAlgNode* node) const {
    for (size_t i = 0; i < node->inputCount(); ++i) {
      auto src = node->getInput(i);
      if (!dynamic_cast<const RelScan*>(src) && liveouts_.find(src) == liveouts_.end() &&
          !intact_nodes_.count(src)) {
        return false;
      }
    }
    return true;
  }

 private:
  const std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>&
      liveouts_;
  const std::unordered_set<const RelAlgNode*>& intact_nodes_;
};

void add_new_indices_for(
    const RelAlgNode* node,
    std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>&
        new_liveouts,
    const std::unordered_set<size_t>& old_liveouts,
    const std::unordered_set<const RelAlgNode*>& intact_nodes,
    const std::unordered_map<const RelAlgNode*, size_t>& orig_node_sizes) {
  auto live_fields = old_liveouts;
  if (auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    for (size_t i = 0; i < aggregate->getGroupByCount(); ++i) {
      live_fields.insert(i);
    }
  }
  auto it_ok =
      new_liveouts.insert(std::make_pair(node, std::unordered_map<size_t, size_t>{}));
  CHECK(it_ok.second);
  auto& new_indices = it_ok.first->second;
  if (intact_nodes.count(node)) {
    for (size_t i = 0, e = node->size(); i < e; ++i) {
      new_indices.insert(std::make_pair(i, i));
    }
    return;
  }
  if (does_redef_cols(node)) {
    auto node_sz_it = orig_node_sizes.find(node);
    CHECK(node_sz_it != orig_node_sizes.end());
    const auto node_size = node_sz_it->second;
    CHECK_GT(node_size, live_fields.size());
    std::vector<size_t> ordered_indices(live_fields.begin(), live_fields.end());
    std::sort(ordered_indices.begin(), ordered_indices.end());
    for (size_t i = 0; i < ordered_indices.size(); ++i) {
      new_indices.insert(std::make_pair(ordered_indices[i], i));
    }
    return;
  }

  auto append_visible_input_mapping =
      [&](const RelAlgNode* src, size_t& old_base, size_t& new_base) {
        auto src_renum_it = new_liveouts.find(src);
        if (src_renum_it != new_liveouts.end()) {
          for (auto m : src_renum_it->second) {
            new_indices.insert(std::make_pair(old_base + m.first, new_base + m.second));
          }
          new_base += src_renum_it->second.size();
        } else if (dynamic_cast<const RelScan*>(src) || intact_nodes.count(src)) {
          for (size_t i = 0; i < src->size(); ++i) {
            new_indices.insert(std::make_pair(old_base + i, new_base + i));
          }
          new_base += src->size();
        } else {
          CHECK(false);
        }
        auto src_sz_it = orig_node_sizes.find(src);
        CHECK(src_sz_it != orig_node_sizes.end());
        old_base += src_sz_it->second;
      };

  if (const auto join = dynamic_cast<const RelJoin*>(node)) {
    size_t old_base = 0;
    size_t new_base = 0;
    CHECK_EQ(size_t(2), join->inputCount());
    append_visible_input_mapping(join->getInput(0), old_base, new_base);
    if (join->getJoinType() != JoinType::SEMI && join->getJoinType() != JoinType::ANTI) {
      append_visible_input_mapping(join->getInput(1), old_base, new_base);
    }
    return;
  }

  if (const auto left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(node)) {
    size_t old_base = 0;
    size_t new_base = 0;
    CHECK_GE(left_deep_join->inputCount(), size_t(2));
    append_visible_input_mapping(left_deep_join->getInput(0), old_base, new_base);
    for (size_t nesting_level = 1; nesting_level < left_deep_join->inputCount();
         ++nesting_level) {
      switch (left_deep_join->getJoinType(nesting_level)) {
        case JoinType::SEMI:
        case JoinType::ANTI:
          break;
        default:
          append_visible_input_mapping(
              left_deep_join->getInput(nesting_level), old_base, new_base);
          break;
      }
    }
    return;
  }

  std::vector<size_t> ordered_indices;
  for (size_t i = 0, old_base = 0, new_base = 0; i < node->inputCount(); ++i) {
    auto src = node->getInput(i);
    append_visible_input_mapping(src, old_base, new_base);
  }
}

class RexInputRenumberVisitor : public RexDeepCopyVisitor {
 public:
  RexInputRenumberVisitor(
      const std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>&
          new_numbering)
      : node_to_input_renum_(new_numbering) {}
  RetType visitInput(const RexInput* input) const override {
    auto source = input->getSourceNode();
    auto node_it = node_to_input_renum_.find(source);
    if (node_it != node_to_input_renum_.end()) {
      auto old_to_new_num = node_it->second;
      auto renum_it = old_to_new_num.find(input->getIndex());
      CHECK(renum_it != old_to_new_num.end());
      return boost::make_unique<RexInput>(source, renum_it->second);
    }
    return input->deepCopy();
  }

 private:
  const std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>&
      node_to_input_renum_;
};

std::vector<std::unique_ptr<const RexAgg>> renumber_rex_aggs(
    std::vector<std::unique_ptr<const RexAgg>>& agg_exprs,
    const std::unordered_map<size_t, size_t>& new_numbering) {
  std::vector<std::unique_ptr<const RexAgg>> new_exprs;
  for (auto& expr : agg_exprs) {
    if (expr->size() > 0) {
      std::vector<size_t> operands;
      operands.reserve(expr->size());
      for (size_t operand_idx = 0; operand_idx < expr->size(); ++operand_idx) {
        const auto idx_it = new_numbering.find(expr->getOperand(operand_idx));
        CHECK(idx_it != new_numbering.end());
        operands.push_back(idx_it->second);
      }
      new_exprs.push_back(boost::make_unique<RexAgg>(
          expr->getKind(), expr->isDistinct(), expr->getType(), operands));
      continue;
    }
    new_exprs.push_back(std::move(expr));
  }
  return new_exprs;
}

SortField renumber_sort_field(const SortField& old_field,
                              const std::unordered_map<size_t, size_t>& new_numbering) {
  auto field_idx = old_field.getField();
  auto idx_it = new_numbering.find(field_idx);
  if (idx_it != new_numbering.end()) {
    field_idx = idx_it->second;
  }
  return SortField(field_idx, old_field.getSortDir(), old_field.getNullsPosition());
}

std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>> mark_live_columns(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>> live_outs;
  std::vector<const RelAlgNode*> work_set;
  for (auto node_it = nodes.rbegin(); node_it != nodes.rend(); ++node_it) {
    auto node = node_it->get();
    if (dynamic_cast<const RelScan*>(node) || live_outs.count(node) ||
        dynamic_cast<const RelModify*>(node) ||
        dynamic_cast<const RelTableFunction*>(node)) {
      continue;
    }
    std::vector<size_t> all_live(node->size());
    std::iota(all_live.begin(), all_live.end(), size_t(0));
    live_outs.insert(std::make_pair(
        node, std::unordered_set<size_t>(all_live.begin(), all_live.end())));

    work_set.push_back(node);
    while (!work_set.empty()) {
      auto walker = work_set.back();
      work_set.pop_back();
      CHECK(!dynamic_cast<const RelScan*>(walker));
      CHECK(live_outs.count(walker));
      auto live_ins = get_live_ins(walker, live_outs);
      CHECK_EQ(live_ins.size(), walker->inputCount());
      for (size_t i = 0; i < walker->inputCount(); ++i) {
        auto src = walker->getInput(i);
        if (dynamic_cast<const RelScan*>(src) ||
            dynamic_cast<const RelTableFunction*>(src) || live_ins[i].empty()) {
          continue;
        }
        if (!live_outs.count(src)) {
          live_outs.insert(std::make_pair(src, std::unordered_set<size_t>{}));
        }
        auto src_it = live_outs.find(src);
        CHECK(src_it != live_outs.end());
        auto& live_out = src_it->second;
        bool changed = false;
        if (!live_out.empty()) {
          live_out.insert(live_ins[i].begin(), live_ins[i].end());
          changed = true;
        } else {
          for (int idx : live_ins[i]) {
            changed |= live_out.insert(idx).second;
          }
        }
        if (changed) {
          work_set.push_back(src);
        }
      }
    }
  }
  return live_outs;
}

struct BaseColumnRef {
  const RelScan* scan;
  std::string column_name;
  bool not_null;
};

std::optional<BaseColumnRef> resolve_base_column(const RelAlgNode* node,
                                                 const size_t index) {
  CHECK(node);
  CHECK_LT(index, node->size());
  if (auto scan = dynamic_cast<const RelScan*>(node)) {
    const auto& catalog = scan->getCatalog();
    const auto td = scan->getTableDescriptor();
    const auto column_name = scan->getFieldName(index);
    const auto cd = catalog.getMetadataForColumn(td->tableId, column_name);
    if (!cd) {
      return std::nullopt;
    }
    return BaseColumnRef{scan, column_name, cd->columnType.get_notnull()};
  }
  if (dynamic_cast<const RelFilter*>(node) || dynamic_cast<const RelSort*>(node)) {
    return resolve_base_column(node->getInput(0), index);
  }
  if (auto project = dynamic_cast<const RelProject*>(node)) {
    auto input = dynamic_cast<const RexInput*>(project->getProjectAt(index));
    if (!input) {
      return std::nullopt;
    }
    return resolve_base_column(input->getSourceNode(), input->getIndex());
  }
  if (auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    if (index >= aggregate->getGroupByCount()) {
      return std::nullopt;
    }
    return resolve_base_column(aggregate->getInput(0), index);
  }
  if (auto join = dynamic_cast<const RelJoin*>(node)) {
    const auto lhs_size = join->getInput(0)->size();
    if (index < lhs_size) {
      return resolve_base_column(join->getInput(0), index);
    }
    return resolve_base_column(join->getInput(1), index - lhs_size);
  }
  return std::nullopt;
}

bool is_unfiltered_key_domain(const RelAlgNode* node) {
  CHECK(node);
  if (dynamic_cast<const RelScan*>(node)) {
    return true;
  }
  if (const auto sort = dynamic_cast<const RelSort*>(node)) {
    return !sort->isLimitDelivered() && sort->getOffset() == 0 &&
           is_unfiltered_key_domain(node->getInput(0));
  }
  if (auto project = dynamic_cast<const RelProject*>(node)) {
    CHECK_EQ(size_t(1), project->inputCount());
    for (size_t i = 0; i < project->size(); ++i) {
      auto input = dynamic_cast<const RexInput*>(project->getProjectAt(i));
      if (!input || input->getSourceNode() != project->getInput(0)) {
        return false;
      }
    }
    return is_unfiltered_key_domain(project->getInput(0));
  }
  if (auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    return aggregate->getAggExprsCount() == 0 &&
           aggregate->getGroupByCount() == aggregate->size() &&
           is_unfiltered_key_domain(aggregate->getInput(0));
  }
  return false;
}

bool subtree_contains(const RelAlgNode* root, const RelAlgNode* needle) {
  if (root == needle) {
    return true;
  }
  for (size_t i = 0; i < root->inputCount(); ++i) {
    if (subtree_contains(root->getInput(i), needle)) {
      return true;
    }
  }
  return false;
}

bool collect_equi_input_pairs(
    const RexScalar* condition,
    std::vector<std::pair<const RexInput*, const RexInput*>>& pairs) {
  const auto rex_op = dynamic_cast<const RexOperator*>(condition);
  if (!rex_op) {
    return false;
  }
  if (rex_op->getOperator() == kAND) {
    for (size_t i = 0; i < rex_op->size(); ++i) {
      if (!collect_equi_input_pairs(rex_op->getOperand(i), pairs)) {
        return false;
      }
    }
    return true;
  }
  if (rex_op->getOperator() != kEQ || rex_op->size() != 2) {
    return false;
  }
  const auto lhs = dynamic_cast<const RexInput*>(rex_op->getOperand(0));
  const auto rhs = dynamic_cast<const RexInput*>(rex_op->getOperand(1));
  if (!lhs || !rhs) {
    return false;
  }
  pairs.emplace_back(lhs, rhs);
  return true;
}

using ColumnRefPair = std::pair<BaseColumnRef, BaseColumnRef>;

bool matches_foreign_key(const std::vector<ColumnRefPair>& fk_to_ref_pairs) {
  if (fk_to_ref_pairs.empty()) {
    return false;
  }
  const auto fk_scan = fk_to_ref_pairs.front().first.scan;
  const auto ref_scan = fk_to_ref_pairs.front().second.scan;
  for (const auto& [fk_col, ref_col] : fk_to_ref_pairs) {
    if (fk_col.scan != fk_scan || ref_col.scan != ref_scan || !fk_col.not_null) {
      return false;
    }
  }

  const auto fk_td = fk_scan->getTableDescriptor();
  const auto ref_td = ref_scan->getTableDescriptor();
  const auto& catalog = fk_scan->getCatalog();
  for (const auto& constraint : catalog.getTableConstraints(fk_td)) {
    if (constraint.type != Catalog_Namespace::TableConstraintType::ForeignKey ||
        !Catalog_Namespace::table_constraint_is_trusted(
            constraint, g_trust_unenforced_table_constraints) ||
        !constraint.foreign_key_reference ||
        constraint.column_names.size() != fk_to_ref_pairs.size()) {
      continue;
    }
    const auto& reference = *constraint.foreign_key_reference;
    if (reference.table_name != ref_td->tableName ||
        reference.column_names.size() != fk_to_ref_pairs.size()) {
      continue;
    }
    bool all_columns_match = true;
    for (size_t i = 0; i < constraint.column_names.size(); ++i) {
      const auto pair_it = std::find_if(
          fk_to_ref_pairs.begin(), fk_to_ref_pairs.end(), [&](const auto& pair) {
            return pair.first.column_name == constraint.column_names[i] &&
                   pair.second.column_name == reference.column_names[i];
          });
      if (pair_it == fk_to_ref_pairs.end()) {
        all_columns_match = false;
        break;
      }
    }
    if (all_columns_match) {
      return true;
    }
  }
  return false;
}

bool right_side_outputs_are_dead(
    const RelJoin* join,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& live_outs) {
  const auto live_it = live_outs.find(join);
  if (live_it == live_outs.end()) {
    return false;
  }
  const auto lhs_size = join->getInput(0)->size();
  for (const auto live_idx : live_it->second) {
    if (live_idx >= lhs_size) {
      return false;
    }
  }
  return true;
}

using IndexMap = std::unordered_map<size_t, size_t>;
using NodeIndexMap = std::unordered_map<const RelAlgNode*, IndexMap>;

std::unordered_map<const RelAlgNode*, size_t> collect_node_sizes(
    const std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  std::unordered_map<const RelAlgNode*, size_t> node_sizes;
  for (const auto& node : nodes) {
    node_sizes.emplace(node.get(), node->size());
  }
  return node_sizes;
}

IndexMap make_preserved_prefix_map(const size_t old_size, const size_t new_size) {
  CHECK_LE(new_size, old_size);
  IndexMap map;
  for (size_t i = 0; i < new_size; ++i) {
    map.emplace(i, i);
  }
  return map;
}

size_t mapped_size(const IndexMap& map) {
  size_t size = 0;
  for (const auto& [_, new_idx] : map) {
    size = std::max(size, new_idx + 1);
  }
  return size;
}

bool is_identity_map(const IndexMap& map, const size_t old_size) {
  if (map.size() != old_size) {
    return false;
  }
  for (size_t i = 0; i < old_size; ++i) {
    auto it = map.find(i);
    if (it == map.end() || it->second != i) {
      return false;
    }
  }
  return true;
}

void add_input_output_mapping(
    IndexMap& node_map,
    const RelAlgNode* input,
    const size_t old_base,
    const size_t new_base,
    const std::unordered_map<const RelAlgNode*, size_t>& old_node_sizes,
    const NodeIndexMap& input_maps) {
  auto input_map_it = input_maps.find(input);
  if (input_map_it != input_maps.end()) {
    for (const auto& [old_idx, new_idx] : input_map_it->second) {
      node_map.emplace(old_base + old_idx, new_base + new_idx);
    }
    return;
  }

  auto old_size_it = old_node_sizes.find(input);
  CHECK(old_size_it != old_node_sizes.end());
  for (size_t i = 0; i < old_size_it->second; ++i) {
    node_map.emplace(old_base + i, new_base + i);
  }
}

IndexMap compute_shape_preserving_output_map(
    const RelAlgNode* node,
    const std::unordered_map<const RelAlgNode*, size_t>& old_node_sizes,
    const NodeIndexMap& input_maps) {
  CHECK(node);
  CHECK(!does_redef_cols(node));
  CHECK(!dynamic_cast<const RelScan*>(node));

  IndexMap node_map;
  size_t old_base = 0;
  size_t new_base = 0;
  for (size_t i = 0; i < node->inputCount(); ++i) {
    auto input = node->getInput(i);
    add_input_output_mapping(
        node_map, input, old_base, new_base, old_node_sizes, input_maps);

    auto old_size_it = old_node_sizes.find(input);
    CHECK(old_size_it != old_node_sizes.end());
    old_base += old_size_it->second;

    auto input_map_it = input_maps.find(input);
    new_base += input_map_it == input_maps.end() ? input->size()
                                                 : mapped_size(input_map_it->second);
  }
  return node_map;
}

IndexMap compute_output_map_after_input_replacement(
    const RelAlgNode* node,
    const RelAlgNode* old_input,
    const size_t new_input_size,
    const std::unordered_map<const RelAlgNode*, size_t>& old_node_sizes) {
  if (does_redef_cols(node) || dynamic_cast<const RelCompound*>(node)) {
    auto node_size_it = old_node_sizes.find(node);
    CHECK(node_size_it != old_node_sizes.end());
    return make_preserved_prefix_map(node_size_it->second, node_size_it->second);
  }

  NodeIndexMap input_maps;
  auto old_input_size_it = old_node_sizes.find(old_input);
  CHECK(old_input_size_it != old_node_sizes.end());
  input_maps.emplace(
      old_input, make_preserved_prefix_map(old_input_size_it->second, new_input_size));
  return compute_shape_preserving_output_map(node, old_node_sizes, input_maps);
}

class RexInputRebindRenumberVisitor : public RexVisitor<void*> {
 public:
  RexInputRebindRenumberVisitor(const RelAlgNode* old_input,
                                const RelAlgNode* new_input,
                                const IndexMap& old_to_new_index)
      : old_input_(old_input)
      , new_input_(new_input)
      , old_to_new_index_(old_to_new_index) {}

  void* visitInput(const RexInput* rex_input) const override {
    if (rex_input->getSourceNode() == old_input_) {
      auto index_it = old_to_new_index_.find(rex_input->getIndex());
      CHECK(index_it != old_to_new_index_.end());
      rex_input->setSourceNode(new_input_);
      rex_input->setIndex(index_it->second);
    }
    return nullptr;
  }

 private:
  const RelAlgNode* old_input_;
  const RelAlgNode* new_input_;
  const IndexMap& old_to_new_index_;
};

void visit_rex_scalars(RelAlgNode* node, const RexVisitor<void*>& visitor) {
  if (auto project = dynamic_cast<RelProject*>(node)) {
    for (size_t i = 0; i < project->size(); ++i) {
      visitor.visit(project->getProjectAt(i));
    }
    return;
  }
  if (auto filter = dynamic_cast<RelFilter*>(node)) {
    visitor.visit(filter->getCondition());
    return;
  }
  if (auto join = dynamic_cast<RelJoin*>(node)) {
    if (auto condition = join->getCondition()) {
      visitor.visit(condition);
    }
    return;
  }
  if (auto compound = dynamic_cast<RelCompound*>(node)) {
    if (auto filter_expr = compound->getFilterExpr()) {
      visitor.visit(filter_expr);
    }
    for (size_t i = 0; i < compound->getScalarSourcesSize(); ++i) {
      visitor.visit(compound->getScalarSource(i));
    }
    return;
  }
}

std::unordered_map<size_t, size_t> strict_input_map_for_node(
    const RelAlgNode* input,
    const NodeIndexMap& node_maps) {
  auto map_it = node_maps.find(input);
  CHECK(map_it != node_maps.end());
  return map_it->second;
}

void renumber_sort_input(RelSort* sort, const IndexMap& input_map) {
  std::vector<SortField> new_collations;
  for (size_t i = 0; i < sort->collationCount(); ++i) {
    const auto old_field = sort->getCollation(i);
    auto field_it = input_map.find(old_field.getField());
    CHECK(field_it != input_map.end());
    new_collations.emplace_back(
        field_it->second, old_field.getSortDir(), old_field.getNullsPosition());
  }
  sort->setCollation(std::move(new_collations));
}

void renumber_aggregate_input(RelAggregate* aggregate, const IndexMap& input_map) {
  for (size_t i = 0; i < aggregate->getGroupByCount(); ++i) {
    auto group_it = input_map.find(i);
    CHECK(group_it != input_map.end());
    CHECK_EQ(i, group_it->second);
  }
  for (const auto& agg_expr : aggregate->getAggExprs()) {
    for (size_t i = 0; i < agg_expr->size(); ++i) {
      CHECK(input_map.count(agg_expr->getOperand(i)));
    }
  }
  auto old_exprs = aggregate->getAggExprsAndRelease();
  auto new_exprs = renumber_rex_aggs(old_exprs, input_map);
  aggregate->setAggExprs(new_exprs);
}

void rebind_node_input_with_map(RelAlgNode* node,
                                std::shared_ptr<const RelAlgNode> old_input,
                                std::shared_ptr<const RelAlgNode> new_input,
                                const IndexMap& old_to_new_index) {
  if (auto sort = dynamic_cast<RelSort*>(node)) {
    renumber_sort_input(sort, old_to_new_index);
  } else if (auto aggregate = dynamic_cast<RelAggregate*>(node)) {
    renumber_aggregate_input(aggregate, old_to_new_index);
  } else {
    RexInputRebindRenumberVisitor visitor(
        old_input.get(), new_input.get(), old_to_new_index);
    visit_rex_scalars(node, visitor);
  }
  node->RelAlgNode::replaceInput(std::move(old_input), std::move(new_input));
}

void renumber_node_inputs(RelAlgNode* node, const NodeIndexMap& node_maps) {
  RexInputRenumberVisitor renumberer(node_maps);
  if (auto project = dynamic_cast<RelProject*>(node)) {
    auto old_exprs = project->getExpressionsAndRelease();
    std::vector<std::unique_ptr<const RexScalar>> new_exprs;
    for (auto& expr : old_exprs) {
      new_exprs.push_back(renumberer.visit(expr.get()));
    }
    project->setExpressions(new_exprs);
    return;
  }
  if (auto filter = dynamic_cast<RelFilter*>(node)) {
    auto new_condition = renumberer.visit(filter->getCondition());
    filter->setCondition(new_condition);
    return;
  }
  if (auto join = dynamic_cast<RelJoin*>(node)) {
    if (join->getCondition()) {
      auto new_condition = renumberer.visit(join->getCondition());
      join->setCondition(new_condition);
    }
    return;
  }
  if (auto sort = dynamic_cast<RelSort*>(node)) {
    renumber_sort_input(sort, strict_input_map_for_node(sort->getInput(0), node_maps));
    return;
  }
  if (auto aggregate = dynamic_cast<RelAggregate*>(node)) {
    renumber_aggregate_input(
        aggregate, strict_input_map_for_node(aggregate->getInput(0), node_maps));
    return;
  }
  if (auto compound = dynamic_cast<RelCompound*>(node)) {
    if (compound->getFilterExpr()) {
      auto new_filter = renumberer.visit(compound->getFilterExpr());
      compound->setFilterExpr(new_filter);
    }
    std::vector<std::unique_ptr<const RexScalar>> new_sources;
    for (size_t i = 0; i < compound->getScalarSourcesSize(); ++i) {
      new_sources.push_back(renumberer.visit(compound->getScalarSource(i)));
    }
    compound->setScalarSources(new_sources);
  }
}

void propagate_output_renumbering(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    const std::unordered_map<const RelAlgNode*, size_t>& old_node_sizes,
    NodeIndexMap node_maps,
    const std::unordered_set<const RelAlgNode*>& already_rewritten) {
  std::unordered_set<const RelAlgNode*> rewritten = already_rewritten;
  bool changed = false;
  do {
    changed = false;
    for (auto& node_owner : nodes) {
      auto node = node_owner.get();
      if (!node || node_maps.count(node) || rewritten.count(node)) {
        continue;
      }

      bool has_renumbered_input = false;
      for (size_t i = 0; i < node->inputCount(); ++i) {
        has_renumbered_input |= node_maps.count(node->getInput(i)) > 0;
      }
      if (!has_renumbered_input) {
        continue;
      }

      renumber_node_inputs(node, node_maps);
      rewritten.insert(node);
      if (!does_redef_cols(node) && !dynamic_cast<const RelCompound*>(node)) {
        auto node_map =
            compute_shape_preserving_output_map(node, old_node_sizes, node_maps);
        auto old_size_it = old_node_sizes.find(node);
        CHECK(old_size_it != old_node_sizes.end());
        if (!is_identity_map(node_map, old_size_it->second)) {
          node_maps.emplace(node, std::move(node_map));
          changed = true;
        }
      }
    }
  } while (changed);
}

bool can_eliminate_right_fk_join(
    const RelJoin* join,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& live_outs) {
  if (join->getJoinType() != JoinType::INNER || !join->getCondition() ||
      !right_side_outputs_are_dead(join, live_outs)) {
    return false;
  }
  const auto lhs = join->getInput(0);
  const auto rhs = join->getInput(1);
  if (!is_unfiltered_key_domain(rhs)) {
    return false;
  }

  std::vector<std::pair<const RexInput*, const RexInput*>> input_pairs;
  if (!collect_equi_input_pairs(join->getCondition(), input_pairs)) {
    return false;
  }

  std::vector<ColumnRefPair> fk_to_ref_pairs;
  for (const auto& [left_input, right_input] : input_pairs) {
    const RexInput* fk_input = nullptr;
    const RexInput* ref_input = nullptr;
    if (subtree_contains(lhs, left_input->getSourceNode()) &&
        subtree_contains(rhs, right_input->getSourceNode())) {
      fk_input = left_input;
      ref_input = right_input;
    } else if (subtree_contains(lhs, right_input->getSourceNode()) &&
               subtree_contains(rhs, left_input->getSourceNode())) {
      fk_input = right_input;
      ref_input = left_input;
    } else {
      return false;
    }

    auto fk_col = resolve_base_column(fk_input->getSourceNode(), fk_input->getIndex());
    auto ref_col = resolve_base_column(ref_input->getSourceNode(), ref_input->getIndex());
    if (!fk_col || !ref_col) {
      return false;
    }
    fk_to_ref_pairs.emplace_back(std::move(*fk_col), std::move(*ref_col));
  }
  return matches_foreign_key(fk_to_ref_pairs);
}

std::string get_field_name(const RelAlgNode* node, size_t index) {
  CHECK_LT(index, node->size());
  if (auto scan = dynamic_cast<const RelScan*>(node)) {
    return scan->getFieldName(index);
  }
  if (auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    CHECK_EQ(aggregate->size(), aggregate->getFields().size());
    return aggregate->getFieldName(index);
  }
  if (auto join = dynamic_cast<const RelJoin*>(node)) {
    const auto lhs_size = join->getInput(0)->size();
    if (index < lhs_size) {
      return get_field_name(join->getInput(0), index);
    }
    return get_field_name(join->getInput(1), index - lhs_size);
  }
  if (auto project = dynamic_cast<const RelProject*>(node)) {
    return project->getFieldName(index);
  }
  CHECK(dynamic_cast<const RelSort*>(node) || dynamic_cast<const RelFilter*>(node));
  return get_field_name(node->getInput(0), index);
}

void try_insert_coalesceable_proj(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& liveouts,
    std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>&
        du_web) {
  std::vector<std::shared_ptr<RelAlgNode>> new_nodes;
  for (auto node : nodes) {
    new_nodes.push_back(node);
    if (!std::dynamic_pointer_cast<RelFilter>(node)) {
      continue;
    }
    const auto filter = node.get();
    auto liveout_it = liveouts.find(filter);
    CHECK(liveout_it != liveouts.end());
    auto& outs = liveout_it->second;
    if (!any_dead_col_in(filter, outs)) {
      continue;
    }
    auto usrs_it = du_web.find(filter);
    CHECK(usrs_it != du_web.end());
    auto& usrs = usrs_it->second;
    if (usrs.size() != 1 || does_redef_cols(*usrs.begin())) {
      continue;
    }
    auto only_usr = const_cast<RelAlgNode*>(*usrs.begin());

    std::vector<std::unique_ptr<const RexScalar>> exprs;
    std::vector<std::string> fields;
    for (size_t i = 0; i < filter->size(); ++i) {
      exprs.push_back(boost::make_unique<RexInput>(filter, i));
      fields.push_back(get_field_name(filter, i));
    }
    auto project_owner = std::make_shared<RelProject>(exprs, fields, node);
    auto project = project_owner.get();

    only_usr->replaceInput(node, project_owner);
    if (dynamic_cast<const RelJoin*>(only_usr)) {
      RexRebindInputsVisitor visitor(filter, project);
      for (auto usr : du_web[only_usr]) {
        visitor.visitNode(usr);
      }
    }

    liveouts.insert(std::make_pair(project, outs));

    usrs.clear();
    usrs.insert(project);
    du_web.insert(
        std::make_pair(project, std::unordered_set<const RelAlgNode*>{only_usr}));

    new_nodes.push_back(project_owner);
  }
  if (new_nodes.size() > nodes.size()) {
    nodes.swap(new_nodes);
  }
}

std::pair<std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>,
          std::vector<const RelAlgNode*>>
sweep_dead_columns(
    const std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& live_outs,
    const std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    const std::unordered_set<const RelAlgNode*>& intact_nodes,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>&
        du_web,
    const std::unordered_map<const RelAlgNode*, size_t>& orig_node_sizes) {
  std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>
      liveouts_renumbering;
  std::vector<const RelAlgNode*> ready_nodes;
  AvailabilityChecker checker(liveouts_renumbering, intact_nodes);
  for (auto node : nodes) {
    // Ignore empty live_out due to some invalid node
    if (!does_redef_cols(node.get()) || intact_nodes.count(node.get())) {
      continue;
    }
    auto live_pair = live_outs.find(node.get());
    CHECK(live_pair != live_outs.end());
    auto old_live_outs = live_pair->second;
    add_new_indices_for(
        node.get(), liveouts_renumbering, old_live_outs, intact_nodes, orig_node_sizes);
    if (auto aggregate = std::dynamic_pointer_cast<RelAggregate>(node)) {
      auto old_exprs = aggregate->getAggExprsAndRelease();
      std::vector<std::unique_ptr<const RexAgg>> new_exprs;
      auto key_name_it = aggregate->getFields().begin();
      std::vector<std::string> new_fields(key_name_it,
                                          key_name_it + aggregate->getGroupByCount());
      for (size_t i = aggregate->getGroupByCount(), j = 0;
           i < aggregate->getFields().size() && j < old_exprs.size();
           ++i, ++j) {
        if (old_live_outs.count(i)) {
          new_exprs.push_back(std::move(old_exprs[j]));
          new_fields.push_back(aggregate->getFieldName(i));
        }
      }
      aggregate->setAggExprs(new_exprs);
      aggregate->setFields(std::move(new_fields));
    } else if (auto project = std::dynamic_pointer_cast<RelProject>(node)) {
      auto old_exprs = project->getExpressionsAndRelease();
      std::vector<std::unique_ptr<const RexScalar>> new_exprs;
      std::vector<std::string> new_fields;
      for (size_t i = 0; i < old_exprs.size(); ++i) {
        if (old_live_outs.count(i)) {
          new_exprs.push_back(std::move(old_exprs[i]));
          new_fields.push_back(project->getFieldName(i));
        }
      }
      project->setExpressions(new_exprs);
      project->setFields(std::move(new_fields));
    } else {
      CHECK(false);
    }
    auto usrs_it = du_web.find(node.get());
    CHECK(usrs_it != du_web.end());
    for (auto usr : usrs_it->second) {
      if (checker.hasAllSrcReady(usr)) {
        ready_nodes.push_back(usr);
      }
    }
  }
  return {liveouts_renumbering, ready_nodes};
}

void propagate_input_renumbering(
    std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>&
        liveout_renumbering,
    const std::vector<const RelAlgNode*>& ready_nodes,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& old_liveouts,
    const std::unordered_set<const RelAlgNode*>& intact_nodes,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>&
        du_web,
    const std::unordered_map<const RelAlgNode*, size_t>& orig_node_sizes) {
  RexInputRenumberVisitor renumberer(liveout_renumbering);
  AvailabilityChecker checker(liveout_renumbering, intact_nodes);
  std::deque<const RelAlgNode*> work_set(ready_nodes.begin(), ready_nodes.end());
  while (!work_set.empty()) {
    auto walker = work_set.front();
    work_set.pop_front();
    CHECK(!dynamic_cast<const RelScan*>(walker));
    auto node = const_cast<RelAlgNode*>(walker);
    if (auto project = dynamic_cast<RelProject*>(node)) {
      auto old_exprs = project->getExpressionsAndRelease();
      std::vector<std::unique_ptr<const RexScalar>> new_exprs;
      for (auto& expr : old_exprs) {
        new_exprs.push_back(renumberer.visit(expr.get()));
      }
      project->setExpressions(new_exprs);
    } else if (auto aggregate = dynamic_cast<RelAggregate*>(node)) {
      auto src_it = liveout_renumbering.find(node->getInput(0));
      CHECK(src_it != liveout_renumbering.end());
      auto old_exprs = aggregate->getAggExprsAndRelease();
      auto new_exprs = renumber_rex_aggs(old_exprs, src_it->second);
      aggregate->setAggExprs(new_exprs);
    } else if (auto join = dynamic_cast<RelJoin*>(node)) {
      if (join->getCondition()) {
        auto new_condition = renumberer.visit(join->getCondition());
        join->setCondition(new_condition);
      }
    } else if (auto filter = dynamic_cast<RelFilter*>(node)) {
      auto new_condition = renumberer.visit(filter->getCondition());
      filter->setCondition(new_condition);
    } else if (auto sort = dynamic_cast<RelSort*>(node)) {
      auto src_it = liveout_renumbering.find(node->getInput(0));
      CHECK(src_it != liveout_renumbering.end());
      std::vector<SortField> new_collations;
      for (size_t i = 0; i < sort->collationCount(); ++i) {
        new_collations.push_back(
            renumber_sort_field(sort->getCollation(i), src_it->second));
      }
      sort->setCollation(std::move(new_collations));
    } else if (!dynamic_cast<RelLogicalUnion*>(node)) {
      LOG(FATAL) << "Unhandled node type: "
                 << node->toString(RelRexToStringConfig::defaults());
    }

    // Ignore empty live_out due to some invalid node
    if (does_redef_cols(node) || intact_nodes.count(node)) {
      continue;
    }
    auto live_pair = old_liveouts.find(node);
    CHECK(live_pair != old_liveouts.end());
    auto live_out = live_pair->second;
    add_new_indices_for(
        node, liveout_renumbering, live_out, intact_nodes, orig_node_sizes);
    auto usrs_it = du_web.find(walker);
    CHECK(usrs_it != du_web.end());
    for (auto usr : usrs_it->second) {
      if (checker.hasAllSrcReady(usr)) {
        work_set.push_back(usr);
      }
    }
  }
}

}  // namespace

void eliminate_lossless_fk_joins(std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  if (nodes.empty()) {
    return;
  }

  auto sink = nodes.back();
  bool changed = false;
  do {
    changed = false;
    auto live_outs = mark_live_columns(nodes);
    auto web = build_du_web(nodes);
    auto old_node_sizes = collect_node_sizes(nodes);
    for (auto& node : nodes) {
      auto join = std::dynamic_pointer_cast<RelJoin>(node);
      if (!join || !can_eliminate_right_fk_join(join.get(), live_outs)) {
        continue;
      }
      auto users_it = web.find(join.get());
      if (users_it == web.end() || users_it->second.empty()) {
        continue;
      }

      auto kept_input = join->getAndOwnInput(0);
      const auto removed_to_kept_map =
          make_preserved_prefix_map(join->size(), kept_input->size());
      NodeIndexMap changed_node_maps;
      std::unordered_set<const RelAlgNode*> rewritten_users;
      for (auto user : users_it->second) {
        auto user_map = compute_output_map_after_input_replacement(
            user, join.get(), kept_input->size(), old_node_sizes);
        rebind_node_input_with_map(
            const_cast<RelAlgNode*>(user), node, kept_input, removed_to_kept_map);
        auto old_size_it = old_node_sizes.find(user);
        CHECK(old_size_it != old_node_sizes.end());
        if (!is_identity_map(user_map, old_size_it->second)) {
          changed_node_maps.emplace(user, std::move(user_map));
        }
        rewritten_users.insert(user);
      }
      propagate_output_renumbering(
          nodes, old_node_sizes, std::move(changed_node_maps), rewritten_users);
      changed = true;
      break;
    }
    if (changed) {
      cleanup_dead_nodes(nodes);
    }
  } while (changed);
}

void eliminate_dead_columns(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  if (nodes.empty()) {
    return;
  }
  auto root = nodes.back().get();
  if (!root) {
    return;
  }
  CHECK(!dynamic_cast<const RelScan*>(root) && !dynamic_cast<const RelJoin*>(root));
  // Mark
  auto old_liveouts = mark_live_columns(nodes);
  std::unordered_set<const RelAlgNode*> intact_nodes;
  bool has_dead_cols = false;
  for (auto live_pair : old_liveouts) {
    auto node = live_pair.first;
    const auto& outs = live_pair.second;
    if (outs.empty()) {
      LOG(WARNING) << "RA node with no used column: "
                   << node->toString(RelRexToStringConfig::defaults());
      // Ignore empty live_out due to some invalid node
      intact_nodes.insert(node);
    }
    if (any_dead_col_in(node, outs)) {
      has_dead_cols = true;
    } else {
      intact_nodes.insert(node);
    }
  }
  if (!has_dead_cols) {
    return;
  }
  auto web = build_du_web(nodes);
  try_insert_coalesceable_proj(nodes, old_liveouts, web);

  for (auto node : nodes) {
    if (intact_nodes.count(node.get()) || does_redef_cols(node.get())) {
      continue;
    }
    bool intact = true;
    for (size_t i = 0; i < node->inputCount(); ++i) {
      auto source = node->getInput(i);
      if (!dynamic_cast<const RelScan*>(source) && !intact_nodes.count(source)) {
        intact = false;
        break;
      }
    }
    if (intact) {
      intact_nodes.insert(node.get());
    }
  }

  std::unordered_map<const RelAlgNode*, size_t> orig_node_sizes;
  for (auto node : nodes) {
    orig_node_sizes.insert(std::make_pair(node.get(), node->size()));
  }
  // Sweep
  std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>
      liveout_renumbering;
  std::vector<const RelAlgNode*> ready_nodes;
  std::tie(liveout_renumbering, ready_nodes) =
      sweep_dead_columns(old_liveouts, nodes, intact_nodes, web, orig_node_sizes);
  // Propagate
  propagate_input_renumbering(
      liveout_renumbering, ready_nodes, old_liveouts, intact_nodes, web, orig_node_sizes);
}

void eliminate_dead_subqueries(std::vector<std::shared_ptr<RexSubQuery>>& subqueries,
                               RelAlgNode const* root) {
  if (!subqueries.empty()) {
    auto live_ids = RexSubQueryIdCollector::getLiveRexSubQueryIds(root);
    auto sort_live_ids_first = [&live_ids](auto& a, auto& b) {
      return live_ids.count(a->getId()) && !live_ids.count(b->getId());
    };
    std::stable_sort(subqueries.begin(), subqueries.end(), sort_live_ids_first);
    size_t n_dead_subqueries;
    if (live_ids.count(subqueries.front()->getId())) {
      auto first_dead_itr = std::upper_bound(subqueries.cbegin(),
                                             subqueries.cend(),
                                             subqueries.front(),
                                             sort_live_ids_first);
      n_dead_subqueries = subqueries.cend() - first_dead_itr;
    } else {
      n_dead_subqueries = subqueries.size();
    }
    if (n_dead_subqueries) {
      VLOG(1) << "Eliminating " << n_dead_subqueries
              << (n_dead_subqueries == 1 ? " subquery." : " subqueries.");
      subqueries.resize(subqueries.size() - n_dead_subqueries);
      subqueries.shrink_to_fit();
    }
  }
}

namespace {

class RexInputSinker : public RexDeepCopyVisitor {
 public:
  RexInputSinker(const std::unordered_map<size_t, size_t>& old_to_new_idx,
                 const RelAlgNode* new_src)
      : old_to_new_in_idx_(old_to_new_idx), target_(new_src) {}

  RetType visitInput(const RexInput* input) const override {
    CHECK_EQ(target_->inputCount(), size_t(1));
    CHECK_EQ(target_->getInput(0), input->getSourceNode());
    auto idx_it = old_to_new_in_idx_.find(input->getIndex());
    CHECK(idx_it != old_to_new_in_idx_.end());
    return boost::make_unique<RexInput>(target_, idx_it->second);
  }

 private:
  const std::unordered_map<size_t, size_t>& old_to_new_in_idx_;
  const RelAlgNode* target_;
};

class SubConditionReplacer : public RexDeepCopyVisitor {
 public:
  SubConditionReplacer(const std::unordered_map<size_t, std::unique_ptr<const RexScalar>>&
                           idx_to_sub_condition)
      : idx_to_subcond_(idx_to_sub_condition) {}
  RetType visitInput(const RexInput* input) const override {
    auto subcond_it = idx_to_subcond_.find(input->getIndex());
    if (subcond_it != idx_to_subcond_.end()) {
      return RexDeepCopyVisitor::visit(subcond_it->second.get());
    }
    return RexDeepCopyVisitor::visitInput(input);
  }

 private:
  const std::unordered_map<size_t, std::unique_ptr<const RexScalar>>& idx_to_subcond_;
};

}  // namespace

void sink_projected_boolean_expr_to_join(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  auto web = build_du_web(nodes);
  auto liveouts = mark_live_columns(nodes);
  for (auto node : nodes) {
    auto project = std::dynamic_pointer_cast<RelProject>(node);
    // TODO(miyu): relax RelScan limitation
    if (!project || project->isSimple() ||
        !dynamic_cast<const RelScan*>(project->getInput(0))) {
      continue;
    }
    auto usrs_it = web.find(project.get());
    CHECK(usrs_it != web.end());
    auto& usrs = usrs_it->second;
    if (usrs.size() != 1) {
      continue;
    }
    auto join = dynamic_cast<RelJoin*>(const_cast<RelAlgNode*>(*usrs.begin()));
    if (!join) {
      continue;
    }
    auto outs_it = liveouts.find(join);
    CHECK(outs_it != liveouts.end());
    std::unordered_map<size_t, size_t> in_to_out_index;
    std::unordered_set<size_t> boolean_expr_indicies;
    bool discarded = false;
    for (size_t i = 0; i < project->size(); ++i) {
      auto oper = dynamic_cast<const RexOperator*>(project->getProjectAt(i));
      if (oper && oper->getType().get_type() == kBOOLEAN) {
        boolean_expr_indicies.insert(i);
      } else {
        // TODO(miyu): relax?
        if (auto input = dynamic_cast<const RexInput*>(project->getProjectAt(i))) {
          in_to_out_index.insert(std::make_pair(input->getIndex(), i));
        } else {
          discarded = true;
        }
      }
    }
    if (discarded || boolean_expr_indicies.empty()) {
      continue;
    }
    const size_t index_base =
        join->getInput(0) == project.get() ? 0 : join->getInput(0)->size();
    for (auto i : boolean_expr_indicies) {
      auto join_idx = index_base + i;
      if (outs_it->second.count(join_idx)) {
        discarded = true;
        break;
      }
    }
    if (discarded) {
      continue;
    }
    RexInputCollector collector(project.get());
    std::vector<size_t> unloaded_input_indices;
    std::unordered_map<size_t, std::unique_ptr<const RexScalar>> in_idx_to_new_subcond;
    // Given all are dead right after join, safe to sink
    for (auto i : boolean_expr_indicies) {
      auto rex_ins = collector.visit(project->getProjectAt(i));
      for (auto& in : rex_ins) {
        CHECK_EQ(in.getSourceNode(), project->getInput(0));
        if (!in_to_out_index.count(in.getIndex())) {
          auto curr_out_index = project->size() + unloaded_input_indices.size();
          in_to_out_index.insert(std::make_pair(in.getIndex(), curr_out_index));
          unloaded_input_indices.push_back(in.getIndex());
        }
        RexInputSinker sinker(in_to_out_index, project.get());
        in_idx_to_new_subcond.insert(
            std::make_pair(i, sinker.visit(project->getProjectAt(i))));
      }
    }
    if (in_idx_to_new_subcond.empty()) {
      continue;
    }
    std::vector<std::unique_ptr<const RexScalar>> new_projections;
    for (size_t i = 0; i < project->size(); ++i) {
      if (boolean_expr_indicies.count(i)) {
        new_projections.push_back(boost::make_unique<RexInput>(project->getInput(0), 0));
      } else {
        auto rex_input = dynamic_cast<const RexInput*>(project->getProjectAt(i));
        CHECK(rex_input != nullptr);
        new_projections.push_back(rex_input->deepCopy());
      }
    }
    for (auto i : unloaded_input_indices) {
      new_projections.push_back(boost::make_unique<RexInput>(project->getInput(0), i));
    }
    project->setExpressions(new_projections);

    SubConditionReplacer replacer(in_idx_to_new_subcond);
    auto new_condition = replacer.visit(join->getCondition());
    join->setCondition(new_condition);
  }
}

namespace {

class RexInputRedirector : public RexDeepCopyVisitor {
 public:
  RexInputRedirector(const RelAlgNode* old_src, const RelAlgNode* new_src)
      : old_src_(old_src), new_src_(new_src) {}

  RetType visitInput(const RexInput* input) const override {
    CHECK_EQ(old_src_, input->getSourceNode());
    CHECK_NE(old_src_, new_src_);
    auto actual_new_src = new_src_;
    if (auto join = dynamic_cast<const RelJoin*>(new_src_)) {
      actual_new_src = join->getInput(0);
      CHECK_EQ(join->inputCount(), size_t(2));
      auto src2_input_base = actual_new_src->size();
      if (input->getIndex() >= src2_input_base) {
        actual_new_src = join->getInput(1);
        return boost::make_unique<RexInput>(actual_new_src,
                                            input->getIndex() - src2_input_base);
      }
    }

    return boost::make_unique<RexInput>(actual_new_src, input->getIndex());
  }

 private:
  const RelAlgNode* old_src_;
  const RelAlgNode* new_src_;
};

class RelRexRebindInputsVisitor : public RelAlgDagNode::Visitor {
 public:
  RelRexRebindInputsVisitor(const RelAlgNode* old_node, const RelAlgNode* new_node)
      : old_node_(old_node), new_node_(new_node) {}

  bool visit(RexInput const* n, std::string s) {
    if (n->getSourceNode() == old_node_) {
      n->setSourceNode(new_node_);
    }
    return true;
  }

 private:
  const RelAlgNode* old_node_;
  const RelAlgNode* new_node_;
};

void replace_all_usages(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    std::shared_ptr<const RelAlgNode> old_def_node,
    std::shared_ptr<const RelAlgNode> new_def_node,
    std::unordered_map<const RelAlgNode*, std::shared_ptr<RelAlgNode>>& deconst_mapping,
    std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>&
        du_web) {
  auto usrs_it = du_web.find(old_def_node.get());
  RelRexRebindInputsVisitor redirector(old_def_node.get(), new_def_node.get());
  // Rebind all RexInputs
  for (auto node : nodes) {
    if (node) {
      node->accept(redirector, std::to_string(node->getId()));
    }
  }
  // Replace all references amongst nodes to the replaced node with the new node (changing
  // the DAG structure)
  for (auto node : nodes) {
    if (node) {
      node->RelAlgNode::replaceInput(old_def_node, new_def_node);
    }
  }
  CHECK(usrs_it != du_web.end());
  auto new_usrs_it = du_web.find(new_def_node.get());
  CHECK(new_usrs_it != du_web.end());
  new_usrs_it->second.insert(usrs_it->second.begin(), usrs_it->second.end());
  usrs_it->second.clear();
}

}  // namespace

void fold_filters(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  std::unordered_map<const RelAlgNode*, std::shared_ptr<RelAlgNode>> deconst_mapping;
  for (auto node : nodes) {
    deconst_mapping.insert(std::make_pair(node.get(), node));
  }

  auto web = build_du_web(nodes);
  for (auto node_it = nodes.rbegin(); node_it != nodes.rend(); ++node_it) {
    auto& node = *node_it;
    if (auto filter = std::dynamic_pointer_cast<RelFilter>(node)) {
      CHECK_EQ(filter->inputCount(), size_t(1));
      auto src_filter = dynamic_cast<const RelFilter*>(filter->getInput(0));
      if (!src_filter) {
        continue;
      }
      auto siblings_it = web.find(src_filter);
      if (siblings_it == web.end() || siblings_it->second.size() != size_t(1)) {
        continue;
      }
      auto src_it = deconst_mapping.find(src_filter);
      CHECK(src_it != deconst_mapping.end());
      auto folded_filter = std::dynamic_pointer_cast<RelFilter>(src_it->second);
      CHECK(folded_filter);
      // TODO(miyu) : drop filter w/ only expression valued constant TRUE?
      if (auto rex_operator = dynamic_cast<const RexOperator*>(filter->getCondition())) {
        VLOG(1) << "Node ID=" << filter->getId() << " folded into "
                << "ID=" << folded_filter->getId();
        if (logger::fast_logging_check(logger::Severity::DEBUG2)) {
          auto node_str = folded_filter->toString(RelRexToStringConfig::defaults());
          auto [node_substr, post_fix] = ::substring(node_str, g_max_log_length);
          VLOG(2) << "Folded Node (ID: " << folded_filter->getId()
                  << ") contents: " << node_substr << post_fix;
        }
        std::vector<std::unique_ptr<const RexScalar>> operands;
        operands.emplace_back(folded_filter->getAndReleaseCondition());
        auto old_condition = dynamic_cast<const RexOperator*>(operands.back().get());
        CHECK(old_condition && old_condition->getType().get_type() == kBOOLEAN);
        RexInputRedirector redirector(folded_filter.get(), folded_filter->getInput(0));
        operands.push_back(redirector.visit(rex_operator));
        auto other_condition = dynamic_cast<const RexOperator*>(operands.back().get());
        CHECK(other_condition && other_condition->getType().get_type() == kBOOLEAN);
        const bool notnull = old_condition->getType().get_notnull() &&
                             other_condition->getType().get_notnull();
        auto new_condition = std::unique_ptr<const RexScalar>(
            new RexOperator(kAND, operands, SQLTypeInfo(kBOOLEAN, notnull)));
        folded_filter->setCondition(new_condition);
        replace_all_usages(nodes, filter, folded_filter, deconst_mapping, web);
        deconst_mapping.erase(filter.get());
        web.erase(filter.get());
        web[filter->getInput(0)].erase(filter.get());
        node.reset();
      }
    }
  }

  if (!nodes.empty()) {
    auto sink = nodes.back();
    for (auto node_it = std::next(nodes.rbegin()); node_it != nodes.rend(); ++node_it) {
      if (sink) {
        break;
      }
      sink = *node_it;
    }
    CHECK(sink);
    cleanup_dead_nodes(nodes);
  }
}

std::vector<const RexScalar*> find_hoistable_conditions(const RexScalar* condition,
                                                        const RelAlgNode* source,
                                                        const size_t first_col_idx,
                                                        const size_t last_col_idx) {
  if (auto rex_op = dynamic_cast<const RexOperator*>(condition)) {
    switch (rex_op->getOperator()) {
      case kAND: {
        std::vector<const RexScalar*> subconditions;
        size_t complete_subcond_count = 0;
        for (size_t i = 0; i < rex_op->size(); ++i) {
          auto conds = find_hoistable_conditions(
              rex_op->getOperand(i), source, first_col_idx, last_col_idx);
          if (conds.size() == size_t(1)) {
            ++complete_subcond_count;
          }
          subconditions.insert(subconditions.end(), conds.begin(), conds.end());
        }
        if (complete_subcond_count == rex_op->size()) {
          return {rex_op};
        } else {
          return {subconditions};
        }
        break;
      }
      case kEQ: {
        const auto lhs_conds = find_hoistable_conditions(
            rex_op->getOperand(0), source, first_col_idx, last_col_idx);
        const auto rhs_conds = find_hoistable_conditions(
            rex_op->getOperand(1), source, first_col_idx, last_col_idx);
        const auto lhs_in = lhs_conds.size() == 1
                                ? dynamic_cast<const RexInput*>(*lhs_conds.begin())
                                : nullptr;
        const auto rhs_in = rhs_conds.size() == 1
                                ? dynamic_cast<const RexInput*>(*rhs_conds.begin())
                                : nullptr;
        if (lhs_in && rhs_in) {
          return {rex_op};
        }
        return {};
        break;
      }
      default:
        break;
    }
    return {};
  }
  if (auto rex_in = dynamic_cast<const RexInput*>(condition)) {
    if (rex_in->getSourceNode() == source) {
      const auto col_idx = rex_in->getIndex();
      return {col_idx >= first_col_idx && col_idx <= last_col_idx ? condition : nullptr};
    }
    return {};
  }
  return {};
}

class JoinTargetRebaser : public RexDeepCopyVisitor {
 public:
  JoinTargetRebaser(const RelJoin* join, const unsigned old_base)
      : join_(join)
      , old_base_(old_base)
      , src1_base_(join->getInput(0)->size())
      , target_count_(join->size()) {}
  RetType visitInput(const RexInput* input) const override {
    auto curr_idx = input->getIndex();
    CHECK_GE(curr_idx, old_base_);
    CHECK_LT(static_cast<size_t>(curr_idx), target_count_);
    curr_idx -= old_base_;
    if (curr_idx >= src1_base_) {
      return boost::make_unique<RexInput>(join_->getInput(1), curr_idx - src1_base_);
    } else {
      return boost::make_unique<RexInput>(join_->getInput(0), curr_idx);
    }
  }

 private:
  const RelJoin* join_;
  const unsigned old_base_;
  const size_t src1_base_;
  const size_t target_count_;
};

class SubConditionRemover : public RexDeepCopyVisitor {
 public:
  SubConditionRemover(const std::vector<const RexScalar*> sub_conds)
      : sub_conditions_(sub_conds.begin(), sub_conds.end()) {}
  RetType visitOperator(const RexOperator* rex_operator) const override {
    if (sub_conditions_.count(rex_operator)) {
      return boost::make_unique<RexLiteral>(
          true, kBOOLEAN, kBOOLEAN, unsigned(-2147483648), 1, unsigned(-2147483648), 1);
    }
    return RexDeepCopyVisitor::visitOperator(rex_operator);
  }

 private:
  std::unordered_set<const RexScalar*> sub_conditions_;
};

void hoist_filter_cond_to_cross_join(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  std::unordered_set<const RelAlgNode*> visited;
  auto web = build_du_web(nodes);
  for (auto node : nodes) {
    if (visited.count(node.get())) {
      continue;
    }
    visited.insert(node.get());
    auto join = dynamic_cast<RelJoin*>(node.get());
    if (join && join->getJoinType() == JoinType::INNER) {
      // Only allow cross join for now.
      if (auto literal = dynamic_cast<const RexLiteral*>(join->getCondition())) {
        // Assume Calcite always generates an inner join on constant boolean true for
        // cross join.
        if (literal->getType() != kBOOLEAN || !literal->getVal<bool>()) {
          continue;
        }
        size_t first_col_idx = 0;
        const RelFilter* filter = nullptr;
        std::vector<const RelJoin*> join_seq{join};
        for (const RelJoin* curr_join = join; !filter;) {
          auto usrs_it = web.find(curr_join);
          CHECK(usrs_it != web.end());
          if (usrs_it->second.size() != size_t(1)) {
            break;
          }
          auto only_usr = *usrs_it->second.begin();
          if (auto usr_join = dynamic_cast<const RelJoin*>(only_usr)) {
            if (join == usr_join->getInput(1)) {
              const auto src1_offset = usr_join->getInput(0)->size();
              first_col_idx += src1_offset;
            }
            join_seq.push_back(usr_join);
            curr_join = usr_join;
            continue;
          }

          filter = dynamic_cast<const RelFilter*>(only_usr);
          break;
        }
        if (!filter) {
          visited.insert(join_seq.begin(), join_seq.end());
          continue;
        }
        const auto src_join = dynamic_cast<const RelJoin*>(filter->getInput(0));
        CHECK(src_join);
        auto modified_filter = const_cast<RelFilter*>(filter);

        if (src_join == join) {
          std::unique_ptr<const RexScalar> filter_condition(
              modified_filter->getAndReleaseCondition());
          std::unique_ptr<const RexScalar> true_condition =
              boost::make_unique<RexLiteral>(true,
                                             kBOOLEAN,
                                             kBOOLEAN,
                                             unsigned(-2147483648),
                                             1,
                                             unsigned(-2147483648),
                                             1);
          modified_filter->setCondition(true_condition);
          join->setCondition(filter_condition);
          continue;
        }
        const auto src1_base = src_join->getInput(0)->size();
        auto source =
            first_col_idx < src1_base ? src_join->getInput(0) : src_join->getInput(1);
        first_col_idx =
            first_col_idx < src1_base ? first_col_idx : first_col_idx - src1_base;
        auto join_conditions =
            find_hoistable_conditions(filter->getCondition(),
                                      source,
                                      first_col_idx,
                                      first_col_idx + join->size() - 1);
        if (join_conditions.empty()) {
          continue;
        }

        JoinTargetRebaser rebaser(join, first_col_idx);
        if (join_conditions.size() == 1) {
          auto new_join_condition = rebaser.visit(*join_conditions.begin());
          join->setCondition(new_join_condition);
        } else {
          std::vector<std::unique_ptr<const RexScalar>> operands;
          bool notnull = true;
          for (size_t i = 0; i < join_conditions.size(); ++i) {
            operands.emplace_back(rebaser.visit(join_conditions[i]));
            auto old_subcond = dynamic_cast<const RexOperator*>(join_conditions[i]);
            CHECK(old_subcond && old_subcond->getType().get_type() == kBOOLEAN);
            notnull = notnull && old_subcond->getType().get_notnull();
          }
          auto new_join_condition = std::unique_ptr<const RexScalar>(
              new RexOperator(kAND, operands, SQLTypeInfo(kBOOLEAN, notnull)));
          join->setCondition(new_join_condition);
        }

        SubConditionRemover remover(join_conditions);
        auto new_filter_condition = remover.visit(filter->getCondition());
        modified_filter->setCondition(new_filter_condition);
      }
    }
  }
}

namespace {

struct GeoFilterInputUse {
  bool has_filter_input{false};
  bool has_geo_filter_input{false};
};

bool is_geo_rex_operator(const RexOperator* rex_operator) {
  if (rex_operator->getOperator() == kBBOX_INTERSECT) {
    return true;
  }
  if (const auto func = dynamic_cast<const RexFunctionOperator*>(rex_operator)) {
    const auto& name = func->getName();
    return name.rfind("ST_", 0) == 0;
  }
  return false;
}

class GeoFilterInputUseVisitor : public RexVisitor<GeoFilterInputUse> {
 public:
  explicit GeoFilterInputUseVisitor(const RelFilter* filter) : filter_(filter) {}

  GeoFilterInputUse visitInput(const RexInput* input) const override {
    return {input->getSourceNode() == filter_, false};
  }

  GeoFilterInputUse visitOperator(const RexOperator* rex_operator) const override {
    auto result = RexVisitor<GeoFilterInputUse>::visitOperator(rex_operator);
    if (is_geo_rex_operator(rex_operator) && result.has_filter_input) {
      result.has_geo_filter_input = true;
    }
    return result;
  }

 protected:
  GeoFilterInputUse aggregateResult(const GeoFilterInputUse& aggregate,
                                    const GeoFilterInputUse& next_result) const override {
    return {aggregate.has_filter_input || next_result.has_filter_input,
            aggregate.has_geo_filter_input || next_result.has_geo_filter_input};
  }

 private:
  const RelFilter* filter_;
};

}  // namespace

void inline_geo_join_input_filters(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  bool changed = false;
  for (auto& node : nodes) {
    auto join = std::dynamic_pointer_cast<RelJoin>(node);
    if (!join || join->getJoinType() != JoinType::INNER || !join->getCondition()) {
      continue;
    }

    for (size_t input_idx = 0; input_idx < join->inputCount(); ++input_idx) {
      auto filter =
          std::dynamic_pointer_cast<const RelFilter>(join->getAndOwnInput(input_idx));
      if (!filter || !filter->getCondition()) {
        continue;
      }

      GeoFilterInputUseVisitor visitor(filter.get());
      if (!visitor.visit(join->getCondition()).has_geo_filter_input) {
        continue;
      }

      const auto filter_source = filter->getAndOwnInput(0);
      join->replaceInput(filter, filter_source);

      std::vector<std::unique_ptr<const RexScalar>> operands;
      operands.emplace_back(join->getAndReleaseCondition());
      RexDeepCopyVisitor copier;
      operands.emplace_back(copier.visit(filter->getCondition()));
      auto new_condition = std::unique_ptr<const RexScalar>(
          new RexOperator(kAND, operands, SQLTypeInfo(kBOOLEAN, false)));
      join->setCondition(new_condition);

      changed = true;
    }
  }

  if (changed) {
    CHECK(!nodes.empty());
    auto sink = nodes.back();
    cleanup_dead_nodes(nodes);
    CHECK(sink);
  }
}

void sync_field_names_if_necessary(std::shared_ptr<const RelProject> from_node,
                                   RelAlgNode* to_node) noexcept {
  auto from_fields = from_node->getFields();
  if (!from_fields.empty()) {
    if (auto proj_to = dynamic_cast<RelProject*>(to_node);
        proj_to && proj_to->getFields().size() == from_fields.size()) {
      proj_to->setFields(std::move(from_fields));
    } else if (auto agg_to = dynamic_cast<RelAggregate*>(to_node);
               agg_to && agg_to->getFields().size() == from_fields.size()) {
      agg_to->setFields(std::move(from_fields));
    } else if (auto compound_to = dynamic_cast<RelCompound*>(to_node);
               compound_to && compound_to->getFields().size() == from_fields.size()) {
      compound_to->setFields(std::move(from_fields));
    } else if (auto tf_to = dynamic_cast<RelTableFunction*>(to_node);
               tf_to && tf_to->getFields().size() == from_fields.size()) {
      tf_to->setFields(std::move(from_fields));
    }
  }
}

// For some reason, Calcite generates Sort, Project, Sort sequences where the
// two Sort nodes are identical and the Project is identity. Simplify this
// pattern by re-binding the input of the second sort to the input of the first.
void simplify_sort(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  if (nodes.size() < 3) {
    return;
  }
  for (size_t i = 0; i <= nodes.size() - 3;) {
    auto first_sort = std::dynamic_pointer_cast<RelSort>(nodes[i]);
    const auto project = std::dynamic_pointer_cast<const RelProject>(nodes[i + 1]);
    auto second_sort = std::dynamic_pointer_cast<RelSort>(nodes[i + 2]);
    if (first_sort && second_sort && project && project->isIdentity() &&
        *first_sort == *second_sort) {
      sync_field_names_if_necessary(project, /* an input of the second sort */
                                    const_cast<RelAlgNode*>(first_sort->getInput(0)));
      second_sort->replaceInput(second_sort->getAndOwnInput(0),
                                first_sort->getAndOwnInput(0));
      nodes[i].reset();
      nodes[i + 1].reset();
      i += 3;
    } else {
      ++i;
    }
  }

  std::vector<std::shared_ptr<RelAlgNode>> new_nodes;
  for (auto node : nodes) {
    if (!node) {
      continue;
    }
    new_nodes.push_back(node);
  }
  nodes.swap(new_nodes);
}
