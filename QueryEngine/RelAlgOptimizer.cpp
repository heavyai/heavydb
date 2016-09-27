#include "RelAlgOptimizer.h"
#include "RexVisitor.h"

#include <glog/logging.h>

#include <numeric>
#include <unordered_set>
#include <unordered_map>

namespace {

class RexRedirectInputsVisitor : public RexDeepCopyVisitor {
 public:
  RexRedirectInputsVisitor(const std::unordered_set<const RelProject*>& crt_inputs) : crt_projects_(crt_inputs) {}

  RetType visitInput(const RexInput* input) const {
    auto source = dynamic_cast<const RelProject*>(input->getSourceNode());
    if (!source || !crt_projects_.count(source)) {
      return input->deepCopy();
    }
    auto new_source = source->getInput(0);
    auto new_input = dynamic_cast<const RexInput*>(source->getProjectAt(input->getIndex()));
    if (!new_input) {
      return input->deepCopy();
    }
    return boost::make_unique<RexInput>(new_source, new_input->getIndex());
  }

 private:
  const std::unordered_set<const RelProject*>& crt_projects_;
};

void redirect_inputs_of(std::shared_ptr<RelAlgNode> node, const std::unordered_set<const RelProject*>& projects) {
  RexRedirectInputsVisitor visitor(projects);
  std::shared_ptr<const RelProject> src_project = nullptr;
  for (size_t i = 0; i < node->inputCount(); ++i) {
    if (auto project = std::dynamic_pointer_cast<const RelProject>(node->getAndOwnInput(i))) {
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
    if (src_project->size() != src_project->getInput(0)->size()) {
      return;
    }
    for (size_t i = 0; i < src_project->size(); ++i) {
      auto target = dynamic_cast<const RexInput*>(src_project->getProjectAt(i));
      CHECK(target);
      if (i != target->getIndex()) {
        return;
      }
    }
    join->replaceInput(src_project, src_project->getAndOwnInput(0));
    auto other_project = src_project == node->getAndOwnInput(0)
                             ? std::dynamic_pointer_cast<const RelProject>(node->getAndOwnInput(1))
                             : std::dynamic_pointer_cast<const RelProject>(node->getAndOwnInput(0));
    if (other_project && projects.count(other_project.get())) {
      if (other_project->size() != other_project->getInput(0)->size()) {
        return;
      }
      for (size_t i = 0; i < other_project->size(); ++i) {
        auto target = dynamic_cast<const RexInput*>(other_project->getProjectAt(i));
        CHECK(target);
        if (i != target->getIndex()) {
          return;
        }
      }
      join->replaceInput(other_project, other_project->getAndOwnInput(0));
    }
    auto new_condition = visitor.visit(join->getCondition());
    join->setCondition(new_condition);
    return;
  }
  if (auto project = std::dynamic_pointer_cast<RelProject>(node)) {
    std::vector<std::unique_ptr<const RexScalar>> new_exprs;
    for (size_t i = 0; i < project->size(); ++i) {
      new_exprs.push_back(visitor.visit(project->getProjectAt(i)));
    }
    project->setExpressions(new_exprs);
    project->replaceInput(src_project, src_project->getAndOwnInput(0));
    return;
  }
  if (auto filter = std::dynamic_pointer_cast<RelFilter>(node)) {
    if (src_project->size() != src_project->getInput(0)->size()) {
      return;
    }
    for (size_t i = 0; i < src_project->size(); ++i) {
      auto target = dynamic_cast<const RexInput*>(src_project->getProjectAt(i));
      CHECK(target);
      if (i != target->getIndex()) {
        return;
      }
    }
    auto new_condition = visitor.visit(filter->getCondition());
    filter->setCondition(new_condition);
    filter->replaceInput(src_project, src_project->getAndOwnInput(0));
    return;
  }
  if (std::dynamic_pointer_cast<RelSort>(node) && dynamic_cast<const RelScan*>(src_project->getInput(0))) {
    return;
  }
  CHECK(std::dynamic_pointer_cast<RelAggregate>(node) || std::dynamic_pointer_cast<RelSort>(node));
  if (src_project->size() != src_project->getInput(0)->size()) {
    return;
  }
  for (size_t i = 0; i < src_project->size(); ++i) {
    auto target = dynamic_cast<const RexInput*>(src_project->getProjectAt(i));
    CHECK(target);
    if (i != target->getIndex()) {
      return;
    }
  }
  node->replaceInput(src_project, src_project->getAndOwnInput(0));
}

void cleanup_dead_nodes(std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  for (auto nodeIt = nodes.rbegin(); nodeIt != nodes.rend(); ++nodeIt) {
    if (nodeIt->unique()) {
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

}  // namespace

// For now, the only target to eliminate is restricted to project-aggregate pair between scan/sort and join
// TODO(miyu): allow more chance if proved safe
void eliminate_identical_copy(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  std::unordered_set<std::shared_ptr<const RelAlgNode>> copies;
  auto sink = nodes.back();
  for (auto node : nodes) {
    auto aggregate = std::dynamic_pointer_cast<const RelAggregate>(node);
    if (!aggregate || aggregate == sink || !(aggregate->getGroupByCount() == 1 && aggregate->getAggExprsCount() == 0)) {
      continue;
    }
    auto project = std::dynamic_pointer_cast<const RelProject>(aggregate->getAndOwnInput(0));
    if (project && project->size() == aggregate->size() && project->getFields() == aggregate->getFields()) {
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
    auto project = std::dynamic_pointer_cast<const RelProject>(aggregate->getAndOwnInput(0));
    CHECK(project);
    auto new_source = project->getAndOwnInput(0);
    if (std::dynamic_pointer_cast<const RelSort>(new_source) || std::dynamic_pointer_cast<const RelScan>(new_source)) {
      node->replaceInput(last_source, new_source);
    }
  }
  decltype(copies)().swap(copies);

  std::unordered_set<const RelProject*> projects;
  for (auto node : nodes) {
    auto project = std::dynamic_pointer_cast<RelProject>(node);
    if (project && project->isSimple()) {
      projects.insert(project.get());
    }
  }

  for (auto node : nodes) {
    redirect_inputs_of(node, projects);
  }

  cleanup_dead_nodes(nodes);
}

namespace {

class RexInputCollector : public RexVisitor<std::unordered_set<RexInput>> {
 protected:
  typedef std::unordered_set<RexInput> RetType;
  RetType aggregateResult(const RetType& aggregate, const RetType& next_result) const override {
    RetType result(aggregate.begin(), aggregate.end());
    result.insert(next_result.begin(), next_result.end());
    return result;
  }

 public:
  RetType visitInput(const RexInput* input) const override {
    RetType result;
    result.insert(*input);
    return result;
  }
};

std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>> build_du_web(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>> web;
  std::unordered_set<const RelAlgNode*> visited;
  std::vector<const RelAlgNode*> work_set;
  for (auto node : nodes) {
    if (visited.count(node.get())) {
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
      auto it_ok = web.insert(std::make_pair(walker, std::unordered_set<const RelAlgNode*>{}));
      CHECK(it_ok.second);
      visited.insert(walker);
      const auto join = dynamic_cast<const RelJoin*>(walker);
      const auto project = dynamic_cast<const RelProject*>(walker);
      const auto aggregate = dynamic_cast<const RelAggregate*>(walker);
      const auto filter = dynamic_cast<const RelFilter*>(walker);
      const auto sort = dynamic_cast<const RelSort*>(walker);
      CHECK(join || project || aggregate || filter || sort);
      for (size_t i = 0; i < walker->inputCount(); ++i) {
        auto src = walker->getInput(i);
        if (dynamic_cast<const RelScan*>(src)) {
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

std::vector<std::unordered_set<size_t>> get_live_ins(
    const RelAlgNode* node,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& live_outs) {
  if (!node || dynamic_cast<const RelScan*>(node)) {
    return {};
  }
  RexInputCollector collector;
  auto it = live_outs.find(node);
  CHECK(it != live_outs.end());
  auto live_out = it->second;
  if (auto project = dynamic_cast<const RelProject*>(node)) {
    CHECK_EQ(size_t(1), project->inputCount());
    auto src = project->getInput(0);
    std::unordered_set<size_t> live_in;
    for (const auto& idx : live_out) {
      CHECK_LT(idx, project->size());
      auto partial_in = collector.visit(project->getProjectAt(idx));
      for (auto rex_in : partial_in) {
        CHECK_EQ(src, rex_in.getSourceNode());
        live_in.insert(rex_in.getIndex());
      }
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
    bool has_count_start_only{false};
    for (const auto& idx : live_out) {
      if (idx < group_key_count) {
        continue;
      }
      const auto agg_idx = idx - group_key_count;
      CHECK_LT(agg_idx, agg_expr_count);
      const auto arg_idx = aggregate->getAggExprs()[agg_idx]->getOperand();
      if (arg_idx >= 0) {
        live_in.insert(static_cast<size_t>(arg_idx));
      } else if (agg_expr_count == 1) {
        has_count_start_only = true;
      }
    }
    if (has_count_start_only && !group_key_count) {
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
    auto src = filter->getInput(0);
    std::unordered_set<size_t> live_in(live_out.begin(), live_out.end());
    auto rex_ins = collector.visit(filter->getCondition());
    for (const auto& rex_in : rex_ins) {
      CHECK_EQ(src, rex_in.getSourceNode());
      live_in.insert(static_cast<size_t>(rex_in.getIndex()));
    }
    return {live_in};
  }
  return {};
}

bool does_redef_cols(const RelAlgNode* node) {
  return dynamic_cast<const RelAggregate*>(node) || dynamic_cast<const RelProject*>(node);
}

}  // namespace

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
  std::vector<size_t> all_live(root->size());
  std::iota(all_live.begin(), all_live.end(), size_t(0));
  std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>> live_outs{
      {root, std::unordered_set<size_t>(all_live.begin(), all_live.end())}};
  std::vector<const RelAlgNode*> work_set{root};
  for (auto node : nodes) {
    work_set.push_back(node.get());
    while (!work_set.empty()) {
      auto walker = work_set.back();
      work_set.pop_back();
      CHECK(!dynamic_cast<const RelScan*>(walker));
      CHECK(live_outs.count(walker));
      auto live_ins = get_live_ins(walker, live_outs);
      CHECK_EQ(live_ins.size(), walker->inputCount());
      for (size_t i = 0; i < walker->inputCount(); ++i) {
        auto src = walker->getInput(0);
        if (dynamic_cast<const RelScan*>(src) || live_ins[i].empty()) {
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
  std::unordered_set<const RelAlgNode*> dead_nodes;
  bool has_dead_cols = false;
  for (auto live_pair : live_outs) {
    auto node = live_pair.first;
    auto live_out = live_pair.second;
    if (live_out.empty()) {
      dead_nodes.insert(node);
    }
    if (node->size() > live_out.size()) {
      has_dead_cols = true;
    } else {
      CHECK_EQ(node->size(), live_out.size());
    }
  }
  if (!has_dead_cols) {
    return;
  }
  // Sweep
  auto web = build_du_web(nodes);
  for (auto node : dead_nodes) {
    auto node_it = web.find(node);
    CHECK(node_it != web.end());
    LOG(WARNING) << "RA node with no used column: " << node->toString();
    web.erase(node_it);
    for (size_t i = 0; i < node_it->first->inputCount(); ++i) {
      auto src = node_it->first->getInput(i);
      auto src_it = web.find(src);
      if (src_it != web.end() && src_it->second.count(node)) {
        src_it->second.erase(node);
      }
    }
  }
  std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>> new_liveouts;
  std::unordered_map<const RelAlgNode*, size_t> node_to_slot;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto node = nodes[i].get();
    CHECK(node);
    node_to_slot.insert(std::make_pair(node, i));
  }
  for (auto live_pair : live_outs) {
    if (!does_redef_cols(live_pair.first)) {
      continue;
    }
    auto node_it = node_to_slot.find(live_pair.first);
    CHECK(node_it != node_to_slot.end());
    auto node = nodes[node_it->second].get();
    auto live_out = live_pair.second;
    // Ignore empty live_out due to some invalid node
    if (node->size() == live_out.size() || live_out.empty()) {
      continue;
    }
    auto it_ok = new_liveouts.insert(std::make_pair(node, std::unordered_map<size_t, size_t>{}));
    CHECK(it_ok.second);
    auto& new_indices = it_ok.first->second;
    std::vector<size_t> ordered_indices(live_out.begin(), live_out.end());
    std::sort(ordered_indices.begin(), ordered_indices.end());
    for (size_t i = 0; i < ordered_indices.size(); ++i) {
      new_indices.insert(std::make_pair(ordered_indices[i], i));
    }
    if (auto aggregate = dynamic_cast<RelAggregate*>(node)) {
      auto old_exprs = aggregate->getAggExprsAndRelease();
      std::vector<std::unique_ptr<const RexAgg>> new_exprs;
      auto key_name_it = aggregate->getFields().begin();
      std::vector<std::string> new_fields(key_name_it, key_name_it + aggregate->getGroupByCount());
      for (size_t i = aggregate->getGroupByCount(), j = 0; i < aggregate->getFields().size() && j < old_exprs.size();
           ++i, ++j) {
        if (live_out.count(i)) {
          new_exprs.push_back(std::move(old_exprs[j]));
          new_fields.push_back(aggregate->getFieldName(i));
        }
      }
      aggregate->setAggExprs(new_exprs);
      aggregate->setFilds(new_fields);
      continue;
    }
    if (auto project = dynamic_cast<RelProject*>(node)) {
      auto old_exprs = project->getExpressionsAndRelease();
      std::vector<std::unique_ptr<const RexScalar>> new_exprs;
      std::vector<std::string> new_fields;
      for (size_t i = 0; i < old_exprs.size(); ++i) {
        if (live_out.count(i)) {
          new_exprs.push_back(std::move(old_exprs[i]));
          new_fields.push_back(project->getFieldName(i));
        }
      }
      project->setExpressions(new_exprs);
      project->setFilds(new_fields);
      continue;
    }
    CHECK(false);
  }
  // TODO(miyu): propagate
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
    if (first_sort && second_sort && project && project->isIdentity() && *first_sort == *second_sort) {
      second_sort->replaceInput(second_sort->getAndOwnInput(0), first_sort->getAndOwnInput(0));
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
