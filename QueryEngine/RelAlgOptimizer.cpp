#include "RelAlgOptimizer.h"
#include "RexVisitor.h"

#include <glog/logging.h>

#include <numeric>
#include <string>
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

std::unordered_set<const RelProject*> get_visible_projects(const RelAlgNode* root) {
  if (auto project = dynamic_cast<const RelProject*>(root)) {
    return {project};
  }

  if (dynamic_cast<const RelAggregate*>(root) || dynamic_cast<const RelScan*>(root)) {
    return std::unordered_set<const RelProject*>{};
  }

  if (auto join = dynamic_cast<const RelJoin*>(root)) {
    auto lhs_projs = get_visible_projects(join->getInput(0));
    auto rhs_projs = get_visible_projects(join->getInput(1));
    lhs_projs.insert(rhs_projs.begin(), rhs_projs.end());
    return lhs_projs;
  }

  CHECK(dynamic_cast<const RelFilter*>(root) || dynamic_cast<const RelSort*>(root));
  return get_visible_projects(root->getInput(0));
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
  auto visible_projs = get_visible_projects(nodes.back().get());
  for (auto node : nodes) {
    auto project = std::dynamic_pointer_cast<RelProject>(node);
    if (project && project->isSimple() && (!visible_projs.count(project.get()) || !project->isRenaming())) {
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
    const std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>> web;
  std::unordered_set<const RelAlgNode*> visited;
  std::vector<const RelAlgNode*> work_set;
  for (auto node : nodes) {
    if (std::dynamic_pointer_cast<RelScan>(node) || visited.count(node.get())) {
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

size_t pick_always_live_col_idx(const RelAlgNode* node) {
  CHECK(node->size());
  static RexInputCollector collector;
  if (auto filter = dynamic_cast<const RelFilter*>(node)) {
    auto rex_ins = collector.visit(filter->getCondition());
    if (!rex_ins.empty()) {
      return static_cast<size_t>(rex_ins.begin()->getIndex());
    }
    return pick_always_live_col_idx(filter->getInput(0));
  } else if (auto join = dynamic_cast<const RelJoin*>(node)) {
    auto rex_ins = collector.visit(join->getCondition());
    if (!rex_ins.empty()) {
      return static_cast<size_t>(rex_ins.begin()->getIndex());
    }
    if (auto lhs_idx = pick_always_live_col_idx(join->getInput(0))) {
      return lhs_idx;
    }
    if (auto rhs_idx = pick_always_live_col_idx(join->getInput(0))) {
      return rhs_idx + join->getInput(0)->size();
    }
  } else if (auto sort = dynamic_cast<const RelSort*>(node)) {
    if (sort->collationCount()) {
      return sort->getCollation(0).getField();
    }
    return pick_always_live_col_idx(sort->getInput(0));
  }
  return size_t(0);
}

std::vector<std::unordered_set<size_t>> get_live_ins(
    const RelAlgNode* node,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& live_outs) {
  if (!node || dynamic_cast<const RelScan*>(node)) {
    return {};
  }
  static RexInputCollector collector;
  auto it = live_outs.find(node);
  CHECK(it != live_outs.end());
  auto live_out = it->second;
  if (auto project = dynamic_cast<const RelProject*>(node)) {
    CHECK_EQ(size_t(1), project->inputCount());
    std::unordered_set<size_t> live_in;
    for (const auto& idx : live_out) {
      CHECK_LT(idx, project->size());
      auto partial_in = collector.visit(project->getProjectAt(idx));
      for (auto rex_in : partial_in) {
        live_in.insert(rex_in.getIndex());
      }
    }
    if (project->size() == 1 && dynamic_cast<const RexLiteral*>(project->getProjectAt(0))) {
      CHECK(live_in.empty());
      live_in.insert(pick_always_live_col_idx(project->getInput(0)));
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
    std::unordered_set<size_t> live_in(live_out.begin(), live_out.end());
    auto rex_ins = collector.visit(filter->getCondition());
    for (const auto& rex_in : rex_ins) {
      live_in.insert(static_cast<size_t>(rex_in.getIndex()));
    }
    return {live_in};
  }
  return {};
}

bool any_dead_col_in(const RelAlgNode* node, const std::unordered_set<size_t>& live_outs) {
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
  AvailabilityChecker(const std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>& liveouts,
                      const std::unordered_set<const RelAlgNode*>& intact_nodes)
      : liveouts_(liveouts), intact_nodes_(intact_nodes) {}

  bool hasAllSrcReady(const RelAlgNode* node) const {
    for (size_t i = 0; i < node->inputCount(); ++i) {
      auto src = node->getInput(i);
      if (!dynamic_cast<const RelScan*>(src) && liveouts_.find(src) == liveouts_.end() && !intact_nodes_.count(src)) {
        return false;
      }
    }
    return true;
  }

 private:
  const std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>& liveouts_;
  const std::unordered_set<const RelAlgNode*>& intact_nodes_;
};

void add_new_indices_for(const RelAlgNode* node,
                         std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>& new_liveouts,
                         const std::unordered_set<size_t>& old_liveouts,
                         const std::unordered_set<const RelAlgNode*>& intact_nodes) {
  auto live_fields = old_liveouts;
  if (auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    for (size_t i = 0; i < aggregate->getGroupByCount(); ++i) {
      live_fields.insert(i);
    }
  }
  auto it_ok = new_liveouts.insert(std::make_pair(node, std::unordered_map<size_t, size_t>{}));
  CHECK(it_ok.second);
  auto& new_indices = it_ok.first->second;
  if (node->size() == live_fields.size()) {
    for (size_t i = 0, e = node->size(); i < e; ++i) {
      new_indices.insert(std::make_pair(i, i));
    }
    return;
  }
  if (does_redef_cols(node)) {
    CHECK_GT(node->size(), live_fields.size());
    LOG(INFO) << node->toString() << " eliminated " << node->size() - live_fields.size() << " columns.";
    std::vector<size_t> ordered_indices(live_fields.begin(), live_fields.end());
    std::sort(ordered_indices.begin(), ordered_indices.end());
    for (size_t i = 0; i < ordered_indices.size(); ++i) {
      new_indices.insert(std::make_pair(ordered_indices[i], i));
    }
    return;
  }
  std::vector<size_t> ordered_indices;
  for (size_t i = 0, old_base = 0, new_base = 0; i < node->inputCount(); ++i) {
    auto src = node->getInput(i);
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
    old_base += src->size();
  }
}

class RexInputRenumberVisitor : public RexDeepCopyVisitor {
 public:
  RexInputRenumberVisitor(
      const std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>& new_numbering)
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
  const std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>& node_to_input_renum_;
};

std::vector<std::unique_ptr<const RexAgg>> renumber_rex_aggs(std::vector<std::unique_ptr<const RexAgg>>& agg_exprs,
                                                             const std::unordered_map<size_t, size_t>& new_numbering) {
  std::vector<std::unique_ptr<const RexAgg>> new_exprs;
  for (auto& expr : agg_exprs) {
    auto old_idx = expr->getOperand();
    auto idx_it = new_numbering.find(old_idx);
    if (idx_it != new_numbering.end()) {
      new_exprs.push_back(
          boost::make_unique<RexAgg>(expr->getKind(), expr->isDistinct(), expr->getType(), idx_it->second));
    } else {
      new_exprs.push_back(std::move(expr));
    }
  }
  return new_exprs;
}

SortField renumber_sort_field(const SortField& old_field, const std::unordered_map<size_t, size_t>& new_numbering) {
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
    if (dynamic_cast<const RelScan*>(node) || live_outs.count(node)) {
      continue;
    }
    std::vector<size_t> all_live(node->size());
    std::iota(all_live.begin(), all_live.end(), size_t(0));
    live_outs.insert(std::make_pair(node, std::unordered_set<size_t>(all_live.begin(), all_live.end())));
    work_set.push_back(node);
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
  return live_outs;
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

void try_insert_coalesable_proj(std::vector<std::shared_ptr<RelAlgNode>>& nodes,
                                std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& liveouts,
                                std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>& du_web,
                                std::unordered_map<const RelAlgNode*, RelAlgNode*>& deconst_mapping) {
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
    auto only_usr = deconst_mapping[*usrs.begin()];

    std::vector<std::unique_ptr<const RexScalar>> exprs;
    std::vector<std::string> fields;
    for (size_t i = 0; i < filter->size(); ++i) {
      exprs.emplace_back(boost::make_unique<RexAbstractInput>(i));
      fields.push_back("$f" + std::to_string(i));
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
    du_web.insert(std::make_pair(project, std::unordered_set<const RelAlgNode*>{only_usr}));

    new_nodes.push_back(project_owner);
  }
  if (new_nodes.size() > nodes.size()) {
    nodes.swap(new_nodes);
  }
}

std::pair<std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>, std::vector<const RelAlgNode*>>
sweep_dead_columns(const std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& live_outs,
                   const std::vector<std::shared_ptr<RelAlgNode>>& nodes,
                   const std::unordered_set<const RelAlgNode*>& intact_nodes,
                   const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>& du_web) {
  std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>> liveouts_renumbering;
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
    add_new_indices_for(node.get(), liveouts_renumbering, old_live_outs, intact_nodes);
    if (auto aggregate = std::dynamic_pointer_cast<RelAggregate>(node)) {
      auto old_exprs = aggregate->getAggExprsAndRelease();
      std::vector<std::unique_ptr<const RexAgg>> new_exprs;
      auto key_name_it = aggregate->getFields().begin();
      std::vector<std::string> new_fields(key_name_it, key_name_it + aggregate->getGroupByCount());
      for (size_t i = aggregate->getGroupByCount(), j = 0; i < aggregate->getFields().size() && j < old_exprs.size();
           ++i, ++j) {
        if (old_live_outs.count(i)) {
          new_exprs.push_back(std::move(old_exprs[j]));
          new_fields.push_back(aggregate->getFieldName(i));
        }
      }
      aggregate->setAggExprs(new_exprs);
      aggregate->setFields(new_fields);
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
      project->setFields(new_fields);
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
    std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>>& liveout_renumbering,
    const std::vector<const RelAlgNode*>& ready_nodes,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<size_t>>& old_liveouts,
    const std::unordered_map<const RelAlgNode*, RelAlgNode*>& deconst_mapping,
    const std::unordered_set<const RelAlgNode*>& intact_nodes,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>& du_web) {
  RexInputRenumberVisitor renumberer(liveout_renumbering);
  AvailabilityChecker checker(liveout_renumbering, intact_nodes);
  std::deque<const RelAlgNode*> work_set(ready_nodes.begin(), ready_nodes.end());
  while (!work_set.empty()) {
    auto walker = work_set.front();
    work_set.pop_front();
    CHECK(!dynamic_cast<const RelScan*>(walker));
    auto node_it = deconst_mapping.find(walker);
    CHECK(node_it != deconst_mapping.end());
    auto node = node_it->second;
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
      auto new_condition = renumberer.visit(join->getCondition());
      join->setCondition(new_condition);
    } else if (auto filter = dynamic_cast<RelFilter*>(node)) {
      auto new_condition = renumberer.visit(filter->getCondition());
      filter->setCondition(new_condition);
    } else if (auto sort = dynamic_cast<RelSort*>(node)) {
      auto src_it = liveout_renumbering.find(node->getInput(0));
      CHECK(src_it != liveout_renumbering.end());
      std::vector<SortField> new_collations;
      for (size_t i = 0; i < sort->collationCount(); ++i) {
        new_collations.push_back(renumber_sort_field(sort->getCollation(i), src_it->second));
      }
      sort->setCollation(std::move(new_collations));
    } else {
      CHECK(false);
    }

    // Ignore empty live_out due to some invalid node
    if (does_redef_cols(node) || intact_nodes.count(node)) {
      continue;
    }
    auto live_pair = old_liveouts.find(node);
    CHECK(live_pair != old_liveouts.end());
    auto live_out = live_pair->second;
    add_new_indices_for(node, liveout_renumbering, live_out, intact_nodes);
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
      LOG(WARNING) << "RA node with no used column: " << node->toString();
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
  // Patch
  std::unordered_map<const RelAlgNode*, RelAlgNode*> deconst_mapping;
  for (auto node : nodes) {
    deconst_mapping.insert(std::make_pair(node.get(), node.get()));
  }
  auto web = build_du_web(nodes);
  try_insert_coalesable_proj(nodes, old_liveouts, web, deconst_mapping);

  // Sweep
  std::unordered_map<const RelAlgNode*, std::unordered_map<size_t, size_t>> liveout_renumbering;
  std::vector<const RelAlgNode*> ready_nodes;
  std::tie(liveout_renumbering, ready_nodes) = sweep_dead_columns(old_liveouts, nodes, intact_nodes, web);
  // Propagate
  propagate_input_renumbering(liveout_renumbering, ready_nodes, old_liveouts, deconst_mapping, intact_nodes, web);
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
