/*
 * Copyright 2023 HEAVY.AI, Inc.
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

// RelAlgDagViewer: Converts a RelRexDag tree or container into a human-readable plan
// string. Use by calling accept() on the RelAlgDagViewer from the root node of the tree,
// then outputting the RelAlgDagViewer using operator<<. Unless you have a
// std::vector<std::shared_ptr<RelAlgNode>> from the Query Engine, in which case you'll
// get better results by calling handleQueryEngineVector() on that vector instead of
// accept.

#pragma once

#include <functional>
#include <optional>
#include <sstream>

class RelAlgDagViewer : public RelAlgDagNode::Visitor {
 public:
  RelAlgDagViewer() {}
  RelAlgDagViewer(bool showing_steps, bool verbose)
      : showing_steps_(showing_steps), verbose_(verbose) {}

  virtual void clear() { *this = RelAlgDagViewer(showing_steps_, verbose_); }

  virtual std::string str() const {
    std::string ret{out_.str()};
    if (!ret.empty()) {
      ret += '\n';
    }
    return ret;
  }

  typedef struct {
    size_t id;
    bool top_node;   // A top node is directly emplaced or directly visited by the user.
    bool used_node;  // A used node is a child of any other node (or is the root node).
  } NodeInfo;
  using id_map = std::unordered_map<std::uintptr_t, NodeInfo>;

  virtual id_map::iterator emplace(RelAlgDagNode const* n) {
    // Ensures we always emplace RelAlgDagNode* to give a consistent uintptr_t key.
    auto it = emplaceVoid(n);
    n->setIdInPlanTree(it->second.id);
    return it;
  }

  // Helper function needed because the Query Engine uses this vector of nodes to store
  // only some of the most important plan nodes from what is conceptually a DAG tree with
  // a single root node. The root node is always the last node in the vector.
  virtual inline void handleQueryEngineVector(
      std::vector<std::shared_ptr<RelAlgNode>> const nodes);

  //////////

 protected:
  virtual id_map::iterator emplaceVoid(void const* n) {
    CHECK(n);
    auto [it, id_is_new] = ids_.emplace(
        std::uintptr_t(n), NodeInfo{next_id_, top_call_, /*used_node=*/false});
    if (id_is_new) {
      ++next_id_;
    }
    return it;
  }

  void beginNextLine(std::optional<size_t> step = {}) {
    if (!already_indented_) {
      if (needs_colon_) {
        out_ << ":";
      }
      if (needs_newline_) {
        out_ << "\n";
      } else {
        needs_newline_ = true;
      }
      if (showing_steps_) {
        if (!step || *step == 0) {
          out_ << "           ";
        } else {
          out_ << "STEP " << std::to_string(*step) << ":    ";
        }
      }
      out_ << std::string(indent_level_ * indent_spaces_, ' ');
    } else {
      already_indented_ = false;
    }
    ++line_number_;
  }

  //////////

  struct BreadthFirstSearch : public RelAlgDagNode::Visitor {
    BreadthFirstSearch(std::function<void(RelAlgDagNode const*)> callback)
        : callback(callback) {}

    virtual void search(RelAlgDagNode const* root) {
      nodes.clear();
      root->accept(*this, {});
      while (!nodes.empty()) {
        auto temp = std::move(nodes);
        for (auto& node : temp) {
          if (node) {
            node->acceptChildren(*this);
          }
        }
      }
    }

    std::vector<RelAlgDagNode const*> nodes;

   protected:
    virtual bool visitAny(RelAlgDagNode const* n, std::string s) override {
      nodes.push_back(n);
      callback(n);
      return /*recurse=*/false;
    }
    std::function<void(RelAlgDagNode const*)> callback;
  };  // struct BreadthFirstSearch

  struct CollectImmediateChildren : public RelAlgDagNode::Visitor {
    std::vector<std::pair<std::string, RelAlgDagNode const*>> nodes;

   protected:
    virtual bool visitAny(RelAlgDagNode const* n, std::string s) override {
      nodes.emplace_back(s, n);
      return /*recurse=*/false;
    }
  };  // struct CollectImmediateChildren

  //////////

  template <typename T>
  void visitChild(T child, std::string name) {
    static_assert(std::is_base_of_v<RelAlgDagNode, std::remove_pointer_t<T>>);
    auto it = ids_.find(std::uintptr_t(static_cast<RelAlgDagNode const*>(child)));
    CHECK(it != ids_.end());
    it->second.used_node = true;  // Node is a child node.
    already_indented_ = true;
    needs_colon_ = true;
    child->accept(*this, name);
    needs_colon_ = false;
  }

  template <typename T>
  bool innerVisit(T t, std::string name, bool recurse = true) {
    static_assert(std::is_base_of_v<RelAlgDagNode, std::remove_pointer_t<T>>);

    // Top nodes are directly emplaced by or directly visited by the user and must start
    // with all the default settings, except that we want to preserve any node ID's that
    // were assigned during the user's previous calls.
    bool const top_call{top_call_};
    if (top_call_) {
      auto ids = std::move(ids_);
      auto next_id = std::move(next_id_);
      auto line_number = std::move(line_number_);
      clear();
      ids_ = std::move(ids);
      next_id_ = std::move(next_id);
      line_number_ = line_number;
    }

    if (!t) {
      return /*recurse=*/false;
    }
    id_map::iterator it = emplace(t);  // Needs the original top_call_ setting.
    CHECK(it != ids_.end());

    // Output some info about this node.
    beginNextLine(t->getStepNumber());
    std::string key_text{!name.empty() ? ("[" + name + "] ") : ""};
    out_ << "#" << it->second.id << " " << key_text;

    RelRexToStringConfig cfg;
    cfg.skip_input_nodes = true;
    cfg.attributes_only = true;
    out_ << t->toString(cfg);

    // Recurse to child nodes.
    // Non-top child nodes always will be recursed.
    // Top child nodes normally won't be recursed because they'll be visited with their
    // own direct call from the user (or from operator<< for the user). If we're here from
    // a direct user call (top_call is true), the node will be recursed.
    if (recurse) {
      top_call_ = false;  // Must turn off top_call_ before recursing to any children.
      ++indent_level_;
      CollectImmediateChildren visitor;
      t->acceptChildren(visitor);
      for (auto [key, child] : visitor.nodes) {
        if (child) {
          if (!verbose_) {
            // Only show RelScan or RexInput nodes when verbose_ is true.
            if (dynamic_cast<RelScan const*>(child) ||
                dynamic_cast<RexInput const*>(child)) {
              continue;
            }
          }
          beginNextLine(child->getStepNumber());
          visitChild(child, key);
        }
      }
      --indent_level_;
    }

    // Resets top_call_ to clear the out_ buffer if user directly calls visit again.
    top_call_ = top_call;

    return /*recurse=*/false;
  }

  //////////

  bool visit(RelAggregate const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RelCompound const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RelFilter const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RelJoin const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RelLeftDeepInnerJoin const* n, std::string s) override {
    return innerVisit(n, s);
  }

  bool visit(RelLogicalUnion const* n, std::string s) override {
    return innerVisit(n, s);
  }

  bool visit(RelLogicalValues const* n, std::string s) override {
    return innerVisit(n, s);
  }

  bool visit(RelModify const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RelProject const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RelScan const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RelSort const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RelTableFunction const* n, std::string s) override {
    return innerVisit(n, s);
  }

  bool visit(RelTranslatedJoin const* n, std::string s) override {
    return innerVisit(n, s);
  }

  //////////

  bool visit(RexAbstractInput const* n, std::string s) override {
    return innerVisit(n, s);
  }

  bool visit(RexCase const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RexFunctionOperator const* n, std::string s) override {
    return innerVisit(n, s);
  }

  bool visit(RexInput const* n, std::string s) override {
    // As a special case, don't print out the addtional lines for the source child under a
    // RexInput node. The information is already shown inline elsewhere in the plan.
    auto ret = innerVisit(n, s, /*recurse=*/false);
    auto it =
        ids_.find(std::uintptr_t(static_cast<RelAlgDagNode const*>(n->getSourceNode())));
    if (it != ids_.end()) {
      out_ << ": source #" << it->second.id;
    }
    return ret;
  }

  bool visit(RexLiteral const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RexOperator const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RexRef const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RexAgg const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RexSubQuery const* n, std::string s) override { return innerVisit(n, s); }

  bool visit(RexWindowFunctionOperator const* n, std::string s) override {
    return innerVisit(n, s);
  }

  //////////

  std::ostringstream out_;
  size_t indent_level_{0};
  size_t line_number_{1};
  bool showing_steps_{false};
  id_map ids_;
  size_t next_id_{1};
  bool top_call_{true};
  bool already_indented_{false};
  bool needs_newline_{false};
  bool needs_colon_{false};
  bool verbose_{false};

  static constexpr size_t indent_spaces_{4};
};  // class RelAlgDagViewer

inline void RelAlgDagViewer::handleQueryEngineVector(
    std::vector<std::shared_ptr<RelAlgNode>> const nodes) {
  if (nodes.empty()) {
    return;
  }

  // Determines if a steps column needs to be displayed.
  // TODO(sy): Do this recursively from the root node?
  for (auto const& node : nodes) {
    if (node && node->getStepNumber() > 1) {
      showing_steps_ = true;
      break;
    }
  }

  // Ensure that the root node is always node #1.
  emplace(nodes.back().get());  // The last node in the vector is the root node.

  // Breadth-first search starting at the root node to assign node ID's to all used nodes,
  // in order of importance. BFS is done instead of recursing which would give a
  // depth-first search and a less helpful ordering of node ID's.
  top_call_ = false;
  BreadthFirstSearch bfs([this](RelAlgDagNode const* n) {
    auto it = emplace(n);
    CHECK(it != ids_.end());
    it->second.used_node = true;
  });
  bfs.search(nodes.back().get());
  top_call_ = true;

  // Visit all nodes, building the text output.
  nodes.back()->accept(*this, {});
}

inline std::ostream& operator<<(std::ostream& os, RelAlgDagViewer const& dagv) {
  os << dagv.str();
  return os;
}
