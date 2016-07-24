#ifdef HAVE_CALCITE
#include "RelAlgAbstractInterpreter.h"
#include "CalciteDeserializerUtils.h"
#include "JsonAccessors.h"
#include "RexVisitor.h"

#include <glog/logging.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <string>
#include <unordered_set>
#include <unordered_map>

unsigned RelAlgNode::crt_id_ = 1;

namespace {

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

 private:
  const RelAlgNode* old_input_;
  const RelAlgNode* new_input_;
};

// Creates an output with n columns.
std::vector<RexInput> n_outputs(const RelAlgNode* node, const size_t n) {
  std::vector<RexInput> outputs;
  for (size_t i = 0; i < n; ++i) {
    outputs.emplace_back(node, i);
  }
  return outputs;
}

typedef std::vector<RexInput> RANodeOutput;

RANodeOutput get_node_output(const RelAlgNode* ra_node) {
  RANodeOutput outputs;
  const auto scan_node = dynamic_cast<const RelScan*>(ra_node);
  if (scan_node) {
    // Scan node has no inputs, output contains all columns in the table.
    CHECK_EQ(size_t(0), scan_node->inputCount());
    return n_outputs(scan_node, scan_node->size());
  }
  const auto project_node = dynamic_cast<const RelProject*>(ra_node);
  if (project_node) {
    // Project output count doesn't depend on the input
    CHECK_EQ(size_t(1), project_node->inputCount());
    return n_outputs(project_node, project_node->size());
  }
  const auto filter_node = dynamic_cast<const RelFilter*>(ra_node);
  if (filter_node) {
    // Filter preserves shape
    CHECK_EQ(size_t(1), filter_node->inputCount());
    const auto prev_out = get_node_output(filter_node->getInput(0));
    return n_outputs(filter_node, prev_out.size());
  }
  const auto aggregate_node = dynamic_cast<const RelAggregate*>(ra_node);
  if (aggregate_node) {
    // Aggregate output count doesn't depend on the input
    CHECK_EQ(size_t(1), aggregate_node->inputCount());
    return n_outputs(aggregate_node, aggregate_node->size());
  }
  const auto compound_node = dynamic_cast<const RelCompound*>(ra_node);
  if (compound_node) {
    // Compound output count doesn't depend on the input
    CHECK_EQ(size_t(1), compound_node->inputCount());
    return n_outputs(compound_node, compound_node->size());
  }
  const auto join_node = dynamic_cast<const RelJoin*>(ra_node);
  if (join_node) {
    // Join concatenates the outputs from the inputs and the output
    // directly references the nodes in the input.
    CHECK_EQ(size_t(2), join_node->inputCount());
    auto lhs_out = get_node_output(join_node->getInput(0));
    const auto rhs_out = get_node_output(join_node->getInput(1));
    lhs_out.insert(lhs_out.end(), rhs_out.begin(), rhs_out.end());
    return lhs_out;
  }
  const auto sort_node = dynamic_cast<const RelSort*>(ra_node);
  if (sort_node) {
    // Sort preserves shape
    CHECK_EQ(size_t(1), sort_node->inputCount());
    const auto prev_out = get_node_output(sort_node->getInput(0));
    return n_outputs(sort_node, prev_out.size());
  }
  CHECK(false);
  return outputs;
}

}  // namespace

void RelProject::replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input.get(), input.get());
  for (const auto& scalar_expr : scalar_exprs_) {
    rebind_inputs.visit(scalar_expr.get());
  }
}

bool RelProject::isIdentity() const {
  if (!isSimple()) {
    return false;
  }
  CHECK_EQ(size_t(1), inputCount());
  const auto source = getInput(0);
  if (dynamic_cast<const RelJoin*>(source)) {
    return false;
  }
  const auto source_shape = get_node_output(source);
  if (source_shape.size() != scalar_exprs_.size()) {
    return false;
  }
  for (size_t i = 0; i < scalar_exprs_.size(); ++i) {
    const auto& scalar_expr = scalar_exprs_[i];
    const auto input = dynamic_cast<const RexInput*>(scalar_expr.get());
    CHECK(input);
    CHECK_EQ(source, input->getSourceNode());
    // We should add the additional check that input->getIndex() != source_shape[i].getIndex(),
    // but Calcite doesn't generate the right Sort-Project-Sort sequence when joins are involved.
    if (input->getSourceNode() != source_shape[i].getSourceNode()) {
      return false;
    }
  }
  return true;
}

void RelJoin::replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input.get(), input.get());
  if (condition_) {
    rebind_inputs.visit(condition_.get());
  }
}

void RelFilter::replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input.get(), input.get());
  rebind_inputs.visit(filter_.get());
}

void RelCompound::replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input.get(), input.get());
  for (const auto& scalar_source : scalar_sources_) {
    rebind_inputs.visit(scalar_source.get());
  }
  if (filter_expr_) {
    rebind_inputs.visit(filter_expr_.get());
  }
}

namespace {

unsigned node_id(const rapidjson::Value& ra_node) noexcept {
  const auto& id = field(ra_node, "id");
  return std::stoi(json_str(id));
}

std::unique_ptr<RexAbstractInput> parse_abstract_input(const rapidjson::Value& expr) noexcept {
  const auto& input = field(expr, "input");
  return std::unique_ptr<RexAbstractInput>(new RexAbstractInput(json_i64(input)));
}

std::unique_ptr<RexLiteral> parse_literal(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  const auto& literal = field(expr, "literal");
  const auto type = to_sql_type(json_str(field(expr, "type")));
  const auto target_type = to_sql_type(json_str(field(expr, "target_type")));
  const auto scale = json_i64(field(expr, "scale"));
  const auto precision = json_i64(field(expr, "precision"));
  const auto type_scale = json_i64(field(expr, "type_scale"));
  const auto type_precision = json_i64(field(expr, "type_precision"));
  switch (type) {
    case kDECIMAL:
      return std::unique_ptr<RexLiteral>(
          new RexLiteral(json_i64(literal), type, target_type, scale, precision, type_scale, type_precision));
    case kDOUBLE:
      return std::unique_ptr<RexLiteral>(
          new RexLiteral(json_double(literal), type, target_type, scale, precision, type_scale, type_precision));
    case kTEXT:
      return std::unique_ptr<RexLiteral>(
          new RexLiteral(json_str(literal), type, target_type, scale, precision, type_scale, type_precision));
    case kBOOLEAN:
      return std::unique_ptr<RexLiteral>(
          new RexLiteral(json_bool(literal), type, target_type, scale, precision, type_scale, type_precision));
    case kNULLT:
      return std::unique_ptr<RexLiteral>(new RexLiteral(target_type));
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      SQLTypeInfo lit_ti(type, false);
      throw std::runtime_error("Unsupported literal type " + lit_ti.get_type_name());
    }
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

std::unique_ptr<const RexScalar> parse_scalar_expr(const rapidjson::Value& expr);

SQLTypeInfo parse_type(const rapidjson::Value& type_obj) {
  CHECK(type_obj.IsObject() && (type_obj.MemberCount() >= 2 && type_obj.MemberCount() <= 4));
  const auto type = to_sql_type(json_str(field(type_obj, "type")));
  const auto nullable = json_bool(field(type_obj, "nullable"));
  const bool has_precision = type_obj.MemberCount() >= 3;
  const bool has_scale = type_obj.MemberCount() == 4;
  const int precision = has_precision ? json_i64(field(type_obj, "precision")) : 0;
  const int scale = has_scale ? json_i64(field(type_obj, "scale")) : 0;
  SQLTypeInfo ti(type, !nullable);
  ti.set_precision(precision);
  ti.set_scale(scale);
  return ti;
}

std::unique_ptr<RexOperator> parse_operator(const rapidjson::Value& expr) {
  const auto op_name = json_str(field(expr, "op"));
  const bool is_quantifier = op_name == std::string("PG_ANY") || op_name == std::string("PG_ALL");
  const auto op = is_quantifier ? kFUNCTION : to_sql_op(op_name);
  const auto& operators_json_arr = field(expr, "operands");
  CHECK(operators_json_arr.IsArray());
  std::vector<std::unique_ptr<const RexScalar>> operands;
  for (auto operators_json_arr_it = operators_json_arr.Begin(); operators_json_arr_it != operators_json_arr.End();
       ++operators_json_arr_it) {
    operands.emplace_back(parse_scalar_expr(*operators_json_arr_it));
  }
  const auto type_it = expr.FindMember("type");
  CHECK(type_it != expr.MemberEnd());
  const auto ti = parse_type(type_it->value);
  return std::unique_ptr<RexOperator>(op == kFUNCTION ? new RexFunctionOperator(op_name, operands, ti)
                                                      : new RexOperator(op, operands, ti));
}

std::unique_ptr<RexCase> parse_case(const rapidjson::Value& expr) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(2));
  std::unique_ptr<const RexScalar> else_expr;
  std::vector<std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>> expr_pair_list;
  for (auto operands_it = operands.Begin(); operands_it != operands.End();) {
    auto when_expr = parse_scalar_expr(*operands_it++);
    if (operands_it == operands.End()) {
      else_expr = std::move(when_expr);
      break;
    }
    auto then_expr = parse_scalar_expr(*operands_it++);
    expr_pair_list.emplace_back(std::move(when_expr), std::move(then_expr));
  }
  return std::unique_ptr<RexCase>(new RexCase(expr_pair_list, else_expr));
}

std::vector<std::string> strings_from_json_array(const rapidjson::Value& json_str_arr) noexcept {
  CHECK(json_str_arr.IsArray());
  std::vector<std::string> fields;
  for (auto json_str_arr_it = json_str_arr.Begin(); json_str_arr_it != json_str_arr.End(); ++json_str_arr_it) {
    CHECK(json_str_arr_it->IsString());
    fields.emplace_back(json_str_arr_it->GetString());
  }
  return fields;
}

std::vector<size_t> indices_from_json_array(const rapidjson::Value& json_idx_arr) noexcept {
  CHECK(json_idx_arr.IsArray());
  std::vector<size_t> indices;
  for (auto json_idx_arr_it = json_idx_arr.Begin(); json_idx_arr_it != json_idx_arr.End(); ++json_idx_arr_it) {
    CHECK(json_idx_arr_it->IsInt());
    CHECK_GE(json_idx_arr_it->GetInt(), 0);
    indices.emplace_back(json_idx_arr_it->GetInt());
  }
  return indices;
}

std::string json_node_to_string(const rapidjson::Value& node) noexcept {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  node.Accept(writer);
  return buffer.GetString();
}

std::unique_ptr<const RexAgg> parse_aggregate_expr(const rapidjson::Value& expr) {
  const auto agg = to_agg_kind(json_str(field(expr, "agg")));
  const auto distinct = json_bool(field(expr, "distinct"));
  const auto agg_ti = parse_type(field(expr, "type"));
  const auto operands = indices_from_json_array(field(expr, "operands"));
  CHECK_LE(operands.size(), size_t(1));
  return std::unique_ptr<const RexAgg>(new RexAgg(agg, distinct, agg_ti, operands.empty() ? -1 : operands[0]));
}

std::unique_ptr<const RexScalar> parse_scalar_expr(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  if (expr.IsObject() && expr.HasMember("input")) {
    return std::unique_ptr<const RexScalar>(parse_abstract_input(expr));
  }
  if (expr.IsObject() && expr.HasMember("literal")) {
    return std::unique_ptr<const RexScalar>(parse_literal(expr));
  }
  if (expr.IsObject() && expr.HasMember("op")) {
    const auto op_str = json_str(field(expr, "op"));
    if (op_str == std::string("CASE")) {
      return std::unique_ptr<const RexScalar>(parse_case(expr));
    }
    return std::unique_ptr<const RexScalar>(parse_operator(expr));
  }
  throw QueryNotSupported("Expression node " + json_node_to_string(expr) + " not supported");
}

JoinType to_join_type(const std::string& join_type_name) {
  if (join_type_name == "inner") {
    return JoinType::INNER;
  }
  if (join_type_name == "left") {
    return JoinType::LEFT;
  }
  throw QueryNotSupported("Join type (" + join_type_name + ") not supported");
}

std::unique_ptr<const RexScalar> disambiguate_rex(const RexScalar*, const RANodeOutput&);

std::unique_ptr<const RexOperator> disambiguate_operator(const RexOperator* rex_operator,
                                                         const RANodeOutput& ra_output) noexcept {
  std::vector<std::unique_ptr<const RexScalar>> disambiguated_operands;
  for (size_t i = 0; i < rex_operator->size(); ++i) {
    disambiguated_operands.emplace_back(disambiguate_rex(rex_operator->getOperand(i), ra_output));
  }
  return rex_operator->getDisambiguated(disambiguated_operands);
}

std::unique_ptr<const RexCase> disambiguate_case(const RexCase* rex_case, const RANodeOutput& ra_output) {
  std::vector<std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>>
      disambiguated_expr_pair_list;
  for (size_t i = 0; i < rex_case->branchCount(); ++i) {
    auto disambiguated_when = disambiguate_rex(rex_case->getWhen(i), ra_output);
    auto disambiguated_then = disambiguate_rex(rex_case->getThen(i), ra_output);
    disambiguated_expr_pair_list.emplace_back(std::move(disambiguated_when), std::move(disambiguated_then));
  }
  std::unique_ptr<const RexScalar> disambiguated_else{disambiguate_rex(rex_case->getElse(), ra_output)};
  return std::unique_ptr<const RexCase>(new RexCase(disambiguated_expr_pair_list, disambiguated_else));
}

std::unique_ptr<const RexScalar> disambiguate_rex(const RexScalar* rex_scalar, const RANodeOutput& ra_output) {
  const auto rex_abstract_input = dynamic_cast<const RexAbstractInput*>(rex_scalar);
  if (rex_abstract_input) {
    CHECK_LT(static_cast<size_t>(rex_abstract_input->getIndex()), ra_output.size());
    return std::unique_ptr<const RexInput>(new RexInput(ra_output[rex_abstract_input->getIndex()]));
  }
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex_scalar);
  if (rex_operator) {
    return disambiguate_operator(rex_operator, ra_output);
  }
  const auto rex_case = dynamic_cast<const RexCase*>(rex_scalar);
  if (rex_case) {
    return disambiguate_case(rex_case, ra_output);
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_scalar);
  CHECK(rex_literal);
  return std::unique_ptr<const RexLiteral>(new RexLiteral(*rex_literal));
}

void bind_project_to_input(RelProject* project_node, const RANodeOutput& input) noexcept {
  CHECK_EQ(size_t(1), project_node->inputCount());
  std::vector<std::unique_ptr<const RexScalar>> disambiguated_exprs;
  for (size_t i = 0; i < project_node->size(); ++i) {
    disambiguated_exprs.emplace_back(disambiguate_rex(project_node->getProjectAt(i), input));
  }
  project_node->setExpressions(disambiguated_exprs);
}

void bind_inputs(const std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  for (auto ra_node : nodes) {
    const auto filter_node = std::dynamic_pointer_cast<RelFilter>(ra_node);
    if (filter_node) {
      CHECK_EQ(size_t(1), filter_node->inputCount());
      auto disambiguated_condition =
          disambiguate_rex(filter_node->getCondition(), get_node_output(filter_node->getInput(0)));
      filter_node->setCondition(disambiguated_condition);
      continue;
    }
    const auto join_node = std::dynamic_pointer_cast<RelJoin>(ra_node);
    if (join_node) {
      CHECK_EQ(size_t(2), join_node->inputCount());
      auto disambiguated_condition = disambiguate_rex(join_node->getCondition(), get_node_output(join_node.get()));
      join_node->setCondition(disambiguated_condition);
      continue;
    }
    const auto project_node = std::dynamic_pointer_cast<RelProject>(ra_node);
    if (project_node) {
      bind_project_to_input(project_node.get(), get_node_output(project_node->getInput(0)));
      continue;
    }
  }
}

void mark_nops(const std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  for (auto node : nodes) {
    const auto agg_node = std::dynamic_pointer_cast<RelAggregate>(node);
    if (!agg_node || agg_node->getAggExprsCount()) {
      continue;
    }
    CHECK_EQ(size_t(1), node->inputCount());
    const auto agg_input_node = dynamic_cast<const RelAggregate*>(node->getInput(0));
    if (agg_input_node && !agg_input_node->getAggExprsCount() &&
        agg_node->getGroupByCount() == agg_input_node->getGroupByCount()) {
      agg_node->markAsNop();
    }
  }
}

std::vector<const Rex*> reproject_targets(const RelProject* simple_project,
                                          const std::vector<const Rex*>& target_exprs) noexcept {
  std::vector<const Rex*> result;
  for (size_t i = 0; i < simple_project->size(); ++i) {
    const auto input_rex = dynamic_cast<const RexInput*>(simple_project->getProjectAt(i));
    CHECK(input_rex);
    CHECK_LT(static_cast<size_t>(input_rex->getIndex()), target_exprs.size());
    result.push_back(target_exprs[input_rex->getIndex()]);
  }
  return result;
}

void create_compound(std::vector<std::shared_ptr<RelAlgNode>>& nodes, const std::vector<size_t>& pattern) noexcept {
  CHECK_GE(pattern.size(), size_t(2));
  CHECK_LE(pattern.size(), size_t(4));

  std::unique_ptr<const RexScalar> filter_rex;
  std::vector<std::unique_ptr<const RexScalar>> scalar_sources;
  size_t groupby_count{0};
  std::vector<std::string> fields;
  std::vector<const RexAgg*> agg_exprs;
  std::vector<const Rex*> target_exprs;
  bool first_project{true};
  bool is_agg{false};
  for (const auto node_idx : pattern) {
    const auto ra_node = nodes[node_idx];
    const auto ra_filter = std::dynamic_pointer_cast<RelFilter>(ra_node);
    if (ra_filter) {
      CHECK(!filter_rex);
      filter_rex.reset(ra_filter->getAndReleaseCondition());
      CHECK(filter_rex);
      continue;
    }
    const auto ra_project = std::dynamic_pointer_cast<RelProject>(ra_node);
    if (ra_project) {
      fields = ra_project->getFields();
      if (first_project) {
        CHECK_EQ(size_t(1), ra_project->inputCount());
        // Rebind the input of the project to the input of the filter itself
        // since we know that we'll evaluate the filter on the fly, with no
        // intermediate buffer.
        const auto filter_input = dynamic_cast<const RelFilter*>(ra_project->getInput(0));
        if (filter_input) {
          CHECK_EQ(size_t(1), filter_input->inputCount());
          bind_project_to_input(ra_project.get(), get_node_output(filter_input->getInput(0)));
        }
        scalar_sources = ra_project->getExpressionsAndRelease();
        for (const auto& scalar_expr : scalar_sources) {
          target_exprs.push_back(scalar_expr.get());
        }
        first_project = false;
      } else {
        CHECK(ra_project->isSimple());
        target_exprs = reproject_targets(ra_project.get(), target_exprs);
      }
      continue;
    }
    const auto ra_aggregate = std::dynamic_pointer_cast<RelAggregate>(ra_node);
    if (ra_aggregate) {
      is_agg = true;
      fields = ra_aggregate->getFields();
      agg_exprs = ra_aggregate->getAggregatesAndRelease();
      groupby_count = ra_aggregate->getGroupByCount();
      decltype(target_exprs){}.swap(target_exprs);
      CHECK_LE(groupby_count, scalar_sources.size());
      for (size_t group_idx = 0; group_idx < groupby_count; ++group_idx) {
        const auto rex_ref = new RexRef(group_idx + 1);
        target_exprs.push_back(rex_ref);
        scalar_sources.emplace_back(rex_ref);
      }
      for (const auto rex_agg : agg_exprs) {
        target_exprs.push_back(rex_agg);
      }
      continue;
    }
  }
  auto compound_node =
      std::make_shared<RelCompound>(filter_rex, target_exprs, groupby_count, agg_exprs, fields, scalar_sources, is_agg);
  auto old_node = nodes[pattern.back()];
  nodes[pattern.back()] = compound_node;
  auto first_node = nodes[pattern.front()];
  CHECK_EQ(size_t(1), first_node->inputCount());
  compound_node->addManagedInput(first_node->getAndOwnInput(0));
  for (size_t i = 0; i < pattern.size() - 1; ++i) {
    nodes[pattern[i]].reset();
  }
  for (auto node : nodes) {
    if (!node) {
      continue;
    }
    node->replaceInput(old_node, compound_node);
  }
}

class RANodeIterator : public std::vector<std::shared_ptr<RelAlgNode>>::const_iterator {
  typedef std::shared_ptr<RelAlgNode> ElementType;
  typedef std::vector<ElementType>::const_iterator Super;
  typedef std::vector<ElementType> Container;

 public:
  enum class AdvancingMode { DUChain, InOrder };

  explicit RANodeIterator(const Container& nodes)
      : Super(nodes.begin()),
        owner_(nodes),
        nodeCount_([&nodes]() -> size_t {
          size_t non_zero_count = 0;
          for (const auto& node : nodes) {
            if (node) {
              ++non_zero_count;
            }
          }
          return non_zero_count;
        }()) {}

  explicit operator size_t() { return std::distance(owner_.begin(), *static_cast<Super*>(this)); }

  RANodeIterator operator++() = delete;

  void advance(AdvancingMode mode) {
    Super& super = *this;
    switch (mode) {
      case AdvancingMode::DUChain: {
        size_t use_count = 0;
        Super only_use = owner_.end();
        for (Super nodeIt = std::next(super); nodeIt != owner_.end(); ++nodeIt) {
          if (!*nodeIt) {
            continue;
          }
          for (size_t i = 0; i < (*nodeIt)->inputCount(); ++i) {
            if ((*super) == (*nodeIt)->getAndOwnInput(i)) {
              ++use_count;
              if (1 == use_count) {
                only_use = nodeIt;
              } else {
                super = owner_.end();
                return;
              }
            }
          }
        }
        super = only_use;
      } break;
      case AdvancingMode::InOrder:
        for (size_t i = 0; i != owner_.size(); ++i) {
          if (!vistied_.count(i)) {
            super = owner_.begin();
            std::advance(super, i);
            return;
          }
        }
        super = owner_.end();
        break;
      default:
        CHECK(false);
    }
  }

  bool allVisited() { return vistied_.size() == nodeCount_; }

  const ElementType& operator*() {
    vistied_.insert(size_t(*this));
    Super& super = *this;
    return *super;
  }

  const ElementType* operator->() { return &(operator*()); }

 private:
  const Container& owner_;
  const size_t nodeCount_;
  std::unordered_set<size_t> vistied_;
};

void coalesce_nodes(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  enum class CoalesceState { Initial, Filter, FirstProject, Aggregate };
  std::vector<size_t> crt_pattern;
  CoalesceState crt_state{CoalesceState::Initial};

  for (RANodeIterator nodeIt(nodes); !nodeIt.allVisited();) {
    const auto ra_node = nodeIt != nodes.end() ? *nodeIt : nullptr;
    switch (crt_state) {
      case CoalesceState::Initial: {
        if (std::dynamic_pointer_cast<const RelFilter>(ra_node)) {
          crt_pattern.push_back(size_t(nodeIt));
          crt_state = CoalesceState::Filter;
          nodeIt.advance(RANodeIterator::AdvancingMode::DUChain);
        } else if (std::dynamic_pointer_cast<const RelProject>(ra_node)) {
          crt_pattern.push_back(size_t(nodeIt));
          crt_state = CoalesceState::FirstProject;
          nodeIt.advance(RANodeIterator::AdvancingMode::DUChain);
        } else {
          nodeIt.advance(RANodeIterator::AdvancingMode::InOrder);
        }
        break;
      }
      case CoalesceState::Filter: {
        if (std::dynamic_pointer_cast<const RelProject>(ra_node)) {
          crt_pattern.push_back(size_t(nodeIt));
          crt_state = CoalesceState::FirstProject;
          nodeIt.advance(RANodeIterator::AdvancingMode::DUChain);
        } else {
          crt_state = CoalesceState::Initial;
          decltype(crt_pattern)().swap(crt_pattern);
        }
        break;
      }
      case CoalesceState::FirstProject: {
        if (std::dynamic_pointer_cast<const RelAggregate>(ra_node)) {
          crt_pattern.push_back(size_t(nodeIt));
          crt_state = CoalesceState::Aggregate;
          nodeIt.advance(RANodeIterator::AdvancingMode::DUChain);
        } else {
          crt_state = CoalesceState::Initial;
          if (crt_pattern.size() >= 2) {
            create_compound(nodes, crt_pattern);
          }
          decltype(crt_pattern)().swap(crt_pattern);
        }
        break;
      }
      case CoalesceState::Aggregate: {
        if (std::dynamic_pointer_cast<const RelProject>(ra_node) &&
            std::static_pointer_cast<RelProject>(ra_node)->isSimple()) {
          crt_pattern.push_back(size_t(nodeIt));
          nodeIt.advance(RANodeIterator::AdvancingMode::InOrder);
        }
        crt_state = CoalesceState::Initial;
        CHECK_GE(crt_pattern.size(), size_t(2));
        create_compound(nodes, crt_pattern);
        decltype(crt_pattern)().swap(crt_pattern);
        break;
      }
      default:
        CHECK(false);
    }
  }
  if (crt_state == CoalesceState::FirstProject || crt_state == CoalesceState::Aggregate) {
    if (crt_pattern.size() >= 2) {
      create_compound(nodes, crt_pattern);
    }
    CHECK(!crt_pattern.empty());
  }
}

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
  for (auto nodeIt = nodes.rbegin(); nodeIt != nodes.rend(); ++nodeIt) {
    if (nodeIt->unique()) {
      nodeIt->reset();
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
    if (first_sort && second_sort && project && project->isIdentity() && *first_sort == *second_sort) {
      second_sort->replaceInput(second_sort->getAndOwnInput(0), first_sort->getAndOwnInput(0));
      nodes[i].reset();
      nodes[i + 1].reset();
      i += 3;
    } else {
      ++i;
    }
  }
}

int64_t get_int_literal_field(const rapidjson::Value& obj, const char field[], const int64_t default_val) noexcept {
  const auto it = obj.FindMember(field);
  if (it == obj.MemberEnd()) {
    return default_val;
  }
  std::unique_ptr<RexLiteral> lit(parse_literal(it->value));
  CHECK_EQ(kDECIMAL, lit->getType());
  CHECK_EQ(unsigned(0), lit->getScale());
  CHECK_EQ(unsigned(0), lit->getTypeScale());
  return lit->getVal<int64_t>();
}

void check_empty_inputs_field(const rapidjson::Value& node) noexcept {
  const auto& inputs_json = field(node, "inputs");
  CHECK(inputs_json.IsArray() && !inputs_json.Size());
}

class RaAbstractInterp {
 public:
  RaAbstractInterp(const rapidjson::Value& query_ast, const Catalog_Namespace::Catalog& cat)
      : query_ast_(query_ast), cat_(cat) {}

  std::shared_ptr<const RelAlgNode> run() {
    const auto& rels = field(query_ast_, "rels");
    CHECK(rels.IsArray());
    try {
      dispatchNodes(rels);
    } catch (const QueryNotSupported&) {
      throw;
    }
    CHECK(!nodes_.empty());
    bind_inputs(nodes_);
    mark_nops(nodes_);
    eliminate_identical_copy(nodes_);
    coalesce_nodes(nodes_);
    simplify_sort(nodes_);
    CHECK(nodes_.back().unique());
    return nodes_.back();
  }

 private:
  void dispatchNodes(const rapidjson::Value& rels) {
    for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
      const auto& crt_node = *rels_it;
      const auto id = node_id(crt_node);
      CHECK_EQ(static_cast<size_t>(id), nodes_.size());
      CHECK(crt_node.IsObject());
      std::shared_ptr<RelAlgNode> ra_node = nullptr;
      const auto rel_op = json_str(field(crt_node, "relOp"));
      if (rel_op == std::string("LogicalTableScan")) {
        ra_node = dispatchTableScan(crt_node);
      } else if (rel_op == std::string("LogicalProject")) {
        ra_node = dispatchProject(crt_node);
      } else if (rel_op == std::string("LogicalFilter")) {
        ra_node = dispatchFilter(crt_node);
      } else if (rel_op == std::string("LogicalAggregate")) {
        ra_node = dispatchAggregate(crt_node);
      } else if (rel_op == std::string("LogicalJoin")) {
        ra_node = dispatchJoin(crt_node);
      } else if (rel_op == std::string("LogicalSort")) {
        ra_node = dispatchSort(crt_node);
      } else {
        throw QueryNotSupported(std::string("Node ") + rel_op + " not supported yet");
      }
      nodes_.push_back(ra_node);
    }
  }

  std::shared_ptr<RelScan> dispatchTableScan(const rapidjson::Value& scan_ra) {
    check_empty_inputs_field(scan_ra);
    CHECK(scan_ra.IsObject());
    const auto td = getTableFromScanNode(scan_ra);
    const auto field_names = getFieldNamesFromScanNode(scan_ra);
    return std::make_shared<RelScan>(td, field_names);
  }

  std::shared_ptr<RelProject> dispatchProject(const rapidjson::Value& proj_ra) {
    const auto inputs = getRelAlgInputs(proj_ra);
    CHECK_EQ(size_t(1), inputs.size());
    const auto& exprs_json = field(proj_ra, "exprs");
    CHECK(exprs_json.IsArray());
    std::vector<std::unique_ptr<const RexScalar>> exprs;
    for (auto exprs_json_it = exprs_json.Begin(); exprs_json_it != exprs_json.End(); ++exprs_json_it) {
      exprs.emplace_back(parse_scalar_expr(*exprs_json_it));
    }
    const auto& fields = field(proj_ra, "fields");
    return std::make_shared<RelProject>(exprs, strings_from_json_array(fields), inputs.front());
  }

  std::shared_ptr<RelFilter> dispatchFilter(const rapidjson::Value& filter_ra) {
    const auto inputs = getRelAlgInputs(filter_ra);
    CHECK_EQ(size_t(1), inputs.size());
    const auto id = node_id(filter_ra);
    CHECK(id);
    auto condition = parse_scalar_expr(field(filter_ra, "condition"));
    return std::make_shared<RelFilter>(condition, inputs.front());
  }

  std::shared_ptr<RelAggregate> dispatchAggregate(const rapidjson::Value& agg_ra) {
    const auto inputs = getRelAlgInputs(agg_ra);
    CHECK_EQ(size_t(1), inputs.size());
    const auto fields = strings_from_json_array(field(agg_ra, "fields"));
    const auto group = indices_from_json_array(field(agg_ra, "group"));
    for (size_t i = 0; i < group.size(); ++i) {
      CHECK_EQ(i, group[i]);
    }
    const auto& aggs_json_arr = field(agg_ra, "aggs");
    CHECK(aggs_json_arr.IsArray());
    std::vector<std::unique_ptr<const RexAgg>> aggs;
    for (auto aggs_json_arr_it = aggs_json_arr.Begin(); aggs_json_arr_it != aggs_json_arr.End(); ++aggs_json_arr_it) {
      aggs.emplace_back(parse_aggregate_expr(*aggs_json_arr_it));
    }
    return std::make_shared<RelAggregate>(group.size(), aggs, fields, inputs.front());
  }

  std::shared_ptr<RelJoin> dispatchJoin(const rapidjson::Value& join_ra) {
    const auto inputs = getRelAlgInputs(join_ra);
    CHECK_EQ(size_t(2), inputs.size());
    const auto join_type = to_join_type(json_str(field(join_ra, "joinType")));
    auto filter_rex = parse_scalar_expr(field(join_ra, "condition"));
    return std::make_shared<RelJoin>(inputs[0], inputs[1], filter_rex, join_type);
  }

  std::shared_ptr<RelSort> dispatchSort(const rapidjson::Value& sort_ra) {
    const auto inputs = getRelAlgInputs(sort_ra);
    CHECK_EQ(size_t(1), inputs.size());
    std::vector<SortField> collation;
    const auto& collation_arr = field(sort_ra, "collation");
    CHECK(collation_arr.IsArray());
    for (auto collation_arr_it = collation_arr.Begin(); collation_arr_it != collation_arr.End(); ++collation_arr_it) {
      const size_t field_idx = json_i64(field(*collation_arr_it, "field"));
      const SortDirection sort_dir = json_str(field(*collation_arr_it, "direction")) == std::string("DESCENDING")
                                         ? SortDirection::Descending
                                         : SortDirection::Ascending;
      const NullSortedPosition null_pos = json_str(field(*collation_arr_it, "nulls")) == std::string("FIRST")
                                              ? NullSortedPosition::First
                                              : NullSortedPosition::Last;
      collation.emplace_back(field_idx, sort_dir, null_pos);
    }
    const auto limit = get_int_literal_field(sort_ra, "fetch", 0);
    const auto offset = get_int_literal_field(sort_ra, "offset", 0);
    return std::make_shared<RelSort>(collation, limit, offset, inputs.front());
  }

  const TableDescriptor* getTableFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto& table_json = field(scan_ra, "table");
    CHECK(table_json.IsArray());
    CHECK_EQ(unsigned(3), table_json.Size());
    const auto td = cat_.getMetadataForTable(table_json[2].GetString());
    CHECK(td);
    return td;
  }

  std::vector<std::string> getFieldNamesFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto& fields_json = field(scan_ra, "fieldNames");
    return strings_from_json_array(fields_json);
  }

  std::vector<std::shared_ptr<const RelAlgNode>> getRelAlgInputs(const rapidjson::Value& node) {
    if (node.HasMember("inputs")) {
      const auto str_input_ids = strings_from_json_array(field(node, "inputs"));
      std::vector<std::shared_ptr<const RelAlgNode>> ra_inputs;
      for (const auto str_id : str_input_ids) {
        ra_inputs.push_back(nodes_[std::stoi(str_id)]);
      }
      return ra_inputs;
    }
    return {prev(node)};
  }

  std::shared_ptr<const RelAlgNode> prev(const rapidjson::Value& crt_node) {
    const auto id = node_id(crt_node);
    CHECK(id);
    CHECK_EQ(static_cast<size_t>(id), nodes_.size());
    return nodes_.back();
  }

  const rapidjson::Value& query_ast_;
  const Catalog_Namespace::Catalog& cat_;
  std::vector<std::shared_ptr<RelAlgNode>> nodes_;
};

}  // namespace

std::shared_ptr<const RelAlgNode> ra_interpret(const rapidjson::Value& query_ast,
                                               const Catalog_Namespace::Catalog& cat) {
  RaAbstractInterp interp(query_ast, cat);
  return interp.run();
}

std::string tree_string(const RelAlgNode* ra, const size_t indent) {
  std::string result = std::string(indent, ' ') + ra->toString() + "\n";
  for (size_t i = 0; i < ra->inputCount(); ++i) {
    result += tree_string(ra->getInput(i), indent + 2);
  }
  return result;
}
#endif  // HAVE_CALCITE
