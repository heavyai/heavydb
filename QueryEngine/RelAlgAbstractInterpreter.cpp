#ifdef HAVE_CALCITE
#include "RelAlgAbstractInterpreter.h"
#include "CalciteDeserializerUtils.h"
#include "JsonAccessors.h"
#include "RexVisitor.h"

#include <glog/logging.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <string>
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

void RelProject::replaceInput(const RelAlgNode* old_input, const RelAlgNode* input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input, input);
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

void RelJoin::replaceInput(const RelAlgNode* old_input, const RelAlgNode* input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input, input);
  if (condition_) {
    rebind_inputs.visit(condition_.get());
  }
}

void RelFilter::replaceInput(const RelAlgNode* old_input, const RelAlgNode* input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input, input);
  rebind_inputs.visit(filter_.get());
}

void RelCompound::replaceInput(const RelAlgNode* old_input, const RelAlgNode* input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input, input);
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

RexAbstractInput* parse_abstract_input(const rapidjson::Value& expr) noexcept {
  const auto& input = field(expr, "input");
  return new RexAbstractInput(json_i64(input));
}

RexLiteral* parse_literal(const rapidjson::Value& expr) noexcept {
  CHECK(expr.IsObject());
  const auto& literal = field(expr, "literal");
  const auto type = to_sql_type(json_str(field(expr, "type")));
  const auto scale = json_i64(field(expr, "scale"));
  const auto precision = json_i64(field(expr, "precision"));
  const auto type_scale = json_i64(field(expr, "type_scale"));
  const auto type_precision = json_i64(field(expr, "type_precision"));
  switch (type) {
    case kDECIMAL:
      return new RexLiteral(json_i64(literal), type, scale, precision, type_scale, type_precision);
    case kDOUBLE:
      return new RexLiteral(json_double(literal), type, scale, precision, type_scale, type_precision);
    case kTEXT:
      return new RexLiteral(json_str(literal), type, scale, precision, type_scale, type_precision);
    case kBOOLEAN:
      return new RexLiteral(json_bool(literal), type, scale, precision, type_scale, type_precision);
    case kNULLT:
      return new RexLiteral();
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

RexScalar* parse_scalar_expr(const rapidjson::Value& expr);

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

RexOperator* parse_operator(const rapidjson::Value& expr) {
  const auto op_name = json_str(field(expr, "op"));
  const bool is_quantifier = op_name == std::string("PG_ANY") || op_name == std::string("PG_ALL");
  const auto op = is_quantifier ? kFUNCTION : to_sql_op(op_name);
  const auto& operators_json_arr = field(expr, "operands");
  CHECK(operators_json_arr.IsArray());
  std::vector<const RexScalar*> operands;
  for (auto operators_json_arr_it = operators_json_arr.Begin(); operators_json_arr_it != operators_json_arr.End();
       ++operators_json_arr_it) {
    operands.push_back(parse_scalar_expr(*operators_json_arr_it));
  }
  if (op == kFUNCTION) {
    return new RexFunctionOperator(op_name, operands);
  }
  const auto type_it = expr.FindMember("type");
  CHECK_EQ(op == kCAST, type_it != expr.MemberEnd());
  SQLTypeInfo ti;
  if (op == kCAST) {
    ti = parse_type(type_it->value);
  }
  return new RexOperator(op, operands, ti);
}

RexCase* parse_case(const rapidjson::Value& expr) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(2));
  const RexScalar* else_expr{nullptr};
  std::vector<std::pair<const RexScalar*, const RexScalar*>> expr_pair_list;
  for (auto operands_it = operands.Begin(); operands_it != operands.End();) {
    const auto when_expr = parse_scalar_expr(*operands_it++);
    if (operands_it == operands.End()) {
      else_expr = when_expr;
      break;
    }
    const auto then_expr = parse_scalar_expr(*operands_it++);
    expr_pair_list.emplace_back(when_expr, then_expr);
  }
  return new RexCase(expr_pair_list, else_expr);
}

std::vector<std::string> strings_from_json_array(const rapidjson::Value& json_str_arr) {
  CHECK(json_str_arr.IsArray());
  std::vector<std::string> fields;
  for (auto json_str_arr_it = json_str_arr.Begin(); json_str_arr_it != json_str_arr.End(); ++json_str_arr_it) {
    CHECK(json_str_arr_it->IsString());
    fields.emplace_back(json_str_arr_it->GetString());
  }
  return fields;
}

std::vector<size_t> indices_from_json_array(const rapidjson::Value& json_idx_arr) {
  CHECK(json_idx_arr.IsArray());
  std::vector<size_t> indices;
  for (auto json_idx_arr_it = json_idx_arr.Begin(); json_idx_arr_it != json_idx_arr.End(); ++json_idx_arr_it) {
    CHECK(json_idx_arr_it->IsInt());
    CHECK_GE(json_idx_arr_it->GetInt(), 0);
    indices.emplace_back(json_idx_arr_it->GetInt());
  }
  return indices;
}

std::string json_node_to_string(const rapidjson::Value& node) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  node.Accept(writer);
  return buffer.GetString();
}

RexAgg* parse_aggregate_expr(const rapidjson::Value& expr) {
  const auto agg = to_agg_kind(json_str(field(expr, "agg")));
  const auto distinct = json_bool(field(expr, "distinct"));
  const auto agg_ti = parse_type(field(expr, "type"));
  const auto operands = indices_from_json_array(field(expr, "operands"));
  CHECK_LE(operands.size(), size_t(1));
  return new RexAgg(agg, distinct, agg_ti, operands.empty() ? -1 : operands[0]);
}

RexScalar* parse_scalar_expr(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  if (expr.IsObject() && expr.HasMember("input")) {
    return parse_abstract_input(expr);
  }
  if (expr.IsObject() && expr.HasMember("literal")) {
    return parse_literal(expr);
  }
  if (expr.IsObject() && expr.HasMember("op")) {
    const auto op_str = json_str(field(expr, "op"));
    if (op_str == std::string("CASE")) {
      return parse_case(expr);
    }
    return parse_operator(expr);
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

const RexScalar* disambiguate_rex(const RexScalar*, const RANodeOutput&);

const RexOperator* disambiguate_operator(const RexOperator* rex_operator, const RANodeOutput& ra_output) {
  std::vector<const RexScalar*> disambiguated_operands;
  for (size_t i = 0; i < rex_operator->size(); ++i) {
    disambiguated_operands.push_back(disambiguate_rex(rex_operator->getOperand(i), ra_output));
  }
  return rex_operator->getDisambiguated(disambiguated_operands);
}

const RexCase* disambiguate_case(const RexCase* rex_case, const RANodeOutput& ra_output) {
  std::vector<std::pair<const RexScalar*, const RexScalar*>> disambiguated_expr_pair_list;
  for (size_t i = 0; i < rex_case->branchCount(); ++i) {
    const auto disambiguated_when = disambiguate_rex(rex_case->getWhen(i), ra_output);
    const auto disambiguated_then = disambiguate_rex(rex_case->getThen(i), ra_output);
    disambiguated_expr_pair_list.emplace_back(disambiguated_when, disambiguated_then);
  }
  return new RexCase(disambiguated_expr_pair_list, disambiguate_rex(rex_case->getElse(), ra_output));
}

const RexScalar* disambiguate_rex(const RexScalar* rex_scalar, const RANodeOutput& ra_output) {
  const auto rex_abstract_input = dynamic_cast<const RexAbstractInput*>(rex_scalar);
  if (rex_abstract_input) {
    CHECK_LT(static_cast<size_t>(rex_abstract_input->getIndex()), ra_output.size());
    return new RexInput(ra_output[rex_abstract_input->getIndex()]);
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
  return new RexLiteral(*rex_literal);
}

void bind_project_to_input(RelProject* project_node, const RANodeOutput& input) {
  CHECK_EQ(size_t(1), project_node->inputCount());
  std::vector<const RexScalar*> disambiguated_exprs;
  for (size_t i = 0; i < project_node->size(); ++i) {
    disambiguated_exprs.push_back(disambiguate_rex(project_node->getProjectAt(i), input));
  }
  project_node->setExpressions(disambiguated_exprs);
}

void bind_inputs(const std::vector<RelAlgNode*>& nodes) {
  for (auto ra_node : nodes) {
    const auto filter_node = dynamic_cast<RelFilter*>(ra_node);
    if (filter_node) {
      CHECK_EQ(size_t(1), filter_node->inputCount());
      const auto disambiguated_condition =
          disambiguate_rex(filter_node->getCondition(), get_node_output(filter_node->getInput(0)));
      filter_node->setCondition(disambiguated_condition);
      continue;
    }
    const auto join_node = dynamic_cast<RelJoin*>(ra_node);
    if (join_node) {
      CHECK_EQ(size_t(2), join_node->inputCount());
      const auto disambiguated_condition = disambiguate_rex(join_node->getCondition(), get_node_output(join_node));
      join_node->setCondition(disambiguated_condition);
      continue;
    }
    const auto project_node = dynamic_cast<RelProject*>(ra_node);
    if (project_node) {
      bind_project_to_input(project_node, get_node_output(project_node->getInput(0)));
      continue;
    }
  }
}

void mark_nops(const std::vector<RelAlgNode*>& nodes) {
  for (auto node : nodes) {
    const auto agg_node = dynamic_cast<RelAggregate*>(node);
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
                                          const std::vector<const Rex*>& target_exprs) {
  std::vector<const Rex*> result;
  for (size_t i = 0; i < simple_project->size(); ++i) {
    const auto input_rex = dynamic_cast<const RexInput*>(simple_project->getProjectAt(i));
    CHECK(input_rex);
    CHECK_LT(static_cast<size_t>(input_rex->getIndex()), target_exprs.size());
    result.push_back(target_exprs[input_rex->getIndex()]);
  }
  return result;
}

void create_compound(std::vector<RelAlgNode*>& nodes, const std::vector<size_t>& pattern) {
  CHECK_GE(pattern.size(), size_t(2));
  CHECK_LE(pattern.size(), size_t(4));
  const RexScalar* filter_rex{nullptr};
  std::vector<const RexScalar*> scalar_sources;
  size_t groupby_count{0};
  std::vector<std::string> fields;
  std::vector<const RexAgg*> agg_exprs;
  std::vector<const Rex*> target_exprs;
  bool first_project{true};
  bool is_agg{false};
  for (const auto node_idx : pattern) {
    const auto ra_node = nodes[node_idx];
    const auto ra_filter = dynamic_cast<RelFilter*>(ra_node);
    if (ra_filter) {
      CHECK(!filter_rex);
      filter_rex = ra_filter->getAndReleaseCondition();
      CHECK(filter_rex);
      continue;
    }
    const auto ra_project = dynamic_cast<RelProject*>(ra_node);
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
          bind_project_to_input(ra_project, get_node_output(filter_input->getInput(0)));
        }
        scalar_sources = ra_project->getExpressionsAndRelease();
        for (const auto scalar_expr : scalar_sources) {
          target_exprs.push_back(scalar_expr);
        }
        first_project = false;
      } else {
        CHECK(ra_project->isSimple());
        target_exprs = reproject_targets(ra_project, target_exprs);
      }
      continue;
    }
    const auto ra_aggregate = dynamic_cast<RelAggregate*>(ra_node);
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
        scalar_sources.push_back(rex_ref);
      }
      for (const auto rex_agg : agg_exprs) {
        target_exprs.push_back(rex_agg);
      }
      continue;
    }
  }
  auto compound_node =
      new RelCompound(filter_rex, target_exprs, groupby_count, agg_exprs, fields, scalar_sources, is_agg);
  auto old_node = nodes[pattern.back()];
  nodes[pattern.back()] = compound_node;
  auto first_node = nodes[pattern.front()];
  CHECK_EQ(size_t(1), first_node->inputCount());
  compound_node->addInput(first_node->getInputAndRelease(0));
  for (size_t i = 0; i < pattern.size() - 1; ++i) {
    nodes[pattern[i]] = nullptr;
  }
  for (auto node : nodes) {
    if (!node) {
      continue;
    }
    node->replaceInput(old_node, compound_node);
  }
  // Since the last node isn't an input, we need to manually delete it.
  // TODO(alex): figure out a better ownership model
  if (pattern.back() == nodes.size() - 1) {
    delete old_node;
  }
}

void coalesce_nodes(std::vector<RelAlgNode*>& nodes) {
  enum class CoalesceState { Initial, Filter, FirstProject, Aggregate };
  std::vector<size_t> crt_pattern;
  CoalesceState crt_state{CoalesceState::Initial};
  for (size_t i = 0; i < nodes.size();) {
    const auto ra_node = nodes[i];
    switch (crt_state) {
      case CoalesceState::Initial: {
        if (dynamic_cast<const RelFilter*>(ra_node)) {
          crt_pattern.push_back(i);
          crt_state = CoalesceState::Filter;
        } else if (dynamic_cast<const RelProject*>(ra_node)) {
          crt_pattern.push_back(i);
          crt_state = CoalesceState::FirstProject;
        }
        ++i;
        break;
      }
      case CoalesceState::Filter: {
        if (dynamic_cast<const RelProject*>(ra_node)) {
          crt_pattern.push_back(i);
          crt_state = CoalesceState::FirstProject;
          ++i;
        } else {
          crt_state = CoalesceState::Initial;
          decltype(crt_pattern)().swap(crt_pattern);
        }
        break;
      }
      case CoalesceState::FirstProject: {
        if (dynamic_cast<const RelAggregate*>(ra_node)) {
          crt_pattern.push_back(i);
          crt_state = CoalesceState::Aggregate;
          ++i;
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
        if (dynamic_cast<const RelProject*>(ra_node) && static_cast<RelProject*>(ra_node)->isSimple()) {
          crt_pattern.push_back(i);
          ++i;
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

// For some reason, Calcite generates Sort, Project, Sort sequences where the
// two Sort nodes are identical and the Project is identity. Simplify this
// pattern by re-binding the input of the second sort to the input of the first.
void simplify_sort(std::vector<RelAlgNode*>& nodes) {
  if (nodes.size() < 3) {
    return;
  }
  for (size_t i = 0; i <= nodes.size() - 3;) {
    auto first_sort = dynamic_cast<RelSort*>(nodes[i]);
    const auto project = dynamic_cast<const RelProject*>(nodes[i + 1]);
    auto second_sort = dynamic_cast<RelSort*>(nodes[i + 2]);
    if (first_sort && second_sort && project && project->isIdentity() && *first_sort == *second_sort) {
      second_sort->replaceInput(second_sort->getInput(0), first_sort->getInputAndRelease(0));
      nodes[i] = nodes[i + 1] = nullptr;
      i += 3;
    } else {
      ++i;
    }
  }
}

int64_t get_int_literal_field(const rapidjson::Value& obj, const char field[], const int64_t default_val) {
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

// As of now, only Join nodes specify inputs and we only support relational
// algebra trees. For full SQL support we'll need DAG relational algebra and thus
// forward edges, which are specified by the inputs field in non-join nodes.
void check_no_inputs_field(const rapidjson::Value& node) {
  CHECK(node.IsObject());
  if (node.HasMember("inputs")) {
    throw QueryNotSupported("This query structure is not supported yet");
  }
}

void check_empty_inputs_field(const rapidjson::Value& node) {
  const auto& inputs_json = field(node, "inputs");
  CHECK(inputs_json.IsArray() && !inputs_json.Size());
}

class RaAbstractInterp {
 public:
  RaAbstractInterp(const rapidjson::Value& query_ast, const Catalog_Namespace::Catalog& cat)
      : query_ast_(query_ast), cat_(cat) {}

  std::unique_ptr<const RelAlgNode> run() {
    const auto& rels = field(query_ast_, "rels");
    CHECK(rels.IsArray());
    try {
      dispatchNodes(rels);
    } catch (const QueryNotSupported&) {
      CHECK(!nodes_.empty());
      // TODO(alex): Not great, need to figure out a better ownership model.
      delete nodes_.back();
      throw;
    }
    CHECK(!nodes_.empty());
    bind_inputs(nodes_);
    mark_nops(nodes_);
    coalesce_nodes(nodes_);
    simplify_sort(nodes_);
    return std::unique_ptr<const RelAlgNode>(nodes_.back());
  }

 private:
  void dispatchNodes(const rapidjson::Value& rels) {
    for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
      const auto& crt_node = *rels_it;
      const auto id = node_id(crt_node);
      CHECK_EQ(static_cast<size_t>(id), nodes_.size());
      CHECK(crt_node.IsObject());
      RelAlgNode* ra_node = nullptr;
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

  RelScan* dispatchTableScan(const rapidjson::Value& scan_ra) {
    check_empty_inputs_field(scan_ra);
    CHECK(scan_ra.IsObject());
    const auto td = getTableFromScanNode(scan_ra);
    const auto field_names = getFieldNamesFromScanNode(scan_ra);
    return new RelScan(td, field_names);
  }

  RelProject* dispatchProject(const rapidjson::Value& proj_ra) {
    check_no_inputs_field(proj_ra);
    const auto& exprs_json = field(proj_ra, "exprs");
    CHECK(exprs_json.IsArray());
    std::vector<const RexScalar*> exprs;
    for (auto exprs_json_it = exprs_json.Begin(); exprs_json_it != exprs_json.End(); ++exprs_json_it) {
      exprs.push_back(parse_scalar_expr(*exprs_json_it));
    }
    const auto& fields = field(proj_ra, "fields");
    return new RelProject(exprs, strings_from_json_array(fields), prev(proj_ra));
  }

  RelFilter* dispatchFilter(const rapidjson::Value& filter_ra) {
    check_no_inputs_field(filter_ra);
    const auto id = node_id(filter_ra);
    CHECK(id);
    return new RelFilter(parse_scalar_expr(field(filter_ra, "condition")), prev(filter_ra));
  }

  const RelAlgNode* prev(const rapidjson::Value& crt_node) {
    const auto id = node_id(crt_node);
    CHECK(id);
    CHECK_EQ(static_cast<size_t>(id), nodes_.size());
    return nodes_.back();
  }

  RelAggregate* dispatchAggregate(const rapidjson::Value& agg_ra) {
    check_no_inputs_field(agg_ra);
    const auto fields = strings_from_json_array(field(agg_ra, "fields"));
    const auto group = indices_from_json_array(field(agg_ra, "group"));
    for (size_t i = 0; i < group.size(); ++i) {
      CHECK_EQ(i, group[i]);
    }
    const auto& aggs_json_arr = field(agg_ra, "aggs");
    CHECK(aggs_json_arr.IsArray());
    std::vector<const RexAgg*> aggs;
    for (auto aggs_json_arr_it = aggs_json_arr.Begin(); aggs_json_arr_it != aggs_json_arr.End(); ++aggs_json_arr_it) {
      aggs.push_back(parse_aggregate_expr(*aggs_json_arr_it));
    }
    return new RelAggregate(group.size(), aggs, fields, prev(agg_ra));
  }

  RelJoin* dispatchJoin(const rapidjson::Value& join_ra) {
    const auto join_type = to_join_type(json_str(field(join_ra, "joinType")));
    const auto filter_rex = parse_scalar_expr(field(join_ra, "condition"));
    const auto str_input_indices = strings_from_json_array(field(join_ra, "inputs"));
    CHECK_EQ(size_t(2), str_input_indices.size());
    std::vector<size_t> input_indices;
    for (const auto& str_index : str_input_indices) {
      input_indices.push_back(std::stoi(str_index));
    }
    CHECK_LT(input_indices[0], nodes_.size());
    CHECK_LT(input_indices[1], nodes_.size());
    return new RelJoin(nodes_[input_indices[0]], nodes_[input_indices[1]], filter_rex, join_type);
  }

  RelSort* dispatchSort(const rapidjson::Value& sort_ra) {
    check_no_inputs_field(sort_ra);
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
    return new RelSort(collation, limit, offset, prev(sort_ra));
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

  const rapidjson::Value& query_ast_;
  const Catalog_Namespace::Catalog& cat_;
  std::vector<RelAlgNode*> nodes_;
};

}  // namespace

std::unique_ptr<const RelAlgNode> ra_interpret(const rapidjson::Value& query_ast,
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
