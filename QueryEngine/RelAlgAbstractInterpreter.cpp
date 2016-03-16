#ifdef HAVE_CALCITE
#include "RelAlgAbstractInterpreter.h"
#include "RelAlgValidator.h"
#include "CalciteDeserializerUtils.h"
#include "RexVisitor.h"

#include "../Analyzer/Analyzer.h"
#include "../Parser/ParserNode.h"

#include <glog/logging.h>

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

}  // namespace

void RelProject::replaceInput(const RelAlgNode* old_input, const RelAlgNode* input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input, input);
  for (const auto& scalar_expr : scalar_exprs_) {
    rebind_inputs.visit(scalar_expr.get());
  }
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

// Checked json field retrieval.
const rapidjson::Value& field(const rapidjson::Value& obj, const char field[]) noexcept {
  CHECK(obj.IsObject());
  const auto field_it = obj.FindMember(field);
  CHECK(field_it != obj.MemberEnd());
  return field_it->value;
}

const int64_t json_i64(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsInt64());
  return obj.GetInt64();
}

const std::string json_str(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsString());
  return obj.GetString();
}

const bool json_bool(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsBool());
  return obj.GetBool();
}

const double json_double(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsDouble());
  return obj.GetDouble();
}

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

RexOperator* parse_operator(const rapidjson::Value& expr) {
  const auto op = to_sql_op(json_str(field(expr, "op")));
  const auto& operators_json_arr = field(expr, "operands");
  CHECK(operators_json_arr.IsArray());
  std::vector<const RexScalar*> operands;
  for (auto operators_json_arr_it = operators_json_arr.Begin(); operators_json_arr_it != operators_json_arr.End();
       ++operators_json_arr_it) {
    operands.push_back(parse_scalar_expr(*operators_json_arr_it));
  }
  return new RexOperator(op, operands);
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

RexAgg* parse_aggregate_expr(const rapidjson::Value& expr) {
  const auto agg = to_agg_kind(json_str(field(expr, "agg")));
  const auto distinct = json_bool(field(expr, "distinct"));
  const auto& type_json = field(expr, "type");
  CHECK(type_json.IsObject() && (type_json.MemberCount() == 2 || type_json.MemberCount() == 4));
  const auto type = to_sql_type(json_str(field(type_json, "type")));
  const auto nullable = json_bool(field(type_json, "nullable"));
  const bool has_precision = type_json.MemberCount() == 4;
  const int precision = has_precision ? json_i64(field(type_json, "precision")) : 0;
  const int scale = has_precision ? json_i64(field(type_json, "scale")) : 0;
  const auto operands = indices_from_json_array(field(expr, "operands"));
  CHECK_LE(operands.size(), size_t(1));
  SQLTypeInfo agg_ti(type, !nullable);
  agg_ti.set_precision(precision);
  agg_ti.set_scale(scale);
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
    return parse_operator(expr);
  }
  CHECK(false);
  return nullptr;
}

RelJoinType to_join_type(const std::string& join_type_name) {
  if (join_type_name == "inner") {
    return RelJoinType::INNER;
  }
  if (join_type_name == "left") {
    return RelJoinType::LEFT;
  }
  CHECK(false);
  return RelJoinType::INNER;
}

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

const RexScalar* disambiguate_rex(const RexScalar* rex_scalar, const RANodeOutput& ra_output) {
  const auto rex_abstract_input = dynamic_cast<const RexAbstractInput*>(rex_scalar);
  if (rex_abstract_input) {
    CHECK_LT(static_cast<size_t>(rex_abstract_input->getIndex()), ra_output.size());
    return new RexInput(ra_output[rex_abstract_input->getIndex()]);
  }
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex_scalar);
  if (rex_operator) {
    std::vector<const RexScalar*> disambiguated_operands;
    for (size_t i = 0; i < rex_operator->size(); ++i) {
      disambiguated_operands.push_back(disambiguate_rex(rex_operator->getOperand(i), ra_output));
    }
    return new RexOperator(rex_operator->getOperator(), disambiguated_operands);
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
    auto filter_node = dynamic_cast<RelFilter*>(ra_node);
    if (filter_node) {
      CHECK_EQ(size_t(1), filter_node->inputCount());
      const auto disambiguated_condition =
          disambiguate_rex(filter_node->getCondition(), get_node_output(filter_node->getInput(0)));
      filter_node->setCondition(disambiguated_condition);
      continue;
    }
    const auto project_node = dynamic_cast<RelProject*>(ra_node);
    if (project_node) {
      bind_project_to_input(project_node, get_node_output(project_node->getInput(0)));
      continue;
    }
  }
}

enum class CoalesceState { Initial, Filter, FirstProject, Aggregate };

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
  std::vector<size_t> group_indices;
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
      group_indices = ra_aggregate->getGroupIndices();
      decltype(target_exprs){}.swap(target_exprs);
      for (const auto group_idx : group_indices) {
        CHECK_LT(group_idx, scalar_sources.size());
        target_exprs.push_back(scalar_sources[group_idx]);
      }
      for (const auto rex_agg : agg_exprs) {
        target_exprs.push_back(rex_agg);
      }
      continue;
    }
  }
  auto compound_node =
      new RelCompound(filter_rex, target_exprs, group_indices, agg_exprs, fields, scalar_sources, is_agg);
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
}

void coalesce_nodes(std::vector<RelAlgNode*>& nodes) {
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
        CHECK(dynamic_cast<const RelProject*>(ra_node));  // TODO: is filter always followed by project?
        crt_pattern.push_back(i);
        crt_state = CoalesceState::FirstProject;
        ++i;
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
  // TODO(alex): wrap-up this function
}

class RaAbstractInterp {
 public:
  RaAbstractInterp(const rapidjson::Value& query_ast, const Catalog_Namespace::Catalog& cat)
      : query_ast_(query_ast), cat_(cat) {}

  std::unique_ptr<const RelAlgNode> run() {
    const auto& rels = field(query_ast_, "rels");
    CHECK(rels.IsArray());
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
        CHECK(false);
      }
      nodes_.push_back(ra_node);
    }
    CHECK(!nodes_.empty());
    bind_inputs(nodes_);
    coalesce_nodes(nodes_);
    CHECK(is_valid_rel_alg(nodes_.back()));
    return std::unique_ptr<const RelAlgNode>(nodes_.back());
  }

 private:
  RelScan* dispatchTableScan(const rapidjson::Value& scan_ra) {
    CHECK(scan_ra.IsObject());
    const auto td = getTableFromScanNode(scan_ra);
    const auto field_names = getFieldNamesFromScanNode(scan_ra);
    return new RelScan(td, field_names);
  }

  RelProject* dispatchProject(const rapidjson::Value& proj_ra) {
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
    const auto fields = strings_from_json_array(field(agg_ra, "fields"));
    const auto group = indices_from_json_array(field(agg_ra, "group"));
    const auto& aggs_json_arr = field(agg_ra, "aggs");
    CHECK(aggs_json_arr.IsArray());
    std::vector<const RexAgg*> aggs;
    for (auto aggs_json_arr_it = aggs_json_arr.Begin(); aggs_json_arr_it != aggs_json_arr.End(); ++aggs_json_arr_it) {
      aggs.push_back(parse_aggregate_expr(*aggs_json_arr_it));
    }
    return new RelAggregate(group, aggs, fields, prev(agg_ra));
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
    return new RelSort(collation, prev(sort_ra));
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

namespace {

SQLTypeInfo build_adjusted_type_info(const SQLTypes sql_type, const int type_scale, const int type_precision) {
  SQLTypeInfo type_ti(sql_type, 0, 0, false);
  type_ti.set_scale(type_scale);
  type_ti.set_precision(type_precision);
  if (type_ti.is_number() && !type_scale) {
    switch (type_precision) {
      case 5:
        return SQLTypeInfo(kSMALLINT, false);
      case 10:
        return SQLTypeInfo(kINT, false);
      case 19:
        return SQLTypeInfo(kBIGINT, false);
      default:
        CHECK(false);
    }
  }
  return type_ti;
}

SQLTypeInfo build_type_info(const SQLTypes sql_type, const int scale, const int precision) {
  SQLTypeInfo ti(sql_type, 0, 0, false);
  ti.set_scale(scale);
  ti.set_precision(precision);
  return ti;
}

std::shared_ptr<Analyzer::Expr> translate_literal(const RexLiteral* rex_literal) {
  const auto lit_ti = build_type_info(rex_literal->getType(), rex_literal->getScale(), rex_literal->getPrecision());
  const auto target_ti =
      build_adjusted_type_info(rex_literal->getType(), rex_literal->getTypeScale(), rex_literal->getTypePrecision());
  switch (rex_literal->getType()) {
    case kDECIMAL: {
      const auto val = rex_literal->getVal<int64_t>();
      const int precision = rex_literal->getPrecision();
      const int scale = rex_literal->getScale();
      auto lit_expr =
          scale ? Parser::FixedPtLiteral::analyzeValue(val, scale, precision) : Parser::IntLiteral::analyzeValue(val);
      return scale && lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kTEXT: {
      return Parser::StringLiteral::analyzeValue(rex_literal->getVal<std::string>());
    }
    case kBOOLEAN: {
      Datum d;
      d.boolval = rex_literal->getVal<bool>();
      return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
    }
    case kDOUBLE: {
      Datum d;
      d.doubleval = rex_literal->getVal<double>();
      auto lit_expr = makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
      return lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kNULLT: {
      return makeExpr<Analyzer::Constant>(kNULLT, true);
    }
    default: { LOG(FATAL) << "Unexpected literal type " << lit_ti.get_type_name(); }
  }
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> translate_input(const RexInput* rex_input,
                                                const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                                const Catalog_Namespace::Catalog& cat) {
  const auto source = rex_input->getSourceNode();
  const auto it_rte_idx = input_to_nest_level.find(source);
  CHECK(it_rte_idx != input_to_nest_level.end());
  const int rte_idx = it_rte_idx->second;
  const auto scan_source = dynamic_cast<const RelScan*>(source);
  const auto& in_metainfo = source->getOutputMetainfo();
  if (scan_source) {
    // We're at leaf (scan) level and not supposed to have input metadata,
    // the name and type information come directly from the catalog.
    CHECK(in_metainfo.empty());
    const auto& field_names = scan_source->getFieldNames();
    CHECK_LT(static_cast<size_t>(rex_input->getIndex()), field_names.size());
    const auto& col_name = field_names[rex_input->getIndex()];
    const auto table_desc = scan_source->getTableDescriptor();
    const auto cd = cat.getMetadataForColumn(table_desc->tableId, col_name);
    CHECK(cd);
    return std::make_shared<Analyzer::ColumnVar>(cd->columnType, table_desc->tableId, cd->columnId, rte_idx);
  }
  CHECK(!in_metainfo.empty());
  CHECK_GE(rte_idx, 0);
  const size_t col_id = rex_input->getIndex();
  CHECK_LT(col_id, in_metainfo.size());
  return std::make_shared<Analyzer::ColumnVar>(in_metainfo[col_id].get_type_info(), -source->getId(), col_id, rte_idx);
}

std::shared_ptr<Analyzer::Expr> remove_cast(const std::shared_ptr<Analyzer::Expr> expr) {
  const auto cast_expr = std::dynamic_pointer_cast<const Analyzer::UOper>(expr);
  return cast_expr && cast_expr->get_optype() == kCAST ? cast_expr->get_own_operand() : expr;
}

std::shared_ptr<Analyzer::Expr> translate_uoper(const RexOperator* rex_operator,
                                                const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                                const Catalog_Namespace::Catalog& cat) {
  CHECK_EQ(size_t(1), rex_operator->size());
  const auto operand_expr = translate_scalar_rex(rex_operator->getOperand(0), input_to_nest_level, cat);
  const auto sql_op = rex_operator->getOperator();
  switch (sql_op) {
    case kCAST: {
      const auto rex_cast = dynamic_cast<const RexCast*>(rex_operator);
      CHECK(rex_cast);
      SQLTypeInfo target_ti(rex_cast->getTargetType(), !rex_cast->getNullable());
      if (target_ti.is_time()) {  // TODO(alex): check and unify with the rest of the cases
        return operand_expr->add_cast(target_ti);
      }
      return std::make_shared<Analyzer::UOper>(target_ti, false, sql_op, operand_expr);
    }
    case kNOT:
    case kISNULL: {
      return std::make_shared<Analyzer::UOper>(kBOOLEAN, sql_op, operand_expr);
    }
    case kISNOTNULL: {
      auto is_null = std::make_shared<Analyzer::UOper>(kBOOLEAN, kISNULL, operand_expr);
      return std::make_shared<Analyzer::UOper>(kBOOLEAN, kNOT, is_null);
    }
    case kMINUS: {
      const auto& ti = operand_expr->get_type_info();
      return std::make_shared<Analyzer::UOper>(ti, false, kUMINUS, operand_expr);
    }
    case kUNNEST: {
      const auto& ti = operand_expr->get_type_info();
      CHECK(ti.is_array());
      return makeExpr<Analyzer::UOper>(ti.get_elem_type(), false, kUNNEST, operand_expr);
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> translate_oper(const RexOperator* rex_operator,
                                               const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                               const Catalog_Namespace::Catalog& cat) {
  CHECK_GT(rex_operator->size(), size_t(0));
  if (rex_operator->size() == 1) {
    return translate_uoper(rex_operator, input_to_nest_level, cat);
  }
  const auto sql_op = rex_operator->getOperator();
  auto lhs = translate_scalar_rex(rex_operator->getOperand(0), input_to_nest_level, cat);
  for (size_t i = 1; i < rex_operator->size(); ++i) {
    auto rhs = translate_scalar_rex(rex_operator->getOperand(i), input_to_nest_level, cat);
    if (sql_op == kEQ || sql_op == kNE) {
      lhs = remove_cast(lhs);
      rhs = remove_cast(rhs);
    }
    lhs = Parser::OperExpr::normalize(sql_op, kONE, lhs, rhs);
  }
  return lhs;
}

}  // namespace

std::shared_ptr<Analyzer::Expr> translate_scalar_rex(
    const RexScalar* rex,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const Catalog_Namespace::Catalog& cat) {
  const auto rex_input = dynamic_cast<const RexInput*>(rex);
  if (rex_input) {
    return translate_input(rex_input, input_to_nest_level, cat);
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex);
  if (rex_literal) {
    return translate_literal(rex_literal);
  }
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex);
  if (rex_operator) {
    return translate_oper(rex_operator, input_to_nest_level, cat);
  }
  CHECK(false);
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> translate_aggregate_rex(
    const RexAgg* rex,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const Catalog_Namespace::Catalog& cat,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  const auto agg_kind = rex->getKind();
  const bool is_distinct = rex->isDistinct();
  const auto operand = rex->getOperand();
  const bool takes_arg{operand >= 0};
  if (takes_arg) {
    CHECK_LT(operand, static_cast<ssize_t>(scalar_sources.size()));
  }
  const auto arg_expr = takes_arg ? scalar_sources[operand] : nullptr;
  const auto agg_ti = get_agg_type(agg_kind, arg_expr.get());
  return makeExpr<Analyzer::AggExpr>(agg_ti, agg_kind, arg_expr, is_distinct);
}

std::string tree_string(const RelAlgNode* ra, const size_t indent) {
  std::string result = std::string(indent, ' ') + ra->toString() + "\n";
  for (size_t i = 0; i < ra->inputCount(); ++i) {
    result += tree_string(ra->getInput(i), indent + 2);
  }
  return result;
}
#endif  // HAVE_CALCITE
