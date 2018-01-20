/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "RelAlgAbstractInterpreter.h"
#include "CalciteDeserializerUtils.h"
#include "JsonAccessors.h"
#include "RelLeftDeepInnerJoin.h"
#include "RelAlgExecutor.h"
#include "RelAlgOptimizer.h"
#include "RexVisitor.h"

#include <glog/logging.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <string>
#include <unordered_set>

namespace {

const unsigned FIRST_RA_NODE_ID = 1;

}  // namespace

thread_local unsigned RelAlgNode::crt_id_ = FIRST_RA_NODE_ID;

void RelAlgNode::resetRelAlgFirstId() noexcept {
  crt_id_ = FIRST_RA_NODE_ID;
}

void RexSubQuery::setExecutionResult(const std::shared_ptr<const ExecutionResult> result) {
  auto row_set = result->getRows();
  CHECK(row_set);
  CHECK_EQ(size_t(1), row_set->colCount());
  type_ = row_set->getColType(0);
  result_ = result;
}

namespace {

class RexRebindInputsVisitor : public RexVisitor<void*> {
 public:
  RexRebindInputsVisitor(const RelAlgNode* old_input, const RelAlgNode* new_input)
      : old_input_(old_input), new_input_(new_input) {}

  void* visitInput(const RexInput* rex_input) const override {
    const auto old_source = rex_input->getSourceNode();
    if (old_source == old_input_) {
      const auto left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(new_input_);
      if (left_deep_join) {
        rebind_inputs_from_left_deep_join(rex_input, left_deep_join);
        return nullptr;
      }
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

}  // namespace

void RelProject::replaceInput(std::shared_ptr<const RelAlgNode> old_input, std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input.get(), input.get());
  for (const auto& scalar_expr : scalar_exprs_) {
    rebind_inputs.visit(scalar_expr.get());
  }
}

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
    auto lhs_out = n_outputs(join_node->getInput(0), get_node_output(join_node->getInput(0)).size());
    const auto rhs_out = n_outputs(join_node->getInput(1), get_node_output(join_node->getInput(1)).size());
    lhs_out.insert(lhs_out.end(), rhs_out.begin(), rhs_out.end());
    return lhs_out;
  }
  const auto multi_join_node = dynamic_cast<const RelMultiJoin*>(ra_node);
  if (multi_join_node) {
    CHECK_LT(size_t(2), multi_join_node->inputCount());
    auto first_out = n_outputs(multi_join_node->getInput(0), get_node_output(multi_join_node->getInput(0)).size());
    for (size_t i = 1; i < multi_join_node->inputCount(); ++i) {
      const auto next_out =
          n_outputs(multi_join_node->getInput(i), get_node_output(multi_join_node->getInput(i)).size());
      first_out.insert(first_out.end(), next_out.begin(), next_out.end());
    }
    return first_out;
  }
  const auto sort_node = dynamic_cast<const RelSort*>(ra_node);
  if (sort_node) {
    // Sort preserves shape
    CHECK_EQ(size_t(1), sort_node->inputCount());
    const auto prev_out = get_node_output(sort_node->getInput(0));
    return n_outputs(sort_node, prev_out.size());
  }
  const auto logical_values_node = dynamic_cast<const RelLogicalValues*>(ra_node);
  if (logical_values_node) {
    CHECK_EQ(size_t(0), logical_values_node->inputCount());
    return n_outputs(logical_values_node, logical_values_node->size());
  }
  CHECK(false);
  return outputs;
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

namespace {

bool isRenamedInput(const RelAlgNode* node, const size_t index, const std::string& new_name) {
  CHECK_LT(index, node->size());
  if (auto join = dynamic_cast<const RelJoin*>(node)) {
    CHECK_EQ(size_t(2), join->inputCount());
    const auto lhs_size = join->getInput(0)->size();
    if (index < lhs_size) {
      return isRenamedInput(join->getInput(0), index, new_name);
    }
    CHECK_GE(index, lhs_size);
    return isRenamedInput(join->getInput(1), index - lhs_size, new_name);
  }

  if (auto scan = dynamic_cast<const RelScan*>(node)) {
    return new_name != scan->getFieldName(index);
  }

  if (auto aggregate = dynamic_cast<const RelAggregate*>(node)) {
    return new_name != aggregate->getFieldName(index);
  }

  if (auto project = dynamic_cast<const RelProject*>(node)) {
    return new_name != project->getFieldName(index);
  }

  if (auto logical_values = dynamic_cast<const RelLogicalValues*>(node)) {
    const auto& tuple_type = logical_values->getTupleType();
    CHECK_LT(index, tuple_type.size());
    return new_name != tuple_type[index].get_resname();
  }

  CHECK(dynamic_cast<const RelSort*>(node) || dynamic_cast<const RelFilter*>(node));
  return isRenamedInput(node->getInput(0), index, new_name);
}

}  // namespace

bool RelProject::isRenaming() const {
  if (!isSimple())
    return false;
  CHECK_EQ(scalar_exprs_.size(), fields_.size());
  for (size_t i = 0; i < fields_.size(); ++i) {
    auto rex_in = dynamic_cast<const RexInput*>(scalar_exprs_[i].get());
    CHECK(rex_in);
    if (isRenamedInput(rex_in->getSourceNode(), rex_in->getIndex(), fields_[i])) {
      return true;
    }
  }
  return false;
}

void RelMultiJoin::append(std::shared_ptr<RelJoin> join) {
  if (sequence_.empty()) {
    const auto src0 = join->getAndOwnInput(0);
    addManagedInput(src0);
  } else {
    CHECK_EQ(sequence_.back().get(), join->getInput(0));
  }
  addManagedInput(join->getAndOwnInput(1));
  sequence_.push_back(join);
  RexDeepCopyVisitor copier;
  auto condition = copier.visit(join->getCondition());
  conditions_.push_back(std::move(condition));
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

std::shared_ptr<RelAlgNode> RelProject::deepCopy() const {
  RexDeepCopyVisitor copier;
  std::vector<std::unique_ptr<const RexScalar>> exprs_copy;
  for (auto& expr : scalar_exprs_) {
    exprs_copy.push_back(copier.visit(expr.get()));
  }
  return std::make_shared<RelProject>(exprs_copy, fields_, inputs_[0]);
}

std::shared_ptr<RelAlgNode> RelFilter::deepCopy() const {
  RexDeepCopyVisitor copier;
  auto filter_copy = copier.visit(filter_.get());
  return std::make_shared<RelFilter>(filter_copy, inputs_[0]);
}

std::shared_ptr<RelAlgNode> RelAggregate::deepCopy() const {
  std::vector<std::unique_ptr<const RexAgg>> aggs_copy;
  for (auto& agg : agg_exprs_) {
    auto copy = agg->deepCopy();
    aggs_copy.push_back(std::move(copy));
  }
  return std::make_shared<RelAggregate>(groupby_count_, aggs_copy, fields_, inputs_[0]);
}

std::shared_ptr<RelAlgNode> RelJoin::deepCopy() const {
  RexDeepCopyVisitor copier;
  auto condition_copy = copier.visit(condition_.get());
  return std::make_shared<RelJoin>(inputs_[0], inputs_[1], condition_copy, join_type_);
}

std::shared_ptr<RelAlgNode> RelCompound::deepCopy() const {
  RexDeepCopyVisitor copier;
  auto filter_copy = filter_expr_ ? copier.visit(filter_expr_.get()) : nullptr;
  std::unordered_map<const Rex*, const Rex*> old_to_new_target;
  std::vector<const RexAgg*> aggs_copy;
  for (auto& agg : agg_exprs_) {
    auto copy = agg->deepCopy();
    old_to_new_target.insert(std::make_pair(agg.get(), copy.get()));
    aggs_copy.push_back(copy.release());
  }
  std::vector<std::unique_ptr<const RexScalar>> sources_copy;
  for (size_t i = 0; i < scalar_sources_.size(); ++i) {
    auto copy = copier.visit(scalar_sources_[i].get());
    old_to_new_target.insert(std::make_pair(scalar_sources_[i].get(), copy.get()));
    sources_copy.push_back(std::move(copy));
  }
  std::vector<const Rex*> target_exprs_copy;
  for (auto target : target_exprs_) {
    auto target_it = old_to_new_target.find(target);
    CHECK(target_it != old_to_new_target.end());
    target_exprs_copy.push_back(target_it->second);
  }
  auto new_compound = std::make_shared<RelCompound>(
      filter_copy, target_exprs_copy, groupby_count_, aggs_copy, fields_, sources_copy, is_agg_);
  new_compound->addManagedInput(inputs_[0]);
  return new_compound;
}

std::shared_ptr<RelAlgNode> RelSort::deepCopy() const {
  return std::make_shared<RelSort>(collation_, limit_, offset_, inputs_[0]);
}

namespace std {
template <>
struct hash<std::pair<const RelAlgNode*, int>> {
  size_t operator()(const std::pair<const RelAlgNode*, int>& input_col) const {
    auto ptr_val = reinterpret_cast<const int64_t*>(&input_col.first);
    return static_cast<int64_t>(*ptr_val) ^ input_col.second;
  }
};
}  // std

namespace {

std::set<std::pair<const RelAlgNode*, int>> get_equiv_cols(const RelAlgNode* node, const size_t which_col) {
  std::set<std::pair<const RelAlgNode*, int>> work_set;
  auto walker = node;
  auto curr_col = which_col;
  while (true) {
    work_set.insert(std::make_pair(walker, curr_col));
    if (dynamic_cast<const RelScan*>(walker) || dynamic_cast<const RelJoin*>(walker)) {
      break;
    }
    CHECK_EQ(size_t(1), walker->inputCount());
    auto only_source = walker->getInput(0);
    if (auto project = dynamic_cast<const RelProject*>(walker)) {
      if (auto input = dynamic_cast<const RexInput*>(project->getProjectAt(curr_col))) {
        const auto join_source = dynamic_cast<const RelJoin*>(only_source);
        if (join_source) {
          CHECK_EQ(size_t(2), join_source->inputCount());
          auto lhs = join_source->getInput(0);
          CHECK((input->getIndex() < lhs->size() && lhs == input->getSourceNode()) ||
                join_source->getInput(1) == input->getSourceNode());
        } else {
          CHECK_EQ(input->getSourceNode(), only_source);
        }
        curr_col = input->getIndex();
      } else {
        break;
      }
    } else if (auto aggregate = dynamic_cast<const RelAggregate*>(walker)) {
      if (curr_col >= aggregate->getGroupByCount()) {
        break;
      }
    }
    walker = only_source;
  }
  return work_set;
}

}  // namespace

bool RelSort::hasEquivCollationOf(const RelSort& that) const {
  if (collation_.size() != that.collation_.size()) {
    return false;
  }

  for (size_t i = 0, e = collation_.size(); i < e; ++i) {
    auto this_sort_key = collation_[i];
    auto that_sort_key = that.collation_[i];
    if (this_sort_key.getSortDir() != that_sort_key.getSortDir()) {
      return false;
    }
    if (this_sort_key.getNullsPosition() != that_sort_key.getNullsPosition()) {
      return false;
    }
    auto this_equiv_keys = get_equiv_cols(this, this_sort_key.getField());
    auto that_equiv_keys = get_equiv_cols(&that, that_sort_key.getField());
    std::vector<std::pair<const RelAlgNode*, int>> intersect;
    std::set_intersection(this_equiv_keys.begin(),
                          this_equiv_keys.end(),
                          that_equiv_keys.begin(),
                          that_equiv_keys.end(),
                          std::back_inserter(intersect));
    if (intersect.empty()) {
      return false;
    }
  }
  return true;
}

namespace {

unsigned node_id(const rapidjson::Value& ra_node) noexcept {
  const auto& id = field(ra_node, "id");
  return std::stoi(json_str(id));
}

// The parse_* functions below de-serialize expressions as they come from Calcite.
// RelAlgAbstractInterpreter will take care of making the representation easy to
// navigate for lower layers, for example by replacing RexAbstractInput with RexInput.

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
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return std::unique_ptr<RexLiteral>(
          new RexLiteral(json_i64(literal), type, target_type, scale, precision, type_scale, type_precision));
    case kDOUBLE: {
      if (literal.IsDouble()) {
        return std::unique_ptr<RexLiteral>(
            new RexLiteral(json_double(literal), type, target_type, scale, precision, type_scale, type_precision));
      }
      CHECK(literal.IsInt64());
      return std::unique_ptr<RexLiteral>(new RexLiteral(
          static_cast<double>(json_i64(literal)), type, target_type, scale, precision, type_scale, type_precision));
    }
    case kTEXT:
      return std::unique_ptr<RexLiteral>(
          new RexLiteral(json_str(literal), type, target_type, scale, precision, type_scale, type_precision));
    case kBOOLEAN:
      return std::unique_ptr<RexLiteral>(
          new RexLiteral(json_bool(literal), type, target_type, scale, precision, type_scale, type_precision));
    case kNULLT:
      return std::unique_ptr<RexLiteral>(new RexLiteral(target_type));
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

std::unique_ptr<const RexScalar> parse_scalar_expr(const rapidjson::Value& expr,
                                                   const Catalog_Namespace::Catalog& cat,
                                                   RelAlgExecutor* ra_executor);

std::unique_ptr<const RexSubQuery> parse_subquery(const rapidjson::Value& expr,
                                                  const Catalog_Namespace::Catalog& cat,
                                                  RelAlgExecutor* ra_executor);

SQLTypeInfo parse_type(const rapidjson::Value& type_obj) {
  CHECK(type_obj.IsObject() && (type_obj.MemberCount() >= 2 && type_obj.MemberCount() <= 4));
  const auto type = to_sql_type(json_str(field(type_obj, "type")));
  const auto nullable = json_bool(field(type_obj, "nullable"));
  const auto precision_it = type_obj.FindMember("precision");
  const int precision = precision_it != type_obj.MemberEnd() ? json_i64(precision_it->value) : 0;
  const auto scale_it = type_obj.FindMember("scale");
  const int scale = scale_it != type_obj.MemberEnd() ? json_i64(scale_it->value) : 0;
  SQLTypeInfo ti(type, !nullable);
  ti.set_precision(precision);
  ti.set_scale(scale);
  return ti;
}

std::unique_ptr<RexOperator> parse_operator(const rapidjson::Value& expr,
                                            const Catalog_Namespace::Catalog& cat,
                                            RelAlgExecutor* ra_executor) {
  const auto op_name = json_str(field(expr, "op"));
  const bool is_quantifier = op_name == std::string("PG_ANY") || op_name == std::string("PG_ALL");
  const auto op = is_quantifier ? kFUNCTION : to_sql_op(op_name);
  const auto& operators_json_arr = field(expr, "operands");
  CHECK(operators_json_arr.IsArray());
  std::vector<std::unique_ptr<const RexScalar>> operands;
  for (auto operators_json_arr_it = operators_json_arr.Begin(); operators_json_arr_it != operators_json_arr.End();
       ++operators_json_arr_it) {
    operands.emplace_back(parse_scalar_expr(*operators_json_arr_it, cat, ra_executor));
  }
  const auto type_it = expr.FindMember("type");
  CHECK(type_it != expr.MemberEnd());
  const auto ti = parse_type(type_it->value);
  if (op == kIN && expr.HasMember("subquery")) {
    auto subquery = parse_subquery(expr, cat, ra_executor);
    operands.emplace_back(std::move(subquery));
  }
  return std::unique_ptr<RexOperator>(op == kFUNCTION ? new RexFunctionOperator(op_name, operands, ti)
                                                      : new RexOperator(op, operands, ti));
}

std::unique_ptr<RexCase> parse_case(const rapidjson::Value& expr,
                                    const Catalog_Namespace::Catalog& cat,
                                    RelAlgExecutor* ra_executor) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(2));
  std::unique_ptr<const RexScalar> else_expr;
  std::vector<std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>> expr_pair_list;
  for (auto operands_it = operands.Begin(); operands_it != operands.End();) {
    auto when_expr = parse_scalar_expr(*operands_it++, cat, ra_executor);
    if (operands_it == operands.End()) {
      else_expr = std::move(when_expr);
      break;
    }
    auto then_expr = parse_scalar_expr(*operands_it++, cat, ra_executor);
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
  if (operands.size() > 1 && (operands.size() != 2 || agg != kAPPROX_COUNT_DISTINCT)) {
    throw QueryNotSupported("Multiple arguments for aggregates aren't supported");
  }
  return std::unique_ptr<const RexAgg>(new RexAgg(agg, distinct, agg_ti, operands));
}

std::unique_ptr<const RexScalar> parse_scalar_expr(const rapidjson::Value& expr,
                                                   const Catalog_Namespace::Catalog& cat,
                                                   RelAlgExecutor* ra_executor) {
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
      return std::unique_ptr<const RexScalar>(parse_case(expr, cat, ra_executor));
    }
    if (op_str == std::string("$SCALAR_QUERY")) {
      return std::unique_ptr<const RexScalar>(parse_subquery(expr, cat, ra_executor));
    }
    return std::unique_ptr<const RexScalar>(parse_operator(expr, cat, ra_executor));
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
    auto operand = rex_operator->getOperand(i);
    if (dynamic_cast<const RexSubQuery*>(operand)) {
      disambiguated_operands.emplace_back(rex_operator->getOperandAndRelease(i));
    } else {
      disambiguated_operands.emplace_back(disambiguate_rex(operand, ra_output));
    }
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

// The inputs used by scalar expressions are given as indices in the serialized
// representation of the query. This is hard to navigate; make the relationship
// explicit by creating RexInput expressions which hold a pointer to the source
// relational algebra node and the index relative to the output of that node.
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
    const auto projected_expr = project_node->getProjectAt(i);
    if (dynamic_cast<const RexSubQuery*>(projected_expr)) {
      disambiguated_exprs.emplace_back(project_node->getProjectAtAndRelease(i));
    } else {
      disambiguated_exprs.emplace_back(disambiguate_rex(projected_expr, input));
    }
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
      : Super(nodes.begin()), owner_(nodes), nodeCount_([&nodes]() -> size_t {
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
          if (!visited_.count(i)) {
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

  bool allVisited() { return visited_.size() == nodeCount_; }

  const ElementType& operator*() {
    visited_.insert(size_t(*this));
    Super& super = *this;
    return *super;
  }

  const ElementType* operator->() { return &(operator*()); }

 private:
  const Container& owner_;
  const size_t nodeCount_;
  std::unordered_set<size_t> visited_;
};

#ifdef ENABLE_EQUIJOIN_FOLD
template <typename T>
const RexOperator* get_equals_operator(const T* node) {
  auto top_operator = dynamic_cast<const RexOperator*>(node->getCondition());
  if (!top_operator) {
    return nullptr;
  }

  std::vector<const RexOperator*> work_set{top_operator};
  while (!work_set.empty()) {
    auto rex_operator = work_set.back();
    work_set.pop_back();
    if (rex_operator->getOperator() == kAND) {
      for (size_t i = 0; i < rex_operator->size(); ++i) {
        if (auto sub_operator = dynamic_cast<const RexOperator*>(rex_operator->getOperand(i))) {
          work_set.push_back(sub_operator);
        }
      }
      continue;
    }
    if (!IS_EQUIVALENCE(rex_operator->getOperator())) {
      continue;
    }
    std::unordered_set<const RelAlgNode*> source_nodes;
    for (size_t i = 0; i < rex_operator->size(); ++i) {
      if (auto rex_in = dynamic_cast<const RexInput*>(rex_operator->getOperand(i))) {
        source_nodes.insert(rex_in->getSourceNode());
      }
    }
    if (source_nodes.size() == size_t(2)) {
      return rex_operator;
    }
  }
  return nullptr;
}

const RexOperator* get_equijoin_condition(const RelJoin* join) {
  if (!join || join->getJoinType() != JoinType::INNER) {
    return nullptr;
  }
  return get_equals_operator(join);
}

std::vector<const RelJoin*> collect_coalesceable_joins(
    const RelJoin* head,
    const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>>& du_web,
    std::vector<const RexScalar*>& condition_set) {
  CHECK(head);
  auto first_condition = get_equijoin_condition(head);
  if (!first_condition) {
    return {};
  }

  condition_set.push_back(first_condition);
  std::vector<const RelJoin*> join_seq;
  const size_t first_table_size = head->getInput(0)->size();
  for (auto walker = head; walker;) {
    join_seq.push_back(walker);
    auto usrs_it = du_web.find(walker);
    CHECK(usrs_it != du_web.end());
    auto& usrs = usrs_it->second;
    CHECK(!usrs.empty());
    if (usrs.size() > size_t(1)) {
      break;
    }
    auto usr_join = dynamic_cast<const RelJoin*>(*usrs.begin());
    if (usr_join) {
      if (usr_join->getInput(0) != join_seq.back() || !dynamic_cast<const RelScan*>(head->getInput(1))) {
        break;
      }
      auto equal_condition = get_equijoin_condition(usr_join);
      if (!equal_condition) {
        // Allow a loop join in the end of sequence for now
        join_seq.push_back(usr_join);
        condition_set.push_back(usr_join->getCondition());
        break;
      }
      auto left_operand = dynamic_cast<const RexInput*>(equal_condition->getOperand(0));
      CHECK(left_operand);
      if (left_operand->getSourceNode() == walker) {
        if (left_operand->getIndex() >= first_table_size) {
          break;
        }
      } else {
        auto right_operand = dynamic_cast<const RexInput*>(equal_condition->getOperand(1));
        CHECK(right_operand);
        if (right_operand->getSourceNode() == walker) {
          if (right_operand->getIndex() >= first_table_size) {
            break;
          }
        }
      }
      condition_set.push_back(equal_condition);
    }
    walker = usr_join;
  }

  if (join_seq.empty()) {
    condition_set.clear();
  } else {
    auto usrs_it = du_web.find(join_seq.back());
    CHECK(usrs_it != du_web.end());
    for (auto usr : usrs_it->second) {
      if (auto filter = dynamic_cast<const RelFilter*>(usr)) {
        if (get_equals_operator(filter)) {
          return {};
        }
      }
    }
  }
  return join_seq;
}

class RexInputRedirector : public RexDeepCopyVisitor {
 public:
  RexInputRedirector(const std::unordered_set<const RelJoin*>& joins) : join_set_(joins) {}
  RetType visitInput(const RexInput* input) const override { return getNonJoinInput(*input); }

  void visitNode(RelAlgNode* node) const {
    if (dynamic_cast<RelAggregate*>(node) || dynamic_cast<RelSort*>(node)) {
      return;
    }
    if (auto join = dynamic_cast<RelJoin*>(node)) {
      auto new_condition = visit(join->getCondition());
      join->setCondition(new_condition);
      return;
    }
    if (auto project = dynamic_cast<RelProject*>(node)) {
      std::vector<std::unique_ptr<const RexScalar>> new_exprs;
      for (size_t i = 0; i < project->size(); ++i) {
        new_exprs.push_back(visit(project->getProjectAt(i)));
      }
      project->setExpressions(new_exprs);
      return;
    }
    if (auto filter = dynamic_cast<RelFilter*>(node)) {
      auto new_condition = visit(filter->getCondition());
      filter->setCondition(new_condition);
      return;
    }
    CHECK(false);
  }

 private:
  RetType getNonJoinInput(const RexInput& rex_in) const {
    const auto crt_source = rex_in.getSourceNode();
    const auto col_id = rex_in.getIndex();
    CHECK_LE(0, col_id);
    const auto join = dynamic_cast<const RelJoin*>(crt_source);
    if (!join || !join_set_.count(join)) {
      return boost::make_unique<RexInput>(crt_source, col_id);
    }
    const auto lhs = join->getInput(0);
    const auto rhs = join->getInput(1);
    CHECK(!dynamic_cast<const RelJoin*>(rhs));
    const auto src0_base = static_cast<unsigned>(lhs->size());
    if (col_id >= src0_base) {
      return boost::make_unique<RexInput>(rhs, col_id - src0_base);
    }
    if (dynamic_cast<const RelJoin*>(lhs)) {
      return getNonJoinInput(RexInput(lhs, col_id));
    }
    return boost::make_unique<RexInput>(lhs, col_id);
  }

  const std::unordered_set<const RelJoin*>& join_set_;
};

void redirect_inputs(std::shared_ptr<RelMultiJoin> multi_join,
                     const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>> du_web,
                     const std::unordered_map<const RelAlgNode*, std::shared_ptr<RelAlgNode>>& deconst_mapping) {
  CHECK(multi_join);
  CHECK_LE(size_t(2), multi_join->joinCount());
  std::unordered_set<const RelJoin*> joins;
  for (size_t i = 0; i < multi_join->joinCount(); ++i) {
    joins.insert(multi_join->getJoinAt(i));
  }
  RexInputRedirector redirector(joins);
  auto tail = multi_join->getJoinAt(multi_join->joinCount() - 1);
  auto usrs_it = du_web.find(tail);
  CHECK(usrs_it != du_web.end());
  for (auto usr : usrs_it->second) {
    auto usr_it = deconst_mapping.find(usr);
    CHECK(usr_it != deconst_mapping.end());
    redirector.visitNode(usr_it->second.get());
  }

  std::vector<std::unique_ptr<const RexScalar>> new_conditions;
  for (size_t i = 0; i < multi_join->joinCount(); ++i) {
    auto old_condition = multi_join->getConditions()[i].get();
    CHECK(old_condition);
    auto new_condition = redirector.visit(old_condition);
    new_conditions.push_back(std::move(new_condition));
  }
  multi_join->setConditions(new_conditions);
}

class JoinKeyReplacer : public RexDeepCopyVisitor {
 public:
  JoinKeyReplacer(const RelAlgNode* last_lhs,
                  const std::unordered_map<size_t, size_t>& lhs_new_numbering,
                  const RelAlgNode* last_rhs,
                  const std::unordered_map<size_t, size_t>& rhs_new_numbering)
      : lhs_(last_lhs),
        lhs_old_to_new_idx_(lhs_new_numbering),
        rhs_(last_rhs),
        rhs_old_to_new_idx_(rhs_new_numbering) {}
  RetType visitInput(const RexInput* input) const override {
    auto source = input->getSourceNode();
    if (source == lhs_) {
      auto renum_it = lhs_old_to_new_idx_.find(input->getIndex());
      if (renum_it != lhs_old_to_new_idx_.end()) {
        return boost::make_unique<RexInput>(input->getSourceNode(), renum_it->second);
      } else {
        return input->deepCopy();
      }
    }
    CHECK(source == rhs_);
    auto renum_it = rhs_old_to_new_idx_.find(input->getIndex());
    if (renum_it != lhs_old_to_new_idx_.end()) {
      return boost::make_unique<RexInput>(lhs_, renum_it->second);
    } else {
      return input->deepCopy();
    }
  }

 private:
  const RelAlgNode* lhs_;
  const std::unordered_map<size_t, size_t>& lhs_old_to_new_idx_;
  const RelAlgNode* rhs_;
  const std::unordered_map<size_t, size_t>& rhs_old_to_new_idx_;
};

std::unordered_map<const RelAlgNode*, size_t> get_node_to_col_base(const RelMultiJoin* multi_join) {
  std::unordered_map<const RelAlgNode*, size_t> node_to_col_base;
  for (size_t i = 0; i < multi_join->joinCount(); ++i) {
    auto join = multi_join->getJoinAt(i);
    auto lhs = join->getInput(0);
    auto rhs = join->getInput(1);
    node_to_col_base.insert(std::make_pair(lhs, 0));
    node_to_col_base.insert(std::make_pair(rhs, lhs->size()));
  }
  auto tail = multi_join->getJoinAt(multi_join->joinCount() - 1);
  node_to_col_base.insert(std::make_pair(tail, 0));
  return node_to_col_base;
}

void replace_equijoin_keys(const std::shared_ptr<RelMultiJoin> multi_join,
                           const std::vector<const RexScalar*>& condition_set,
                           const std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>> du_web,
                           const std::unordered_map<const RelAlgNode*, std::shared_ptr<RelAlgNode>>& deconst_mapping) {
  CHECK_LE(size_t(2), multi_join->joinCount());
  auto tail = multi_join->getJoinAt(multi_join->joinCount() - 1);
  auto last_lhs = tail->getInput(0);
  auto last_rhs = tail->getInput(1);
  auto usrs_it = du_web.find(tail);
  CHECK(usrs_it != du_web.end());
  CHECK_EQ(condition_set.size(), multi_join->joinCount());
  auto node_to_col_base = get_node_to_col_base(multi_join.get());
  std::unordered_map<size_t, size_t> lhs_old_to_new_idx;
  std::unordered_map<size_t, size_t> rhs_old_to_new_idx;
  std::unordered_map<size_t, size_t> old_to_new_flattened_idx;
  for (size_t i = 0; i < condition_set.size(); ++i) {
    auto rex_operator = dynamic_cast<const RexOperator*>(condition_set[i]);
    if (!rex_operator || !IS_EQUIVALENCE(rex_operator->getOperator())) {
      continue;
    }
    auto lhs_table = multi_join->getJoinAt(i)->getInput(0);
    auto rhs_table = multi_join->getJoinAt(i)->getInput(1);
    CHECK(node_to_col_base.count(lhs_table));
    CHECK(node_to_col_base.count(rhs_table));
    const auto lhs_col_base = node_to_col_base[lhs_table];
    const auto rhs_col_base = node_to_col_base[rhs_table];
    auto lhs = dynamic_cast<const RexInput*>(rex_operator->getOperand(0));
    auto rhs = dynamic_cast<const RexInput*>(rex_operator->getOperand(1));
    CHECK(lhs && rhs);
    if (lhs->getSourceNode() == lhs_table) {
      CHECK_EQ(rhs->getSourceNode(), rhs_table);
    } else {
      CHECK_EQ(rhs->getSourceNode(), lhs_table);
      CHECK_EQ(lhs->getSourceNode(), rhs_table);
      std::swap(lhs, rhs);
    }
    // Direct users of tail join need special handling due to indirect pointing of inputs.
    if (rhs->getSourceNode() == last_rhs) {
      rhs_old_to_new_idx.insert(std::make_pair(rhs->getIndex(), lhs->getIndex() + lhs_col_base));
    } else {
      lhs_old_to_new_idx.insert(std::make_pair(rhs->getIndex() + rhs_col_base, lhs->getIndex() + lhs_col_base));
    }
    old_to_new_flattened_idx.insert(std::make_pair(rhs->getIndex() + rhs_col_base, lhs->getIndex() + lhs_col_base));
  }

  JoinKeyReplacer replacer(last_lhs, lhs_old_to_new_idx, last_rhs, rhs_old_to_new_idx);
  std::vector<const RelAlgNode*> direct_usrs(usrs_it->second.begin(), usrs_it->second.end());
  std::vector<const RelAlgNode*> indirect_usrs;
  // Direct users of tail join need special handling due to indirect pointing of inputs.
  while (!direct_usrs.empty()) {
    auto walker = direct_usrs.back();
    direct_usrs.pop_back();
    auto walker_it = deconst_mapping.find(walker);
    CHECK(walker_it != deconst_mapping.end());
    if (auto project = std::dynamic_pointer_cast<RelProject>(walker_it->second)) {
      std::vector<std::unique_ptr<const RexScalar>> new_exprs;
      for (size_t i = 0; i < project->size(); ++i) {
        new_exprs.push_back(replacer.visit(project->getProjectAt(i)));
      }
      project->setExpressions(new_exprs);
      continue;
    }
    if (auto filter = std::dynamic_pointer_cast<RelFilter>(walker_it->second)) {
      auto new_condition = replacer.visit(filter->getCondition());
      filter->setCondition(new_condition);
      auto filter_usrs_it = du_web.find(filter.get());
      CHECK(filter_usrs_it != du_web.end());
      indirect_usrs.insert(indirect_usrs.end(), filter_usrs_it->second.begin(), filter_usrs_it->second.end());
      continue;
    }
    throw std::runtime_error(walker->toString() + " after join not supported yet");
  }

  RexInputRenumber<true> renumber(old_to_new_flattened_idx);
  while (!indirect_usrs.empty()) {
    auto walker = indirect_usrs.back();
    indirect_usrs.pop_back();
    auto walker_it = deconst_mapping.find(walker);
    CHECK(walker_it != deconst_mapping.end());
    if (auto project = std::dynamic_pointer_cast<RelProject>(walker_it->second)) {
      std::vector<std::unique_ptr<const RexScalar>> new_exprs;
      for (size_t i = 0; i < project->size(); ++i) {
        new_exprs.push_back(renumber.visit(project->getProjectAt(i)));
      }
      project->setExpressions(new_exprs);
      continue;
    }
    throw std::runtime_error(walker->toString() + " after join + filter not supported yet");
  }
}

void coalesce_joins(std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  std::unordered_map<const RelAlgNode*, std::shared_ptr<RelAlgNode>> deconst_mapping;
  for (auto node : nodes) {
    deconst_mapping.insert(std::make_pair(node.get(), node));
  }
  auto web = build_du_web(nodes);

  std::unordered_set<const RelAlgNode*> visited;
  std::vector<std::shared_ptr<RelAlgNode>> new_nodes;
  for (auto node : nodes) {
    if (visited.count(node.get())) {
      continue;
    }
    visited.insert(node.get());
    if (auto join = std::dynamic_pointer_cast<RelJoin>(node)) {
      std::vector<const RexScalar*> condition_set;
      auto sequence = collect_coalesceable_joins(join.get(), web, condition_set);
      if (sequence.size() < 2) {
        new_nodes.push_back(node);
        continue;
      }
      CHECK_EQ(deconst_mapping.count(sequence.back()), size_t(1));
      auto tail = deconst_mapping[sequence.back()];
      CHECK_EQ(web.count(tail.get()), size_t(1));
      for (auto usr : web[tail.get()]) {
        // TODO(miyu): check if safe to relax this limitation
        if (dynamic_cast<const RelJoin*>(usr) || dynamic_cast<const RelMultiJoin*>(usr)) {
          new_nodes.push_back(node);
          continue;
        }
      }
      // TODO(miyu): relax this limitation after we support duplicate input tables
      //             esp. for self-join.
      std::unordered_set<const RelAlgNode*> inputs_to_sequence;
      for (auto j : sequence) {
        if (j == sequence.front()) {
          inputs_to_sequence.insert(j->getInput(0));
        }
        inputs_to_sequence.insert(j->getInput(1));
      }
      if (inputs_to_sequence.size() != sequence.size() + 1) {
        new_nodes.push_back(node);
        continue;
      }

      visited.insert(sequence.begin(), sequence.end());
      std::vector<std::shared_ptr<RelJoin>> managed_sequence;
      for (auto j : sequence) {
        CHECK_EQ(deconst_mapping.count(j), size_t(1));
        auto mj = std::dynamic_pointer_cast<RelJoin>(deconst_mapping[j]);
        CHECK(mj);
        managed_sequence.push_back(mj);
      }
      auto multi_join = std::make_shared<RelMultiJoin>(managed_sequence);
      for (auto usr : web[tail.get()]) {
        CHECK_EQ(deconst_mapping.count(usr), size_t(1));
        deconst_mapping[usr]->replaceInput(tail, multi_join);
      }
      replace_equijoin_keys(multi_join, condition_set, web, deconst_mapping);
      redirect_inputs(multi_join, web, deconst_mapping);
      new_nodes.push_back(multi_join);
    } else {
      new_nodes.push_back(node);
    }
  }
  nodes.swap(new_nodes);
}
#endif

void coalesce_nodes(std::vector<std::shared_ptr<RelAlgNode>>& nodes,
                    const std::vector<const RelAlgNode*>& left_deep_joins) {
#ifdef ENABLE_EQUIJOIN_FOLD
  if (left_deep_joins.empty()) {
    coalesce_joins(nodes);
  }
#endif
  enum class CoalesceState { Initial, Filter, FirstProject, Aggregate };
  std::vector<size_t> crt_pattern;
  CoalesceState crt_state{CoalesceState::Initial};

  for (RANodeIterator nodeIt(nodes); !nodeIt.allVisited();) {
    const auto ra_node = nodeIt != nodes.end() ? *nodeIt : nullptr;
    switch (crt_state) {
      case CoalesceState::Initial: {
        if (std::dynamic_pointer_cast<const RelFilter>(ra_node) &&
            std::find(left_deep_joins.begin(), left_deep_joins.end(), ra_node.get()) == left_deep_joins.end()) {
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

// Create an in-memory, easy to navigate relational algebra DAG from its serialized,
// JSON representation. Also, apply high level optimizations which can be expressed
// through relational algebra extended with RelCompound. The RelCompound node is an
// equivalent representation for sequences of RelFilter, RelProject and RelAggregate
// nodes. This coalescing minimizes the amount of intermediate buffers required to
// evaluate a query. Lower level optimizations are taken care by lower levels, mainly
// RelAlgTranslator and the IR code generation.
class RelAlgAbstractInterpreter {
 public:
  RelAlgAbstractInterpreter(const rapidjson::Value& query_ast,
                            const Catalog_Namespace::Catalog& cat,
                            RelAlgExecutor* ra_executor)
      : query_ast_(query_ast), cat_(cat), ra_executor_(ra_executor) {}

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
    simplify_sort(nodes_);
    sink_projected_boolean_expr_to_join(nodes_);
    eliminate_identical_copy(nodes_);
    fold_filters(nodes_);
    std::vector<const RelAlgNode*> filtered_left_deep_joins;
    std::vector<const RelAlgNode*> left_deep_joins;
    for (const auto& node : nodes_) {
      const auto left_deep_join_root = get_left_deep_join_root(node);
      // The filter which starts a left-deep join pattern must not be coalesced
      // since it contains (part of) the join condition.
      if (left_deep_join_root) {
        left_deep_joins.push_back(left_deep_join_root.get());
        if (std::dynamic_pointer_cast<const RelFilter>(left_deep_join_root)) {
          filtered_left_deep_joins.push_back(left_deep_join_root.get());
        }
      }
    }
    if (filtered_left_deep_joins.empty()) {
      hoist_filter_cond_to_cross_join(nodes_);
    }
    eliminate_dead_columns(nodes_);
    coalesce_nodes(nodes_, left_deep_joins);
    CHECK(nodes_.back().unique());
    create_left_deep_join(nodes_);
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
      if (rel_op == std::string("EnumerableTableScan")) {
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
      } else if (rel_op == std::string("LogicalValues")) {
        ra_node = dispatchLogicalValues(crt_node);
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
      exprs.emplace_back(parse_scalar_expr(*exprs_json_it, cat_, ra_executor_));
    }
    const auto& fields = field(proj_ra, "fields");
    return std::make_shared<RelProject>(exprs, strings_from_json_array(fields), inputs.front());
  }

  std::shared_ptr<RelFilter> dispatchFilter(const rapidjson::Value& filter_ra) {
    const auto inputs = getRelAlgInputs(filter_ra);
    CHECK_EQ(size_t(1), inputs.size());
    const auto id = node_id(filter_ra);
    CHECK(id);
    auto condition = parse_scalar_expr(field(filter_ra, "condition"), cat_, ra_executor_);
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
    if (agg_ra.HasMember("groups") || agg_ra.HasMember("indicator")) {
      throw QueryNotSupported("GROUP BY extensions not supported");
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
    auto filter_rex = parse_scalar_expr(field(join_ra, "condition"), cat_, ra_executor_);
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
    auto limit = get_int_literal_field(sort_ra, "fetch", -1);
    if (limit == 0) {
      throw QueryNotSupported("LIMIT 0 not supported");
    }
    const auto offset = get_int_literal_field(sort_ra, "offset", 0);
    return std::make_shared<RelSort>(collation, limit > 0 ? limit : 0, offset, inputs.front());
  }

  std::shared_ptr<RelLogicalValues> dispatchLogicalValues(const rapidjson::Value& logical_values_ra) {
    const auto& tuple_type_arr = field(logical_values_ra, "type");
    CHECK(tuple_type_arr.IsArray());
    std::vector<TargetMetaInfo> tuple_type;
    for (auto tuple_type_arr_it = tuple_type_arr.Begin(); tuple_type_arr_it != tuple_type_arr.End();
         ++tuple_type_arr_it) {
      const auto component_type = parse_type(*tuple_type_arr_it);
      const auto component_name = json_str(field(*tuple_type_arr_it, "name"));
      tuple_type.emplace_back(component_name, component_type);
    }
    const auto& inputs_arr = field(logical_values_ra, "inputs");
    CHECK(inputs_arr.IsArray());
    const auto& tuples_arr = field(logical_values_ra, "tuples");
    CHECK(tuples_arr.IsArray());
    if (inputs_arr.Size() || tuples_arr.Size()) {
      throw QueryNotSupported("Non-empty LogicalValues not supported yet");
    }
    return std::make_shared<RelLogicalValues>(tuple_type);
  }

  const TableDescriptor* getTableFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto& table_json = field(scan_ra, "table");
    CHECK(table_json.IsArray());
    CHECK_EQ(unsigned(2), table_json.Size());
    const auto td = cat_.getMetadataForTable(table_json[1].GetString());
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
  RelAlgExecutor* ra_executor_;
};

std::shared_ptr<const RelAlgNode> ra_interpret(const rapidjson::Value& query_ast,
                                               const Catalog_Namespace::Catalog& cat,
                                               RelAlgExecutor* ra_executor) {
  RelAlgAbstractInterpreter interp(query_ast, cat, ra_executor);
  return interp.run();
}

std::unique_ptr<const RexSubQuery> parse_subquery(const rapidjson::Value& expr,
                                                  const Catalog_Namespace::Catalog& cat,
                                                  RelAlgExecutor* ra_executor) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(0));
  const auto& subquery_ast = field(expr, "subquery");

  const auto ra = ra_interpret(subquery_ast, cat, ra_executor);
  auto subquery = new RexSubQuery(ra);
  ra_executor->registerSubquery(subquery);
  return std::unique_ptr<const RexSubQuery>(subquery);
}

}  // namespace

// Driver for the query de-serialization and high level optimization.
std::shared_ptr<const RelAlgNode> deserialize_ra_dag(const std::string& query_ra,
                                                     const Catalog_Namespace::Catalog& cat,
                                                     RelAlgExecutor* ra_executor) {
  rapidjson::Document query_ast;
  query_ast.Parse(query_ra.c_str());
  CHECK(!query_ast.HasParseError());
  CHECK(query_ast.IsObject());
  RelAlgNode::resetRelAlgFirstId();
  return ra_interpret(query_ast, cat, ra_executor);
}

// Prints the relational algebra as a tree; useful for debugging.
std::string tree_string(const RelAlgNode* ra, const size_t indent) {
  std::string result = std::string(indent, ' ') + ra->toString() + "\n";
  for (size_t i = 0; i < ra->inputCount(); ++i) {
    result += tree_string(ra->getInput(i), indent + 2);
  }
  return result;
}
