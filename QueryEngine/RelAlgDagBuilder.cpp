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

#include "RelAlgDagBuilder.h"
#include "CalciteDeserializerUtils.h"
#include "DateTimePlusRewrite.h"
#include "DateTimeTranslator.h"
#include "DeepCopyVisitor.h"
#include "Descriptors/RelAlgExecutionDescriptor.h"
#include "ExpressionRewrite.h"
#include "ExtensionFunctionsBinding.h"
#include "ExtensionFunctionsWhitelist.h"
#include "JsonAccessors.h"
#include "RelAlgOptimizer.h"
#include "RelLeftDeepInnerJoin.h"
#include "Shared/sqldefs.h"

#include <rapidjson/error/en.h>
#include <rapidjson/error/error.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <boost/functional/hash.hpp>

#include <string>
#include <unordered_set>

using namespace std::literals;

namespace {

const unsigned FIRST_RA_NODE_ID = 1;

}  // namespace

thread_local unsigned RelAlgNode::crt_id_ = FIRST_RA_NODE_ID;

void RelAlgNode::resetRelAlgFirstId() noexcept {
  crt_id_ = FIRST_RA_NODE_ID;
}

namespace {

class RebindInputsVisitor : public DeepCopyVisitor {
 public:
  RebindInputsVisitor(const RelAlgNode* old_input, const RelAlgNode* new_input)
      : old_input_(old_input), new_input_(new_input) {}

  RetType visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    if (col_ref->getNode() == old_input_) {
      auto left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(new_input_);
      if (left_deep_join) {
        return rebind_inputs_from_left_deep_join(col_ref, left_deep_join);
      }
      return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          col_ref->get_type_info(), new_input_, col_ref->getIndex());
    }
    return col_ref->deep_copy();
  }

 protected:
  const RelAlgNode* old_input_;
  const RelAlgNode* new_input_;
};

class RebindReindexInputsVisitor : public RebindInputsVisitor {
 public:
  RebindReindexInputsVisitor(
      const RelAlgNode* old_input,
      const RelAlgNode* new_input,
      const std::optional<std::unordered_map<unsigned, unsigned>>& old_to_new_index_map)
      : RebindInputsVisitor(old_input, new_input), mapping_(old_to_new_index_map) {}

  RetType visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    auto res = RebindInputsVisitor::visitColumnRef(col_ref);
    if (mapping_) {
      auto new_col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(res.get());
      CHECK(new_col_ref);
      auto it = mapping_->find(new_col_ref->getIndex());
      CHECK(it != mapping_->end());
      return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          new_col_ref->get_type_info(), new_col_ref->getNode(), it->second);
    }
    return res;
  }

 protected:
  const std::optional<std::unordered_map<unsigned, unsigned>>& mapping_;
};

}  // namespace

void RelAlgNode::print() const {
  std::cout << toString() << std::endl;
}

void RelProject::replaceInput(
    std::shared_ptr<const RelAlgNode> old_input,
    std::shared_ptr<const RelAlgNode> input,
    std::optional<std::unordered_map<unsigned, unsigned>> old_to_new_index_map) {
  RelAlgNode::replaceInput(old_input, input);
  RebindReindexInputsVisitor visitor(old_input.get(), input.get(), old_to_new_index_map);
  for (size_t i = 0; i < exprs_.size(); ++i) {
    exprs_[i] = visitor.visit(exprs_[i].get());
  }
}

void RelProject::appendInput(std::string new_field_name, hdk::ir::ExprPtr expr) {
  fields_.emplace_back(std::move(new_field_name));
  exprs_.emplace_back(std::move(expr));
}

template <typename T>
bool is_one_of(const RelAlgNode* node) {
  return dynamic_cast<const T*>(node);
}

template <typename T1, typename T2, typename... Ts>
bool is_one_of(const RelAlgNode* node) {
  return dynamic_cast<const T1*>(node) || is_one_of<T2, Ts...>(node);
}

// TODO: always simply use node->size()
size_t getNodeColumnCount(const RelAlgNode* node) {
  // Nodes that don't depend on input.
  if (is_one_of<RelScan,
                RelProject,
                RelAggregate,
                RelCompound,
                RelTableFunction,
                RelLogicalUnion,
                RelLogicalValues>(node)) {
    return node->size();
  }

  // Nodes that preserve size.
  if (is_one_of<RelFilter, RelSort>(node)) {
    CHECK_EQ(size_t(1), node->inputCount());
    return getNodeColumnCount(node->getInput(0));
  }

  // Join concatenates the outputs from the inputs.
  if (is_one_of<RelJoin>(node)) {
    CHECK_EQ(size_t(2), node->inputCount());
    return getNodeColumnCount(node->getInput(0)) + getNodeColumnCount(node->getInput(1));
  }

  LOG(FATAL) << "Unhandled ra_node type: " << ::toString(node);
  return 0;
}

hdk::ir::ExprPtrVector genColumnRefs(const RelAlgNode* node, size_t count) {
  hdk::ir::ExprPtrVector res;
  res.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    res.emplace_back(
        hdk::ir::makeExpr<hdk::ir::ColumnRef>(getColumnType(node, i), node, i));
  }
  return res;
}

hdk::ir::ExprPtrVector getNodeColumnRefs(const RelAlgNode* node) {
  // Nodes that don't depend on input.
  if (is_one_of<RelScan,
                RelProject,
                RelAggregate,
                RelCompound,
                RelTableFunction,
                RelLogicalUnion,
                RelLogicalValues,
                RelFilter,
                RelSort>(node)) {
    return genColumnRefs(node, getNodeColumnCount(node));
  }

  if (is_one_of<RelJoin>(node)) {
    CHECK_EQ(size_t(2), node->inputCount());
    auto lhs_out =
        genColumnRefs(node->getInput(0), getNodeColumnCount(node->getInput(0)));
    const auto rhs_out =
        genColumnRefs(node->getInput(1), getNodeColumnCount(node->getInput(1)));
    lhs_out.insert(lhs_out.end(), rhs_out.begin(), rhs_out.end());
    return lhs_out;
  }

  LOG(FATAL) << "Unhandled ra_node type: " << ::toString(node);
  return {};
}

hdk::ir::ExprPtr getNodeColumnRef(const RelAlgNode* node, unsigned index) {
  if (is_one_of<RelScan,
                RelProject,
                RelAggregate,
                RelCompound,
                RelTableFunction,
                RelLogicalUnion,
                RelLogicalValues,
                RelFilter,
                RelSort>(node)) {
    CHECK_LT(index, node->size());
    return hdk::ir::makeExpr<hdk::ir::ColumnRef>(getColumnType(node, index), node, index);
  }

  if (is_one_of<RelJoin>(node)) {
    CHECK_EQ(size_t(2), node->inputCount());
    auto lhs_size = node->getInput(0)->size();
    if (index < lhs_size) {
      return getNodeColumnRef(node->getInput(0), index);
    }
    return getNodeColumnRef(node->getInput(1), index - lhs_size);
  }

  LOG(FATAL) << "Unhandled node type: " << ::toString(node);
  return nullptr;
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
  const auto source_shape = getNodeColumnRefs(source);
  if (source_shape.size() != exprs_.size()) {
    return false;
  }
  for (size_t i = 0; i < exprs_.size(); ++i) {
    const auto col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(exprs_[i].get());
    CHECK(col_ref);
    CHECK_EQ(source, col_ref->getNode());
    // We should add the additional check that input->getIndex() !=
    // source_shape[i].getIndex(), but Calcite doesn't generate the right
    // Sort-Project-Sort sequence when joins are involved.
    if (col_ref->getNode() !=
        dynamic_cast<const hdk::ir::ColumnRef*>(source_shape[i].get())->getNode()) {
      return false;
    }
  }
  return true;
}

namespace {

bool isRenamedInput(const RelAlgNode* node,
                    const size_t index,
                    const std::string& new_name) {
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

  if (auto table_func = dynamic_cast<const RelTableFunction*>(node)) {
    return new_name != table_func->getFieldName(index);
  }

  if (auto logical_values = dynamic_cast<const RelLogicalValues*>(node)) {
    const auto& tuple_type = logical_values->getTupleType();
    CHECK_LT(index, tuple_type.size());
    return new_name != tuple_type[index].get_resname();
  }

  CHECK(dynamic_cast<const RelSort*>(node) || dynamic_cast<const RelFilter*>(node) ||
        dynamic_cast<const RelLogicalUnion*>(node));
  return isRenamedInput(node->getInput(0), index, new_name);
}

}  // namespace

bool RelProject::isRenaming() const {
  if (!isSimple()) {
    return false;
  }
  CHECK_EQ(exprs_.size(), fields_.size());
  for (size_t i = 0; i < fields_.size(); ++i) {
    auto col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(exprs_[i].get());
    CHECK(col_ref);
    if (isRenamedInput(col_ref->getNode(), col_ref->getIndex(), fields_[i])) {
      return true;
    }
  }
  return false;
}

void RelAggregate::replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                                std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RebindInputsVisitor visitor(old_input.get(), input.get());
  for (size_t i = 0; i < aggs_.size(); ++i) {
    aggs_[i] = visitor.visit(aggs_[i].get());
  }
}

void RelJoin::replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                           std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  if (condition_) {
    RebindInputsVisitor visitor(old_input.get(), input.get());
    condition_ = visitor.visit(condition_.get());
  }
}

void RelFilter::replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                             std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RebindInputsVisitor visitor(old_input.get(), input.get());
  condition_ = visitor.visit(condition_.get());
}

void RelCompound::replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                               std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RebindInputsVisitor visitor(old_input.get(), input.get());
  if (filter_) {
    filter_ = visitor.visit(filter_.get());
  }
  for (size_t i = 0; i < groupby_exprs_.size(); ++i) {
    groupby_exprs_[i] = visitor.visit(groupby_exprs_[i].get());
  }
  for (size_t i = 0; i < exprs_.size(); ++i) {
    exprs_[i] = visitor.visit(exprs_[i].get());
  }
}

RelProject::RelProject(RelProject const& rhs)
    : RelAlgNode(rhs)
    , exprs_(rhs.exprs_)
    , fields_(rhs.fields_)
    , hint_applied_(false)
    , hints_(std::make_unique<Hints>()) {
  if (rhs.hint_applied_) {
    for (auto const& kv : *rhs.hints_) {
      addHint(kv.second);
    }
  }
}

RelLogicalValues::RelLogicalValues(RelLogicalValues const& rhs)
    : RelAlgNode(rhs), tuple_type_(rhs.tuple_type_), values_(rhs.values_) {}

RelFilter::RelFilter(RelFilter const& rhs)
    : RelAlgNode(rhs), condition_(rhs.condition_) {}

RelAggregate::RelAggregate(RelAggregate const& rhs)
    : RelAlgNode(rhs)
    , groupby_count_(rhs.groupby_count_)
    , aggs_(rhs.aggs_)
    , fields_(rhs.fields_)
    , hint_applied_(false)
    , hints_(std::make_unique<Hints>()) {
  if (rhs.hint_applied_) {
    for (auto const& kv : *rhs.hints_) {
      addHint(kv.second);
    }
  }
}

RelJoin::RelJoin(RelJoin const& rhs)
    : RelAlgNode(rhs)
    , condition_(rhs.condition_)
    , join_type_(rhs.join_type_)
    , hint_applied_(false)
    , hints_(std::make_unique<Hints>()) {
  if (rhs.hint_applied_) {
    for (auto const& kv : *rhs.hints_) {
      addHint(kv.second);
    }
  }
}

RelCompound::RelCompound(RelCompound const& rhs)
    : RelAlgNode(rhs)
    , filter_(rhs.filter_)
    , groupby_count_(rhs.groupby_count_)
    , fields_(rhs.fields_)
    , is_agg_(rhs.is_agg_)
    , groupby_exprs_(rhs.groupby_exprs_)
    , exprs_(rhs.exprs_)
    , hint_applied_(false)
    , hints_(std::make_unique<Hints>()) {
  if (rhs.hint_applied_) {
    for (auto const& kv : *rhs.hints_) {
      addHint(kv.second);
    }
  }
}

void RelTableFunction::replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                                    std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RebindInputsVisitor visitor(old_input.get(), input.get());
  for (size_t i = 0; i < table_func_input_exprs_.size(); ++i) {
    table_func_input_exprs_[i] = visitor.visit(table_func_input_exprs_[i].get());
  }
}

int32_t RelTableFunction::countConstantArgs() const {
  int32_t literal_args = 0;
  for (const auto& arg : table_func_input_exprs_) {
    if (hdk::ir::expr_is<hdk::ir::Constant>(arg)) {
      literal_args += 1;
    }
  }
  return literal_args;
}

RelTableFunction::RelTableFunction(RelTableFunction const& rhs)
    : RelAlgNode(rhs)
    , function_name_(rhs.function_name_)
    , fields_(rhs.fields_)
    , col_input_exprs_(rhs.col_input_exprs_)
    , table_func_input_exprs_(rhs.table_func_input_exprs_) {}

namespace std {
template <>
struct hash<std::pair<const RelAlgNode*, int>> {
  size_t operator()(const std::pair<const RelAlgNode*, int>& input_col) const {
    auto ptr_val = reinterpret_cast<const int64_t*>(&input_col.first);
    return static_cast<int64_t>(*ptr_val) ^ input_col.second;
  }
};
}  // namespace std

namespace {

std::set<std::pair<const RelAlgNode*, int>> get_equiv_cols(const RelAlgNode* node,
                                                           const size_t which_col) {
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
      if (auto col_ref =
              dynamic_cast<const hdk::ir::ColumnRef*>(project->getExpr(curr_col).get())) {
        const auto join_source = dynamic_cast<const RelJoin*>(only_source);
        if (join_source) {
          CHECK_EQ(size_t(2), join_source->inputCount());
          auto lhs = join_source->getInput(0);
          CHECK((col_ref->getIndex() < lhs->size() && lhs == col_ref->getNode()) ||
                join_source->getInput(1) == col_ref->getNode());
        } else {
          CHECK_EQ(col_ref->getNode(), only_source);
        }
        curr_col = col_ref->getIndex();
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

// class RelLogicalUnion methods

RelLogicalUnion::RelLogicalUnion(RelAlgInputs inputs, bool is_all)
    : RelAlgNode(std::move(inputs)), is_all_(is_all) {
  CHECK_EQ(2u, inputs_.size());
  if (!is_all_) {
    throw QueryNotSupported("UNION without ALL is not supported yet.");
  }
}

size_t RelLogicalUnion::size() const {
  return inputs_.front()->size();
}

std::string RelLogicalUnion::toString() const {
  return cat(::typeName(this), "(is_all(", is_all_, "))");
}

size_t RelLogicalUnion::toHash() const {
  if (!hash_) {
    hash_ = typeid(RelLogicalUnion).hash_code();
    boost::hash_combine(*hash_, is_all_);
  }
  return *hash_;
}

std::string RelLogicalUnion::getFieldName(const size_t i) const {
  if (auto const* input = dynamic_cast<RelCompound const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  } else if (auto const* input = dynamic_cast<RelProject const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  } else if (auto const* input = dynamic_cast<RelLogicalUnion const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  } else if (auto const* input = dynamic_cast<RelAggregate const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  } else if (auto const* input = dynamic_cast<RelScan const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  } else if (auto const* input =
                 dynamic_cast<RelTableFunction const*>(inputs_[0].get())) {
    return input->getFieldName(i);
  }
  UNREACHABLE() << "Unhandled input type: " << ::toString(inputs_.front());
  return {};
}

void RelLogicalUnion::checkForMatchingMetaInfoTypes() const {
  std::vector<TargetMetaInfo> const& tmis0 = inputs_[0]->getOutputMetainfo();
  std::vector<TargetMetaInfo> const& tmis1 = inputs_[1]->getOutputMetainfo();
  if (tmis0.size() != tmis1.size()) {
    VLOG(2) << "tmis0.size() = " << tmis0.size() << " != " << tmis1.size()
            << " = tmis1.size()";
    throw std::runtime_error("Subqueries of a UNION must have matching data types.");
  }
  for (size_t i = 0; i < tmis0.size(); ++i) {
    if (tmis0[i].get_type_info() != tmis1[i].get_type_info()) {
      SQLTypeInfo const& ti0 = tmis0[i].get_type_info();
      SQLTypeInfo const& ti1 = tmis1[i].get_type_info();
      VLOG(2) << "Types do not match for UNION:\n  tmis0[" << i
              << "].get_type_info().to_string() = " << ti0.to_string() << "\n  tmis1["
              << i << "].get_type_info().to_string() = " << ti1.to_string();
      if (ti0.is_dict_encoded_string() && ti1.is_dict_encoded_string() &&
          ti0.get_comp_param() != ti1.get_comp_param()) {
        throw std::runtime_error(
            "Taking the UNION of different text-encoded dictionaries is not yet "
            "supported. This may be resolved by using shared dictionaries. For example, "
            "by making one a shared dictionary reference to the other.");
      } else {
        throw std::runtime_error(
            "Subqueries of a UNION must have the exact same data types.");
      }
    }
  }
}

namespace {

unsigned node_id(const rapidjson::Value& ra_node) noexcept {
  const auto& id = field(ra_node, "id");
  return std::stoi(json_str(id));
}

std::string json_node_to_string(const rapidjson::Value& node) noexcept {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  node.Accept(writer);
  return buffer.GetString();
}

hdk::ir::ExprPtr parse_expr(const rapidjson::Value& expr,
                            int db_id,
                            SchemaProviderPtr schema_provider,
                            RelAlgDagBuilder& root_dag_builder,
                            const hdk::ir::ExprPtrVector& ra_output);

hdk::ir::TimeUnit precisionToTimeUnit(int precision) {
  switch (precision) {
    case 0:
      return hdk::ir::TimeUnit::kSecond;
    case 3:
      return hdk::ir::TimeUnit::kMilli;
    case 6:
      return hdk::ir::TimeUnit::kMicro;
    case 9:
      return hdk::ir::TimeUnit::kNano;
    default:
      throw std::runtime_error("Unsupported datetime precision: " +
                               std::to_string(precision));
  }
}

const hdk::ir::Type* buildType(hdk::ir::Context& ctx,
                               const std::string& type_name,
                               bool nullable,
                               int precision,
                               int scale) {
  if (type_name == std::string("BIGINT")) {
    return ctx.int64(nullable);
  }
  if (type_name == std::string("INTEGER")) {
    return ctx.int32(nullable);
  }
  if (type_name == std::string("TINYINT")) {
    return ctx.int8(nullable);
  }
  if (type_name == std::string("SMALLINT")) {
    return ctx.int16(nullable);
  }
  if (type_name == std::string("FLOAT")) {
    return ctx.fp32(nullable);
  }
  if (type_name == std::string("REAL")) {
    return ctx.fp32(nullable);
  }
  if (type_name == std::string("DOUBLE")) {
    return ctx.fp64(nullable);
  }
  if (type_name == std::string("DECIMAL")) {
    return ctx.decimal64(precision, scale, nullable);
  }
  if (type_name == std::string("CHAR") || type_name == std::string("VARCHAR")) {
    return ctx.text(nullable);
  }
  if (type_name == std::string("BOOLEAN")) {
    return ctx.boolean(nullable);
  }
  if (type_name == std::string("TIMESTAMP")) {
    return ctx.timestamp(precisionToTimeUnit(precision), nullable);
  }
  if (type_name == std::string("DATE")) {
    return ctx.date64(hdk::ir::TimeUnit::kSecond, nullable);
  }
  if (type_name == std::string("TIME")) {
    return ctx.time64(precisionToTimeUnit(precision), nullable);
  }
  if (type_name == std::string("NULL")) {
    return ctx.null();
  }
  if (type_name == std::string("ARRAY")) {
    return ctx.arrayVarLen(ctx.null(), 4, nullable);
  }
  if (type_name == std::string("INTERVAL_DAY") ||
      type_name == std::string("INTERVAL_HOUR") ||
      type_name == std::string("INTERVAL_MINUTE") ||
      type_name == std::string("INTERVAL_SECOND")) {
    return ctx.interval64(hdk::ir::TimeUnit::kMilli, nullable);
  }
  if (type_name == std::string("INTERVAL_MONTH") ||
      type_name == std::string("INTERVAL_YEAR")) {
    return ctx.interval64(hdk::ir::TimeUnit::kMonth, nullable);
  }
  if (type_name == std::string("TEXT")) {
    return ctx.text(nullable);
  }

  throw std::runtime_error("Unsupported type: " + type_name);
}

const hdk::ir::Type* parseType(const rapidjson::Value& type_obj) {
  if (type_obj.IsArray()) {
    throw QueryNotSupported("Composite types are not currently supported.");
  }
  CHECK(type_obj.IsObject() && type_obj.MemberCount() >= 2)
      << json_node_to_string(type_obj);
  auto type_name = json_str(field(type_obj, "type"));
  auto nullable = json_bool(field(type_obj, "nullable"));
  auto precision_it = type_obj.FindMember("precision");
  int precision =
      precision_it != type_obj.MemberEnd() ? json_i64(precision_it->value) : 0;
  auto scale_it = type_obj.FindMember("scale");
  int scale = scale_it != type_obj.MemberEnd() ? json_i64(scale_it->value) : 0;

  return buildType(hdk::ir::Context::defaultCtx(), type_name, nullable, precision, scale);
}

SqlWindowFunctionKind parse_window_function_kind(const std::string& name) {
  if (name == "ROW_NUMBER") {
    return SqlWindowFunctionKind::ROW_NUMBER;
  }
  if (name == "RANK") {
    return SqlWindowFunctionKind::RANK;
  }
  if (name == "DENSE_RANK") {
    return SqlWindowFunctionKind::DENSE_RANK;
  }
  if (name == "PERCENT_RANK") {
    return SqlWindowFunctionKind::PERCENT_RANK;
  }
  if (name == "CUME_DIST") {
    return SqlWindowFunctionKind::CUME_DIST;
  }
  if (name == "NTILE") {
    return SqlWindowFunctionKind::NTILE;
  }
  if (name == "LAG") {
    return SqlWindowFunctionKind::LAG;
  }
  if (name == "LEAD") {
    return SqlWindowFunctionKind::LEAD;
  }
  if (name == "FIRST_VALUE") {
    return SqlWindowFunctionKind::FIRST_VALUE;
  }
  if (name == "LAST_VALUE") {
    return SqlWindowFunctionKind::LAST_VALUE;
  }
  if (name == "AVG") {
    return SqlWindowFunctionKind::AVG;
  }
  if (name == "MIN") {
    return SqlWindowFunctionKind::MIN;
  }
  if (name == "MAX") {
    return SqlWindowFunctionKind::MAX;
  }
  if (name == "SUM") {
    return SqlWindowFunctionKind::SUM;
  }
  if (name == "COUNT") {
    return SqlWindowFunctionKind::COUNT;
  }
  if (name == "$SUM0") {
    return SqlWindowFunctionKind::SUM_INTERNAL;
  }
  throw std::runtime_error("Unsupported window function: " + name);
}

SortDirection parse_sort_direction(const rapidjson::Value& collation) {
  return json_str(field(collation, "direction")) == std::string("DESCENDING")
             ? SortDirection::Descending
             : SortDirection::Ascending;
}

NullSortedPosition parse_nulls_position(const rapidjson::Value& collation) {
  return json_str(field(collation, "nulls")) == std::string("FIRST")
             ? NullSortedPosition::First
             : NullSortedPosition::Last;
}

std::vector<hdk::ir::OrderEntry> parseWindowOrderCollation(const rapidjson::Value& arr) {
  std::vector<hdk::ir::OrderEntry> collation;
  size_t field_idx = 0;
  for (auto it = arr.Begin(); it != arr.End(); ++it, ++field_idx) {
    const auto sort_dir = parse_sort_direction(*it);
    const auto null_pos = parse_nulls_position(*it);
    collation.emplace_back(field_idx,
                           sort_dir == SortDirection::Descending,
                           null_pos == NullSortedPosition::First);
  }
  return collation;
}

struct WindowBound {
  bool unbounded;
  bool preceding;
  bool following;
  bool is_current_row;
  hdk::ir::ExprPtr offset;
  int order_key;
};

WindowBound parse_window_bound(const rapidjson::Value& window_bound_obj,
                               int db_id,
                               SchemaProviderPtr schema_provider,
                               RelAlgDagBuilder& root_dag_builder,
                               const hdk::ir::ExprPtrVector& ra_output) {
  CHECK(window_bound_obj.IsObject());
  WindowBound window_bound;
  window_bound.unbounded = json_bool(field(window_bound_obj, "unbounded"));
  window_bound.preceding = json_bool(field(window_bound_obj, "preceding"));
  window_bound.following = json_bool(field(window_bound_obj, "following"));
  window_bound.is_current_row = json_bool(field(window_bound_obj, "is_current_row"));
  const auto& offset_field = field(window_bound_obj, "offset");
  if (offset_field.IsObject()) {
    window_bound.offset =
        parse_expr(offset_field, db_id, schema_provider, root_dag_builder, ra_output);
  } else {
    CHECK(offset_field.IsNull());
  }
  window_bound.order_key = json_i64(field(window_bound_obj, "order_key"));
  return window_bound;
}

hdk::ir::ExprPtr parse_subquery_expr(const rapidjson::Value& expr,
                                     int db_id,
                                     SchemaProviderPtr schema_provider,
                                     RelAlgDagBuilder& root_dag_builder) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(0));
  const auto& subquery_ast = field(expr, "subquery");

  RelAlgDagBuilder subquery_dag(root_dag_builder, subquery_ast, db_id, schema_provider);
  auto node = subquery_dag.getRootNodeShPtr();
  auto subquery =
      hdk::ir::makeExpr<hdk::ir::ScalarSubquery>(getColumnType(node.get(), 0), node);
  root_dag_builder.registerSubquery(subquery);
  return subquery;
}

std::vector<std::string> strings_from_json_array(
    const rapidjson::Value& json_str_arr) noexcept {
  CHECK(json_str_arr.IsArray());
  std::vector<std::string> fields;
  for (auto json_str_arr_it = json_str_arr.Begin(); json_str_arr_it != json_str_arr.End();
       ++json_str_arr_it) {
    CHECK(json_str_arr_it->IsString());
    fields.emplace_back(json_str_arr_it->GetString());
  }
  return fields;
}

std::vector<size_t> indices_from_json_array(
    const rapidjson::Value& json_idx_arr) noexcept {
  CHECK(json_idx_arr.IsArray());
  std::vector<size_t> indices;
  for (auto json_idx_arr_it = json_idx_arr.Begin(); json_idx_arr_it != json_idx_arr.End();
       ++json_idx_arr_it) {
    CHECK(json_idx_arr_it->IsInt());
    CHECK_GE(json_idx_arr_it->GetInt(), 0);
    indices.emplace_back(json_idx_arr_it->GetInt());
  }
  return indices;
}

hdk::ir::ExprPtr parseInput(const rapidjson::Value& expr,
                            const hdk::ir::ExprPtrVector& ra_output) {
  const auto& input = field(expr, "input");
  CHECK_LT(json_i64(input), ra_output.size());
  return ra_output[json_i64(input)];
}

SQLTypeInfo build_type_info(const SQLTypes sql_type,
                            const int scale,
                            const int precision) {
  SQLTypeInfo ti(sql_type, 0, 0, true);
  if (ti.is_decimal()) {
    ti.set_scale(scale);
    ti.set_precision(precision);
  }
  return ti;
}

hdk::ir::ExprPtr parseLiteral(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  const auto& literal = field(expr, "literal");
  const auto type = to_sql_type(json_str(field(expr, "type")));
  const auto target_type = to_sql_type(json_str(field(expr, "target_type")));
  const auto scale = json_i64(field(expr, "scale"));
  const auto precision = json_i64(field(expr, "precision"));
  const auto type_scale = json_i64(field(expr, "type_scale"));
  const auto type_precision = json_i64(field(expr, "type_precision"));
  if (literal.IsNull()) {
    return hdk::ir::makeExpr<hdk::ir::Constant>(target_type, true, Datum{0});
  }

  auto lit_ti = build_type_info(type, scale, precision);
  auto target_ti = build_type_info(target_type, type_scale, type_precision);
  switch (type) {
    case kINT:
    case kBIGINT: {
      Datum d;
      d.bigintval = json_i64(literal);
      return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, d);
    }
    case kDECIMAL: {
      int64_t val = json_i64(literal);
      if (target_ti.is_fp() && !scale) {
        return make_fp_constant(val, target_ti);
      }
      auto lit_expr = scale ? Analyzer::analyzeFixedPtValue(val, scale, precision)
                            : Analyzer::analyzeIntValue(val);
      return lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kTEXT: {
      return Analyzer::analyzeStringValue(json_str(literal));
    }
    case kBOOLEAN: {
      Datum d;
      d.boolval = json_bool(literal);
      return hdk::ir::makeExpr<hdk::ir::Constant>(kBOOLEAN, false, d);
    }
    case kDOUBLE: {
      Datum d;
      if (literal.IsDouble()) {
        d.doubleval = json_double(literal);
      } else if (literal.IsInt64()) {
        d.doubleval = static_cast<double>(literal.GetInt64());
      } else if (literal.IsUint64()) {
        d.doubleval = static_cast<double>(literal.GetUint64());
      } else {
        UNREACHABLE() << "Unhandled type: " << literal.GetType();
      }
      auto lit_expr = hdk::ir::makeExpr<hdk::ir::Constant>(kDOUBLE, false, d);
      return lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH: {
      Datum d;
      d.bigintval = json_i64(literal);
      return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, d);
    }
    case kTIME:
    case kTIMESTAMP: {
      Datum d;
      d.bigintval = type == kTIMESTAMP && precision > 0 ? json_i64(literal)
                                                        : json_i64(literal) / 1000;
      return hdk::ir::makeExpr<hdk::ir::Constant>(
          SQLTypeInfo(type, precision, 0, false), false, d);
    }
    case kDATE: {
      Datum d;
      d.bigintval = json_i64(literal) * 24 * 3600;
      return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, d);
    }
    case kNULLT: {
      if (target_ti.is_array()) {
        hdk::ir::ExprPtrVector args;
        // defaulting to valid sub-type for convenience
        target_ti.set_subtype(kBOOLEAN);
        return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(target_ti, args, true);
      }
      return hdk::ir::makeExpr<hdk::ir::Constant>(target_type, true, Datum{0});
    }
    default: {
      LOG(FATAL) << "Unexpected literal type " << lit_ti.get_type_name();
    }
  }
  return nullptr;
}

hdk::ir::ExprPtr parse_case_expr(const rapidjson::Value& expr,
                                 int db_id,
                                 SchemaProviderPtr schema_provider,
                                 RelAlgDagBuilder& root_dag_builder,
                                 const hdk::ir::ExprPtrVector& ra_output) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(2));
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
  hdk::ir::ExprPtr else_expr;
  for (auto operands_it = operands.Begin(); operands_it != operands.End();) {
    auto when_expr =
        parse_expr(*operands_it++, db_id, schema_provider, root_dag_builder, ra_output);
    if (operands_it == operands.End()) {
      else_expr = std::move(when_expr);
      break;
    }
    auto then_expr =
        parse_expr(*operands_it++, db_id, schema_provider, root_dag_builder, ra_output);
    expr_list.emplace_back(std::move(when_expr), std::move(then_expr));
  }
  return Analyzer::normalizeCaseExpr(expr_list, else_expr, nullptr);
}

hdk::ir::ExprPtrVector parseExprArray(const rapidjson::Value& arr,
                                      int db_id,
                                      SchemaProviderPtr schema_provider,
                                      RelAlgDagBuilder& root_dag_builder,
                                      const hdk::ir::ExprPtrVector& ra_output) {
  hdk::ir::ExprPtrVector exprs;
  for (auto it = arr.Begin(); it != arr.End(); ++it) {
    exprs.emplace_back(
        parse_expr(*it, db_id, schema_provider, root_dag_builder, ra_output));
  }
  return exprs;
}

hdk::ir::ExprPtrVector parseWindowOrderExprs(const rapidjson::Value& arr,
                                             int db_id,
                                             SchemaProviderPtr schema_provider,
                                             RelAlgDagBuilder& root_dag_builder,
                                             const hdk::ir::ExprPtrVector& ra_output) {
  hdk::ir::ExprPtrVector exprs;
  for (auto it = arr.Begin(); it != arr.End(); ++it) {
    exprs.emplace_back(parse_expr(
        field(*it, "field"), db_id, schema_provider, root_dag_builder, ra_output));
  }
  return exprs;
}

hdk::ir::ExprPtr makeUOper(SQLOps op, hdk::ir::ExprPtr arg, const SQLTypeInfo& ti) {
  switch (op) {
    case kCAST: {
      CHECK_NE(kNULLT, ti.get_type());
      const auto& arg_ti = arg->get_type_info();
      if (arg_ti.is_string() && ti.is_string()) {
        return arg;
      }
      if (ti.is_time() || arg_ti.is_string()) {
        // TODO(alex): check and unify with the rest of the cases
        // Do not propogate encoding on small dates
        return ti.is_date_in_days() ? arg->add_cast(SQLTypeInfo(kDATE, false))
                                    : arg->add_cast(ti);
      }
      if (!arg_ti.is_string() && ti.is_string()) {
        return arg->add_cast(ti);
      }
      return std::make_shared<hdk::ir::UOper>(ti, false, op, arg);
    }
    case kNOT:
    case kISNULL: {
      return std::make_shared<hdk::ir::UOper>(kBOOLEAN, op, arg);
    }
    case kISNOTNULL: {
      auto is_null = std::make_shared<hdk::ir::UOper>(kBOOLEAN, kISNULL, arg);
      return std::make_shared<hdk::ir::UOper>(kBOOLEAN, kNOT, is_null);
    }
    case kMINUS: {
      return std::make_shared<hdk::ir::UOper>(arg->get_type_info(), false, kUMINUS, arg);
    }
    case kUNNEST: {
      const auto& arg_ti = arg->get_type_info();
      CHECK(arg_ti.is_array());
      return hdk::ir::makeExpr<hdk::ir::UOper>(
          arg_ti.get_elem_type(), false, kUNNEST, arg);
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

hdk::ir::ExprPtr maybeMakeDateExpr(SQLOps op,
                                   const hdk::ir::ExprPtrVector& operands,
                                   const SQLTypeInfo& ti) {
  if (op != kPLUS && op != kMINUS) {
    return nullptr;
  }
  if (operands.size() != 2) {
    return nullptr;
  }

  auto& lhs = operands[0];
  auto& rhs = operands[1];
  auto& lhs_ti = lhs->get_type_info();
  auto& rhs_ti = rhs->get_type_info();
  if (!lhs_ti.is_timestamp() && !lhs_ti.is_date()) {
    if (lhs_ti.get_type() == kTIME) {
      throw std::runtime_error("DateTime addition/subtraction not supported for TIME.");
    }
    return nullptr;
  }
  if (rhs_ti.get_type() == kTIMESTAMP || rhs_ti.get_type() == kDATE) {
    if (lhs_ti.is_high_precision_timestamp() || rhs_ti.is_high_precision_timestamp()) {
      throw std::runtime_error(
          "High Precision timestamps are not supported for TIMESTAMPDIFF operation. "
          "Use "
          "DATEDIFF.");
    }
    auto bigint_ti = SQLTypeInfo(kBIGINT, false);
    const auto datediff_field =
        (ti.get_type() == kINTERVAL_DAY_TIME) ? dtSECOND : dtMONTH;
    auto result =
        hdk::ir::makeExpr<hdk::ir::DatediffExpr>(bigint_ti, datediff_field, rhs, lhs);
    // multiply 1000 to result since expected result should be in millisecond precision.
    if (ti.get_type() == kINTERVAL_DAY_TIME) {
      return hdk::ir::makeExpr<hdk::ir::BinOper>(
          bigint_ti.get_type(),
          kMULTIPLY,
          kONE,
          result,
          hdk::ir::Constant::make(bigint_ti, 1000));
    } else {
      return result;
    }
  }
  if (op == kPLUS) {
    auto dt_plus =
        hdk::ir::makeExpr<hdk::ir::FunctionOper>(lhs_ti, "DATETIME_PLUS", operands);
    const auto date_trunc = rewrite_to_date_trunc(dt_plus.get());
    if (date_trunc) {
      return date_trunc;
    }
  }
  const auto interval = fold_expr(rhs.get());
  auto interval_ti = interval->get_type_info();
  auto bigint_ti = SQLTypeInfo(kBIGINT, false);
  const auto interval_lit = std::dynamic_pointer_cast<hdk::ir::Constant>(interval);
  if (interval_ti.get_type() == kINTERVAL_DAY_TIME) {
    hdk::ir::ExprPtr interval_sec;
    if (interval_lit) {
      interval_sec = hdk::ir::Constant::make(
          bigint_ti,
          (op == kMINUS ? -interval_lit->get_constval().bigintval
                        : interval_lit->get_constval().bigintval) /
              1000);
    } else {
      interval_sec =
          hdk::ir::makeExpr<hdk::ir::BinOper>(bigint_ti.get_type(),
                                              kDIVIDE,
                                              kONE,
                                              interval,
                                              hdk::ir::Constant::make(bigint_ti, 1000));
      if (op == kMINUS) {
        interval_sec =
            std::make_shared<hdk::ir::UOper>(bigint_ti, false, kUMINUS, interval_sec);
      }
    }
    return hdk::ir::makeExpr<hdk::ir::DateaddExpr>(lhs_ti, daSECOND, interval_sec, lhs);
  }
  CHECK(interval_ti.get_type() == kINTERVAL_YEAR_MONTH);
  const auto interval_months =
      op == kMINUS ? std::make_shared<hdk::ir::UOper>(bigint_ti, false, kUMINUS, interval)
                   : interval;
  return hdk::ir::makeExpr<hdk::ir::DateaddExpr>(lhs_ti, daMONTH, interval_months, lhs);
}

std::pair<hdk::ir::ExprPtr, SQLQualifier> getQuantifiedBinOperRhs(
    const hdk::ir::ExprPtr& expr,
    hdk::ir::ExprPtr orig_expr) {
  if (auto fn_oper = dynamic_cast<const hdk::ir::FunctionOper*>(expr.get())) {
    auto& fn_name = fn_oper->getName();
    if (fn_name == "PG_ANY" || fn_name == "PG_ALL") {
      CHECK_EQ(fn_oper->getArity(), (size_t)1);
      return std::make_pair(fn_oper->getOwnArg(0), (fn_name == "PG_ANY") ? kANY : kALL);
    }
  } else if (auto uoper = dynamic_cast<const hdk::ir::UOper*>(expr.get())) {
    if (uoper->get_optype() == kCAST) {
      return getQuantifiedBinOperRhs(uoper->get_own_operand(), orig_expr);
    }
  }

  return std::make_pair(orig_expr, kONE);
}

std::pair<hdk::ir::ExprPtr, SQLQualifier> getQuantifiedBinOperRhs(
    const hdk::ir::ExprPtr& expr) {
  return getQuantifiedBinOperRhs(expr, expr);
}

bool supportedLowerBound(const WindowBound& window_bound) {
  return window_bound.unbounded && window_bound.preceding && !window_bound.following &&
         !window_bound.is_current_row && !window_bound.offset &&
         window_bound.order_key == 0;
}

bool supportedUpperBound(const WindowBound& window_bound,
                         SqlWindowFunctionKind kind,
                         const hdk::ir::ExprPtrVector& order_keys) {
  const bool to_current_row = !window_bound.unbounded && !window_bound.preceding &&
                              !window_bound.following && window_bound.is_current_row &&
                              !window_bound.offset && window_bound.order_key == 1;
  switch (kind) {
    case SqlWindowFunctionKind::ROW_NUMBER:
    case SqlWindowFunctionKind::RANK:
    case SqlWindowFunctionKind::DENSE_RANK:
    case SqlWindowFunctionKind::CUME_DIST: {
      return to_current_row;
    }
    default: {
      return order_keys.empty()
                 ? (window_bound.unbounded && !window_bound.preceding &&
                    window_bound.following && !window_bound.is_current_row &&
                    !window_bound.offset && window_bound.order_key == 2)
                 : to_current_row;
    }
  }
}

hdk::ir::ExprPtr parseWindowFunction(const rapidjson::Value& json_expr,
                                     const std::string& op_name,
                                     const hdk::ir::ExprPtrVector& operands,
                                     SQLTypeInfo ti,
                                     int db_id,
                                     SchemaProviderPtr schema_provider,
                                     RelAlgDagBuilder& root_dag_builder,
                                     const hdk::ir::ExprPtrVector& ra_output) {
  const auto& partition_keys_arr = field(json_expr, "partition_keys");
  auto partition_keys = parseExprArray(
      partition_keys_arr, db_id, schema_provider, root_dag_builder, ra_output);
  const auto& order_keys_arr = field(json_expr, "order_keys");
  auto order_keys = parseWindowOrderExprs(
      order_keys_arr, db_id, schema_provider, root_dag_builder, ra_output);
  const auto collation = parseWindowOrderCollation(order_keys_arr);
  const auto kind = parse_window_function_kind(op_name);
  // Adjust type for SUM window function.
  if (kind == SqlWindowFunctionKind::SUM_INTERNAL && ti.is_integer()) {
    ti = SQLTypeInfo(kBIGINT, ti.get_notnull());
  }
  const auto lower_bound = parse_window_bound(field(json_expr, "lower_bound"),
                                              db_id,
                                              schema_provider,
                                              root_dag_builder,
                                              ra_output);
  const auto upper_bound = parse_window_bound(field(json_expr, "upper_bound"),
                                              db_id,
                                              schema_provider,
                                              root_dag_builder,
                                              ra_output);
  bool is_rows = json_bool(field(json_expr, "is_rows"));
  ti.set_notnull(false);

  if (!supportedLowerBound(lower_bound) ||
      !supportedUpperBound(upper_bound, kind, order_keys) ||
      ((kind == SqlWindowFunctionKind::ROW_NUMBER) != is_rows)) {
    throw std::runtime_error("Frame specification not supported");
  }

  if (window_function_is_value(kind)) {
    CHECK_GE(operands.size(), 1u);
    ti = operands[0]->get_type_info();
  }

  return hdk::ir::makeExpr<hdk::ir::WindowFunction>(
      ti, kind, operands, partition_keys, order_keys, collation);
}

hdk::ir::ExprPtr parseLike(const std::string& fn_name,
                           const hdk::ir::ExprPtrVector& operands) {
  CHECK(operands.size() == 2 || operands.size() == 3);
  auto& arg = operands[0];
  auto& like = operands[1];
  if (!dynamic_cast<const hdk::ir::Constant*>(like.get())) {
    throw std::runtime_error("The matching pattern must be a literal.");
  }
  auto escape = (operands.size() == 3) ? operands[2] : nullptr;
  bool is_ilike = fn_name == "PG_ILIKE"sv;
  return Analyzer::getLikeExpr(arg, like, escape, is_ilike, false);
}

hdk::ir::ExprPtr parseRegexp(const hdk::ir::ExprPtrVector& operands) {
  CHECK(operands.size() == 2 || operands.size() == 3);
  auto& arg = operands[0];
  auto& pattern = operands[1];
  if (!dynamic_cast<const hdk::ir::Constant*>(pattern.get())) {
    throw std::runtime_error("The matching pattern must be a literal.");
  }
  const auto escape = (operands.size() == 3) ? operands[2] : nullptr;
  return Analyzer::getRegexpExpr(arg, pattern, escape, false);
}

hdk::ir::ExprPtr parseLikely(const hdk::ir::ExprPtrVector& operands) {
  CHECK(operands.size() == 1);
  return hdk::ir::makeExpr<hdk::ir::LikelihoodExpr>(operands[0], 0.9375);
}

hdk::ir::ExprPtr parseUnlikely(const hdk::ir::ExprPtrVector& operands) {
  CHECK(operands.size() == 1);
  return hdk::ir::makeExpr<hdk::ir::LikelihoodExpr>(operands[0], 0.0625);
}

inline void validateDatetimeDatepartArgument(const hdk::ir::Constant* literal_expr) {
  if (!literal_expr || literal_expr->get_is_null()) {
    throw std::runtime_error("The 'DatePart' argument must be a not 'null' literal.");
  }
}

hdk::ir::ExprPtr parseExtract(const std::string& fn_name,
                              const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(2));
  auto& timeunit = operands[0];
  auto timeunit_lit = dynamic_cast<const hdk::ir::Constant*>(timeunit.get());
  validateDatetimeDatepartArgument(timeunit_lit);
  auto& from_expr = operands[1];
  if (fn_name == "PG_DATE_TRUNC"sv) {
    return DateTruncExpr::generate(from_expr, *timeunit_lit->get_constval().stringval);
  } else {
    CHECK(fn_name == "PG_EXTRACT"sv);
    return ExtractExpr::generate(from_expr, *timeunit_lit->get_constval().stringval);
  }
}

hdk::ir::ExprPtr parseDateadd(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(3));
  auto& timeunit = operands[0];
  auto timeunit_lit = dynamic_cast<const hdk::ir::Constant*>(timeunit.get());
  validateDatetimeDatepartArgument(timeunit_lit);
  auto& number_units = operands[1];
  auto number_units_const = dynamic_cast<const hdk::ir::Constant*>(number_units.get());
  if (number_units_const && number_units_const->get_is_null()) {
    throw std::runtime_error("The 'Interval' argument literal must not be 'null'.");
  }
  auto cast_number_units = number_units->add_cast(SQLTypeInfo(kBIGINT, false));
  auto& datetime = operands[2];
  auto& datetime_ti = datetime->get_type_info();
  if (datetime_ti.get_type() == kTIME) {
    throw std::runtime_error("DateAdd operation not supported for TIME.");
  }
  auto field = to_dateadd_field(*timeunit_lit->get_constval().stringval);
  int dim = datetime_ti.get_dimension();
  return hdk::ir::makeExpr<hdk::ir::DateaddExpr>(
      SQLTypeInfo(kTIMESTAMP, dim, 0, false), field, cast_number_units, datetime);
}

hdk::ir::ExprPtr parseDatediff(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(3));
  auto& timeunit = operands[0];
  auto timeunit_lit = dynamic_cast<const hdk::ir::Constant*>(timeunit.get());
  validateDatetimeDatepartArgument(timeunit_lit);
  auto& start = operands[1];
  auto& end = operands[2];
  auto field = to_datediff_field(*timeunit_lit->get_constval().stringval);
  return hdk::ir::makeExpr<hdk::ir::DatediffExpr>(
      SQLTypeInfo(kBIGINT, false), field, start, end);
}

hdk::ir::ExprPtr parseDatepart(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(2));
  auto& timeunit = operands[0];
  auto timeunit_lit = dynamic_cast<const hdk::ir::Constant*>(timeunit.get());
  validateDatetimeDatepartArgument(timeunit_lit);
  auto& from_expr = operands[1];
  return ExtractExpr::generate(
      from_expr, to_datepart_field(*timeunit_lit->get_constval().stringval));
}

hdk::ir::ExprPtr parseLength(const std::string& fn_name,
                             const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  auto& str_arg = operands[0];
  return hdk::ir::makeExpr<hdk::ir::CharLengthExpr>(str_arg->decompress(),
                                                    fn_name == "CHAR_LENGTH"sv);
}

hdk::ir::ExprPtr parseKeyForString(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  auto& arg_ti = operands[0]->get_type_info();
  if (!arg_ti.is_string() || arg_ti.is_varlen()) {
    throw std::runtime_error("KEY_FOR_STRING expects a dictionary encoded text column.");
  }
  return hdk::ir::makeExpr<hdk::ir::KeyForStringExpr>(operands[0]);
}

hdk::ir::ExprPtr parseWidthBucket(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(4));
  auto target_value = operands[0];
  auto lower_bound = operands[1];
  auto upper_bound = operands[2];
  auto partition_count = operands[3];
  if (!partition_count->get_type_info().is_integer()) {
    throw std::runtime_error(
        "PARTITION_COUNT expression of width_bucket function expects an integer type.");
  }
  auto check_numeric_type =
      [](const std::string& col_name, const hdk::ir::Expr* expr, bool allow_null_type) {
        if (expr->get_type_info().get_type() == kNULLT) {
          if (!allow_null_type) {
            throw std::runtime_error(
                col_name + " expression of width_bucket function expects non-null type.");
          }
          return;
        }
        if (!expr->get_type_info().is_number()) {
          throw std::runtime_error(
              col_name + " expression of width_bucket function expects a numeric type.");
        }
      };
  // target value may have null value
  check_numeric_type("TARGET_VALUE", target_value.get(), true);
  check_numeric_type("LOWER_BOUND", lower_bound.get(), false);
  check_numeric_type("UPPER_BOUND", upper_bound.get(), false);

  auto cast_to_double_if_necessary = [](hdk::ir::ExprPtr arg) {
    const auto& arg_ti = arg->get_type_info();
    if (arg_ti.get_type() != kDOUBLE) {
      const auto& double_ti = SQLTypeInfo(kDOUBLE, arg_ti.get_notnull());
      return arg->add_cast(double_ti);
    }
    return arg;
  };
  target_value = cast_to_double_if_necessary(target_value);
  lower_bound = cast_to_double_if_necessary(lower_bound);
  upper_bound = cast_to_double_if_necessary(upper_bound);
  return hdk::ir::makeExpr<hdk::ir::WidthBucketExpr>(
      target_value, lower_bound, upper_bound, partition_count);
}

hdk::ir::ExprPtr parseSampleRatio(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  auto arg = operands[0];
  auto& arg_ti = operands[0]->get_type_info();
  if (arg_ti.get_type() != kDOUBLE) {
    const auto& double_ti = SQLTypeInfo(kDOUBLE, arg_ti.get_notnull());
    arg = arg->add_cast(double_ti);
  }
  return hdk::ir::makeExpr<hdk::ir::SampleRatioExpr>(arg);
}

hdk::ir::ExprPtr parseCurrentUser() {
  std::string user{"SESSIONLESS_USER"};
  return Analyzer::getUserLiteral(user);
}

hdk::ir::ExprPtr parseLower(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  if (operands[0]->get_type_info().is_dict_encoded_string() ||
      dynamic_cast<const hdk::ir::Constant*>(operands[0].get())) {
    return hdk::ir::makeExpr<hdk::ir::LowerExpr>(operands[0]);
  }

  throw std::runtime_error(
      "LOWER expects a dictionary encoded text column or a literal.");
}

hdk::ir::ExprPtr parseCardinality(const std::string& fn_name,
                                  const hdk::ir::ExprPtrVector& operands,
                                  const SQLTypeInfo& ti) {
  CHECK_EQ(operands.size(), size_t(1));
  auto& arg = operands[0];
  auto& arg_ti = arg->get_type_info();
  if (!arg_ti.is_array()) {
    throw std::runtime_error(fn_name + " expects an array expression.");
  }
  if (arg_ti.get_subtype() == kARRAY) {
    throw std::runtime_error(fn_name + " expects one-dimension array expression.");
  }
  auto array_size = arg_ti.get_size();
  auto array_elem_size = arg_ti.get_elem_type().get_array_context_logical_size();

  if (array_size > 0) {
    if (array_elem_size <= 0) {
      throw std::runtime_error(fn_name + ": unexpected array element type.");
    }
    // Return cardinality of a fixed length array
    return hdk::ir::Constant::make(ti, array_size / array_elem_size);
  }
  // Variable length array cardinality will be calculated at runtime
  return hdk::ir::makeExpr<hdk::ir::CardinalityExpr>(arg);
}

hdk::ir::ExprPtr parseItem(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(2));
  auto& base = operands[0];
  auto& index = operands[1];
  return hdk::ir::makeExpr<hdk::ir::BinOper>(
      base->get_type_info().get_elem_type(), false, kARRAY_AT, kONE, base, index);
}

hdk::ir::ExprPtr parseCurrentDate(time_t now) {
  constexpr bool is_null = false;
  Datum datum;
  datum.bigintval = now - now % (24 * 60 * 60);  // Assumes 0 < now.
  return hdk::ir::makeExpr<hdk::ir::Constant>(kDATE, is_null, datum);
}

hdk::ir::ExprPtr parseCurrentTime(time_t now) {
  constexpr bool is_null = false;
  Datum datum;
  datum.bigintval = now % (24 * 60 * 60);  // Assumes 0 < now.
  return hdk::ir::makeExpr<hdk::ir::Constant>(kTIME, is_null, datum);
}

hdk::ir::ExprPtr parseCurrentTimestamp(time_t now) {
  Datum d;
  d.bigintval = now;
  return hdk::ir::makeExpr<hdk::ir::Constant>(kTIMESTAMP, false, d, false);
}

hdk::ir::ExprPtr parseDatetime(const hdk::ir::ExprPtrVector& operands, time_t now) {
  CHECK_EQ(operands.size(), size_t(1));
  auto arg_lit = dynamic_cast<const hdk::ir::Constant*>(operands[0].get());
  const std::string datetime_err{R"(Only DATETIME('NOW') supported for now.)"};
  if (!arg_lit || arg_lit->get_is_null()) {
    throw std::runtime_error(datetime_err);
  }
  CHECK(arg_lit->get_type_info().is_string());
  if (*arg_lit->get_constval().stringval != "NOW"sv) {
    throw std::runtime_error(datetime_err);
  }
  return parseCurrentTimestamp(now);
}

hdk::ir::ExprPtr parseHPTLiteral(const hdk::ir::ExprPtrVector& operands,
                                 const SQLTypeInfo& ti) {
  /* since calcite uses Avatica package called DateTimeUtils to parse timestamp strings.
     Therefore any string having fractional seconds more 3 places after the decimal
     (milliseconds) will get truncated to 3 decimal places, therefore we lose precision
     (us|ns). Issue: [BE-2461] Here we are hijacking literal cast to Timestamp(6|9) from
     calcite and translating them to generate our own casts.
  */
  CHECK_EQ(operands.size(), size_t(1));
  auto& arg = operands[0];
  auto& arg_ti = arg->get_type_info();
  if (!arg_ti.is_string()) {
    throw std::runtime_error(
        "High precision timestamp cast argument must be a string. Input type is: " +
        arg_ti.get_type_name());
  } else if (!ti.is_high_precision_timestamp()) {
    throw std::runtime_error(
        "Cast target type should be high precision timestamp. Input type is: " +
        ti.get_type_name());
  } else if (ti.get_dimension() != 6 && ti.get_dimension() != 9) {
    throw std::runtime_error(
        "Cast target type should be TIMESTAMP(6|9). Input type is: TIMESTAMP(" +
        std::to_string(ti.get_dimension()) + ")");
  }

  return arg->add_cast(ti);
}

hdk::ir::ExprPtr parseAbs(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
  auto& arg = operands[0];
  auto& arg_ti = arg->get_type_info();
  CHECK(arg_ti.is_number());
  auto zero = hdk::ir::Constant::make(arg_ti, 0);
  auto lt_zero = Analyzer::normalizeOperExpr(kLT, kONE, arg, zero);
  auto uminus_operand = hdk::ir::makeExpr<hdk::ir::UOper>(arg_ti, false, kUMINUS, arg);
  expr_list.emplace_back(lt_zero, uminus_operand);
  return Analyzer::normalizeCaseExpr(expr_list, arg, nullptr);
}

hdk::ir::ExprPtr parseSign(const hdk::ir::ExprPtrVector& operands) {
  CHECK_EQ(operands.size(), size_t(1));
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
  auto& arg = operands[0];
  auto& arg_ti = arg->get_type_info();
  CHECK(arg_ti.is_number());
  // For some reason, Rex based DAG checker marks SIGN as non-cacheable.
  // To duplicate this behavior with no Rex, non-cacheable zero constant
  // is used here.
  // TODO: revise this part in checker and probably remove this flag here.
  const auto zero = hdk::ir::Constant::make(arg_ti, 0, false);
  const auto lt_zero =
      hdk::ir::makeExpr<hdk::ir::BinOper>(kBOOLEAN, kLT, kONE, arg, zero);
  expr_list.emplace_back(lt_zero, hdk::ir::Constant::make(arg_ti, -1));
  const auto eq_zero =
      hdk::ir::makeExpr<hdk::ir::BinOper>(kBOOLEAN, kEQ, kONE, arg, zero);
  expr_list.emplace_back(eq_zero, hdk::ir::Constant::make(arg_ti, 0));
  const auto gt_zero =
      hdk::ir::makeExpr<hdk::ir::BinOper>(kBOOLEAN, kGT, kONE, arg, zero);
  expr_list.emplace_back(gt_zero, hdk::ir::Constant::make(arg_ti, 1));
  return Analyzer::normalizeCaseExpr(expr_list, nullptr, nullptr);
}

hdk::ir::ExprPtr parseRound(const std::string& fn_name,
                            const hdk::ir::ExprPtrVector& operands,
                            const SQLTypeInfo& ti) {
  auto args = operands;

  if (args.size() == 1) {
    // push a 0 constant if 2nd operand is missing.
    // this needs to be done as calcite returns
    // only the 1st operand without defaulting the 2nd one
    // when the user did not specify the 2nd operand.
    SQLTypes t = kSMALLINT;
    Datum d;
    d.smallintval = 0;
    args.push_back(hdk::ir::makeExpr<hdk::ir::Constant>(t, false, d));
  }

  // make sure we have only 2 operands
  CHECK(args.size() == 2);

  if (!args[0]->get_type_info().is_number()) {
    throw std::runtime_error("Only numeric 1st operands are supported");
  }

  // the 2nd operand does not need to be a constant
  // it can happily reference another integer column
  if (!args[1]->get_type_info().is_integer()) {
    throw std::runtime_error("Only integer 2nd operands are supported");
  }

  // Calcite may upcast decimals in a way that is
  // incompatible with the extension function input. Play it safe and stick with the
  // argument type instead.
  const SQLTypeInfo ret_ti =
      args[0]->get_type_info().is_decimal() ? args[0]->get_type_info() : ti;

  return hdk::ir::makeExpr<hdk::ir::FunctionOperWithCustomTypeHandling>(
      ret_ti, fn_name, args);
}

hdk::ir::ExprPtr parseArrayFunction(const hdk::ir::ExprPtrVector& operands,
                                    const SQLTypeInfo& ti) {
  if (ti.get_subtype() == kNULLT) {
    auto res_type = ti;
    CHECK(res_type.get_type() == kARRAY);

    // FIX-ME:  Deal with NULL arrays
    if (operands.size() > 0) {
      const auto first_element_logical_type =
          get_nullable_logical_type_info(operands[0]->get_type_info());

      auto diff_elem_itr =
          std::find_if(operands.begin(),
                       operands.end(),
                       [first_element_logical_type](const auto expr) {
                         return first_element_logical_type !=
                                get_nullable_logical_type_info(expr->get_type_info());
                       });
      if (diff_elem_itr != operands.end()) {
        throw std::runtime_error(
            "Element " + std::to_string(diff_elem_itr - operands.begin()) +
            " is not of the same type as other elements of the array. Consider casting "
            "to force this condition.\nElement Type: " +
            get_nullable_logical_type_info((*diff_elem_itr)->get_type_info())
                .to_string() +
            "\nArray type: " + first_element_logical_type.to_string());
      }

      if (first_element_logical_type.is_string() &&
          !first_element_logical_type.is_dict_encoded_string()) {
        res_type.set_subtype(first_element_logical_type.get_type());
        res_type.set_compression(kENCODING_FIXED);
      } else if (first_element_logical_type.is_dict_encoded_string()) {
        res_type.set_subtype(first_element_logical_type.get_type());
        res_type.set_comp_param(TRANSIENT_DICT_ID);
      } else {
        res_type.set_subtype(first_element_logical_type.get_type());
        res_type.set_scale(first_element_logical_type.get_scale());
        res_type.set_precision(first_element_logical_type.get_precision());
      }

      return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(res_type, operands);
    } else {
      // defaulting to valid sub-type for convenience
      res_type.set_subtype(kBOOLEAN);
      return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(res_type, operands);
    }
  } else {
    return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(ti, operands);
  }
}

hdk::ir::ExprPtr parseFunctionOperator(const std::string& fn_name,
                                       const hdk::ir::ExprPtrVector operands,
                                       const SQLTypeInfo& ti,
                                       RelAlgDagBuilder& root_dag_builder) {
  if (fn_name == "PG_ANY"sv || fn_name == "PG_ALL"sv) {
    return hdk::ir::makeExpr<hdk::ir::FunctionOper>(ti, fn_name, operands);
  }
  if (fn_name == "LIKE"sv || fn_name == "PG_ILIKE"sv) {
    return parseLike(fn_name, operands);
  }
  if (fn_name == "REGEXP_LIKE"sv) {
    return parseRegexp(operands);
  }
  if (fn_name == "LIKELY"sv) {
    return parseLikely(operands);
  }
  if (fn_name == "UNLIKELY"sv) {
    return parseUnlikely(operands);
  }
  if (fn_name == "PG_EXTRACT"sv || fn_name == "PG_DATE_TRUNC"sv) {
    return parseExtract(fn_name, operands);
  }
  if (fn_name == "DATEADD"sv) {
    return parseDateadd(operands);
  }
  if (fn_name == "DATEDIFF"sv) {
    return parseDatediff(operands);
  }
  if (fn_name == "DATEPART"sv) {
    return parseDatepart(operands);
  }
  if (fn_name == "LENGTH"sv || fn_name == "CHAR_LENGTH"sv) {
    return parseLength(fn_name, operands);
  }
  if (fn_name == "KEY_FOR_STRING"sv) {
    return parseKeyForString(operands);
  }
  if (fn_name == "WIDTH_BUCKET"sv) {
    return parseWidthBucket(operands);
  }
  if (fn_name == "SAMPLE_RATIO"sv) {
    return parseSampleRatio(operands);
  }
  if (fn_name == "CURRENT_USER"sv) {
    return parseCurrentUser();
  }
  if (root_dag_builder.config().exec.enable_experimental_string_functions &&
      fn_name == "LOWER"sv) {
    return parseLower(operands);
  }
  if (fn_name == "CARDINALITY"sv || fn_name == "ARRAY_LENGTH"sv) {
    return parseCardinality(fn_name, operands, ti);
  }
  if (fn_name == "ITEM"sv) {
    return parseItem(operands);
  }
  if (fn_name == "CURRENT_DATE"sv) {
    return parseCurrentDate(root_dag_builder.now());
  }
  if (fn_name == "CURRENT_TIME"sv) {
    return parseCurrentTime(root_dag_builder.now());
  }
  if (fn_name == "CURRENT_TIMESTAMP"sv) {
    return parseCurrentTimestamp(root_dag_builder.now());
  }
  if (fn_name == "NOW"sv) {
    return parseCurrentTimestamp(root_dag_builder.now());
  }
  if (fn_name == "DATETIME"sv) {
    return parseDatetime(operands, root_dag_builder.now());
  }
  if (fn_name == "usTIMESTAMP"sv || fn_name == "nsTIMESTAMP"sv) {
    return parseHPTLiteral(operands, ti);
  }
  if (fn_name == "ABS"sv) {
    return parseAbs(operands);
  }
  if (fn_name == "SIGN"sv) {
    return parseSign(operands);
  }
  if (fn_name == "CEIL"sv || fn_name == "FLOOR"sv) {
    return hdk::ir::makeExpr<hdk::ir::FunctionOperWithCustomTypeHandling>(
        ti, fn_name, operands);
  }
  if (fn_name == "ROUND"sv) {
    return parseRound(fn_name, operands, ti);
  }
  if (fn_name == "DATETIME_PLUS"sv) {
    auto dt_plus = hdk::ir::makeExpr<hdk::ir::FunctionOper>(ti, fn_name, operands);
    const auto date_trunc = rewrite_to_date_trunc(dt_plus.get());
    if (date_trunc) {
      return date_trunc;
    }
    return parseDateadd(operands);
  }
  if (fn_name == "/INT"sv) {
    CHECK_EQ(operands.size(), size_t(2));
    return Analyzer::normalizeOperExpr(kDIVIDE, kONE, operands[0], operands[1]);
  }
  if (fn_name == "Reinterpret"sv) {
    CHECK_EQ(operands.size(), size_t(1));
    return operands[0];
  }
  if (fn_name == "OFFSET_IN_FRAGMENT"sv) {
    CHECK_EQ(operands.size(), size_t(0));
    return hdk::ir::makeExpr<hdk::ir::OffsetInFragment>();
  }
  if (fn_name == "ARRAY"sv) {
    // Var args; currently no check.  Possible fix-me -- can array have 0 elements?
    return parseArrayFunction(operands, ti);
  }

  if (fn_name == "||"sv || fn_name == "SUBSTRING"sv) {
    SQLTypeInfo ret_ti(kTEXT, false);
    return hdk::ir::makeExpr<hdk::ir::FunctionOper>(ret_ti, fn_name, operands);
  }
  // Reset possibly wrong return type of rex_function to the return
  // type of the optimal valid implementation. The return type can be
  // wrong in the case of multiple implementations of UDF functions
  // that have different return types but Calcite specifies the return
  // type according to the first implementation.
  auto args = operands;
  SQLTypeInfo ret_ti;
  try {
    auto ext_func_sig = bind_function(fn_name, args);

    auto ext_func_args = ext_func_sig.getArgs();
    CHECK_EQ(args.size(), ext_func_args.size());
    for (size_t i = 0; i < args.size(); i++) {
      // fold casts on constants
      if (auto constant = std::dynamic_pointer_cast<hdk::ir::Constant>(args[i])) {
        auto ext_func_arg_ti = ext_arg_type_to_type_info(ext_func_args[i]);
        if (ext_func_arg_ti != args[i]->get_type_info()) {
          args[i] = constant->add_cast(ext_func_arg_ti);
        }
      }
    }

    ret_ti = ext_arg_type_to_type_info(ext_func_sig.getRet());
  } catch (ExtensionFunctionBindingError& e) {
    LOG(WARNING) << "RelAlgTranslator::translateFunction: " << e.what();
    throw;
  }

  // By default, the extension function type will not allow nulls. If one of the arguments
  // is nullable, the extension function must also explicitly allow nulls.
  bool arguments_not_null = true;
  for (const auto& arg_expr : args) {
    if (!arg_expr->get_type_info().get_notnull()) {
      arguments_not_null = false;
      break;
    }
  }
  ret_ti.set_notnull(arguments_not_null);

  return hdk::ir::makeExpr<hdk::ir::FunctionOper>(ret_ti, fn_name, std::move(args));
}

bool isAggSupportedForType(const SQLAgg& agg_kind, const SQLTypeInfo& arg_ti) {
  if ((agg_kind == kMIN || agg_kind == kMAX || agg_kind == kSUM || agg_kind == kAVG) &&
      !(arg_ti.is_number() || arg_ti.is_boolean() || arg_ti.is_time())) {
    return false;
  }

  return true;
}

hdk::ir::ExprPtr parseAggregateExpr(const rapidjson::Value& json_expr,
                                    RelAlgDagBuilder& root_dag_builder,
                                    const hdk::ir::ExprPtrVector& sources) {
  auto agg_str = json_str(field(json_expr, "agg"));
  if (agg_str == "APPROX_QUANTILE") {
    LOG(INFO) << "APPROX_QUANTILE is deprecated. Please use APPROX_PERCENTILE instead.";
  }
  auto agg_kind = to_agg_kind(agg_str);
  auto is_distinct = json_bool(field(json_expr, "distinct"));
  auto operands = indices_from_json_array(field(json_expr, "operands"));
  if (operands.size() > 1 &&
      (operands.size() != 2 ||
       (agg_kind != kAPPROX_COUNT_DISTINCT && agg_kind != kAPPROX_QUANTILE))) {
    throw QueryNotSupported("Multiple arguments for aggregates aren't supported");
  }

  hdk::ir::ExprPtr arg_expr;
  std::shared_ptr<hdk::ir::Constant> arg1;  // 2nd aggregate parameter
  if (operands.size() > 0) {
    const auto operand = operands[0];
    CHECK_LT(operand, sources.size());
    CHECK_LE(operands.size(), 2u);
    arg_expr = sources[operand];

    if (agg_kind == kAPPROX_COUNT_DISTINCT && operands.size() == 2) {
      arg1 = std::dynamic_pointer_cast<hdk::ir::Constant>(sources[operands[1]]);
      if (!arg1 || arg1->get_type_info().get_type() != kINT ||
          arg1->get_constval().intval < 1 || arg1->get_constval().intval > 100) {
        throw std::runtime_error(
            "APPROX_COUNT_DISTINCT's second parameter should be SMALLINT literal between "
            "1 and 100");
      }
    } else if (agg_kind == kAPPROX_QUANTILE) {
      // If second parameter is not given then APPROX_MEDIAN is assumed.
      if (operands.size() == 2) {
        arg1 = std::dynamic_pointer_cast<hdk::ir::Constant>(
            std::dynamic_pointer_cast<hdk::ir::Constant>(sources[operands[1]])
                ->add_cast(SQLTypeInfo(kDOUBLE)));
      } else {
        Datum median;
        median.doubleval = 0.5;
        arg1 = std::make_shared<hdk::ir::Constant>(kDOUBLE, false, median);
      }
    }
    auto& arg_ti = arg_expr->get_type_info();
    if (!isAggSupportedForType(agg_kind, arg_ti)) {
      throw std::runtime_error("Aggregate on " + arg_ti.get_type_name() +
                               " is not supported yet.");
    }
  }
  auto agg_ti = get_agg_type(
      agg_kind, arg_expr.get(), root_dag_builder.config().exec.group_by.bigint_count);
  return hdk::ir::makeExpr<hdk::ir::AggExpr>(
      agg_ti, agg_kind, arg_expr, is_distinct, arg1);
}

hdk::ir::ExprPtr parse_operator_expr(const rapidjson::Value& json_expr,
                                     int db_id,
                                     SchemaProviderPtr schema_provider,
                                     RelAlgDagBuilder& root_dag_builder,
                                     const hdk::ir::ExprPtrVector& ra_output) {
  const auto op_name = json_str(field(json_expr, "op"));
  const bool is_quantifier =
      op_name == std::string("PG_ANY") || op_name == std::string("PG_ALL");
  const auto op = is_quantifier ? kFUNCTION : to_sql_op(op_name);
  const auto& operators_json_arr = field(json_expr, "operands");
  CHECK(operators_json_arr.IsArray());
  auto operands = parseExprArray(
      operators_json_arr, db_id, schema_provider, root_dag_builder, ra_output);
  const auto type_it = json_expr.FindMember("type");
  CHECK(type_it != json_expr.MemberEnd());
  auto type = parseType(type_it->value);
  auto ti = type->toTypeInfo();

  if (op == kIN && json_expr.HasMember("subquery")) {
    CHECK_EQ(operands.size(), (size_t)1);
    auto subquery =
        parse_subquery_expr(json_expr, db_id, schema_provider, root_dag_builder);
    SQLTypeInfo ti(kBOOLEAN);
    return hdk::ir::makeExpr<hdk::ir::InSubquery>(
        ti,
        operands[0],
        dynamic_cast<const hdk::ir::ScalarSubquery*>(subquery.get())->getNodeShared());
  } else if (json_expr.FindMember("partition_keys") != json_expr.MemberEnd()) {
    return parseWindowFunction(json_expr,
                               op_name,
                               operands,
                               ti,
                               db_id,
                               schema_provider,
                               root_dag_builder,
                               ra_output);

  } else if (op == kFUNCTION) {
    return parseFunctionOperator(op_name, operands, ti, root_dag_builder);
  } else {
    CHECK_GE(operands.size(), (size_t)1);

    if (operands.size() == 1) {
      return makeUOper(op, operands[0], ti);
    }

    if (auto res = maybeMakeDateExpr(op, operands, ti)) {
      return res;
    }

    auto res = operands[0];
    for (size_t i = 1; i < operands.size(); ++i) {
      auto [rhs, qual] = getQuantifiedBinOperRhs(operands[i]);
      res = Analyzer::normalizeOperExpr(op, qual, res, rhs, nullptr);
    }
    return res;
  }
}

hdk::ir::ExprPtr parse_expr(const rapidjson::Value& expr,
                            int db_id,
                            SchemaProviderPtr schema_provider,
                            RelAlgDagBuilder& root_dag_builder,
                            const hdk::ir::ExprPtrVector& ra_output) {
  CHECK(expr.IsObject());
  if (expr.IsObject() && expr.HasMember("input")) {
    return parseInput(expr, ra_output);
  }
  if (expr.IsObject() && expr.HasMember("literal")) {
    return parseLiteral(expr);
  }
  if (expr.IsObject() && expr.HasMember("op")) {
    hdk::ir::ExprPtr res;
    const auto op_str = json_str(field(expr, "op"));
    if (op_str == std::string("CASE")) {
      res = parse_case_expr(expr, db_id, schema_provider, root_dag_builder, ra_output);
    } else if (op_str == std::string("$SCALAR_QUERY")) {
      res = parse_subquery_expr(expr, db_id, schema_provider, root_dag_builder);
    } else {
      res =
          parse_operator_expr(expr, db_id, schema_provider, root_dag_builder, ra_output);
    }
    CHECK(res);

    return res;
  }
  throw QueryNotSupported("Expression node " + json_node_to_string(expr) +
                          " not supported");
}

JoinType to_join_type(const std::string& join_type_name) {
  if (join_type_name == "inner") {
    return JoinType::INNER;
  }
  if (join_type_name == "left") {
    return JoinType::LEFT;
  }
  if (join_type_name == "semi") {
    return JoinType::SEMI;
  }
  if (join_type_name == "anti") {
    return JoinType::ANTI;
  }
  throw QueryNotSupported("Join type (" + join_type_name + ") not supported");
}

void handleQueryHint(const std::vector<std::shared_ptr<RelAlgNode>>& nodes,
                     RelAlgDagBuilder* dag_builder) noexcept {
  // query hint is delivered by the above three nodes
  // when a query block has top-sort node, a hint is registered to
  // one of the node which locates at the nearest from the sort node
  for (auto node : nodes) {
    Hints* hint_delivered = nullptr;
    const auto agg_node = std::dynamic_pointer_cast<RelAggregate>(node);
    if (agg_node) {
      if (agg_node->hasDeliveredHint()) {
        hint_delivered = agg_node->getDeliveredHints();
      }
    }
    const auto project_node = std::dynamic_pointer_cast<RelProject>(node);
    if (project_node) {
      if (project_node->hasDeliveredHint()) {
        hint_delivered = project_node->getDeliveredHints();
      }
    }
    const auto compound_node = std::dynamic_pointer_cast<RelCompound>(node);
    if (compound_node) {
      if (compound_node->hasDeliveredHint()) {
        hint_delivered = compound_node->getDeliveredHints();
      }
    }
    if (hint_delivered && !hint_delivered->empty()) {
      dag_builder->registerQueryHints(node, hint_delivered);
    }
  }
}

void mark_nops(const std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  for (auto node : nodes) {
    const auto agg_node = std::dynamic_pointer_cast<RelAggregate>(node);
    if (!agg_node || agg_node->getAggsCount()) {
      continue;
    }
    CHECK_EQ(size_t(1), node->inputCount());
    const auto agg_input_node = dynamic_cast<const RelAggregate*>(node->getInput(0));
    if (agg_input_node && !agg_input_node->getAggsCount() &&
        agg_node->getGroupByCount() == agg_input_node->getGroupByCount()) {
      agg_node->markAsNop();
    }
  }
}

hdk::ir::ExprPtrVector reprojectExprs(const RelProject* simple_project,
                                      const hdk::ir::ExprPtrVector& exprs) noexcept {
  hdk::ir::ExprPtrVector result;
  for (size_t i = 0; i < simple_project->size(); ++i) {
    const auto col_ref =
        dynamic_cast<const hdk::ir::ColumnRef*>(simple_project->getExpr(i).get());
    CHECK(col_ref);
    CHECK_LT(static_cast<size_t>(col_ref->getIndex()), exprs.size());
    result.push_back(exprs[col_ref->getIndex()]);
  }
  return result;
}

/**
 * The InputReplacementVisitor visitor visits each node in a given relational algebra
 * expression and replaces the inputs to that expression with inputs from a different
 * node in the RA tree. Used for coalescing nodes with complex expressions.
 */
class InputReplacementVisitor : public DeepCopyVisitor {
 public:
  InputReplacementVisitor(const RelAlgNode* node_to_keep,
                          const hdk::ir::ExprPtrVector& exprs,
                          const hdk::ir::ExprPtrVector* groupby_exprs = nullptr)
      : node_to_keep_(node_to_keep), exprs_(exprs), groupby_exprs_(groupby_exprs) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    if (col_ref->getNode() == node_to_keep_) {
      const auto index = col_ref->getIndex();
      CHECK_LT(index, exprs_.size());
      return visit(exprs_[index].get());
    }
    return col_ref->deep_copy();
  }

  hdk::ir::ExprPtr visitGroupColumnRef(
      const hdk::ir::GroupColumnRef* col_ref) const override {
    CHECK(groupby_exprs_);
    CHECK_LE(col_ref->getIndex(), groupby_exprs_->size());
    return visit((*groupby_exprs_)[col_ref->getIndex() - 1].get());
  }

 private:
  const RelAlgNode* node_to_keep_;
  const hdk::ir::ExprPtrVector& exprs_;
  const hdk::ir::ExprPtrVector* groupby_exprs_;
};

void create_compound(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    const std::vector<size_t>& pattern,
    std::unordered_map<size_t, RegisteredQueryHint>& query_hints) noexcept {
  CHECK_GE(pattern.size(), size_t(2));
  CHECK_LE(pattern.size(), size_t(4));

  hdk::ir::ExprPtr filter_expr;
  size_t groupby_count{0};
  std::vector<std::string> fields;
  hdk::ir::ExprPtrVector exprs;
  hdk::ir::ExprPtrVector groupby_exprs;
  bool first_project{true};
  bool is_agg{false};
  RelAlgNode* last_node{nullptr};

  size_t node_hash{0};
  std::optional<RegisteredQueryHint> registered_query_hint;
  for (const auto node_idx : pattern) {
    const auto ra_node = nodes[node_idx];
    auto registered_query_hint_it = query_hints.find(ra_node->toHash());
    if (registered_query_hint_it != query_hints.end()) {
      node_hash = registered_query_hint_it->first;
      registered_query_hint = registered_query_hint_it->second;
    }
    const auto ra_filter = std::dynamic_pointer_cast<RelFilter>(ra_node);
    if (ra_filter) {
      CHECK(!filter_expr);
      filter_expr = ra_filter->getConditionExprShared();
      CHECK(filter_expr);
      last_node = ra_node.get();
      continue;
    }
    const auto ra_project = std::dynamic_pointer_cast<RelProject>(ra_node);
    if (ra_project) {
      fields = ra_project->getFields();

      if (first_project) {
        CHECK_EQ(size_t(1), ra_project->inputCount());
        // Rebind the input of the project to the input of the filter itself
        // since we know that we'll evaluate the filter on the fly, with no
        // intermediate buffer. There are cases when filter is not a part of
        // the pattern, detect it by simply cehcking current filter rex.
        const auto filter_input = dynamic_cast<const RelFilter*>(ra_project->getInput(0));
        if (filter_input && filter_expr) {
          CHECK_EQ(size_t(1), filter_input->inputCount());
          ra_project->replaceInput(ra_project->getAndOwnInput(0),
                                   filter_input->getAndOwnInput(0));
        }
        for (auto& expr : ra_project->getExprs()) {
          exprs.push_back(expr);
        }
        first_project = false;
      } else {
        if (ra_project->isSimple()) {
          exprs = reprojectExprs(ra_project.get(), exprs);
        } else {
          // TODO(adb): This is essentially a more general case of simple project, we
          // could likely merge the two
          hdk::ir::ExprPtrVector new_exprs;
          InputReplacementVisitor visitor(last_node, exprs, &groupby_exprs);
          for (size_t i = 0; i < ra_project->size(); ++i) {
            auto expr = ra_project->getExpr(i);
            if (auto col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(expr.get())) {
              const auto index = col_ref->getIndex();
              CHECK_LT(index, exprs.size());
              new_exprs.push_back(exprs[index]);
            } else {
              new_exprs.push_back(visitor.visit(expr.get()));
            }
          }
          exprs = std::move(new_exprs);
        }
      }
      last_node = ra_node.get();
      continue;
    }
    const auto ra_aggregate = std::dynamic_pointer_cast<RelAggregate>(ra_node);
    if (ra_aggregate) {
      is_agg = true;
      fields = ra_aggregate->getFields();
      groupby_count = ra_aggregate->getGroupByCount();
      groupby_exprs.swap(exprs);
      CHECK_LE(groupby_count, groupby_exprs.size());
      InputReplacementVisitor visitor(last_node, groupby_exprs);
      for (size_t group_idx = 0; group_idx < groupby_count; ++group_idx) {
        exprs.push_back(hdk::ir::makeExpr<hdk::ir::GroupColumnRef>(
            groupby_exprs[group_idx]->get_type_info(), group_idx + 1));
      }
      for (auto& expr : ra_aggregate->getAggs()) {
        exprs.push_back(visitor.visit(expr.get()));
      }
      last_node = ra_node.get();
      continue;
    }
  }

  auto compound_node = std::make_shared<RelCompound>(filter_expr,
                                                     std::move(exprs),
                                                     groupby_count,
                                                     std::move(groupby_exprs),
                                                     fields,
                                                     is_agg);
  auto old_node = nodes[pattern.back()];
  nodes[pattern.back()] = compound_node;
  auto first_node = nodes[pattern.front()];
  CHECK_EQ(size_t(1), first_node->inputCount());
  compound_node->addManagedInput(first_node->getAndOwnInput(0));
  if (registered_query_hint) {
    // pass the registered hint from the origin node to newly created compound node
    // where it is coalesced
    query_hints.erase(node_hash);
    query_hints.emplace(compound_node->toHash(), *registered_query_hint);
  }
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
  using ElementType = std::shared_ptr<RelAlgNode>;
  using Super = std::vector<ElementType>::const_iterator;
  using Container = std::vector<ElementType>;

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

  explicit operator size_t() {
    return std::distance(owner_.begin(), *static_cast<Super*>(this));
  }

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
        break;
      }
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

bool input_can_be_coalesced(const RelAlgNode* parent_node,
                            const size_t index,
                            const bool first_rex_is_input) {
  if (auto agg_node = dynamic_cast<const RelAggregate*>(parent_node)) {
    if (index == 0 && agg_node->getGroupByCount() > 0) {
      return true;
    } else {
      // Is an aggregated target, only allow the project to be elided if the aggregate
      // target is simply passed through (i.e. if the top level expression attached to
      // the project node is a RexInput expression)
      return first_rex_is_input;
    }
  }
  return first_rex_is_input;
}

/**
 * CoalesceSecondaryProjectVisitor visits each relational algebra expression node in a
 * given input and determines whether or not the input is a candidate for coalescing
 * into the parent RA node. Intended for use only on the inputs of a RelProject node.
 */
class CoalesceSecondaryProjectVisitor : public ScalarExprVisitor<bool> {
 public:
  bool visitColumnRef(const hdk::ir::ColumnRef* col_ref) const final {
    // The top level expression node is checked before we apply the visitor. If we get
    // here, this input rex is a child of another rex node, and we handle the can be
    // coalesced check slightly differently
    return input_can_be_coalesced(col_ref->getNode(), col_ref->getIndex(), false);
  }

  bool visitConstant(const hdk::ir::Constant*) const final { return false; }

  bool visitInSubquery(const hdk::ir::InSubquery*) const final { return false; }

  bool visitScalarSubquery(const hdk::ir::ScalarSubquery*) const final { return false; }

 protected:
  bool aggregateResult(const bool& aggregate, const bool& next_result) const final {
    return aggregate && next_result;
  }

  bool defaultResult() const final { return true; }
};

// Detect the window function SUM pattern: CASE WHEN COUNT() > 0 THEN SUM ELSE 0
bool is_window_function_sum(const hdk::ir::Expr* expr) {
  const auto case_expr = dynamic_cast<const hdk::ir::CaseExpr*>(expr);
  if (case_expr && case_expr->get_expr_pair_list().size() == 1) {
    const hdk::ir::Expr* then = case_expr->get_expr_pair_list().front().second.get();

    // Allow optional cast.
    const auto cast = dynamic_cast<const hdk::ir::UOper*>(then);
    if (cast && cast->get_optype() == kCAST) {
      then = cast->get_operand();
    }

    const auto then_window = dynamic_cast<const hdk::ir::WindowFunction*>(then);
    if (then_window && then_window->getKind() == SqlWindowFunctionKind::SUM_INTERNAL) {
      return true;
    }
  }
  return false;
}

// Detect both window function operators and window function operators embedded in case
// statements (for null handling)
bool is_window_function_expr(const hdk::ir::Expr* expr) {
  if (dynamic_cast<const hdk::ir::WindowFunction*>(expr) != nullptr) {
    return true;
  }

  // unwrap from casts, if they exist
  const auto cast = dynamic_cast<const hdk::ir::UOper*>(expr);
  if (cast && cast->get_optype() == kCAST) {
    return is_window_function_expr(cast->get_operand());
  }

  if (is_window_function_sum(expr)) {
    return true;
  }

  // Check for Window Function AVG:
  // (CASE WHEN count > 0 THEN sum ELSE 0) / COUNT
  const auto div = dynamic_cast<const hdk::ir::BinOper*>(expr);
  if (div && div->get_optype() == kDIVIDE) {
    const auto case_expr =
        dynamic_cast<const hdk::ir::CaseExpr*>(div->get_left_operand());
    const auto second_window =
        dynamic_cast<const hdk::ir::WindowFunction*>(div->get_right_operand());
    if (case_expr && second_window &&
        second_window->getKind() == SqlWindowFunctionKind::COUNT) {
      if (is_window_function_sum(case_expr)) {
        return true;
      }
    }
  }
  return false;
}

void coalesce_nodes(std::vector<std::shared_ptr<RelAlgNode>>& nodes,
                    const std::vector<const RelAlgNode*>& left_deep_joins,
                    std::unordered_map<size_t, RegisteredQueryHint>& query_hints) {
  enum class CoalesceState { Initial, Filter, FirstProject, Aggregate };
  std::vector<size_t> crt_pattern;
  CoalesceState crt_state{CoalesceState::Initial};

  auto reset_state = [&crt_pattern, &crt_state]() {
    crt_state = CoalesceState::Initial;
    std::vector<size_t>().swap(crt_pattern);
  };

  for (RANodeIterator nodeIt(nodes); !nodeIt.allVisited();) {
    const auto ra_node = nodeIt != nodes.end() ? *nodeIt : nullptr;
    switch (crt_state) {
      case CoalesceState::Initial: {
        if (std::dynamic_pointer_cast<const RelFilter>(ra_node) &&
            std::find(left_deep_joins.begin(), left_deep_joins.end(), ra_node.get()) ==
                left_deep_joins.end()) {
          crt_pattern.push_back(size_t(nodeIt));
          crt_state = CoalesceState::Filter;
          nodeIt.advance(RANodeIterator::AdvancingMode::DUChain);
        } else if (auto project_node =
                       std::dynamic_pointer_cast<const RelProject>(ra_node)) {
          if (project_node->hasWindowFunctionExpr()) {
            nodeIt.advance(RANodeIterator::AdvancingMode::InOrder);
          } else {
            crt_pattern.push_back(size_t(nodeIt));
            crt_state = CoalesceState::FirstProject;
            nodeIt.advance(RANodeIterator::AdvancingMode::DUChain);
          }
        } else {
          nodeIt.advance(RANodeIterator::AdvancingMode::InOrder);
        }
        break;
      }
      case CoalesceState::Filter: {
        if (auto project_node = std::dynamic_pointer_cast<const RelProject>(ra_node)) {
          // Given we now add preceding projects for all window functions following
          // RelFilter nodes, the following should never occur
          CHECK(!project_node->hasWindowFunctionExpr());
          crt_pattern.push_back(size_t(nodeIt));
          crt_state = CoalesceState::FirstProject;
          nodeIt.advance(RANodeIterator::AdvancingMode::DUChain);
        } else {
          reset_state();
        }
        break;
      }
      case CoalesceState::FirstProject: {
        if (std::dynamic_pointer_cast<const RelAggregate>(ra_node)) {
          crt_pattern.push_back(size_t(nodeIt));
          crt_state = CoalesceState::Aggregate;
          nodeIt.advance(RANodeIterator::AdvancingMode::DUChain);
        } else {
          if (crt_pattern.size() >= 2) {
            create_compound(nodes, crt_pattern, query_hints);
          }
          reset_state();
        }
        break;
      }
      case CoalesceState::Aggregate: {
        if (auto project_node = std::dynamic_pointer_cast<const RelProject>(ra_node)) {
          if (!project_node->hasWindowFunctionExpr()) {
            // TODO(adb): overloading the simple project terminology again here
            bool is_simple_project{true};
            for (auto& expr : project_node->getExprs()) {
              // If the top level scalar rex is an input node, we can bypass the visitor
              if (auto col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(expr.get())) {
                if (!input_can_be_coalesced(
                        col_ref->getNode(), col_ref->getIndex(), true)) {
                  is_simple_project = false;
                  break;
                }
                continue;
              }
              CoalesceSecondaryProjectVisitor visitor;
              if (!visitor.visit(expr.get())) {
                is_simple_project = false;
                break;
              }
            }
            if (is_simple_project) {
              crt_pattern.push_back(size_t(nodeIt));
              nodeIt.advance(RANodeIterator::AdvancingMode::InOrder);
            }
          }
        }
        CHECK_GE(crt_pattern.size(), size_t(2));
        create_compound(nodes, crt_pattern, query_hints);
        reset_state();
        break;
      }
      default:
        CHECK(false);
    }
  }
  if (crt_state == CoalesceState::FirstProject || crt_state == CoalesceState::Aggregate) {
    if (crt_pattern.size() >= 2) {
      create_compound(nodes, crt_pattern, query_hints);
    }
    CHECK(!crt_pattern.empty());
  }
}

class ReplacementExprVisitor : public DeepCopyVisitor {
 public:
  ReplacementExprVisitor() {}

  ReplacementExprVisitor(
      std::unordered_map<const hdk::ir::Expr*, hdk::ir::ExprPtr> replacements)
      : replacements_(std::move(replacements)) {}

  void addReplacement(const hdk::ir::Expr* from, hdk::ir::ExprPtr to) {
    replacements_[from] = to;
  }

  hdk::ir::ExprPtr visit(const hdk::ir::Expr* expr) const {
    auto it = replacements_.find(expr);
    if (it != replacements_.end()) {
      return it->second;
    }
    return DeepCopyVisitor::visit(expr);
  }

 private:
  std::unordered_map<const hdk::ir::Expr*, hdk::ir::ExprPtr> replacements_;
};

/**
 * Propagate an input backwards in the RA tree. With the exception of joins, all inputs
 * must be carried through the RA tree. This visitor takes as a parameter a source
 * projection RA Node, then checks to see if any inputs do not reference the source RA
 * node (which implies the inputs reference a node farther back in the tree). The input
 * is then backported to the source projection node, and a new input is generated which
 * references the input on the source RA node, thereby carrying the input through the
 * intermediate query step.
 */
class InputBackpropagationVisitor : public DeepCopyVisitor {
 public:
  InputBackpropagationVisitor(RelProject* node) : node_(node) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    if (col_ref->getNode() != node_) {
      auto cur_index = col_ref->getIndex();
      auto cur_source_node = col_ref->getNode();
      auto it = replacements_.find(std::make_pair(cur_source_node, cur_index));
      if (it != replacements_.end()) {
        return it->second;
      } else {
        std::string field_name = "";
        if (auto cur_project_node = dynamic_cast<const RelProject*>(cur_source_node)) {
          field_name = cur_project_node->getFieldName(cur_index);
        }
        node_->appendInput(field_name, col_ref->deep_copy());
        auto expr = hdk::ir::makeExpr<hdk::ir::ColumnRef>(
            getColumnType(node_, node_->size() - 1), node_, node_->size() - 1);
        replacements_[std::make_pair(cur_source_node, cur_index)] = expr;
        return expr;
      }
    } else {
      return DeepCopyVisitor::visitColumnRef(col_ref);
    }
  }

 protected:
  using InputReplacements =
      std::unordered_map<std::pair<const RelAlgNode*, unsigned>,
                         hdk::ir::ExprPtr,
                         boost::hash<std::pair<const RelAlgNode*, unsigned>>>;

  mutable RelProject* node_;
  mutable InputReplacements replacements_;
};

void propagate_hints_to_new_project(
    std::shared_ptr<RelProject> prev_node,
    std::shared_ptr<RelProject> new_node,
    std::unordered_map<size_t, RegisteredQueryHint>& query_hints) {
  auto delivered_hints = prev_node->getDeliveredHints();
  bool needs_propagate_hints = !delivered_hints->empty();
  if (needs_propagate_hints) {
    for (auto& kv : *delivered_hints) {
      new_node->addHint(kv.second);
    }
    auto prev_it = query_hints.find(prev_node->toHash());
    // query hint for the prev projection node should be registered
    CHECK(prev_it != query_hints.end());
    query_hints.emplace(new_node->toHash(), prev_it->second);
  }
}

/**
 * Detect the presence of window function operators nested inside expressions. Separate
 * the window function operator from the expression, computing the expression as a
 * subsequent step and replacing the window function operator with a RexInput. Also move
 * all input nodes to the newly created project node.
 * In pseudocode:
 * for each rex in project list:
 *    detect window function expression
 *    if window function expression:
 *        copy window function expression
 *        replace window function expression in base expression w/ input
 *        add base expression to new project node after the current node
 *        replace base expression in current project node with the window function
 expression copy
 */
void separate_window_function_expressions(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    std::unordered_map<size_t, RegisteredQueryHint>& query_hints) {
  std::list<std::shared_ptr<RelAlgNode>> node_list(nodes.begin(), nodes.end());

  for (auto node_itr = node_list.begin(); node_itr != node_list.end(); ++node_itr) {
    const auto node = *node_itr;
    auto window_func_project_node = std::dynamic_pointer_cast<RelProject>(node);
    if (!window_func_project_node) {
      continue;
    }

    // map scalar expression index in the project node to window function ptr
    std::unordered_map<size_t, hdk::ir::ExprPtr> embedded_window_function_exprs;

    // Iterate the target exprs of the project node and check for window function
    // expressions. If an embedded expression exists, save it in the
    // embedded_window_function_expressions map and split the expression into a window
    // function expression and a parent expression in a subsequent project node
    for (size_t i = 0; i < window_func_project_node->size(); i++) {
      auto expr = window_func_project_node->getExpr(i);
      if (is_window_function_expr(expr.get())) {
        // top level window function exprs are fine
        continue;
      }

      std::list<const hdk::ir::Expr*> window_function_exprs;
      expr->find_expr(is_window_function_expr, window_function_exprs);
      if (!window_function_exprs.empty()) {
        const auto ret = embedded_window_function_exprs.insert(std::make_pair(
            i,
            const_cast<hdk::ir::Expr*>(window_function_exprs.front())->get_shared_ptr()));
        CHECK(ret.second);
      }
    }

    if (!embedded_window_function_exprs.empty()) {
      hdk::ir::ExprPtrVector new_exprs;

      auto window_func_exprs = window_func_project_node->getExprs();
      for (size_t rex_idx = 0; rex_idx < window_func_exprs.size(); ++rex_idx) {
        const auto embedded_window_func_expr_pair =
            embedded_window_function_exprs.find(rex_idx);
        if (embedded_window_func_expr_pair == embedded_window_function_exprs.end()) {
          new_exprs.emplace_back(hdk::ir::makeExpr<hdk::ir::ColumnRef>(
              window_func_project_node->getExpr(rex_idx)->get_type_info(),
              window_func_project_node.get(),
              rex_idx));
        } else {
          const auto window_func_expr_idx = embedded_window_func_expr_pair->first;
          CHECK_LT(window_func_expr_idx, window_func_exprs.size());

          const auto& window_func_expr = embedded_window_func_expr_pair->second;
          auto window_func_parent_expr = window_func_exprs[window_func_expr_idx].get();

          // Replace window func expr with ColumnRef
          auto window_func_result_input_expr =
              hdk::ir::makeExpr<hdk::ir::ColumnRef>(window_func_expr->get_type_info(),
                                                    window_func_project_node.get(),
                                                    window_func_expr_idx);
          std::unordered_map<const hdk::ir::Expr*, hdk::ir::ExprPtr> replacements;
          replacements[window_func_expr.get()] = window_func_result_input_expr;
          ReplacementExprVisitor visitor(std::move(replacements));
          auto new_parent_expr = visitor.visit(window_func_parent_expr);

          // Put the parent expr in the new scalar exprs
          new_exprs.emplace_back(std::move(new_parent_expr));

          // Put the window func expr in cur scalar exprs
          window_func_exprs[window_func_expr_idx] = window_func_expr;
        }
      }

      CHECK_EQ(window_func_exprs.size(), new_exprs.size());
      window_func_project_node->setExpressions(std::move(window_func_exprs));

      // Ensure any inputs from the node containing the expression (the "new" node)
      // exist on the window function project node, e.g. if we had a binary operation
      // involving an aggregate value or column not included in the top level
      // projection list.
      InputBackpropagationVisitor visitor(window_func_project_node.get());
      for (size_t i = 0; i < new_exprs.size(); i++) {
        if (dynamic_cast<const hdk::ir::ColumnRef*>(new_exprs[i].get())) {
          // ignore top level inputs, these were copied directly from the previous
          // node
          continue;
        }
        new_exprs[i] = visitor.visit(new_exprs[i].get());
      }

      // Build the new project node and insert it into the list after the project node
      // containing the window function
      auto new_project =
          std::make_shared<RelProject>(std::move(new_exprs),
                                       window_func_project_node->getFields(),
                                       window_func_project_node);
      propagate_hints_to_new_project(window_func_project_node, new_project, query_hints);
      node_list.insert(std::next(node_itr), new_project);

      // Rebind all the following inputs
      for (auto rebind_itr = std::next(node_itr, 2); rebind_itr != node_list.end();
           rebind_itr++) {
        (*rebind_itr)->replaceInput(window_func_project_node, new_project);
      }
    }
  }
  nodes.assign(node_list.begin(), node_list.end());
}

using InputSet = std::unordered_set<std::pair<const RelAlgNode*, unsigned>,
                                    boost::hash<std::pair<const RelAlgNode*, unsigned>>>;

class InputCollector : public ScalarExprVisitor<InputSet> {
 public:
  InputSet visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    InputSet result;
    result.emplace(col_ref->getNode(), col_ref->getIndex());
    return result;
  }

 protected:
  InputSet aggregateResult(const InputSet& aggregate,
                           const InputSet& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

/**
 * Inserts a simple project before any project containing a window function node. Forces
 * all window function inputs into a single contiguous buffer for centralized processing
 * (e.g. in distributed mode). This is also needed when a window function node is preceded
 * by a filter node, both for correctness (otherwise a window operator will be coalesced
 * with its preceding filter node and be computer over unfiltered results, and for
 * performance, as currently filter nodes that are not coalesced into projects keep all
 * columns from the table as inputs, and hence bring everything in memory.
 * Once the new project has been created, the inputs in the
 * window function project must be rewritten to read from the new project, and to index
 * off the projected exprs in the new project.
 */
void add_window_function_pre_project(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    const bool always_add_project_if_first_project_is_window_expr,
    std::unordered_map<size_t, RegisteredQueryHint>& query_hints) {
  std::list<std::shared_ptr<RelAlgNode>> node_list(nodes.begin(), nodes.end());
  size_t project_node_counter{0};
  for (auto node_itr = node_list.begin(); node_itr != node_list.end(); ++node_itr) {
    const auto node = *node_itr;

    auto window_func_project_node = std::dynamic_pointer_cast<RelProject>(node);
    if (!window_func_project_node) {
      continue;
    }
    project_node_counter++;
    if (!window_func_project_node->hasWindowFunctionExpr()) {
      // this projection node does not have a window function
      // expression -- skip to the next node in the DAG.
      continue;
    }

    const auto prev_node_itr = std::prev(node_itr);
    const auto prev_node = *prev_node_itr;
    CHECK(prev_node);

    auto filter_node = std::dynamic_pointer_cast<RelFilter>(prev_node);

    auto scan_node = std::dynamic_pointer_cast<RelScan>(prev_node);
    const bool has_multi_fragment_scan_input =
        (scan_node && scan_node->getNumFragments() > 1) ? true : false;

    // We currently add a preceding project node in one of two conditions:
    // 1. always_add_project_if_first_project_is_window_expr = true, which
    // we currently only set for distributed, but could also be set to support
    // multi-frag window function inputs, either if we can detect that an input table
    // is multi-frag up front, or using a retry mechanism like we do for join filter
    // push down.
    // TODO(todd): Investigate a viable approach for the above.
    // 2. Regardless of #1, if the window function project node is preceded by a
    // filter node. This is required both for correctness and to avoid pulling
    // all source input columns into memory since non-coalesced filter node
    // inputs are currently not pruned or eliminated via dead column elimination.
    // Note that we expect any filter node followed by a project node to be coalesced
    // into a single compound node in RelAlgDagBuilder::coalesce_nodes, and that action
    // prunes unused inputs.
    // TODO(todd): Investigate whether the shotgun filter node issue affects other
    // query plans, i.e. filters before joins, and whether there is a more general
    // approach to solving this (will still need the preceding project node for
    // window functions preceded by filter nodes for correctness though)

    if (!((always_add_project_if_first_project_is_window_expr &&
           project_node_counter == 1) ||
          filter_node || has_multi_fragment_scan_input)) {
      continue;
    }

    InputSet inputs;
    InputCollector input_collector;
    for (size_t i = 0; i < window_func_project_node->size(); i++) {
      auto new_inputs = input_collector.visit(window_func_project_node->getExpr(i).get());
      inputs.insert(new_inputs.begin(), new_inputs.end());
    }

    // Note: Technically not required since we are mapping old inputs to new input
    // indices, but makes the re-mapping of inputs easier to follow.
    std::vector<std::pair<const RelAlgNode*, unsigned>> sorted_inputs(inputs.begin(),
                                                                      inputs.end());
    std::sort(sorted_inputs.begin(),
              sorted_inputs.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    hdk::ir::ExprPtrVector exprs;
    std::vector<std::string> fields;
    std::unordered_map<unsigned, unsigned> old_index_to_new_index;
    for (auto& input : sorted_inputs) {
      CHECK_EQ(input.first, prev_node.get());
      auto res =
          old_index_to_new_index.insert(std::make_pair(input.second, exprs.size()));
      CHECK(res.second);
      exprs.emplace_back(hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          getColumnType(input.first, input.second), input.first, input.second));
      fields.emplace_back("");
    }

    auto new_project = std::make_shared<RelProject>(std::move(exprs), fields, prev_node);
    propagate_hints_to_new_project(window_func_project_node, new_project, query_hints);
    node_list.insert(node_itr, new_project);
    window_func_project_node->replaceInput(
        prev_node, new_project, old_index_to_new_index);
  }

  nodes.assign(node_list.begin(), node_list.end());
}

int64_t get_int_literal_field(const rapidjson::Value& obj,
                              const char field[],
                              const int64_t default_val) noexcept {
  const auto it = obj.FindMember(field);
  if (it == obj.MemberEnd()) {
    return default_val;
  }
  auto expr = parseLiteral(it->value);
  CHECK(expr->get_type_info().is_integer());
  return dynamic_cast<const hdk::ir::Constant*>(expr.get())->intVal();
}

void check_empty_inputs_field(const rapidjson::Value& node) noexcept {
  const auto& inputs_json = field(node, "inputs");
  CHECK(inputs_json.IsArray() && !inputs_json.Size());
}

TableInfoPtr getTableFromScanNode(int db_id,
                                  SchemaProviderPtr schema_provider,
                                  const rapidjson::Value& scan_ra) {
  const auto& table_json = field(scan_ra, "table");
  CHECK(table_json.IsArray());
  CHECK_EQ(unsigned(2), table_json.Size());
  const auto info = schema_provider->getTableInfo(db_id, table_json[1].GetString());
  CHECK(info);
  return info;
}

std::vector<std::string> getFieldNamesFromScanNode(const rapidjson::Value& scan_ra) {
  const auto& fields_json = field(scan_ra, "fieldNames");
  return strings_from_json_array(fields_json);
}

}  // namespace

bool RelProject::hasWindowFunctionExpr() const {
  for (const auto& expr : exprs_) {
    if (is_window_function_expr(expr.get())) {
      return true;
    }
  }
  return false;
}
namespace details {

class RelAlgDispatcher {
 public:
  RelAlgDispatcher(int db_id, SchemaProviderPtr schema_provider)
      : db_id_(db_id), schema_provider_(schema_provider) {}

  std::vector<std::shared_ptr<RelAlgNode>> run(const rapidjson::Value& rels,
                                               RelAlgDagBuilder& root_dag_builder) {
    for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
      const auto& crt_node = *rels_it;
      const auto id = node_id(crt_node);
      CHECK_EQ(static_cast<size_t>(id), nodes_.size());
      CHECK(crt_node.IsObject());
      std::shared_ptr<RelAlgNode> ra_node = nullptr;
      const auto rel_op = json_str(field(crt_node, "relOp"));
      if (rel_op == std::string("EnumerableTableScan") ||
          rel_op == std::string("LogicalTableScan")) {
        ra_node = dispatchTableScan(crt_node);
      } else if (rel_op == std::string("LogicalProject")) {
        ra_node = dispatchProject(crt_node, root_dag_builder);
      } else if (rel_op == std::string("LogicalFilter")) {
        ra_node = dispatchFilter(crt_node, root_dag_builder);
      } else if (rel_op == std::string("LogicalAggregate")) {
        ra_node = dispatchAggregate(crt_node, root_dag_builder);
      } else if (rel_op == std::string("LogicalJoin")) {
        ra_node = dispatchJoin(crt_node, root_dag_builder);
      } else if (rel_op == std::string("LogicalSort")) {
        ra_node = dispatchSort(crt_node);
      } else if (rel_op == std::string("LogicalValues")) {
        ra_node = dispatchLogicalValues(crt_node);
      } else if (rel_op == std::string("LogicalTableFunctionScan")) {
        ra_node = dispatchTableFunction(crt_node, root_dag_builder);
      } else if (rel_op == std::string("LogicalUnion")) {
        ra_node = dispatchUnion(crt_node);
      } else {
        throw QueryNotSupported(std::string("Node ") + rel_op + " not supported yet");
      }
      nodes_.push_back(ra_node);
    }

    return std::move(nodes_);
  }

 private:
  std::shared_ptr<RelScan> dispatchTableScan(const rapidjson::Value& scan_ra) {
    check_empty_inputs_field(scan_ra);
    CHECK(scan_ra.IsObject());
    const auto tinfo = getTableFromScanNode(db_id_, schema_provider_, scan_ra);
    const auto field_names = getFieldNamesFromScanNode(scan_ra);
    std::vector<ColumnInfoPtr> infos;
    infos.reserve(field_names.size());
    for (auto& col_name : field_names) {
      infos.emplace_back(schema_provider_->getColumnInfo(*tinfo, col_name));
      CHECK(infos.back());
    }
    auto scan_node = std::make_shared<RelScan>(tinfo, std::move(infos));
    if (scan_ra.HasMember("hints")) {
      getRelAlgHints(scan_ra, scan_node);
    }
    return scan_node;
  }

  std::shared_ptr<RelProject> dispatchProject(const rapidjson::Value& proj_ra,
                                              RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(proj_ra);
    CHECK_EQ(size_t(1), inputs.size());
    const auto& exprs_json = field(proj_ra, "exprs");
    CHECK(exprs_json.IsArray());
    hdk::ir::ExprPtrVector exprs;
    for (auto exprs_json_it = exprs_json.Begin(); exprs_json_it != exprs_json.End();
         ++exprs_json_it) {
      exprs.emplace_back(parse_expr(*exprs_json_it,
                                    db_id_,
                                    schema_provider_,
                                    root_dag_builder,
                                    getNodeColumnRefs(inputs[0].get())));
    }
    const auto& fields = field(proj_ra, "fields");
    if (proj_ra.HasMember("hints")) {
      auto project_node = std::make_shared<RelProject>(
          std::move(exprs), strings_from_json_array(fields), inputs.front());
      getRelAlgHints(proj_ra, project_node);
      return project_node;
    }
    return std::make_shared<RelProject>(
        std::move(exprs), strings_from_json_array(fields), inputs.front());
  }

  std::shared_ptr<RelFilter> dispatchFilter(const rapidjson::Value& filter_ra,
                                            RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(filter_ra);
    CHECK_EQ(size_t(1), inputs.size());
    const auto id = node_id(filter_ra);
    CHECK(id);
    auto condition_expr = parse_expr(field(filter_ra, "condition"),
                                     db_id_,
                                     schema_provider_,
                                     root_dag_builder,
                                     getNodeColumnRefs(inputs[0].get()));
    return std::make_shared<RelFilter>(std::move(condition_expr), inputs.front());
  }

  std::shared_ptr<RelAggregate> dispatchAggregate(const rapidjson::Value& agg_ra,
                                                  RelAlgDagBuilder& root_dag_builder) {
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
    hdk::ir::ExprPtrVector input_exprs = getInputExprsForAgg(inputs[0].get());
    hdk::ir::ExprPtrVector aggs;
    for (auto aggs_json_arr_it = aggs_json_arr.Begin();
         aggs_json_arr_it != aggs_json_arr.End();
         ++aggs_json_arr_it) {
      auto agg = parseAggregateExpr(*aggs_json_arr_it, root_dag_builder, input_exprs);
      aggs.push_back(agg);
    }
    auto agg_node = std::make_shared<RelAggregate>(
        group.size(), std::move(aggs), fields, inputs.front());
    if (agg_ra.HasMember("hints")) {
      getRelAlgHints(agg_ra, agg_node);
    }
    return agg_node;
  }

  std::shared_ptr<RelJoin> dispatchJoin(const rapidjson::Value& join_ra,
                                        RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(join_ra);
    CHECK_EQ(size_t(2), inputs.size());
    const auto join_type = to_join_type(json_str(field(join_ra, "joinType")));
    auto ra_outputs = genColumnRefs(inputs[0].get(), inputs[0]->size());
    const auto ra_outputs2 = genColumnRefs(inputs[1].get(), inputs[1]->size());
    ra_outputs.insert(ra_outputs.end(), ra_outputs2.begin(), ra_outputs2.end());
    auto condition = parse_expr(field(join_ra, "condition"),
                                db_id_,
                                schema_provider_,
                                root_dag_builder,
                                ra_outputs);
    auto join_node =
        std::make_shared<RelJoin>(inputs[0], inputs[1], std::move(condition), join_type);
    if (join_ra.HasMember("hints")) {
      getRelAlgHints(join_ra, join_node);
    }
    return join_node;
  }

  std::shared_ptr<RelSort> dispatchSort(const rapidjson::Value& sort_ra) {
    const auto inputs = getRelAlgInputs(sort_ra);
    CHECK_EQ(size_t(1), inputs.size());
    std::vector<SortField> collation;
    const auto& collation_arr = field(sort_ra, "collation");
    CHECK(collation_arr.IsArray());
    for (auto collation_arr_it = collation_arr.Begin();
         collation_arr_it != collation_arr.End();
         ++collation_arr_it) {
      const size_t field_idx = json_i64(field(*collation_arr_it, "field"));
      const auto sort_dir = parse_sort_direction(*collation_arr_it);
      const auto null_pos = parse_nulls_position(*collation_arr_it);
      collation.emplace_back(field_idx, sort_dir, null_pos);
    }
    auto limit = get_int_literal_field(sort_ra, "fetch", -1);
    const auto offset = get_int_literal_field(sort_ra, "offset", 0);
    auto ret = std::make_shared<RelSort>(
        collation, limit > 0 ? limit : 0, offset, inputs.front());
    ret->setEmptyResult(limit == 0);
    return ret;
  }

  std::vector<TargetMetaInfo> parseTupleType(const rapidjson::Value& tuple_type_arr) {
    CHECK(tuple_type_arr.IsArray());
    std::vector<TargetMetaInfo> tuple_type;
    for (auto tuple_type_arr_it = tuple_type_arr.Begin();
         tuple_type_arr_it != tuple_type_arr.End();
         ++tuple_type_arr_it) {
      auto component_type = parseType(*tuple_type_arr_it);
      const auto component_name = json_str(field(*tuple_type_arr_it, "name"));
      tuple_type.emplace_back(component_name, component_type->toTypeInfo());
    }
    return tuple_type;
  }

  std::shared_ptr<RelTableFunction> dispatchTableFunction(
      const rapidjson::Value& table_func_ra,
      RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(table_func_ra);
    const auto& invocation = field(table_func_ra, "invocation");
    CHECK(invocation.IsObject());

    const auto& operands = field(invocation, "operands");
    CHECK(operands.IsArray());
    CHECK_GE(operands.Size(), unsigned(0));

    hdk::ir::ExprPtrVector col_input_exprs;
    hdk::ir::ExprPtrVector table_func_input_exprs;
    std::vector<std::string> fields;

    for (auto exprs_json_it = operands.Begin(); exprs_json_it != operands.End();
         ++exprs_json_it) {
      const auto& expr_json = *exprs_json_it;
      CHECK(expr_json.IsObject());
      if (expr_json.HasMember("op")) {
        const auto op_str = json_str(field(expr_json, "op"));
        if (op_str == "CAST" && expr_json.HasMember("type")) {
          const auto& expr_type = field(expr_json, "type");
          CHECK(expr_type.IsObject());
          CHECK(expr_type.HasMember("type"));
          const auto& expr_type_name = json_str(field(expr_type, "type"));
          if (expr_type_name == "CURSOR") {
            CHECK(expr_json.HasMember("operands"));
            const auto& expr_operands = field(expr_json, "operands");
            CHECK(expr_operands.IsArray());
            if (expr_operands.Size() != 1) {
              throw std::runtime_error(
                  "Table functions currently only support one ResultSet input");
            }
            auto pos = field(expr_operands[0], "input").GetInt();
            CHECK_LT(pos, inputs.size());
            for (size_t i = inputs[pos]->size(); i > 0; i--) {
              unsigned col_idx = static_cast<unsigned>(inputs[pos]->size() - i);
              table_func_input_exprs.emplace_back(hdk::ir::makeExpr<hdk::ir::ColumnRef>(
                  getColumnType(inputs[pos].get(), col_idx), inputs[pos].get(), col_idx));
              col_input_exprs.push_back(table_func_input_exprs.back());
            }
            continue;
          }
        }
      }
      hdk::ir::ExprPtrVector ra_output;
      for (auto& node : inputs) {
        auto node_output = getNodeColumnRefs(node.get());
        ra_output.insert(ra_output.end(), node_output.begin(), node_output.end());
      }
      table_func_input_exprs.emplace_back(parse_expr(
          *exprs_json_it, db_id_, schema_provider_, root_dag_builder, ra_output));
    }

    const auto& op_name = field(invocation, "op");
    CHECK(op_name.IsString());

    const auto& row_types = field(table_func_ra, "rowType");
    std::vector<TargetMetaInfo> tuple_type = parseTupleType(row_types);
    // Calcite doesn't always put proper type for the table function result.
    if (op_name.GetString() == "sort_column_limit"s ||
        op_name.GetString() == "ct_binding_scalar_multiply"s ||
        op_name.GetString() == "column_list_safe_row_sum"s ||
        op_name.GetString() == "ct_named_rowmul_output"s) {
      CHECK_EQ(tuple_type.size(), 1);
      tuple_type[0] = TargetMetaInfo(tuple_type[0].get_resname(),
                                     col_input_exprs[0]->get_type_info());
    } else if (op_name.GetString() == "ct_scalar_1_arg_runtime_sizing"s) {
      CHECK_EQ(tuple_type.size(), 1);
      if (table_func_input_exprs[0]->get_type_info().is_integer()) {
        tuple_type[0] =
            TargetMetaInfo(tuple_type[0].get_resname(), SQLTypeInfo(kBIGINT, false));
      } else {
        CHECK(table_func_input_exprs[0]->get_type_info().is_fp());
        tuple_type[0] =
            TargetMetaInfo(tuple_type[0].get_resname(), SQLTypeInfo(kDOUBLE, false));
      }
    } else if (op_name.GetString() == "ct_templated_no_cursor_user_constant_sizer"s) {
      CHECK_EQ(tuple_type.size(), 1);
      tuple_type[0] = TargetMetaInfo(tuple_type[0].get_resname(),
                                     table_func_input_exprs[0]->get_type_info());
    } else if (op_name.GetString() == "ct_binding_column2"s) {
      CHECK_EQ(tuple_type.size(), 1);
      if (col_input_exprs[0]->get_type_info().is_string()) {
        tuple_type[0] = TargetMetaInfo(tuple_type[0].get_resname(),
                                       col_input_exprs[0]->get_type_info());
      }
    }
    for (size_t i = 0; i < tuple_type.size(); i++) {
      fields.emplace_back("");
    }
    return std::make_shared<RelTableFunction>(op_name.GetString(),
                                              inputs,
                                              fields,
                                              std::move(col_input_exprs),
                                              std::move(table_func_input_exprs),
                                              std::move(tuple_type));
  }

  std::shared_ptr<RelLogicalValues> dispatchLogicalValues(
      const rapidjson::Value& logical_values_ra) {
    const auto& tuple_type_arr = field(logical_values_ra, "type");
    std::vector<TargetMetaInfo> tuple_type = parseTupleType(tuple_type_arr);
    const auto& inputs_arr = field(logical_values_ra, "inputs");
    CHECK(inputs_arr.IsArray());
    const auto& tuples_arr = field(logical_values_ra, "tuples");
    CHECK(tuples_arr.IsArray());

    if (inputs_arr.Size()) {
      throw QueryNotSupported("Inputs not supported in logical values yet.");
    }

    std::vector<hdk::ir::ExprPtrVector> values;
    if (tuples_arr.Size()) {
      for (const auto& row : tuples_arr.GetArray()) {
        CHECK(row.IsArray());
        const auto values_json = row.GetArray();
        if (!values.empty()) {
          CHECK_EQ(values[0].size(), values_json.Size());
        }
        values.emplace_back(hdk::ir::ExprPtrVector{});
        for (const auto& value : values_json) {
          CHECK(value.IsObject());
          CHECK(value.HasMember("literal"));
          values.back().emplace_back(parseLiteral(value));
        }
      }
    }

    return std::make_shared<RelLogicalValues>(tuple_type, std::move(values));
  }

  std::shared_ptr<RelLogicalUnion> dispatchUnion(
      const rapidjson::Value& logical_union_ra) {
    auto inputs = getRelAlgInputs(logical_union_ra);
    auto const& all_type_bool = field(logical_union_ra, "all");
    CHECK(all_type_bool.IsBool());
    return std::make_shared<RelLogicalUnion>(std::move(inputs), all_type_bool.GetBool());
  }

  RelAlgInputs getRelAlgInputs(const rapidjson::Value& node) {
    if (node.HasMember("inputs")) {
      const auto str_input_ids = strings_from_json_array(field(node, "inputs"));
      RelAlgInputs ra_inputs;
      for (const auto& str_id : str_input_ids) {
        ra_inputs.push_back(nodes_[std::stoi(str_id)]);
      }
      return ra_inputs;
    }
    return {prev(node)};
  }

  std::pair<std::string, std::string> getKVOptionPair(std::string& str, size_t& pos) {
    auto option = str.substr(0, pos);
    std::string delim = "=";
    size_t delim_pos = option.find(delim);
    auto key = option.substr(0, delim_pos);
    auto val = option.substr(delim_pos + 1, option.length());
    str.erase(0, pos + delim.length() + 1);
    return {key, val};
  }

  ExplainedQueryHint parseHintString(std::string& hint_string) {
    std::string white_space_delim = " ";
    int l = hint_string.length();
    hint_string = hint_string.erase(0, 1).substr(0, l - 2);
    size_t pos = 0;
    if ((pos = hint_string.find("options:")) != std::string::npos) {
      // need to parse hint options
      std::vector<std::string> tokens;
      std::string hint_name = hint_string.substr(0, hint_string.find(white_space_delim));
      auto hint_type = RegisteredQueryHint::translateQueryHint(hint_name);
      bool kv_list_op = false;
      std::string raw_options = hint_string.substr(pos + 8, hint_string.length() - 2);
      if (raw_options.find('{') != std::string::npos) {
        kv_list_op = true;
      } else {
        CHECK(raw_options.find('[') != std::string::npos);
      }
      auto t1 = raw_options.erase(0, 1);
      raw_options = t1.substr(0, t1.length() - 1);
      std::string op_delim = ", ";
      if (kv_list_op) {
        // kv options
        std::unordered_map<std::string, std::string> kv_options;
        while ((pos = raw_options.find(op_delim)) != std::string::npos) {
          auto kv_pair = getKVOptionPair(raw_options, pos);
          kv_options.emplace(kv_pair.first, kv_pair.second);
        }
        // handle the last kv pair
        auto kv_pair = getKVOptionPair(raw_options, pos);
        kv_options.emplace(kv_pair.first, kv_pair.second);
        return {hint_type, true, false, true, kv_options};
      } else {
        std::vector<std::string> list_options;
        while ((pos = raw_options.find(op_delim)) != std::string::npos) {
          list_options.emplace_back(raw_options.substr(0, pos));
          raw_options.erase(0, pos + white_space_delim.length() + 1);
        }
        // handle the last option
        list_options.emplace_back(raw_options.substr(0, pos));
        return {hint_type, true, false, false, list_options};
      }
    } else {
      // marker hint: no extra option for this hint
      std::string hint_name = hint_string.substr(0, hint_string.find(white_space_delim));
      auto hint_type = RegisteredQueryHint::translateQueryHint(hint_name);
      return {hint_type, true, true, false};
    }
  }

  void getRelAlgHints(const rapidjson::Value& json_node,
                      std::shared_ptr<RelAlgNode> node) {
    std::string hint_explained = json_str(field(json_node, "hints"));
    size_t pos = 0;
    std::string delim = "|";
    std::vector<std::string> hint_list;
    while ((pos = hint_explained.find(delim)) != std::string::npos) {
      hint_list.emplace_back(hint_explained.substr(0, pos));
      hint_explained.erase(0, pos + delim.length());
    }
    // handling the last one
    hint_list.emplace_back(hint_explained.substr(0, pos));

    const auto agg_node = std::dynamic_pointer_cast<RelAggregate>(node);
    if (agg_node) {
      for (std::string& hint : hint_list) {
        auto parsed_hint = parseHintString(hint);
        agg_node->addHint(parsed_hint);
      }
    }
    const auto project_node = std::dynamic_pointer_cast<RelProject>(node);
    if (project_node) {
      for (std::string& hint : hint_list) {
        auto parsed_hint = parseHintString(hint);
        project_node->addHint(parsed_hint);
      }
    }
    const auto scan_node = std::dynamic_pointer_cast<RelScan>(node);
    if (scan_node) {
      for (std::string& hint : hint_list) {
        auto parsed_hint = parseHintString(hint);
        scan_node->addHint(parsed_hint);
      }
    }
    const auto join_node = std::dynamic_pointer_cast<RelJoin>(node);
    if (join_node) {
      for (std::string& hint : hint_list) {
        auto parsed_hint = parseHintString(hint);
        join_node->addHint(parsed_hint);
      }
    }

    const auto compound_node = std::dynamic_pointer_cast<RelCompound>(node);
    if (compound_node) {
      for (std::string& hint : hint_list) {
        auto parsed_hint = parseHintString(hint);
        compound_node->addHint(parsed_hint);
      }
    }
  }

  std::shared_ptr<const RelAlgNode> prev(const rapidjson::Value& crt_node) {
    const auto id = node_id(crt_node);
    CHECK(id);
    CHECK_EQ(static_cast<size_t>(id), nodes_.size());
    return nodes_.back();
  }

  int db_id_;
  SchemaProviderPtr schema_provider_;
  std::vector<std::shared_ptr<RelAlgNode>> nodes_;
};

}  // namespace details

void RelAlgDag::eachNode(std::function<void(RelAlgNode const*)> const& callback) const {
  for (auto const& node : nodes_) {
    if (node) {
      callback(node.get());
    }
  }
}

void RelAlgDag::registerQueryHints(std::shared_ptr<RelAlgNode> node,
                                   Hints* hints_delivered) {
  bool detect_columnar_output_hint = false;
  bool detect_rowwise_output_hint = false;
  RegisteredQueryHint query_hint = RegisteredQueryHint::fromConfig(*config_);
  for (auto it = hints_delivered->begin(); it != hints_delivered->end(); it++) {
    auto target = it->second;
    auto hint_type = it->first;
    switch (hint_type) {
      case QueryHint::kCpuMode: {
        query_hint.registerHint(QueryHint::kCpuMode);
        query_hint.cpu_mode = true;
        break;
      }
      case QueryHint::kColumnarOutput: {
        detect_columnar_output_hint = true;
        break;
      }
      case QueryHint::kRowwiseOutput: {
        detect_rowwise_output_hint = true;
        break;
      }
      default:
        break;
    }
  }
  // we have four cases depending on 1) enable_columnar_output flag
  // and 2) query hint status: columnar_output and rowwise_output
  // case 1. enable_columnar_output = true
  // case 1.a) columnar_output = true (so rowwise_output = false);
  // case 1.b) rowwise_output = true (so columnar_output = false);
  // case 2. enable_columnar_output = false
  // case 2.a) columnar_output = true (so rowwise_output = false);
  // case 2.b) rowwise_output = true (so columnar_output = false);
  // case 1.a --> use columnar output
  // case 1.b --> use rowwise output
  // case 2.a --> use columnar output
  // case 2.b --> use rowwise output
  if (detect_columnar_output_hint && detect_rowwise_output_hint) {
    VLOG(1) << "Two hints 1) columnar output and 2) rowwise output are enabled together, "
            << "so skip them and use the runtime configuration "
               "\"enable_columnar_output\"";
  } else if (detect_columnar_output_hint && !detect_rowwise_output_hint) {
    if (config_->rs.enable_columnar_output) {
      VLOG(1) << "We already enable columnar output by default "
                 "(g_enable_columnar_output = true), so skip this columnar output hint";
    } else {
      query_hint.registerHint(QueryHint::kColumnarOutput);
      query_hint.columnar_output = true;
    }
  } else if (!detect_columnar_output_hint && detect_rowwise_output_hint) {
    if (!config_->rs.enable_columnar_output) {
      VLOG(1) << "We already use the default rowwise output (g_enable_columnar_output "
                 "= false), so skip this rowwise output hint";
    } else {
      query_hint.registerHint(QueryHint::kRowwiseOutput);
      query_hint.rowwise_output = true;
    }
  }
  query_hint_.emplace(node->toHash(), query_hint);
}

void RelAlgDag::resetQueryExecutionState() {
  for (auto& node : nodes_) {
    if (node) {
      node->resetQueryExecutionState();
    }
  }
}

RelAlgDagBuilder::RelAlgDagBuilder(RelAlgDagBuilder& root_dag_builder,
                                   const rapidjson::Value& query_ast,
                                   int db_id,
                                   SchemaProviderPtr schema_provider)
    : RelAlgDag(root_dag_builder.config_, root_dag_builder.now())
    , db_id_(db_id)
    , schema_provider_(schema_provider) {
  build(query_ast, root_dag_builder);
}

RelAlgDagBuilder::RelAlgDagBuilder(const std::string& query_ra,
                                   int db_id,
                                   SchemaProviderPtr schema_provider,
                                   ConfigPtr config)
    : RelAlgDag(config), db_id_(db_id), schema_provider_(schema_provider) {
  rapidjson::Document query_ast;
  query_ast.Parse(query_ra.c_str());
  VLOG(2) << "Parsing query RA JSON: " << query_ra;
  if (query_ast.HasParseError()) {
    query_ast.GetParseError();
    LOG(ERROR) << "Failed to parse RA tree from Calcite (offset "
               << query_ast.GetErrorOffset() << "):\n"
               << rapidjson::GetParseError_En(query_ast.GetParseError());
    VLOG(1) << "Failed to parse query RA: " << query_ra;
    throw std::runtime_error(
        "Failed to parse relational algebra tree. Possible query syntax error.");
  }
  CHECK(query_ast.IsObject());
  RelAlgNode::resetRelAlgFirstId();
  build(query_ast, *this);
}

void RelAlgDagBuilder::build(const rapidjson::Value& query_ast,
                             RelAlgDagBuilder& lead_dag_builder) {
  const auto& rels = field(query_ast, "rels");
  CHECK(rels.IsArray());
  try {
    nodes_ =
        details::RelAlgDispatcher(db_id_, schema_provider_).run(rels, lead_dag_builder);
  } catch (const QueryNotSupported&) {
    throw;
  }
  CHECK(!nodes_.empty());

  handleQueryHint(nodes_, this);
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
  eliminate_dead_subqueries(subqueries_, nodes_.back().get());
  separate_window_function_expressions(nodes_, query_hint_);
  add_window_function_pre_project(
      nodes_,
      false /* always_add_project_if_first_project_is_window_expr */,
      query_hint_);
  coalesce_nodes(nodes_, left_deep_joins, query_hint_);
  CHECK(nodes_.back().use_count() == 1);
  create_left_deep_join(nodes_);
  CHECK(nodes_.size());
  root_ = nodes_.back();
}

// Return tree with depth represented by indentations.
std::string tree_string(const RelAlgNode* ra, const size_t depth) {
  std::string result = std::string(2 * depth, ' ') + ::toString(ra) + '\n';
  for (size_t i = 0; i < ra->inputCount(); ++i) {
    result += tree_string(ra->getInput(i), depth + 1);
  }
  return result;
}

std::string RelCompound::toString() const {
  return cat(::typeName(this),
             getIdString(),
             "(filter=",
             (filter_ ? filter_->toString() : "null"),
             ", ",
             std::to_string(groupby_count_),
             ", fields=",
             ::toString(fields_),
             ", groupby_exprs=",
             ::toString(groupby_exprs_),
             ", exprs=",
             ::toString(exprs_),
             ", is_agg=",
             std::to_string(is_agg_),
             ", inputs=",
             inputsToString(inputs_),
             ")");
}

size_t RelCompound::toHash() const {
  if (!hash_) {
    hash_ = typeid(RelCompound).hash_code();
    boost::hash_combine(*hash_, filter_ ? filter_->hash() : boost::hash_value("n"));
    boost::hash_combine(*hash_, is_agg_);
    for (auto& expr : exprs_) {
      boost::hash_combine(*hash_, expr->hash());
    }
    for (auto& expr : groupby_exprs_) {
      boost::hash_combine(*hash_, expr->hash());
    }
    boost::hash_combine(*hash_, groupby_count_);
    boost::hash_combine(*hash_, ::toString(fields_));
  }
  return *hash_;
}

SQLTypeInfo getColumnType(const RelAlgNode* node, size_t col_idx) {
  // By default use metainfo.
  const auto& metainfo = node->getOutputMetainfo();
  if (metainfo.size() > col_idx) {
    return metainfo[col_idx].get_type_info();
  }

  // For scans we can use embedded column info.
  const auto scan = dynamic_cast<const RelScan*>(node);
  if (scan) {
    return scan->getColumnTypeBySpi(col_idx + 1);
  }

  // For filter, sort and union we can propagate column type of
  // their sources.
  if (is_one_of<RelFilter, RelSort, RelLogicalUnion>(node)) {
    return getColumnType(node->getInput(0), col_idx);
  }

  // For aggregates we can we can propagate type from group key
  // or extract type from AggExpr
  const auto agg = dynamic_cast<const RelAggregate*>(node);
  if (agg) {
    if (col_idx < agg->getGroupByCount()) {
      return getColumnType(agg->getInput(0), col_idx);
    } else {
      return agg->getAggs()[col_idx - agg->getGroupByCount()]->get_type_info();
    }
  }

  // For logical values we can use its tuple type.
  const auto values = dynamic_cast<const RelLogicalValues*>(node);
  if (values) {
    CHECK_GT(values->size(), col_idx);
    if (values->getTupleType()[col_idx].get_type_info().get_type() == kNULLT) {
      // replace w/ bigint
      return SQLTypeInfo(kBIGINT, false);
    }
    return values->getTupleType()[col_idx].get_type_info();
  }

  // For table functions we can use its tuple type.
  const auto table_fn = dynamic_cast<const RelTableFunction*>(node);
  if (table_fn) {
    CHECK_GT(table_fn->size(), col_idx);
    return table_fn->getTupleType()[col_idx].get_type_info();
  }

  // For projections type can be extracted from Exprs.
  const auto proj = dynamic_cast<const RelProject*>(node);
  if (proj) {
    CHECK_GT(proj->getExprs().size(), col_idx);
    return proj->getExprs()[col_idx]->get_type_info();
  }

  // For joins we can propagate type from one of its sources.
  const auto join = dynamic_cast<const RelJoin*>(node);
  if (join) {
    CHECK_GT(join->size(), col_idx);
    if (col_idx < join->getInput(0)->size()) {
      return getColumnType(join->getInput(0), col_idx);
    } else {
      return getColumnType(join->getInput(1), col_idx - join->getInput(0)->size());
    }
  }

  const auto deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(node);
  if (deep_join) {
    CHECK_GT(deep_join->size(), col_idx);
    unsigned offs = 0;
    for (size_t i = 0; i < join->inputCount(); ++i) {
      auto input = join->getInput(i);
      if (col_idx - offs < input->size()) {
        return getColumnType(input, col_idx - offs);
      }
      offs += input->size();
    }
  }

  // For coumpounds type can be extracted from Exprs.
  const auto compound = dynamic_cast<const RelCompound*>(node);
  if (compound) {
    CHECK_GT(compound->size(), col_idx);
    return compound->getExprs()[col_idx]->get_type_info();
  }

  CHECK(false) << "Missing output metainfo for node " + node->toString() +
                      " col_idx=" + std::to_string(col_idx);
  return {};
}

hdk::ir::ExprPtrVector getInputExprsForAgg(const RelAlgNode* node) {
  hdk::ir::ExprPtrVector res;
  res.reserve(node->size());
  auto project = dynamic_cast<const RelProject*>(node);
  auto compound = dynamic_cast<const RelCompound*>(node);
  if (project || compound) {
    const auto& exprs = project ? project->getExprs() : compound->getExprs();
    for (unsigned col_idx = 0; col_idx < static_cast<unsigned>(exprs.size()); ++col_idx) {
      auto& expr = exprs[col_idx];
      if (dynamic_cast<const hdk::ir::Constant*>(expr.get())) {
        res.emplace_back(expr);
      } else {
        res.emplace_back(
            hdk::ir::makeExpr<hdk::ir::ColumnRef>(expr->get_type_info(), node, col_idx));
      }
    }
  } else if (is_one_of<RelLogicalValues, RelAggregate, RelLogicalUnion, RelTableFunction>(
                 node)) {
    res = getNodeColumnRefs(node);
  } else {
    CHECK(false) << "Unexpected node: " << node->toString();
  }

  return res;
}
