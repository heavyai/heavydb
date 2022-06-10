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
#include "Descriptors/RelAlgExecutionDescriptor.h"
#include "JsonAccessors.h"
#include "RelAlgOptimizer.h"
#include "RelLeftDeepInnerJoin.h"
#include "RexVisitor.h"
#include "Shared/sqldefs.h"

#include <rapidjson/error/en.h>
#include <rapidjson/error/error.h>
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

void RexSubQuery::setExecutionResult(
    const std::shared_ptr<const ExecutionResult> result) {
  auto row_set = result->getRows();
  CHECK(row_set);
  CHECK_EQ(size_t(1), row_set->colCount());
  *(type_.get()) = row_set->getColType(0);
  (*(result_.get())) = result;
}

std::unique_ptr<RexSubQuery> RexSubQuery::deepCopy() const {
  return std::make_unique<RexSubQuery>(type_, result_, ra_->deepCopy());
}

unsigned RexSubQuery::getId() const {
  return ra_->getId();
}

namespace {

class RexRebindInputsVisitor : public RexVisitor<void*> {
 public:
  RexRebindInputsVisitor(const RelAlgNode* old_input, const RelAlgNode* new_input)
      : old_input_(old_input), new_input_(new_input) {}

  virtual ~RexRebindInputsVisitor() = default;

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
  outputs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    outputs.emplace_back(node, i);
  }
  return outputs;
}

class RexRebindReindexInputsVisitor : public RexRebindInputsVisitor {
 public:
  RexRebindReindexInputsVisitor(
      const RelAlgNode* old_input,
      const RelAlgNode* new_input,
      std::unordered_map<unsigned, unsigned> old_to_new_index_map)
      : RexRebindInputsVisitor(old_input, new_input), mapping_(old_to_new_index_map) {}

  void* visitInput(const RexInput* rex_input) const override {
    RexRebindInputsVisitor::visitInput(rex_input);
    auto mapping_itr = mapping_.find(rex_input->getIndex());
    CHECK(mapping_itr != mapping_.end());
    rex_input->setIndex(mapping_itr->second);
    return nullptr;
  }

 private:
  const std::unordered_map<unsigned, unsigned> mapping_;
};

}  // namespace

void RelProject::replaceInput(
    std::shared_ptr<const RelAlgNode> old_input,
    std::shared_ptr<const RelAlgNode> input,
    std::optional<std::unordered_map<unsigned, unsigned>> old_to_new_index_map) {
  RelAlgNode::replaceInput(old_input, input);
  std::unique_ptr<RexRebindInputsVisitor> rebind_inputs;
  if (old_to_new_index_map) {
    rebind_inputs = std::make_unique<RexRebindReindexInputsVisitor>(
        old_input.get(), input.get(), *old_to_new_index_map);
  } else {
    rebind_inputs =
        std::make_unique<RexRebindInputsVisitor>(old_input.get(), input.get());
  }
  CHECK(rebind_inputs);
  for (const auto& scalar_expr : scalar_exprs_) {
    rebind_inputs->visit(scalar_expr.get());
  }
}

void RelProject::appendInput(std::string new_field_name,
                             std::unique_ptr<const RexScalar> new_input) {
  fields_.emplace_back(std::move(new_field_name));
  scalar_exprs_.emplace_back(std::move(new_input));
}

RANodeOutput get_node_output(const RelAlgNode* ra_node) {
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
    auto lhs_out =
        n_outputs(join_node->getInput(0), get_node_output(join_node->getInput(0)).size());
    const auto rhs_out =
        n_outputs(join_node->getInput(1), get_node_output(join_node->getInput(1)).size());
    lhs_out.insert(lhs_out.end(), rhs_out.begin(), rhs_out.end());
    return lhs_out;
  }
  const auto table_func_node = dynamic_cast<const RelTableFunction*>(ra_node);
  if (table_func_node) {
    // Table Function output count doesn't depend on the input
    return n_outputs(table_func_node, table_func_node->size());
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
  const auto logical_union_node = dynamic_cast<const RelLogicalUnion*>(ra_node);
  if (logical_union_node) {
    return n_outputs(logical_union_node, logical_union_node->size());
  }
  LOG(FATAL) << "Unhandled ra_node type: " << ::toString(ra_node);
  return {};
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
    // We should add the additional check that input->getIndex() !=
    // source_shape[i].getIndex(), but Calcite doesn't generate the right
    // Sort-Project-Sort sequence when joins are involved.
    if (input->getSourceNode() != source_shape[i].getSourceNode()) {
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

void RelJoin::replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                           std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input.get(), input.get());
  if (condition_) {
    rebind_inputs.visit(condition_.get());
  }
}

void RelFilter::replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                             std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input.get(), input.get());
  rebind_inputs.visit(filter_.get());
}

void RelCompound::replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                               std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input.get(), input.get());
  for (const auto& scalar_source : scalar_sources_) {
    rebind_inputs.visit(scalar_source.get());
  }
  if (filter_expr_) {
    rebind_inputs.visit(filter_expr_.get());
  }
}

RelProject::RelProject(RelProject const& rhs)
    : RelAlgNode(rhs)
    , fields_(rhs.fields_)
    , hint_applied_(false)
    , hints_(std::make_unique<Hints>()) {
  RexDeepCopyVisitor copier;
  for (auto const& expr : rhs.scalar_exprs_) {
    scalar_exprs_.push_back(copier.visit(expr.get()));
  }
  if (rhs.hint_applied_) {
    for (auto const& kv : *rhs.hints_) {
      addHint(kv.second);
    }
  }
}

RelLogicalValues::RelLogicalValues(RelLogicalValues const& rhs)
    : RelAlgNode(rhs)
    , tuple_type_(rhs.tuple_type_)
    , values_(RexDeepCopyVisitor::copy(rhs.values_)) {}

RelFilter::RelFilter(RelFilter const& rhs) : RelAlgNode(rhs) {
  RexDeepCopyVisitor copier;
  filter_ = copier.visit(rhs.filter_.get());
}

RelAggregate::RelAggregate(RelAggregate const& rhs)
    : RelAlgNode(rhs)
    , groupby_count_(rhs.groupby_count_)
    , fields_(rhs.fields_)
    , hint_applied_(false)
    , hints_(std::make_unique<Hints>()) {
  agg_exprs_.reserve(rhs.agg_exprs_.size());
  for (auto const& agg : rhs.agg_exprs_) {
    agg_exprs_.push_back(agg->deepCopy());
  }
  if (rhs.hint_applied_) {
    for (auto const& kv : *rhs.hints_) {
      addHint(kv.second);
    }
  }
}

RelJoin::RelJoin(RelJoin const& rhs)
    : RelAlgNode(rhs)
    , join_type_(rhs.join_type_)
    , hint_applied_(false)
    , hints_(std::make_unique<Hints>()) {
  RexDeepCopyVisitor copier;
  condition_ = copier.visit(rhs.condition_.get());
  if (rhs.hint_applied_) {
    for (auto const& kv : *rhs.hints_) {
      addHint(kv.second);
    }
  }
}

namespace {

std::vector<std::unique_ptr<const RexAgg>> copyAggExprs(
    std::vector<std::unique_ptr<const RexAgg>> const& agg_exprs) {
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_copy;
  agg_exprs_copy.reserve(agg_exprs.size());
  for (auto const& agg_expr : agg_exprs) {
    agg_exprs_copy.push_back(agg_expr->deepCopy());
  }
  return agg_exprs_copy;
}

std::vector<std::unique_ptr<const RexScalar>> copyRexScalars(
    std::vector<std::unique_ptr<const RexScalar>> const& scalar_sources) {
  std::vector<std::unique_ptr<const RexScalar>> scalar_sources_copy;
  scalar_sources_copy.reserve(scalar_sources.size());
  RexDeepCopyVisitor copier;
  for (auto const& scalar_source : scalar_sources) {
    scalar_sources_copy.push_back(copier.visit(scalar_source.get()));
  }
  return scalar_sources_copy;
}

std::vector<const Rex*> remapTargetPointers(
    std::vector<std::unique_ptr<const RexAgg>> const& agg_exprs_new,
    std::vector<std::unique_ptr<const RexScalar>> const& scalar_sources_new,
    std::vector<std::unique_ptr<const RexAgg>> const& agg_exprs_old,
    std::vector<std::unique_ptr<const RexScalar>> const& scalar_sources_old,
    std::vector<const Rex*> const& target_exprs_old) {
  std::vector<const Rex*> target_exprs(target_exprs_old);
  std::unordered_map<const Rex*, const Rex*> old_to_new_target(target_exprs.size());
  for (size_t i = 0; i < agg_exprs_new.size(); ++i) {
    old_to_new_target.emplace(agg_exprs_old[i].get(), agg_exprs_new[i].get());
  }
  for (size_t i = 0; i < scalar_sources_new.size(); ++i) {
    old_to_new_target.emplace(scalar_sources_old[i].get(), scalar_sources_new[i].get());
  }
  for (auto& target : target_exprs) {
    auto target_it = old_to_new_target.find(target);
    CHECK(target_it != old_to_new_target.end());
    target = target_it->second;
  }
  return target_exprs;
}

}  // namespace

RelCompound::RelCompound(RelCompound const& rhs)
    : RelAlgNode(rhs)
    , groupby_count_(rhs.groupby_count_)
    , agg_exprs_(copyAggExprs(rhs.agg_exprs_))
    , fields_(rhs.fields_)
    , is_agg_(rhs.is_agg_)
    , scalar_sources_(copyRexScalars(rhs.scalar_sources_))
    , target_exprs_(remapTargetPointers(agg_exprs_,
                                        scalar_sources_,
                                        rhs.agg_exprs_,
                                        rhs.scalar_sources_,
                                        rhs.target_exprs_))
    , hint_applied_(false)
    , hints_(std::make_unique<Hints>()) {
  RexDeepCopyVisitor copier;
  filter_expr_ = rhs.filter_expr_ ? copier.visit(rhs.filter_expr_.get()) : nullptr;
  if (rhs.hint_applied_) {
    for (auto const& kv : *rhs.hints_) {
      addHint(kv.second);
    }
  }
}

void RelTableFunction::replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                                    std::shared_ptr<const RelAlgNode> input) {
  RelAlgNode::replaceInput(old_input, input);
  RexRebindInputsVisitor rebind_inputs(old_input.get(), input.get());
  for (const auto& target_expr : target_exprs_) {
    rebind_inputs.visit(target_expr.get());
  }
  for (const auto& func_input : table_func_inputs_) {
    rebind_inputs.visit(func_input.get());
  }
}

int32_t RelTableFunction::countRexLiteralArgs() const {
  int32_t literal_args = 0;
  for (const auto& arg : table_func_inputs_) {
    const auto rex_literal = dynamic_cast<const RexLiteral*>(arg.get());
    if (rex_literal) {
      literal_args += 1;
    }
  }
  return literal_args;
}

RelTableFunction::RelTableFunction(RelTableFunction const& rhs)
    : RelAlgNode(rhs)
    , function_name_(rhs.function_name_)
    , fields_(rhs.fields_)
    , col_inputs_(rhs.col_inputs_)
    , table_func_inputs_(copyRexScalars(rhs.table_func_inputs_))
    , target_exprs_(copyRexScalars(rhs.target_exprs_)) {
  std::unordered_map<const Rex*, const Rex*> old_to_new_input;
  for (size_t i = 0; i < table_func_inputs_.size(); ++i) {
    old_to_new_input.emplace(rhs.table_func_inputs_[i].get(),
                             table_func_inputs_[i].get());
  }
  for (auto& target : col_inputs_) {
    auto target_it = old_to_new_input.find(target);
    CHECK(target_it != old_to_new_input.end());
    target = target_it->second;
  }
}

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

// Rest of code requires a raw pointer, but RexInput object needs to live somewhere.
RexScalar const* RelLogicalUnion::copyAndRedirectSource(RexScalar const* rex_scalar,
                                                        size_t input_idx) const {
  if (auto const* rex_input_ptr = dynamic_cast<RexInput const*>(rex_scalar)) {
    RexInput rex_input(*rex_input_ptr);
    rex_input.setSourceNode(getInput(input_idx));
    scalar_exprs_.emplace_back(std::make_shared<RexInput const>(std::move(rex_input)));
    return scalar_exprs_.back().get();
  }
  return rex_scalar;
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

// The parse_* functions below de-serialize expressions as they come from Calcite.
// RelAlgDagBuilder will take care of making the representation easy to
// navigate for lower layers, for example by replacing RexAbstractInput with RexInput.

std::unique_ptr<RexAbstractInput> parse_abstract_input(
    const rapidjson::Value& expr) noexcept {
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
  if (literal.IsNull()) {
    return std::unique_ptr<RexLiteral>(new RexLiteral(target_type));
  }
  switch (type) {
    case kINT:
    case kBIGINT:
    case kDECIMAL:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return std::unique_ptr<RexLiteral>(new RexLiteral(json_i64(literal),
                                                        type,
                                                        target_type,
                                                        scale,
                                                        precision,
                                                        type_scale,
                                                        type_precision));
    case kDOUBLE: {
      if (literal.IsDouble()) {
        return std::unique_ptr<RexLiteral>(new RexLiteral(json_double(literal),
                                                          type,
                                                          target_type,
                                                          scale,
                                                          precision,
                                                          type_scale,
                                                          type_precision));
      } else if (literal.IsInt64()) {
        return std::make_unique<RexLiteral>(static_cast<double>(literal.GetInt64()),
                                            type,
                                            target_type,
                                            scale,
                                            precision,
                                            type_scale,
                                            type_precision);

      } else if (literal.IsUint64()) {
        return std::make_unique<RexLiteral>(static_cast<double>(literal.GetUint64()),
                                            type,
                                            target_type,
                                            scale,
                                            precision,
                                            type_scale,
                                            type_precision);
      }
      UNREACHABLE() << "Unhandled type: " << literal.GetType();
    }
    case kTEXT:
      return std::unique_ptr<RexLiteral>(new RexLiteral(json_str(literal),
                                                        type,
                                                        target_type,
                                                        scale,
                                                        precision,
                                                        type_scale,
                                                        type_precision));
    case kBOOLEAN:
      return std::unique_ptr<RexLiteral>(new RexLiteral(json_bool(literal),
                                                        type,
                                                        target_type,
                                                        scale,
                                                        precision,
                                                        type_scale,
                                                        type_precision));
    case kNULLT:
      return std::unique_ptr<RexLiteral>(new RexLiteral(target_type));
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

std::unique_ptr<const RexScalar> parse_scalar_expr(const rapidjson::Value& expr,
                                                   int db_id,
                                                   SchemaProviderPtr schema_provider,
                                                   RelAlgDagBuilder& root_dag_builder);

SQLTypeInfo parse_type(const rapidjson::Value& type_obj) {
  if (type_obj.IsArray()) {
    throw QueryNotSupported("Composite types are not currently supported.");
  }
  CHECK(type_obj.IsObject() && type_obj.MemberCount() >= 2)
      << json_node_to_string(type_obj);
  const auto type = to_sql_type(json_str(field(type_obj, "type")));
  const auto nullable = json_bool(field(type_obj, "nullable"));
  const auto precision_it = type_obj.FindMember("precision");
  const int precision =
      precision_it != type_obj.MemberEnd() ? json_i64(precision_it->value) : 0;
  const auto scale_it = type_obj.FindMember("scale");
  const int scale = scale_it != type_obj.MemberEnd() ? json_i64(scale_it->value) : 0;
  SQLTypeInfo ti(type, !nullable);
  ti.set_precision(precision);
  ti.set_scale(scale);
  return ti;
}

std::vector<std::unique_ptr<const RexScalar>> parse_expr_array(
    const rapidjson::Value& arr,
    int db_id,
    SchemaProviderPtr schema_provider,
    RelAlgDagBuilder& root_dag_builder) {
  std::vector<std::unique_ptr<const RexScalar>> exprs;
  for (auto it = arr.Begin(); it != arr.End(); ++it) {
    exprs.emplace_back(parse_scalar_expr(*it, db_id, schema_provider, root_dag_builder));
  }
  return exprs;
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

std::vector<std::unique_ptr<const RexScalar>> parse_window_order_exprs(
    const rapidjson::Value& arr,
    int db_id,
    SchemaProviderPtr schema_provider,
    RelAlgDagBuilder& root_dag_builder) {
  std::vector<std::unique_ptr<const RexScalar>> exprs;
  for (auto it = arr.Begin(); it != arr.End(); ++it) {
    exprs.emplace_back(
        parse_scalar_expr(field(*it, "field"), db_id, schema_provider, root_dag_builder));
  }
  return exprs;
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

std::vector<SortField> parse_window_order_collation(const rapidjson::Value& arr,
                                                    RelAlgDagBuilder& root_dag_builder) {
  std::vector<SortField> collation;
  size_t field_idx = 0;
  for (auto it = arr.Begin(); it != arr.End(); ++it, ++field_idx) {
    const auto sort_dir = parse_sort_direction(*it);
    const auto null_pos = parse_nulls_position(*it);
    collation.emplace_back(field_idx, sort_dir, null_pos);
  }
  return collation;
}

RexWindowFunctionOperator::RexWindowBound parse_window_bound(
    const rapidjson::Value& window_bound_obj,
    int db_id,
    SchemaProviderPtr schema_provider,
    RelAlgDagBuilder& root_dag_builder) {
  CHECK(window_bound_obj.IsObject());
  RexWindowFunctionOperator::RexWindowBound window_bound;
  window_bound.unbounded = json_bool(field(window_bound_obj, "unbounded"));
  window_bound.preceding = json_bool(field(window_bound_obj, "preceding"));
  window_bound.following = json_bool(field(window_bound_obj, "following"));
  window_bound.is_current_row = json_bool(field(window_bound_obj, "is_current_row"));
  const auto& offset_field = field(window_bound_obj, "offset");
  if (offset_field.IsObject()) {
    window_bound.offset =
        parse_scalar_expr(offset_field, db_id, schema_provider, root_dag_builder);
  } else {
    CHECK(offset_field.IsNull());
  }
  window_bound.order_key = json_i64(field(window_bound_obj, "order_key"));
  return window_bound;
}

std::unique_ptr<const RexSubQuery> parse_subquery(const rapidjson::Value& expr,
                                                  int db_id,
                                                  SchemaProviderPtr schema_provider,
                                                  RelAlgDagBuilder& root_dag_builder) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(0));
  const auto& subquery_ast = field(expr, "subquery");

  RelAlgDagBuilder subquery_dag(root_dag_builder, subquery_ast, db_id, schema_provider);
  auto subquery = std::make_shared<RexSubQuery>(subquery_dag.getRootNodeShPtr());
  root_dag_builder.registerSubquery(subquery);
  return subquery->deepCopy();
}

std::unique_ptr<RexOperator> parse_operator(const rapidjson::Value& expr,
                                            int db_id,
                                            SchemaProviderPtr schema_provider,
                                            RelAlgDagBuilder& root_dag_builder) {
  const auto op_name = json_str(field(expr, "op"));
  const bool is_quantifier =
      op_name == std::string("PG_ANY") || op_name == std::string("PG_ALL");
  const auto op = is_quantifier ? kFUNCTION : to_sql_op(op_name);
  const auto& operators_json_arr = field(expr, "operands");
  CHECK(operators_json_arr.IsArray());
  auto operands =
      parse_expr_array(operators_json_arr, db_id, schema_provider, root_dag_builder);
  const auto type_it = expr.FindMember("type");
  CHECK(type_it != expr.MemberEnd());
  auto ti = parse_type(type_it->value);
  if (op == kIN && expr.HasMember("subquery")) {
    auto subquery = parse_subquery(expr, db_id, schema_provider, root_dag_builder);
    operands.emplace_back(std::move(subquery));
  }
  if (expr.FindMember("partition_keys") != expr.MemberEnd()) {
    const auto& partition_keys_arr = field(expr, "partition_keys");
    auto partition_keys =
        parse_expr_array(partition_keys_arr, db_id, schema_provider, root_dag_builder);
    const auto& order_keys_arr = field(expr, "order_keys");
    auto order_keys = parse_window_order_exprs(
        order_keys_arr, db_id, schema_provider, root_dag_builder);
    const auto collation = parse_window_order_collation(order_keys_arr, root_dag_builder);
    const auto kind = parse_window_function_kind(op_name);
    const auto lower_bound = parse_window_bound(
        field(expr, "lower_bound"), db_id, schema_provider, root_dag_builder);
    const auto upper_bound = parse_window_bound(
        field(expr, "upper_bound"), db_id, schema_provider, root_dag_builder);
    bool is_rows = json_bool(field(expr, "is_rows"));
    ti.set_notnull(false);
    return std::make_unique<RexWindowFunctionOperator>(kind,
                                                       operands,
                                                       partition_keys,
                                                       order_keys,
                                                       collation,
                                                       lower_bound,
                                                       upper_bound,
                                                       is_rows,
                                                       ti);
  }
  return std::unique_ptr<RexOperator>(op == kFUNCTION
                                          ? new RexFunctionOperator(op_name, operands, ti)
                                          : new RexOperator(op, std::move(operands), ti));
}

std::unique_ptr<RexCase> parse_case(const rapidjson::Value& expr,
                                    int db_id,
                                    SchemaProviderPtr schema_provider,
                                    RelAlgDagBuilder& root_dag_builder) {
  const auto& operands = field(expr, "operands");
  CHECK(operands.IsArray());
  CHECK_GE(operands.Size(), unsigned(2));
  std::unique_ptr<const RexScalar> else_expr;
  std::vector<
      std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>>
      expr_pair_list;
  for (auto operands_it = operands.Begin(); operands_it != operands.End();) {
    auto when_expr =
        parse_scalar_expr(*operands_it++, db_id, schema_provider, root_dag_builder);
    if (operands_it == operands.End()) {
      else_expr = std::move(when_expr);
      break;
    }
    auto then_expr =
        parse_scalar_expr(*operands_it++, db_id, schema_provider, root_dag_builder);
    expr_pair_list.emplace_back(std::move(when_expr), std::move(then_expr));
  }
  return std::unique_ptr<RexCase>(new RexCase(expr_pair_list, else_expr));
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

std::unique_ptr<const RexAgg> parse_aggregate_expr(const rapidjson::Value& expr) {
  const auto agg_str = json_str(field(expr, "agg"));
  if (agg_str == "APPROX_QUANTILE") {
    LOG(INFO) << "APPROX_QUANTILE is deprecated. Please use APPROX_PERCENTILE instead.";
  }
  const auto agg = to_agg_kind(agg_str);
  const auto distinct = json_bool(field(expr, "distinct"));
  const auto agg_ti = parse_type(field(expr, "type"));
  const auto operands = indices_from_json_array(field(expr, "operands"));
  if (operands.size() > 1 && (operands.size() != 2 || (agg != kAPPROX_COUNT_DISTINCT &&
                                                       agg != kAPPROX_QUANTILE))) {
    throw QueryNotSupported("Multiple arguments for aggregates aren't supported");
  }
  return std::unique_ptr<const RexAgg>(new RexAgg(agg, distinct, agg_ti, operands));
}

std::unique_ptr<const RexScalar> parse_scalar_expr(const rapidjson::Value& expr,
                                                   int db_id,
                                                   SchemaProviderPtr schema_provider,
                                                   RelAlgDagBuilder& root_dag_builder) {
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
      return std::unique_ptr<const RexScalar>(
          parse_case(expr, db_id, schema_provider, root_dag_builder));
    }
    if (op_str == std::string("$SCALAR_QUERY")) {
      return std::unique_ptr<const RexScalar>(
          parse_subquery(expr, db_id, schema_provider, root_dag_builder));
    }
    return std::unique_ptr<const RexScalar>(
        parse_operator(expr, db_id, schema_provider, root_dag_builder));
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

std::unique_ptr<const RexScalar> disambiguate_rex(const RexScalar*, const RANodeOutput&);

std::unique_ptr<const RexOperator> disambiguate_operator(
    const RexOperator* rex_operator,
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
  const auto rex_window_function_operator =
      dynamic_cast<const RexWindowFunctionOperator*>(rex_operator);
  if (rex_window_function_operator) {
    const auto& partition_keys = rex_window_function_operator->getPartitionKeys();
    std::vector<std::unique_ptr<const RexScalar>> disambiguated_partition_keys;
    for (const auto& partition_key : partition_keys) {
      disambiguated_partition_keys.emplace_back(
          disambiguate_rex(partition_key.get(), ra_output));
    }
    std::vector<std::unique_ptr<const RexScalar>> disambiguated_order_keys;
    const auto& order_keys = rex_window_function_operator->getOrderKeys();
    for (const auto& order_key : order_keys) {
      disambiguated_order_keys.emplace_back(disambiguate_rex(order_key.get(), ra_output));
    }
    return rex_window_function_operator->disambiguatedOperands(
        disambiguated_operands,
        disambiguated_partition_keys,
        disambiguated_order_keys,
        rex_window_function_operator->getCollation());
  }
  return rex_operator->getDisambiguated(disambiguated_operands);
}

std::unique_ptr<const RexCase> disambiguate_case(const RexCase* rex_case,
                                                 const RANodeOutput& ra_output) {
  std::vector<
      std::pair<std::unique_ptr<const RexScalar>, std::unique_ptr<const RexScalar>>>
      disambiguated_expr_pair_list;
  for (size_t i = 0; i < rex_case->branchCount(); ++i) {
    auto disambiguated_when = disambiguate_rex(rex_case->getWhen(i), ra_output);
    auto disambiguated_then = disambiguate_rex(rex_case->getThen(i), ra_output);
    disambiguated_expr_pair_list.emplace_back(std::move(disambiguated_when),
                                              std::move(disambiguated_then));
  }
  std::unique_ptr<const RexScalar> disambiguated_else{
      disambiguate_rex(rex_case->getElse(), ra_output)};
  return std::unique_ptr<const RexCase>(
      new RexCase(disambiguated_expr_pair_list, disambiguated_else));
}

// The inputs used by scalar expressions are given as indices in the serialized
// representation of the query. This is hard to navigate; make the relationship
// explicit by creating RexInput expressions which hold a pointer to the source
// relational algebra node and the index relative to the output of that node.
std::unique_ptr<const RexScalar> disambiguate_rex(const RexScalar* rex_scalar,
                                                  const RANodeOutput& ra_output) {
  const auto rex_abstract_input = dynamic_cast<const RexAbstractInput*>(rex_scalar);
  if (rex_abstract_input) {
    CHECK_LT(static_cast<size_t>(rex_abstract_input->getIndex()), ra_output.size());
    return std::unique_ptr<const RexInput>(
        new RexInput(ra_output[rex_abstract_input->getIndex()]));
  }
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex_scalar);
  if (rex_operator) {
    return disambiguate_operator(rex_operator, ra_output);
  }
  const auto rex_case = dynamic_cast<const RexCase*>(rex_scalar);
  if (rex_case) {
    return disambiguate_case(rex_case, ra_output);
  }
  if (auto const rex_literal = dynamic_cast<const RexLiteral*>(rex_scalar)) {
    return rex_literal->deepCopy();
  } else if (auto const rex_subquery = dynamic_cast<const RexSubQuery*>(rex_scalar)) {
    return rex_subquery->deepCopy();
  } else {
    throw QueryNotSupported("Unable to disambiguate expression of type " +
                            std::string(typeid(*rex_scalar).name()));
  }
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

void bind_table_func_to_input(RelTableFunction* table_func_node,
                              const RANodeOutput& input) noexcept {
  std::vector<std::unique_ptr<const RexScalar>> disambiguated_exprs;
  for (size_t i = 0; i < table_func_node->getTableFuncInputsSize(); ++i) {
    const auto target_expr = table_func_node->getTableFuncInputAt(i);
    if (dynamic_cast<const RexSubQuery*>(target_expr)) {
      disambiguated_exprs.emplace_back(table_func_node->getTableFuncInputAtAndRelease(i));
    } else {
      disambiguated_exprs.emplace_back(disambiguate_rex(target_expr, input));
    }
  }
  table_func_node->setTableFuncInputs(disambiguated_exprs);
}

void bind_inputs(const std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept {
  for (auto ra_node : nodes) {
    const auto filter_node = std::dynamic_pointer_cast<RelFilter>(ra_node);
    if (filter_node) {
      CHECK_EQ(size_t(1), filter_node->inputCount());
      auto disambiguated_condition = disambiguate_rex(
          filter_node->getCondition(), get_node_output(filter_node->getInput(0)));
      filter_node->setCondition(disambiguated_condition);
      continue;
    }
    const auto join_node = std::dynamic_pointer_cast<RelJoin>(ra_node);
    if (join_node) {
      CHECK_EQ(size_t(2), join_node->inputCount());
      auto disambiguated_condition =
          disambiguate_rex(join_node->getCondition(), get_node_output(join_node.get()));
      join_node->setCondition(disambiguated_condition);
      continue;
    }
    const auto project_node = std::dynamic_pointer_cast<RelProject>(ra_node);
    if (project_node) {
      bind_project_to_input(project_node.get(),
                            get_node_output(project_node->getInput(0)));
      continue;
    }
    const auto table_func_node = std::dynamic_pointer_cast<RelTableFunction>(ra_node);
    if (table_func_node) {
      /*
        Collect all inputs from table function input (non-literal)
        arguments.
      */
      RANodeOutput input;
      input.reserve(table_func_node->inputCount());
      for (size_t i = 0; i < table_func_node->inputCount(); i++) {
        auto node_output = get_node_output(table_func_node->getInput(i));
        input.insert(input.end(), node_output.begin(), node_output.end());
      }
      bind_table_func_to_input(table_func_node.get(), input);
    }
  }
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

namespace {

std::vector<const Rex*> reproject_targets(
    const RelProject* simple_project,
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

/**
 * The RexInputReplacement visitor visits each node in a given relational algebra
 * expression and replaces the inputs to that expression with inputs from a different
 * node in the RA tree. Used for coalescing nodes with complex expressions.
 */
class RexInputReplacementVisitor : public RexDeepCopyVisitor {
 public:
  RexInputReplacementVisitor(
      const RelAlgNode* node_to_keep,
      const std::vector<std::unique_ptr<const RexScalar>>& scalar_sources)
      : node_to_keep_(node_to_keep), scalar_sources_(scalar_sources) {}

  // Reproject the RexInput from its current RA Node to the RA Node we intend to keep
  RetType visitInput(const RexInput* input) const final {
    if (input->getSourceNode() == node_to_keep_) {
      const auto index = input->getIndex();
      CHECK_LT(index, scalar_sources_.size());
      return visit(scalar_sources_[index].get());
    } else {
      return input->deepCopy();
    }
  }

 private:
  const RelAlgNode* node_to_keep_;
  const std::vector<std::unique_ptr<const RexScalar>>& scalar_sources_;
};

}  // namespace

void create_compound(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    const std::vector<size_t>& pattern,
    std::unordered_map<size_t, RegisteredQueryHint>& query_hints) noexcept {
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
  RelAlgNode* last_node{nullptr};

  size_t node_hash{0};
  bool hint_registered{false};
  RegisteredQueryHint registered_query_hint = RegisteredQueryHint::defaults();
  for (const auto node_idx : pattern) {
    const auto ra_node = nodes[node_idx];
    auto registered_query_hint_it = query_hints.find(ra_node->toHash());
    if (registered_query_hint_it != query_hints.end()) {
      hint_registered = true;
      node_hash = registered_query_hint_it->first;
      registered_query_hint = registered_query_hint_it->second;
    }
    const auto ra_filter = std::dynamic_pointer_cast<RelFilter>(ra_node);
    if (ra_filter) {
      CHECK(!filter_rex);
      filter_rex.reset(ra_filter->getAndReleaseCondition());
      CHECK(filter_rex);
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
        // intermediate buffer.
        const auto filter_input = dynamic_cast<const RelFilter*>(ra_project->getInput(0));
        if (filter_input) {
          CHECK_EQ(size_t(1), filter_input->inputCount());
          bind_project_to_input(ra_project.get(),
                                get_node_output(filter_input->getInput(0)));
        }
        scalar_sources = ra_project->getExpressionsAndRelease();
        for (const auto& scalar_expr : scalar_sources) {
          target_exprs.push_back(scalar_expr.get());
        }
        first_project = false;
      } else {
        if (ra_project->isSimple()) {
          target_exprs = reproject_targets(ra_project.get(), target_exprs);
        } else {
          // TODO(adb): This is essentially a more general case of simple project, we
          // could likely merge the two
          std::vector<const Rex*> result;
          RexInputReplacementVisitor visitor(last_node, scalar_sources);
          for (size_t i = 0; i < ra_project->size(); ++i) {
            const auto rex = ra_project->getProjectAt(i);
            if (auto rex_input = dynamic_cast<const RexInput*>(rex)) {
              const auto index = rex_input->getIndex();
              CHECK_LT(index, target_exprs.size());
              result.push_back(target_exprs[index]);
            } else {
              scalar_sources.push_back(visitor.visit(rex));
              result.push_back(scalar_sources.back().get());
            }
          }
          target_exprs = result;
        }
      }
      last_node = ra_node.get();
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
      last_node = ra_node.get();
      continue;
    }
  }

  auto compound_node = std::make_shared<RelCompound>(
      filter_rex, target_exprs, groupby_count, agg_exprs, fields, scalar_sources, is_agg);
  auto old_node = nodes[pattern.back()];
  nodes[pattern.back()] = compound_node;
  auto first_node = nodes[pattern.front()];
  CHECK_EQ(size_t(1), first_node->inputCount());
  compound_node->addManagedInput(first_node->getAndOwnInput(0));
  if (hint_registered) {
    // pass the registered hint from the origin node to newly created compound node
    // where it is coalesced
    query_hints.erase(node_hash);
    query_hints.emplace(compound_node->toHash(), registered_query_hint);
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

namespace {

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
class CoalesceSecondaryProjectVisitor : public RexVisitor<bool> {
 public:
  bool visitInput(const RexInput* input) const final {
    // The top level expression node is checked before we apply the visitor. If we get
    // here, this input rex is a child of another rex node, and we handle the can be
    // coalesced check slightly differently
    return input_can_be_coalesced(input->getSourceNode(), input->getIndex(), false);
  }

  bool visitLiteral(const RexLiteral*) const final { return false; }

  bool visitSubQuery(const RexSubQuery*) const final { return false; }

  bool visitRef(const RexRef*) const final { return false; }

 protected:
  bool aggregateResult(const bool& aggregate, const bool& next_result) const final {
    return aggregate && next_result;
  }

  bool defaultResult() const final { return true; }
};

// Detect the window function SUM pattern: CASE WHEN COUNT() > 0 THEN SUM ELSE 0
bool is_window_function_sum(const RexScalar* rex) {
  const auto case_operator = dynamic_cast<const RexCase*>(rex);
  if (case_operator && case_operator->branchCount() == 1) {
    const auto then_window =
        dynamic_cast<const RexWindowFunctionOperator*>(case_operator->getThen(0));
    if (then_window && then_window->getKind() == SqlWindowFunctionKind::SUM_INTERNAL) {
      return true;
    }
  }
  return false;
}

// Detect both window function operators and window function operators embedded in case
// statements (for null handling)
bool is_window_function_operator(const RexScalar* rex) {
  if (dynamic_cast<const RexWindowFunctionOperator*>(rex)) {
    return true;
  }

  // unwrap from casts, if they exist
  const auto rex_cast = dynamic_cast<const RexOperator*>(rex);
  if (rex_cast && rex_cast->getOperator() == kCAST) {
    CHECK_EQ(rex_cast->size(), size_t(1));
    return is_window_function_operator(rex_cast->getOperand(0));
  }

  if (is_window_function_sum(rex)) {
    return true;
  }
  // Check for Window Function AVG:
  // (CASE WHEN count > 0 THEN sum ELSE 0) / COUNT
  const RexOperator* divide_operator = dynamic_cast<const RexOperator*>(rex);
  if (divide_operator && divide_operator->getOperator() == kDIVIDE) {
    CHECK_EQ(divide_operator->size(), size_t(2));
    const auto case_operator =
        dynamic_cast<const RexCase*>(divide_operator->getOperand(0));
    const auto second_window =
        dynamic_cast<const RexWindowFunctionOperator*>(divide_operator->getOperand(1));
    if (case_operator && second_window &&
        second_window->getKind() == SqlWindowFunctionKind::COUNT) {
      if (is_window_function_sum(case_operator)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

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
            for (size_t i = 0; i < project_node->size(); i++) {
              const auto scalar_rex = project_node->getProjectAt(i);
              // If the top level scalar rex is an input node, we can bypass the visitor
              if (auto input_rex = dynamic_cast<const RexInput*>(scalar_rex)) {
                if (!input_can_be_coalesced(
                        input_rex->getSourceNode(), input_rex->getIndex(), true)) {
                  is_simple_project = false;
                  break;
                }
                continue;
              }
              CoalesceSecondaryProjectVisitor visitor;
              if (!visitor.visit(project_node->getProjectAt(i))) {
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

/**
 * WindowFunctionDetectionVisitor detects the presence of embedded Window Function Rex
 * Operators and returns a pointer to the WindowFunctionOperator. Only the first
 * detected operator will be returned (e.g. a binary operator that is WindowFunc1 &
 * WindowFunc2 would return a pointer to WindowFunc1). Neither the window function
 * operator nor its parent expression are modified.
 */
class WindowFunctionDetectionVisitor : public RexVisitor<const RexScalar*> {
 protected:
  // Detect embedded window function expressions in operators
  const RexScalar* visitOperator(const RexOperator* rex_operator) const final {
    if (is_window_function_operator(rex_operator)) {
      return rex_operator;
    }

    const size_t operand_count = rex_operator->size();
    for (size_t i = 0; i < operand_count; ++i) {
      const auto operand = rex_operator->getOperand(i);
      if (is_window_function_operator(operand)) {
        // Handle both RexWindowFunctionOperators and window functions built up from
        // multiple RexScalar objects (e.g. AVG)
        return operand;
      }
      const auto operandResult = visit(operand);
      if (operandResult) {
        return operandResult;
      }
    }

    return defaultResult();
  }

  // Detect embedded window function expressions in case statements. Note that this may
  // manifest as a nested case statement inside a top level case statement, as some
  // window functions (sum, avg) are represented as a case statement. Use the
  // is_window_function_operator helper to detect complete window function expressions.
  const RexScalar* visitCase(const RexCase* rex_case) const final {
    if (is_window_function_operator(rex_case)) {
      return rex_case;
    }

    auto result = defaultResult();
    for (size_t i = 0; i < rex_case->branchCount(); ++i) {
      const auto when = rex_case->getWhen(i);
      result = is_window_function_operator(when) ? when : visit(when);
      if (result) {
        return result;
      }
      const auto then = rex_case->getThen(i);
      result = is_window_function_operator(then) ? then : visit(then);
      if (result) {
        return result;
      }
    }
    if (rex_case->getElse()) {
      auto else_expr = rex_case->getElse();
      result = is_window_function_operator(else_expr) ? else_expr : visit(else_expr);
    }
    return result;
  }

  const RexScalar* aggregateResult(const RexScalar* const& aggregate,
                                   const RexScalar* const& next_result) const final {
    // all methods calling aggregate result should be overriden
    UNREACHABLE();
    return nullptr;
  }

  const RexScalar* defaultResult() const final { return nullptr; }
};

/** Replaces the first occurrence of a WindowFunctionOperator rex with the provided
 * `replacement_rex`. Typically used for splitting a complex rex into two simpler
 * rexes, and forwarding one of the rexes to a later node. The forwarded rex
 * is then replaced with a RexInput using this visitor.
 * Note that for window function replacement, the overloads in this visitor must match
 * the overloads in the detection visitor above, to ensure a detected window function
 * expression is properly replaced.
 */
class RexWindowFuncReplacementVisitor : public RexDeepCopyVisitor {
 public:
  RexWindowFuncReplacementVisitor(std::unique_ptr<const RexScalar> replacement_rex)
      : replacement_rex_(std::move(replacement_rex)) {}

  ~RexWindowFuncReplacementVisitor() { CHECK(replacement_rex_ == nullptr); }

 protected:
  RetType visitOperator(const RexOperator* rex_operator) const final {
    if (should_replace_operand(rex_operator)) {
      return std::move(replacement_rex_);
    }

    const auto rex_window_function_operator =
        dynamic_cast<const RexWindowFunctionOperator*>(rex_operator);
    if (rex_window_function_operator) {
      // Deep copy the embedded window function operator
      return visitWindowFunctionOperator(rex_window_function_operator);
    }

    const size_t operand_count = rex_operator->size();
    std::vector<RetType> new_opnds;
    for (size_t i = 0; i < operand_count; ++i) {
      const auto operand = rex_operator->getOperand(i);
      if (should_replace_operand(operand)) {
        new_opnds.push_back(std::move(replacement_rex_));
      } else {
        new_opnds.emplace_back(visit(rex_operator->getOperand(i)));
      }
    }
    return rex_operator->getDisambiguated(new_opnds);
  }

  RetType visitCase(const RexCase* rex_case) const final {
    if (should_replace_operand(rex_case)) {
      return std::move(replacement_rex_);
    }

    std::vector<std::pair<RetType, RetType>> new_pair_list;
    for (size_t i = 0; i < rex_case->branchCount(); ++i) {
      auto when_operand = rex_case->getWhen(i);
      auto then_operand = rex_case->getThen(i);
      new_pair_list.emplace_back(
          should_replace_operand(when_operand) ? std::move(replacement_rex_)
                                               : visit(when_operand),
          should_replace_operand(then_operand) ? std::move(replacement_rex_)
                                               : visit(then_operand));
    }
    auto new_else = should_replace_operand(rex_case->getElse())
                        ? std::move(replacement_rex_)
                        : visit(rex_case->getElse());
    return std::make_unique<RexCase>(new_pair_list, new_else);
  }

 private:
  bool should_replace_operand(const RexScalar* rex) const {
    return replacement_rex_ && is_window_function_operator(rex);
  }

  mutable std::unique_ptr<const RexScalar> replacement_rex_;
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
class RexInputBackpropagationVisitor : public RexDeepCopyVisitor {
 public:
  RexInputBackpropagationVisitor(RelProject* node) : node_(node) { CHECK(node_); }

 protected:
  RetType visitInput(const RexInput* rex_input) const final {
    if (rex_input->getSourceNode() != node_) {
      const auto cur_index = rex_input->getIndex();
      auto cur_source_node = rex_input->getSourceNode();
      std::string field_name = "";
      if (auto cur_project_node = dynamic_cast<const RelProject*>(cur_source_node)) {
        field_name = cur_project_node->getFieldName(cur_index);
      }
      node_->appendInput(field_name, rex_input->deepCopy());
      return std::make_unique<RexInput>(node_, node_->size() - 1);
    } else {
      return rex_input->deepCopy();
    }
  }

 private:
  mutable RelProject* node_;
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

  WindowFunctionDetectionVisitor visitor;
  for (auto node_itr = node_list.begin(); node_itr != node_list.end(); ++node_itr) {
    const auto node = *node_itr;
    auto window_func_project_node = std::dynamic_pointer_cast<RelProject>(node);
    if (!window_func_project_node) {
      continue;
    }

    // map scalar expression index in the project node to window function ptr
    std::unordered_map<size_t, const RexScalar*> embedded_window_function_expressions;

    // Iterate the target exprs of the project node and check for window function
    // expressions. If an embedded expression exists, save it in the
    // embedded_window_function_expressions map and split the expression into a window
    // function expression and a parent expression in a subsequent project node
    for (size_t i = 0; i < window_func_project_node->size(); i++) {
      const auto scalar_rex = window_func_project_node->getProjectAt(i);
      if (is_window_function_operator(scalar_rex)) {
        // top level window function exprs are fine
        continue;
      }

      if (const auto window_func_rex = visitor.visit(scalar_rex)) {
        const auto ret = embedded_window_function_expressions.insert(
            std::make_pair(i, window_func_rex));
        CHECK(ret.second);
      }
    }

    if (!embedded_window_function_expressions.empty()) {
      std::vector<std::unique_ptr<const RexScalar>> new_scalar_exprs;

      auto window_func_scalar_exprs =
          window_func_project_node->getExpressionsAndRelease();
      for (size_t rex_idx = 0; rex_idx < window_func_scalar_exprs.size(); ++rex_idx) {
        const auto embedded_window_func_expr_pair =
            embedded_window_function_expressions.find(rex_idx);
        if (embedded_window_func_expr_pair ==
            embedded_window_function_expressions.end()) {
          new_scalar_exprs.emplace_back(
              std::make_unique<const RexInput>(window_func_project_node.get(), rex_idx));
        } else {
          const auto window_func_rex_idx = embedded_window_func_expr_pair->first;
          CHECK_LT(window_func_rex_idx, window_func_scalar_exprs.size());

          const auto& window_func_rex = embedded_window_func_expr_pair->second;

          RexDeepCopyVisitor copier;
          auto window_func_rex_copy = copier.visit(window_func_rex);

          auto window_func_parent_expr =
              window_func_scalar_exprs[window_func_rex_idx].get();

          // Replace window func rex with an input rex
          auto window_func_result_input = std::make_unique<const RexInput>(
              window_func_project_node.get(), window_func_rex_idx);
          RexWindowFuncReplacementVisitor replacer(std::move(window_func_result_input));
          auto new_parent_rex = replacer.visit(window_func_parent_expr);

          // Put the parent expr in the new scalar exprs
          new_scalar_exprs.emplace_back(std::move(new_parent_rex));

          // Put the window func expr in cur scalar exprs
          window_func_scalar_exprs[window_func_rex_idx] = std::move(window_func_rex_copy);
        }
      }

      CHECK_EQ(window_func_scalar_exprs.size(), new_scalar_exprs.size());
      window_func_project_node->setExpressions(window_func_scalar_exprs);

      // Ensure any inputs from the node containing the expression (the "new" node)
      // exist on the window function project node, e.g. if we had a binary operation
      // involving an aggregate value or column not included in the top level
      // projection list.
      RexInputBackpropagationVisitor input_visitor(window_func_project_node.get());
      for (size_t i = 0; i < new_scalar_exprs.size(); i++) {
        if (dynamic_cast<const RexInput*>(new_scalar_exprs[i].get())) {
          // ignore top level inputs, these were copied directly from the previous
          // node
          continue;
        }
        new_scalar_exprs[i] = input_visitor.visit(new_scalar_exprs[i].get());
      }

      // Build the new project node and insert it into the list after the project node
      // containing the window function
      auto new_project =
          std::make_shared<RelProject>(std::move(new_scalar_exprs),
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

using RexInputSet = std::unordered_set<RexInput>;

class RexInputCollector : public RexVisitor<RexInputSet> {
 public:
  RexInputSet visitInput(const RexInput* input) const override {
    return RexInputSet{*input};
  }

 protected:
  RexInputSet aggregateResult(const RexInputSet& aggregate,
                              const RexInputSet& next_result) const override {
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

    RexInputSet inputs;
    RexInputCollector input_collector;
    for (size_t i = 0; i < window_func_project_node->size(); i++) {
      auto new_inputs = input_collector.visit(window_func_project_node->getProjectAt(i));
      inputs.insert(new_inputs.begin(), new_inputs.end());
    }

    // Note: Technically not required since we are mapping old inputs to new input
    // indices, but makes the re-mapping of inputs easier to follow.
    std::vector<RexInput> sorted_inputs(inputs.begin(), inputs.end());
    std::sort(sorted_inputs.begin(),
              sorted_inputs.end(),
              [](const auto& a, const auto& b) { return a.getIndex() < b.getIndex(); });

    std::vector<std::unique_ptr<const RexScalar>> scalar_exprs;
    std::vector<std::string> fields;
    std::unordered_map<unsigned, unsigned> old_index_to_new_index;
    for (auto& input : sorted_inputs) {
      CHECK_EQ(input.getSourceNode(), prev_node.get());
      CHECK(old_index_to_new_index
                .insert(std::make_pair(input.getIndex(), scalar_exprs.size()))
                .second);
      scalar_exprs.emplace_back(input.deepCopy());
      fields.emplace_back("");
    }

    auto new_project =
        std::make_shared<RelProject>(std::move(scalar_exprs), fields, prev_node);
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
  std::unique_ptr<RexLiteral> lit(parse_literal(it->value));
  CHECK_EQ(kDECIMAL, lit->getType());
  CHECK_EQ(unsigned(0), lit->getScale());
  CHECK_EQ(unsigned(0), lit->getTargetScale());
  return lit->getVal<int64_t>();
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
  for (const auto& expr : scalar_exprs_) {
    if (is_window_function_operator(expr.get())) {
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
        ra_node = dispatchAggregate(crt_node);
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
    std::vector<std::unique_ptr<const RexScalar>> exprs;
    for (auto exprs_json_it = exprs_json.Begin(); exprs_json_it != exprs_json.End();
         ++exprs_json_it) {
      exprs.emplace_back(
          parse_scalar_expr(*exprs_json_it, db_id_, schema_provider_, root_dag_builder));
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
    auto condition = parse_scalar_expr(
        field(filter_ra, "condition"), db_id_, schema_provider_, root_dag_builder);
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
    for (auto aggs_json_arr_it = aggs_json_arr.Begin();
         aggs_json_arr_it != aggs_json_arr.End();
         ++aggs_json_arr_it) {
      aggs.emplace_back(parse_aggregate_expr(*aggs_json_arr_it));
    }
    if (agg_ra.HasMember("hints")) {
      auto agg_node = std::make_shared<RelAggregate>(
          group.size(), std::move(aggs), fields, inputs.front());
      getRelAlgHints(agg_ra, agg_node);
      return agg_node;
    }
    return std::make_shared<RelAggregate>(
        group.size(), std::move(aggs), fields, inputs.front());
  }

  std::shared_ptr<RelJoin> dispatchJoin(const rapidjson::Value& join_ra,
                                        RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(join_ra);
    CHECK_EQ(size_t(2), inputs.size());
    const auto join_type = to_join_type(json_str(field(join_ra, "joinType")));
    auto filter_rex = parse_scalar_expr(
        field(join_ra, "condition"), db_id_, schema_provider_, root_dag_builder);
    if (join_ra.HasMember("hints")) {
      auto join_node = std::make_shared<RelJoin>(
          inputs[0], inputs[1], std::move(filter_rex), join_type);
      getRelAlgHints(join_ra, join_node);
      return join_node;
    }
    return std::make_shared<RelJoin>(
        inputs[0], inputs[1], std::move(filter_rex), join_type);
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

  std::shared_ptr<RelTableFunction> dispatchTableFunction(
      const rapidjson::Value& table_func_ra,
      RelAlgDagBuilder& root_dag_builder) {
    const auto inputs = getRelAlgInputs(table_func_ra);
    const auto& invocation = field(table_func_ra, "invocation");
    CHECK(invocation.IsObject());

    const auto& operands = field(invocation, "operands");
    CHECK(operands.IsArray());
    CHECK_GE(operands.Size(), unsigned(0));

    std::vector<const Rex*> col_inputs;
    std::vector<std::unique_ptr<const RexScalar>> table_func_inputs;
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
              table_func_inputs.emplace_back(
                  std::make_unique<RexAbstractInput>(col_inputs.size()));
              col_inputs.emplace_back(table_func_inputs.back().get());
            }
            continue;
          }
        }
      }
      table_func_inputs.emplace_back(
          parse_scalar_expr(*exprs_json_it, db_id_, schema_provider_, root_dag_builder));
    }

    const auto& op_name = field(invocation, "op");
    CHECK(op_name.IsString());

    std::vector<std::unique_ptr<const RexScalar>> table_function_projected_outputs;
    const auto& row_types = field(table_func_ra, "rowType");
    CHECK(row_types.IsArray());
    CHECK_GE(row_types.Size(), unsigned(0));
    const auto& row_types_array = row_types.GetArray();
    for (size_t i = 0; i < row_types_array.Size(); i++) {
      // We don't care about the type information in rowType -- replace each output with
      // a reference to be resolved later in the translator
      table_function_projected_outputs.emplace_back(std::make_unique<RexRef>(i));
      fields.emplace_back("");
    }
    return std::make_shared<RelTableFunction>(op_name.GetString(),
                                              inputs,
                                              fields,
                                              col_inputs,
                                              table_func_inputs,
                                              table_function_projected_outputs);
  }

  std::shared_ptr<RelLogicalValues> dispatchLogicalValues(
      const rapidjson::Value& logical_values_ra) {
    const auto& tuple_type_arr = field(logical_values_ra, "type");
    CHECK(tuple_type_arr.IsArray());
    std::vector<TargetMetaInfo> tuple_type;
    for (auto tuple_type_arr_it = tuple_type_arr.Begin();
         tuple_type_arr_it != tuple_type_arr.End();
         ++tuple_type_arr_it) {
      const auto component_type = parse_type(*tuple_type_arr_it);
      const auto component_name = json_str(field(*tuple_type_arr_it, "name"));
      tuple_type.emplace_back(component_name, component_type);
    }
    const auto& inputs_arr = field(logical_values_ra, "inputs");
    CHECK(inputs_arr.IsArray());
    const auto& tuples_arr = field(logical_values_ra, "tuples");
    CHECK(tuples_arr.IsArray());

    if (inputs_arr.Size()) {
      throw QueryNotSupported("Inputs not supported in logical values yet.");
    }

    std::vector<RelLogicalValues::RowValues> values;
    if (tuples_arr.Size()) {
      for (const auto& row : tuples_arr.GetArray()) {
        CHECK(row.IsArray());
        const auto values_json = row.GetArray();
        if (!values.empty()) {
          CHECK_EQ(values[0].size(), values_json.Size());
        }
        values.emplace_back(RelLogicalValues::RowValues{});
        for (const auto& value : values_json) {
          CHECK(value.IsObject());
          CHECK(value.HasMember("literal"));
          values.back().emplace_back(parse_literal(value));
        }
      }
    }

    return std::make_shared<RelLogicalValues>(tuple_type, values);
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
  RegisteredQueryHint query_hint;
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
  // we have four cases depending on 1) g_enable_columnar_output flag
  // and 2) query hint status: columnar_output and rowwise_output
  // case 1. g_enable_columnar_output = true
  // case 1.a) columnar_output = true (so rowwise_output = false);
  // case 1.b) rowwise_output = true (so columnar_output = false);
  // case 2. g_enable_columnar_output = false
  // case 2.a) columnar_output = true (so rowwise_output = false);
  // case 2.b) rowwise_output = true (so columnar_output = false);
  // case 1.a --> use columnar output
  // case 1.b --> use rowwise output
  // case 2.a --> use columnar output
  // case 2.b --> use rowwise output
  if (detect_columnar_output_hint && detect_rowwise_output_hint) {
    VLOG(1) << "Two hints 1) columnar output and 2) rowwise output are enabled together, "
            << "so skip them and use the runtime configuration "
               "\"g_enable_columnar_output\"";
  } else if (detect_columnar_output_hint && !detect_rowwise_output_hint) {
    if (g_enable_columnar_output) {
      VLOG(1) << "We already enable columnar output by default "
                 "(g_enable_columnar_output = true), so skip this columnar output hint";
    } else {
      query_hint.registerHint(QueryHint::kColumnarOutput);
      query_hint.columnar_output = true;
    }
  } else if (!detect_columnar_output_hint && detect_rowwise_output_hint) {
    if (!g_enable_columnar_output) {
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
    : db_id_(db_id), schema_provider_(schema_provider) {
  build(query_ast, root_dag_builder);
}

RelAlgDagBuilder::RelAlgDagBuilder(const std::string& query_ra,
                                   int db_id,
                                   SchemaProviderPtr schema_provider)
    : db_id_(db_id), schema_provider_(schema_provider) {
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
  bind_inputs(nodes_);

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

std::string RexSubQuery::toString() const {
  return cat(::typeName(this), "(", ::toString(ra_.get()), ")");
}

size_t RexSubQuery::toHash() const {
  if (!hash_) {
    hash_ = typeid(RexSubQuery).hash_code();
    boost::hash_combine(*hash_, ra_->toHash());
  }
  return *hash_;
}

std::string RexInput::toString() const {
  const auto scan_node = dynamic_cast<const RelScan*>(node_);
  if (scan_node) {
    auto field_name = scan_node->getFieldName(getIndex());
    auto table_name = scan_node->getTableInfo()->name;
    return ::typeName(this) + "(" + table_name + "." + field_name + ")";
  }
  return cat(::typeName(this),
             "(node=",
             ::toString(node_),
             ", in_index=",
             std::to_string(getIndex()),
             ")");
}

size_t RexInput::toHash() const {
  if (!hash_) {
    hash_ = typeid(RexInput).hash_code();
    boost::hash_combine(*hash_, node_->toHash());
    boost::hash_combine(*hash_, getIndex());
  }
  return *hash_;
}

std::string RelCompound::toString() const {
  return cat(::typeName(this),
             "(",
             (filter_expr_ ? filter_expr_->toString() : "null"),
             ", target_exprs=",
             ::toString(target_exprs_),
             ", ",
             std::to_string(groupby_count_),
             ", agg_exps=",
             ::toString(agg_exprs_),
             ", fields=",
             ::toString(fields_),
             ", scalar_sources=",
             ::toString(scalar_sources_),
             ", is_agg=",
             std::to_string(is_agg_),
             ")");
}

size_t RelCompound::toHash() const {
  if (!hash_) {
    hash_ = typeid(RelCompound).hash_code();
    boost::hash_combine(*hash_,
                        filter_expr_ ? filter_expr_->toHash() : boost::hash_value("n"));
    boost::hash_combine(*hash_, is_agg_);
    for (auto& target_expr : target_exprs_) {
      if (auto rex_scalar = dynamic_cast<const RexScalar*>(target_expr)) {
        boost::hash_combine(*hash_, rex_scalar->toHash());
      }
    }
    for (auto& agg_expr : agg_exprs_) {
      boost::hash_combine(*hash_, agg_expr->toHash());
    }
    for (auto& scalar_source : scalar_sources_) {
      boost::hash_combine(*hash_, scalar_source->toHash());
    }
    boost::hash_combine(*hash_, groupby_count_);
    boost::hash_combine(*hash_, ::toString(fields_));
  }
  return *hash_;
}
