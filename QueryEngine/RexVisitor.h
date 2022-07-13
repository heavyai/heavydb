/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#ifndef QUERYENGINE_REXVISITOR_H
#define QUERYENGINE_REXVISITOR_H

#include "RelAlgDag.h"

#include <memory>

template <class T>
class RexVisitorBase {
 public:
  virtual T visit(const RexScalar* rex_scalar) const {
    CHECK(rex_scalar);
    const auto rex_input = dynamic_cast<const RexInput*>(rex_scalar);
    if (rex_input) {
      return visitInput(rex_input);
    }
    const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_scalar);
    if (rex_literal) {
      return visitLiteral(rex_literal);
    }
    const auto rex_subquery = dynamic_cast<const RexSubQuery*>(rex_scalar);
    if (rex_subquery) {
      return visitSubQuery(rex_subquery);
    }
    const auto rex_operator = dynamic_cast<const RexOperator*>(rex_scalar);
    if (rex_operator) {
      return visitOperator(rex_operator);
    }
    const auto rex_case = dynamic_cast<const RexCase*>(rex_scalar);
    if (rex_case) {
      return visitCase(rex_case);
    }
    const auto rex_ref = dynamic_cast<const RexRef*>(rex_scalar);
    if (rex_ref) {
      return visitRef(rex_ref);
    }
    LOG(FATAL) << "No visit path for "
               << rex_scalar->toString(RelRexToStringConfig::defaults());
    return defaultResult();
  }

  virtual T visitInput(const RexInput*) const = 0;

  virtual T visitLiteral(const RexLiteral*) const = 0;

  virtual T visitSubQuery(const RexSubQuery*) const = 0;

  virtual T visitRef(const RexRef*) const = 0;

  virtual T visitOperator(const RexOperator* rex_operator) const = 0;

  virtual T visitCase(const RexCase* rex_case) const = 0;

 protected:
  virtual T defaultResult() const = 0;
};

template <class T>
class RexVisitor : public RexVisitorBase<T> {
 public:
  T visitInput(const RexInput*) const override { return defaultResult(); }

  T visitLiteral(const RexLiteral*) const override { return defaultResult(); }

  T visitSubQuery(const RexSubQuery*) const override { return defaultResult(); }

  T visitRef(const RexRef*) const override { return defaultResult(); }

  T visitOperator(const RexOperator* rex_operator) const override {
    const size_t operand_count = rex_operator->size();
    T result = defaultResult();
    for (size_t i = 0; i < operand_count; ++i) {
      const auto operand = rex_operator->getOperand(i);
      T operandResult = RexVisitorBase<T>::visit(operand);
      result = aggregateResult(result, operandResult);
    }
    const auto rex_window_func_operator =
        dynamic_cast<const RexWindowFunctionOperator*>(rex_operator);
    if (rex_window_func_operator) {
      return visitWindowFunctionOperator(rex_window_func_operator, result);
    }
    return result;
  }

  T visitCase(const RexCase* rex_case) const override {
    T result = defaultResult();
    for (size_t i = 0; i < rex_case->branchCount(); ++i) {
      const auto when = rex_case->getWhen(i);
      result = aggregateResult(result, RexVisitorBase<T>::visit(when));
      const auto then = rex_case->getThen(i);
      result = aggregateResult(result, RexVisitorBase<T>::visit(then));
    }
    if (rex_case->getElse()) {
      result = aggregateResult(result, RexVisitorBase<T>::visit(rex_case->getElse()));
    }
    return result;
  }

 protected:
  virtual T aggregateResult(const T& aggregate, const T& next_result) const {
    return next_result;
  }

  T defaultResult() const override { return T{}; }

 private:
  T visitWindowFunctionOperator(const RexWindowFunctionOperator* rex_window_func_operator,
                                const T operands_visit_result) const {
    T result = operands_visit_result;
    for (const auto& key : rex_window_func_operator->getPartitionKeys()) {
      T partial_result = RexVisitorBase<T>::visit(key.get());
      result = aggregateResult(result, partial_result);
    }
    for (const auto& key : rex_window_func_operator->getOrderKeys()) {
      T partial_result = RexVisitorBase<T>::visit(key.get());
      result = aggregateResult(result, partial_result);
    }
    return result;
  }
};

class RexDeepCopyVisitor : public RexVisitorBase<std::unique_ptr<const RexScalar>> {
 protected:
  using RetType = std::unique_ptr<const RexScalar>;

  RetType visitInput(const RexInput* input) const override { return input->deepCopy(); }

  RetType visitLiteral(const RexLiteral* literal) const override {
    return literal->deepCopy();
  }

  RetType visitSubQuery(const RexSubQuery* subquery) const override {
    return subquery->deepCopy();
  }

  RetType visitRef(const RexRef* ref) const override { return ref->deepCopy(); }

  RetType visitOperator(const RexOperator* rex_operator) const override {
    const auto rex_window_function_operator =
        dynamic_cast<const RexWindowFunctionOperator*>(rex_operator);
    if (rex_window_function_operator) {
      return visitWindowFunctionOperator(rex_window_function_operator);
    }

    const size_t operand_count = rex_operator->size();
    std::vector<RetType> new_opnds;
    for (size_t i = 0; i < operand_count; ++i) {
      new_opnds.push_back(visit(rex_operator->getOperand(i)));
    }
    return rex_operator->getDisambiguated(new_opnds);
  }

  RetType visitWindowFunctionOperator(
      const RexWindowFunctionOperator* rex_window_function_operator) const {
    const size_t operand_count = rex_window_function_operator->size();
    std::vector<RetType> new_opnds;
    for (size_t i = 0; i < operand_count; ++i) {
      new_opnds.push_back(visit(rex_window_function_operator->getOperand(i)));
    }

    const auto& partition_keys = rex_window_function_operator->getPartitionKeys();
    std::vector<std::unique_ptr<const RexScalar>> disambiguated_partition_keys;
    for (const auto& partition_key : partition_keys) {
      disambiguated_partition_keys.emplace_back(visit(partition_key.get()));
    }
    std::vector<std::unique_ptr<const RexScalar>> disambiguated_order_keys;
    const auto& order_keys = rex_window_function_operator->getOrderKeys();
    for (const auto& order_key : order_keys) {
      disambiguated_order_keys.emplace_back(visit(order_key.get()));
    }
    return rex_window_function_operator->disambiguatedOperands(
        new_opnds,
        disambiguated_partition_keys,
        disambiguated_order_keys,
        rex_window_function_operator->getCollation());
  }

  RetType visitCase(const RexCase* rex_case) const override {
    std::vector<std::pair<RetType, RetType>> new_pair_list;
    for (size_t i = 0; i < rex_case->branchCount(); ++i) {
      new_pair_list.emplace_back(visit(rex_case->getWhen(i)),
                                 visit(rex_case->getThen(i)));
    }
    auto new_else = visit(rex_case->getElse());
    return std::make_unique<RexCase>(new_pair_list, new_else);
  }

 private:
  RetType defaultResult() const override { return nullptr; }

 public:
  using RowValues = std::vector<std::unique_ptr<const RexScalar>>;

  static std::vector<RowValues> copy(std::vector<RowValues> const& rhs) {
    RexDeepCopyVisitor copier;
    std::vector<RowValues> retval;
    retval.reserve(rhs.size());
    for (auto const& row : rhs) {
      retval.push_back({});
      retval.back().reserve(row.size());
      for (auto const& value : row) {
        retval.back().push_back(copier.visit(value.get()));
      }
    }
    return retval;
  }
};

template <bool bAllowMissing>
class RexInputRenumber : public RexDeepCopyVisitor {
 public:
  RexInputRenumber(const std::unordered_map<size_t, size_t>& new_numbering)
      : old_to_new_idx_(new_numbering) {}
  RetType visitInput(const RexInput* input) const override {
    auto renum_it = old_to_new_idx_.find(input->getIndex());
    if (bAllowMissing) {
      if (renum_it != old_to_new_idx_.end()) {
        return std::make_unique<RexInput>(input->getSourceNode(), renum_it->second);
      } else {
        return input->deepCopy();
      }
    } else {
      CHECK(renum_it != old_to_new_idx_.end());
      return std::make_unique<RexInput>(input->getSourceNode(), renum_it->second);
    }
  }

 private:
  const std::unordered_map<size_t, size_t>& old_to_new_idx_;
};

#endif  // QUERYENGINE_REXVISITOR_H
