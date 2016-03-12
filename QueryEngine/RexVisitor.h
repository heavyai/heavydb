#ifndef QUERYENGINE_REXVISITOR_H
#define QUERYENGINE_REXVISITOR_H

#include "RelAlgAbstractInterpreter.h"

template <class T>
class RexVisitor {
 public:
  T visit(const RexScalar* rex_scalar) const {
    const auto rex_input = dynamic_cast<const RexInput*>(rex_scalar);
    if (rex_input) {
      return visitInput(rex_input);
    }
    const auto rex_literal = dynamic_cast<const RexLiteral*>(rex_scalar);
    if (rex_literal) {
      return visitLiteral(rex_literal);
    }
    const auto rex_operator = dynamic_cast<const RexOperator*>(rex_scalar);
    if (rex_operator) {
      return visitOperator(rex_operator);
    }
    CHECK(false);
    return defaultResult();
  }

  virtual T visitInput(const RexInput*) const { return defaultResult(); }

  virtual T visitLiteral(const RexLiteral*) const { return defaultResult(); }

  T visitOperator(const RexOperator* rex_operator) const {
    const size_t operand_count = rex_operator->size();
    T result = defaultResult();
    for (size_t i = 0; i < operand_count; ++i) {
      const auto operand = rex_operator->getOperand(i);
      T operandResult = visit(operand);
      result = aggregateResult(result, operandResult);
    }
    return result;
  }

 protected:
  virtual T aggregateResult(const T& aggregate, const T& next_result) const { return next_result; }

  virtual T defaultResult() const { return T{}; }
};

#endif  // QUERYENGINE_REXVISITOR_H
