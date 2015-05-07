#ifndef QUERYENGINE_EXPRESSIONRANGE_H
#define QUERYENGINE_EXPRESSIONRANGE_H

#include "../Analyzer/Analyzer.h"
#include "../Fragmenter/Fragmenter.h"


class ExpressionRange {
public:
  bool valid;
  int64_t min;
  int64_t max;

  ExpressionRange operator+(const ExpressionRange& other) const;
  ExpressionRange operator-(const ExpressionRange& other) const;
  ExpressionRange operator*(const ExpressionRange& other) const;
  ExpressionRange operator/(const ExpressionRange& other) const;
  ExpressionRange operator||(const ExpressionRange& other) const;

private:
  template<class BinOp>
  ExpressionRange binOp(const ExpressionRange& other, const BinOp& bin_op) const {
    if (!valid || !other.valid) {
      return { false, 0, 0 };
    }
    try {
      std::vector<int64_t> limits {
        bin_op(min, other.min),
        bin_op(min, other.max),
        bin_op(max, other.min),
        bin_op(max, other.max)
      };
      return {
        true,
        *std::min_element(limits.begin(), limits.end()),
        *std::max_element(limits.begin(), limits.end())
      };
    } catch (...) {
      return { false, 0, 0 };
    }
  }
};

ExpressionRange getExpressionRange(
  const Analyzer::Expr*,
  const std::vector<Fragmenter_Namespace::FragmentInfo>&);

#endif  // QUERYENGINE_EXPRESSIONRANGE_H
