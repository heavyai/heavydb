#ifndef QUERYENGINE_EXPRESSIONRANGE_H
#define QUERYENGINE_EXPRESSIONRANGE_H

#include "../Analyzer/Analyzer.h"
#include "../Fragmenter/Fragmenter.h"

#include <boost/multiprecision/cpp_int.hpp>


typedef boost::multiprecision::number<
  boost::multiprecision::cpp_int_backend<
    64, 64,
    boost::multiprecision::signed_magnitude,
    boost::multiprecision::checked,
    void>
  > checked_int64_t;

enum class ExpressionRangeType {
  Invalid,
  Integer,
  FloatingPoint,
};

class ExpressionRange;

template<typename T>
T getMin(const ExpressionRange& other);

template<typename T>
T getMax(const ExpressionRange& other);

class ExpressionRange {
public:
  ExpressionRangeType type;
  bool has_nulls;
  union {
    int64_t int_min;
    double fp_min;
  };
  union {
    int64_t int_max;
    double fp_max;
  };

  ExpressionRange operator+(const ExpressionRange& other) const;
  ExpressionRange operator-(const ExpressionRange& other) const;
  ExpressionRange operator*(const ExpressionRange& other) const;
  ExpressionRange operator/(const ExpressionRange& other) const;
  ExpressionRange operator||(const ExpressionRange& other) const;

private:
  template<class T, class BinOp>
  ExpressionRange binOp(const ExpressionRange& other, const BinOp& bin_op) const {
    if (type == ExpressionRangeType::Invalid || other.type == ExpressionRangeType::Invalid) {
      return { ExpressionRangeType::Invalid, false, { 0 }, { 0 } };
    }
    try {
      std::vector<T> limits {
        bin_op(getMin<T>(*this), getMin<T>(other)),
        bin_op(getMin<T>(*this), getMax<T>(other)),
        bin_op(getMax<T>(*this), getMin<T>(other)),
        bin_op(getMax<T>(*this), getMax<T>(other))
      };
      ExpressionRange result;
      result.type =
        (type == ExpressionRangeType::Integer && other.type == ExpressionRangeType::Integer)
          ? ExpressionRangeType::Integer
          : ExpressionRangeType::FloatingPoint;
      result.has_nulls = has_nulls || other.has_nulls;
      switch (result.type) {
      case ExpressionRangeType::Integer: {
        result.int_min = *std::min_element(limits.begin(), limits.end());
        result.int_max = *std::max_element(limits.begin(), limits.end());
        break;
      }
      case ExpressionRangeType::FloatingPoint: {
        result.fp_min = *std::min_element(limits.begin(), limits.end());
        result.fp_max = *std::max_element(limits.begin(), limits.end());
        break;
      }
      default:
        CHECK(false);
      }
      return result;
    } catch (...) {
      return { ExpressionRangeType::Invalid, false, { 0 }, { 0 } };
    }
  }
};

template<>
inline int64_t getMin<int64_t>(const ExpressionRange& e) {
  return e.int_min;
}

template<>
inline double getMin<double>(const ExpressionRange& e) {
  return e.fp_min;
}

template<>
inline int64_t getMax<int64_t>(const ExpressionRange& e) {
  return e.int_max;
}

template<>
inline double getMax<double>(const ExpressionRange& e) {
  return e.fp_max;
}

class Executor;

ExpressionRange getExpressionRange(
  const Analyzer::Expr*,
  const std::vector<Fragmenter_Namespace::FragmentInfo>&,
  const Executor*);

#endif  // QUERYENGINE_EXPRESSIONRANGE_H
