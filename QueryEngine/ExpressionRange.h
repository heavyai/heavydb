#ifndef QUERYENGINE_EXPRESSIONRANGE_H
#define QUERYENGINE_EXPRESSIONRANGE_H

#include "../Analyzer/Analyzer.h"
#include "../Fragmenter/Fragmenter.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <deque>

typedef boost::multiprecision::number<
    boost::multiprecision::
        cpp_int_backend<64, 64, boost::multiprecision::signed_magnitude, boost::multiprecision::checked, void>>
    checked_int64_t;

enum class ExpressionRangeType {
  Invalid,
  Integer,
  FloatingPoint,
};

class ExpressionRange;

template <typename T>
T getMin(const ExpressionRange& other);

template <typename T>
T getMax(const ExpressionRange& other);

class ExpressionRange {
 public:
  static ExpressionRange makeIntRange(const int64_t int_min,
                                      const int64_t int_max,
                                      const int64_t bucket,
                                      const bool has_nulls) {
    return ExpressionRange(int_min, int_max, bucket, has_nulls);
  }

  static ExpressionRange makeFpRange(const double fp_min, const double fp_max, const bool has_nulls) {
    return ExpressionRange(fp_min, fp_max, has_nulls);
  }

  static ExpressionRange makeInvalidRange() { return ExpressionRange(); }

  int64_t getIntMin() const {
    CHECK(ExpressionRangeType::Integer == type);
    return int_min;
  }

  int64_t getIntMax() const {
    CHECK(ExpressionRangeType::Integer == type);
    return int_max;
  }

  double getFpMin() const {
    CHECK(ExpressionRangeType::FloatingPoint == type);
    return fp_min;
  }

  double getFpMax() const {
    CHECK(ExpressionRangeType::FloatingPoint == type);
    return fp_max;
  }

  ExpressionRangeType getType() const { return type; }

  int64_t getBucket() const { return bucket_; }

  bool hasNulls() const { return has_nulls; }

  ExpressionRange operator+(const ExpressionRange& other) const;
  ExpressionRange operator-(const ExpressionRange& other) const;
  ExpressionRange operator*(const ExpressionRange& other) const;
  ExpressionRange operator/(const ExpressionRange& other) const;
  ExpressionRange operator||(const ExpressionRange& other) const;

 private:
  ExpressionRange(const int64_t int_min_in, const int64_t int_max_in, const int64_t bucket, const bool has_nulls_in)
      : type(ExpressionRangeType::Integer),
        has_nulls(has_nulls_in),
        int_min(int_min_in),
        int_max(int_max_in),
        bucket_(bucket) {}

  ExpressionRange(const double fp_min_in, const double fp_max_in, const bool has_nulls_in)
      : type(ExpressionRangeType::FloatingPoint),
        has_nulls(has_nulls_in),
        fp_min(fp_min_in),
        fp_max(fp_max_in),
        bucket_(0) {}

  ExpressionRange() : type(ExpressionRangeType::Invalid), has_nulls(false), bucket_(0) {}

  template <class T, class BinOp>
  ExpressionRange binOp(const ExpressionRange& other, const BinOp& bin_op) const {
    if (type == ExpressionRangeType::Invalid || other.type == ExpressionRangeType::Invalid) {
      return ExpressionRange::makeInvalidRange();
    }
    try {
      std::vector<T> limits{bin_op(getMin<T>(*this), getMin<T>(other)),
                            bin_op(getMin<T>(*this), getMax<T>(other)),
                            bin_op(getMax<T>(*this), getMin<T>(other)),
                            bin_op(getMax<T>(*this), getMax<T>(other))};
      ExpressionRange result;
      result.type = (type == ExpressionRangeType::Integer && other.type == ExpressionRangeType::Integer)
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
      return ExpressionRange::makeInvalidRange();
    }
  }

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
  int64_t bucket_;
};

template <>
inline int64_t getMin<int64_t>(const ExpressionRange& e) {
  return e.getIntMin();
}

template <>
inline double getMin<double>(const ExpressionRange& e) {
  return e.getFpMin();
}

template <>
inline int64_t getMax<int64_t>(const ExpressionRange& e) {
  return e.getIntMax();
}

template <>
inline double getMax<double>(const ExpressionRange& e) {
  return e.getFpMax();
}

class Executor;

ExpressionRange getExpressionRange(const Analyzer::Expr*,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>&,
                                   const Executor*);

#endif  // QUERYENGINE_EXPRESSIONRANGE_H
