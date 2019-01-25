/* * Copyright 2017 MapD Technologies, Inc.
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

#ifndef QUERYENGINE_EXPRESSIONRANGE_H
#define QUERYENGINE_EXPRESSIONRANGE_H

#include "../Analyzer/Analyzer.h"
#include "../Shared/unreachable.h"

#include <boost/multiprecision/cpp_int.hpp>  //#include <boost/none.hpp>
#include <boost/optional.hpp>

typedef boost::multiprecision::number<
    boost::multiprecision::cpp_int_backend<64,
                                           64,
                                           boost::multiprecision::signed_magnitude,
                                           boost::multiprecision::checked,
                                           void>>
    checked_int64_t;

enum class ExpressionRangeType { Invalid, Integer, Float, Double, Null };

class ExpressionRange;

class InputColDescriptor;

template <typename T>
T getMin(const ExpressionRange& other);

template <typename T>
T getMax(const ExpressionRange& other);

template <typename T>
T get_value_from_datum(const Datum datum, const SQLTypes type_info) noexcept;

class ExpressionRange {
 public:
  static ExpressionRange makeIntRange(const int64_t int_min,
                                      const int64_t int_max,
                                      const int64_t bucket,
                                      const bool has_nulls) {
    return ExpressionRange(int_min, int_max, bucket, has_nulls);
  }

  static ExpressionRange makeDoubleRange(const double fp_min,
                                         const double fp_max,
                                         const bool has_nulls) {
    return ExpressionRange(ExpressionRangeType::Double, fp_min, fp_max, has_nulls);
  }

  static ExpressionRange makeFloatRange(const float fp_min,
                                        const float fp_max,
                                        const bool has_nulls) {
    return ExpressionRange(ExpressionRangeType::Float, fp_min, fp_max, has_nulls);
  }

  static ExpressionRange makeNullRange() {
    return ExpressionRange(ExpressionRangeType::Null);
  }

  static ExpressionRange makeInvalidRange() { return ExpressionRange(); }

  int64_t getIntMin() const {
    CHECK(ExpressionRangeType::Integer == type_);
    return int_min_;
  }

  int64_t getIntMax() const {
    CHECK(ExpressionRangeType::Integer == type_);
    return int_max_;
  }

  double getFpMin() const {
    CHECK(type_ == ExpressionRangeType::Float || type_ == ExpressionRangeType::Double);
    return fp_min_;
  }

  double getFpMax() const {
    CHECK(type_ == ExpressionRangeType::Float || type_ == ExpressionRangeType::Double);
    return fp_max_;
  }

  void setIntMin(const int64_t int_min) {
    CHECK(ExpressionRangeType::Integer == type_);
    int_min_ = int_min;
  }

  void setIntMax(const int64_t int_max) {
    CHECK(ExpressionRangeType::Integer == type_);
    int_max_ = int_max;
  }

  void setIntInvalidRange() {
    CHECK(ExpressionRangeType::Integer == type_);
    int_max_ = -1;
    int_min_ = 0;
  }

  void setFpMin(const double fp_min) {
    CHECK(type_ == ExpressionRangeType::Float || type_ == ExpressionRangeType::Double);
    fp_min_ = fp_min;
  }

  void setFpMax(const double fp_max) {
    CHECK(type_ == ExpressionRangeType::Float || type_ == ExpressionRangeType::Double);
    fp_max_ = fp_max;
  }

  ExpressionRangeType getType() const { return type_; }

  int64_t getBucket() const {
    CHECK(type_ != ExpressionRangeType::Invalid);
    return bucket_;
  }

  bool hasNulls() const {
    CHECK(type_ != ExpressionRangeType::Invalid);
    return has_nulls_;
  }

  void setHasNulls() { has_nulls_ = true; }

  ExpressionRange operator+(const ExpressionRange& other) const;
  ExpressionRange operator-(const ExpressionRange& other) const;
  ExpressionRange operator*(const ExpressionRange& other) const;
  ExpressionRange operator/(const ExpressionRange& other) const;
  ExpressionRange operator||(const ExpressionRange& other) const;

  bool operator==(const ExpressionRange& other) const;

 private:
  ExpressionRange(const int64_t int_min_in,
                  const int64_t int_max_in,
                  const int64_t bucket,
                  const bool has_nulls_in)
      : type_(ExpressionRangeType::Integer)
      , has_nulls_(has_nulls_in)
      , int_min_(int_min_in)
      , int_max_(int_max_in)
      , bucket_(bucket) {}

  ExpressionRange(const ExpressionRangeType type,
                  const double fp_min_in,
                  const double fp_max_in,
                  const bool has_nulls_in)
      : type_(type)
      , has_nulls_(has_nulls_in)
      , fp_min_(fp_min_in)
      , fp_max_(fp_max_in)
      , bucket_(0) {
    CHECK(type_ == ExpressionRangeType::Float || type_ == ExpressionRangeType::Double);
  }

  ExpressionRange()
      : type_(ExpressionRangeType::Invalid), has_nulls_(false), bucket_(0) {}

  explicit ExpressionRange(const ExpressionRangeType type)
      : type_(type), has_nulls_(true), bucket_(0) {
    CHECK(type_ == ExpressionRangeType::Null);
  }

  template <class T, class BinOp>
  ExpressionRange binOp(const ExpressionRange& other, const BinOp& bin_op) const {
    CHECK(type_ == other.type_);
    try {
      std::vector<T> limits{bin_op(getMin<T>(*this), getMin<T>(other)),
                            bin_op(getMin<T>(*this), getMax<T>(other)),
                            bin_op(getMax<T>(*this), getMin<T>(other)),
                            bin_op(getMax<T>(*this), getMax<T>(other))};
      ExpressionRange result;
      result.type_ = type_;
      result.has_nulls_ = has_nulls_ || other.has_nulls_;
      switch (result.type_) {
        case ExpressionRangeType::Integer: {
          result.int_min_ = *std::min_element(limits.begin(), limits.end());
          result.int_max_ = *std::max_element(limits.begin(), limits.end());
          break;
        }
        case ExpressionRangeType::Float:
        case ExpressionRangeType::Double: {
          result.fp_min_ = *std::min_element(limits.begin(), limits.end());
          result.fp_max_ = *std::max_element(limits.begin(), limits.end());
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

  ExpressionRangeType type_;
  bool has_nulls_;
  union {
    int64_t int_min_;
    double fp_min_;
  };
  union {
    int64_t int_max_;
    double fp_max_;
  };
  int64_t bucket_;
};

template <>
inline int64_t getMin<int64_t>(const ExpressionRange& e) {
  return e.getIntMin();
}

template <>
inline float getMin<float>(const ExpressionRange& e) {
  return e.getFpMin();
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
inline float getMax<float>(const ExpressionRange& e) {
  return e.getFpMax();
}

template <>
inline double getMax<double>(const ExpressionRange& e) {
  return e.getFpMax();
}

template <>
inline int64_t get_value_from_datum(const Datum datum,
                                    const SQLTypes type_info) noexcept {
  switch (type_info) {
    case kBOOLEAN:
      return datum.boolval;
    case kTINYINT:
      return datum.tinyintval;
    case kSMALLINT:
      return datum.smallintval;
    case kINT:
      return datum.intval;
    case kBIGINT:
      return datum.bigintval;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return static_cast<int64_t>(datum.timeval);
    case kNUMERIC:
    case kDECIMAL: {
      return datum.bigintval;
      break;
    }
    default:
      UNREACHABLE();
  }
  return (int64_t)0;
}

template <>
inline double get_value_from_datum(const Datum datum, const SQLTypes type_info) noexcept {
  switch (type_info) {
    case kFLOAT:
      return datum.floatval;
    case kDOUBLE:
      return datum.doubleval;
    default:
      UNREACHABLE();
  }
  return 0.0;
}

void apply_int_qual(const Datum const_datum,
                    const SQLTypes const_type,
                    const SQLOps sql_op,
                    ExpressionRange& qual_range);
void apply_fp_qual(const Datum const_datum,
                   const SQLTypes const_type,
                   const SQLOps sql_op,
                   ExpressionRange& qual_range);

ExpressionRange apply_simple_quals(
    const Analyzer::ColumnVar*,
    const ExpressionRange&,
    const boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> = boost::none);

class Executor;
struct InputTableInfo;

ExpressionRange getLeafColumnRange(const Analyzer::ColumnVar*,
                                   const std::vector<InputTableInfo>&,
                                   const Executor*,
                                   const bool is_outer_join_proj);

ExpressionRange getExpressionRange(
    const Analyzer::Expr*,
    const std::vector<InputTableInfo>&,
    const Executor*,
    boost::optional<std::list<std::shared_ptr<Analyzer::Expr>>> = boost::none);

#endif  // QUERYENGINE_EXPRESSIONRANGE_H
