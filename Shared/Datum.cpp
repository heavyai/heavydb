/*
 * Copyright 2021 OmniSci, Inc.
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

/**
 * @file		DatumString.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Functions to convert between strings and Datum
 **/

#include <algorithm>
#include <cassert>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include "DateConverters.h"
#include "DateTimeParser.h"
#include "IR/Context.h"
#include "IR/Type.h"
#include "Logger/Logger.h"
#include "QueryEngine/DateTimeUtils.h"
#include "StringTransform.h"
#include "misc.h"
#include "sqltypes.h"

std::string SQLTypeInfo::type_name[kSQLTYPE_LAST] = {"NULL",
                                                     "BOOLEAN",
                                                     "CHAR",
                                                     "VARCHAR",
                                                     "NUMERIC",
                                                     "DECIMAL",
                                                     "INTEGER",
                                                     "SMALLINT",
                                                     "FLOAT",
                                                     "DOUBLE",
                                                     "TIME",
                                                     "TIMESTAMP",
                                                     "BIGINT",
                                                     "TEXT",
                                                     "DATE",
                                                     "ARRAY",
                                                     "INTERVAL_DAY_TIME",
                                                     "INTERVAL_YEAR_MONTH",
                                                     "TINYINT",
                                                     "EVAL_CONTEXT_TYPE",
                                                     "VOID",
                                                     "CURSOR",
                                                     "COLUMN",
                                                     "COLUMN_LIST"};
std::string SQLTypeInfo::comp_name[kENCODING_LAST] =
    {"NONE", "FIXED", "RL", "DIFF", "DICT", "SPARSE", "COMPRESSED", "DAYS"};

namespace {
// Return decimal_value * 10^dscale
int64_t convert_decimal_value_to_scale_internal(const int64_t decimal_value,
                                                int const dscale) {
  constexpr int max_scale = std::numeric_limits<uint64_t>::digits10;  // 19
  constexpr auto pow10 = shared::powersOf<uint64_t, max_scale + 1>(10);
  if (dscale < 0) {
    if (dscale < -max_scale) {
      return 0;  // +/- 0.09223372036854775807 rounds to 0
    }
    uint64_t const u = std::abs(decimal_value);
    uint64_t const pow = pow10[-dscale];
    uint64_t div = u / pow;
    uint64_t rem = u % pow;
    div += pow / 2 <= rem;
    return decimal_value < 0 ? -div : div;
  } else if (dscale < max_scale) {
    int64_t retval;
#ifdef _WIN32
    return decimal_value * pow10[dscale];
#else
    if (!__builtin_mul_overflow(decimal_value, pow10[dscale], &retval)) {
      return retval;
    }
#endif
  }
  if (decimal_value == 0) {
    return 0;
  }
  throw std::runtime_error("Overflow in DECIMAL-to-DECIMAL conversion.");
}
}  // namespace

int64_t parse_numeric(const std::string_view s, const hdk::ir::Type* type) {
  // if we are given a dimension, first parse to the maximum precision of the string
  // and then convert to the correct size
  auto dtype = type->as<hdk::ir::DecimalType>();
  CHECK(dtype);
  size_t dot = s.find_first_of('.', 0);
  std::string before_dot;
  std::string after_dot;
  if (dot != std::string::npos) {
    // make .99 as 0.99, or std::stoll below throws exception 'std::invalid_argument'
    before_dot = (0 == dot) ? "0" : s.substr(0, dot);
    after_dot = s.substr(dot + 1);
  } else {
    before_dot = s;
    after_dot = "0";
  }
  const bool is_negative = before_dot.find_first_of('-', 0) != std::string::npos;
  const int64_t sign = is_negative ? -1 : 1;
  int64_t result;
  result = std::abs(std::stoll(before_dot));
  int64_t fraction = 0;
  const size_t before_dot_digits = before_dot.length() - (is_negative ? 1 : 0);

  constexpr int max_digits = std::numeric_limits<int64_t>::digits10;
  if (!after_dot.empty()) {
    int64_t next_digit = 0;
    // After dot will be used to scale integer part so make sure it wont overflow
    if (after_dot.size() + before_dot_digits > max_digits) {
      if (before_dot_digits >= max_digits) {
        after_dot = "0";
      } else {
        next_digit = std::stoll(after_dot.substr(max_digits - before_dot_digits, 1));
        after_dot = after_dot.substr(0, max_digits - before_dot_digits);
      }
    }
    fraction = std::stoll(after_dot);
    fraction += next_digit >= 5 ? 1 : 0;
  }

  result = convert_decimal_value_to_scale_internal(result, after_dot.length());
  result += fraction;
  result = convert_decimal_value_to_scale_internal(result,
                                                   dtype->scale() - after_dot.length());
  return result * sign;
}

int64_t parse_numeric(const std::string_view s, SQLTypeInfo& ti) {
  return parse_numeric(s, hdk::ir::Context::defaultCtx().fromTypeInfo(ti));
}

namespace {

// Equal to NULL value for nullable types.
template <typename T>
T minValue(unsigned const fieldsize) {
  static_assert(std::is_signed_v<T>);
  return T(-1) << (fieldsize - 1);
}

template <typename T>
T maxValue(unsigned const fieldsize) {
  return ~minValue<T>(fieldsize);
}

// GCC 10 does not support std::from_chars w/ double, so strtold() is used instead.
// Convert s to long double then round to integer type T.
// It's not assumed that long double is any particular size; it is to be nice to
// users who use floating point values where integers are expected. Some platforms
// may be more accommodating with larger long doubles than others.
template <typename T, typename U = long double>
T parseFloatAsInteger(std::string_view s, const hdk::ir::Type* type) {
  // Use stack memory if s is small enough before resorting to dynamic memory.
  constexpr size_t bufsize = 64;
  char c_str[bufsize];
  std::string str;
  char const* str_begin;
  char* str_end;
  if (s.size() < bufsize) {
    s.copy(c_str, s.size());
    c_str[s.size()] = '\0';
    str_begin = c_str;
  } else {
    str = s;
    str_begin = str.c_str();
  }
  U value = strtold(str_begin, &str_end);
  if (str_begin == str_end) {
    throw std::runtime_error("Unable to parse " + std::string(s) + " to " +
                             type->toString());
  } else if (str_begin + s.size() != str_end) {
    throw std::runtime_error(std::string("Unexpected character \"") + *str_end +
                             "\" encountered in " + type->toString() + " value " +
                             std::string(s));
  }
  value = std::round(value);
  if (!std::isfinite(value)) {
    throw std::runtime_error("Invalid conversion from \"" + std::string(s) + "\" to " +
                             type->toString());
  } else if (value < static_cast<U>(std::numeric_limits<T>::min()) ||
             static_cast<U>(std::numeric_limits<T>::max()) < value) {
    throw std::runtime_error("Integer " + std::string(s) + " is out of range for " +
                             type->toString());
  }
  return static_cast<T>(value);
}

// String ends in either "." or ".0".
inline bool hasCommonSuffix(char const* const ptr, char const* const end) {
  return *ptr == '.' && (ptr + 1 == end || (ptr[1] == '0' && ptr + 2 == end));
}

template <typename T>
T parseInteger(std::string_view s, const hdk::ir::Type* type) {
  T retval{0};
  char const* const end = s.data() + s.size();
  auto [ptr, error_code] = std::from_chars(s.data(), end, retval);
  if (ptr != end) {
    if (error_code != std::errc() || !hasCommonSuffix(ptr, end)) {
      retval = parseFloatAsInteger<T>(s, type);
    }
  } else if (error_code != std::errc()) {
    if (error_code == std::errc::result_out_of_range) {
      throw std::runtime_error("Integer " + std::string(s) + " is out of range for " +
                               type->toString());
    }
    throw std::runtime_error("Invalid conversion from \"" + std::string(s) + "\" to " +
                             type->toString());
  }
  // Bounds checking based on SQLTypeInfo.
  unsigned const fieldsize = type->size() * 8;
  if (fieldsize < 8 * sizeof(T)) {
    if (maxValue<T>(fieldsize) < retval) {
      throw std::runtime_error("Integer " + std::string(s) +
                               " exceeds maximum value for " + type->toString());
    } else if (!type->nullable()) {
      if (retval < minValue<T>(fieldsize)) {
        throw std::runtime_error("Integer " + std::string(s) +
                                 " exceeds minimum value for " + type->toString());
      }
    } else {
      if (retval <= minValue<T>(fieldsize)) {
        throw std::runtime_error("Integer " + std::string(s) +
                                 " exceeds minimum value for nullable " +
                                 type->toString());
      }
    }
  } else if (type->nullable() && retval == std::numeric_limits<T>::min()) {
    throw std::runtime_error("Integer " + std::string(s) +
                             " exceeds minimum value for nullable " + type->toString());
  }
  return retval;
}

template <typename T>
T parseInteger(std::string_view s, SQLTypeInfo const& ti) {
  return parseInteger<T>(s, hdk::ir::Context::defaultCtx().fromTypeInfo(ti));
}

}  // namespace

/*
 * @brief convert string to a datum
 */
Datum StringToDatum(std::string_view s, const hdk::ir::Type* type) {
  Datum d;
  try {
    switch (type->id()) {
      case hdk::ir::Type::kFixedLenArray:
      case hdk::ir::Type::kVarLenArray:
      case hdk::ir::Type::kColumn:
      case hdk::ir::Type::kColumnList:
        break;
      case hdk::ir::Type::kBoolean:
        if (s == "t" || s == "T" || s == "1" || to_upper(std::string(s)) == "TRUE") {
          d.boolval = true;
        } else if (s == "f" || s == "F" || s == "0" ||
                   to_upper(std::string(s)) == "FALSE") {
          d.boolval = false;
        } else {
          throw std::runtime_error("Invalid string for boolean " + std::string(s));
        }
        break;
      case hdk::ir::Type::kInteger:
        switch (type->size()) {
          case 1:
            d.tinyintval = parseInteger<int8_t>(s, type);
            break;
          case 2:
            d.smallintval = parseInteger<int16_t>(s, type);
            break;
          case 4:
            d.intval = parseInteger<int32_t>(s, type);
            break;
          case 8:
            d.bigintval = parseInteger<int64_t>(s, type);
            break;
          default:
            abort();
        }
        break;
      case hdk::ir::Type::kDecimal:
        CHECK_EQ(type->size(), 8);
        d.bigintval = parse_numeric(s, type);
        break;
      case hdk::ir::Type::kFloatingPoint:
        switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
          case hdk::ir::FloatingPointType::kFloat:
            d.floatval = std::stof(std::string(s));
            break;
          case hdk::ir::FloatingPointType::kDouble:
            d.doubleval = std::stod(std::string(s));
            break;
          default:
            abort();
        }
        break;
      case hdk::ir::Type::kDate:
      case hdk::ir::Type::kTime:
      case hdk::ir::Type::kTimestamp: {
        auto unit = type->as<hdk::ir::DateTimeBaseType>()->unit();
        int64_t val;
        if (type->isDate()) {
          val = dateTimeParse<hdk::ir::Type::kDate>(s, unit);
        } else if (type->isTime()) {
          val = dateTimeParse<hdk::ir::Type::kTime>(s, unit);
        } else {
          val = dateTimeParse<hdk::ir::Type::kTimestamp>(s, unit);
        }
        d.bigintval = val;
      } break;
      default:
        throw std::runtime_error("Internal error: invalid type in StringToDatum: " +
                                 type->toString());
    }
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("Invalid conversion from string to " + type->toString());
  } catch (const std::out_of_range&) {
    throw std::runtime_error("Got out of range error during conversion from string to " +
                             type->toString());
  }
  return d;
}

Datum StringToDatum(std::string_view s, SQLTypeInfo& ti) {
  return StringToDatum(s, hdk::ir::Context::defaultCtx().fromTypeInfo(ti));
}

bool DatumEqual(const Datum a, const Datum b, const SQLTypeInfo& ti) {
  switch (ti.get_type()) {
    case kBOOLEAN:
      return a.boolval == b.boolval;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      return a.bigintval == b.bigintval;
    case kINT:
      return a.intval == b.intval;
    case kSMALLINT:
      return a.smallintval == b.smallintval;
    case kTINYINT:
      return a.tinyintval == b.tinyintval;
    case kFLOAT:
      return a.floatval == b.floatval;
    case kDOUBLE:
      return a.doubleval == b.doubleval;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return a.bigintval == b.bigintval;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (ti.get_compression() == kENCODING_DICT) {
        return a.intval == b.intval;
      }
      if (a.stringval == nullptr && b.stringval == nullptr) {
        return true;
      }
      if (a.stringval == nullptr || b.stringval == nullptr) {
        return false;
      }
      return *a.stringval == *b.stringval;
    default:
      return false;
  }
  return false;
}

size_t hash(Datum datum, const SQLTypeInfo& ti) {
  size_t res = 0;
  switch (ti.get_type()) {
    case kBOOLEAN:
      boost::hash_combine(res, datum.boolval);
      break;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      boost::hash_combine(res, datum.bigintval);
      break;
    case kINT:
      boost::hash_combine(res, datum.intval);
      break;
    case kSMALLINT:
      boost::hash_combine(res, datum.smallintval);
      break;
    case kTINYINT:
      boost::hash_combine(res, datum.tinyintval);
      break;
    case kFLOAT:
      boost::hash_combine(res, datum.floatval);
      break;
    case kDOUBLE:
      boost::hash_combine(res, datum.doubleval);
      break;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      boost::hash_combine(res, datum.bigintval);
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (ti.get_compression() == kENCODING_DICT) {
        boost::hash_combine(res, datum.intval);
      } else if (datum.stringval) {
        boost::hash_combine(res, *datum.stringval);
      }
      break;
    default:
      throw std::runtime_error("Internal error: unsupported Datum type: " +
                               ti.get_type_name());
  }
  return res;
}

/*
 * @brief convert datum to string
 */
std::string DatumToString(Datum d, const hdk::ir::Type* type) {
  constexpr size_t buf_size = 64;
  char buf[buf_size];  // Hold "2000-03-01 12:34:56.123456789" and large years.
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
      if (d.boolval) {
        return "t";
      }
      return "f";
    case hdk::ir::Type::kDecimal: {
      CHECK_EQ(type->size(), 8);
      auto precision = type->as<hdk::ir::DecimalType>()->precision();
      auto scale = type->as<hdk::ir::DecimalType>()->scale();
      double v = (double)d.bigintval / pow(10, scale);
      int size = snprintf(buf, buf_size, "%*.*f", precision, scale, v);
      CHECK_LE(0, size) << v << ' ' << type->toString();
      CHECK_LT(size_t(size), buf_size) << v << ' ' << type->toString();
      return buf;
    }
    case hdk::ir::Type::kInteger:
      switch (type->size()) {
        case 1:
          return std::to_string(d.tinyintval);
        case 2:
          return std::to_string(d.smallintval);
        case 4:
          return std::to_string(d.intval);
        case 8:
          return std::to_string(d.bigintval);
        default:
          break;
      }
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          return std::to_string(d.floatval);
        case hdk::ir::FloatingPointType::kDouble:
          return std::to_string(d.doubleval);
        default:
          break;
      }
      break;
    case hdk::ir::Type::kTime: {
      size_t const len = shared::formatHMS(buf, buf_size, d.bigintval);
      CHECK_EQ(8u, len);  // 8 == strlen("HH:MM:SS")
      return buf;
    }
    case hdk::ir::Type::kTimestamp: {
      auto unit = type->as<hdk::ir::TimestampType>()->unit();
      size_t const len = shared::formatDateTime(buf, buf_size, d.bigintval, unit);
      return buf;
    }
    case hdk::ir::Type::kDate: {
      size_t const len = shared::formatDate(buf, buf_size, d.bigintval);
      CHECK_LE(10u, len);  // 10 == strlen("YYYY-MM-DD")
      return buf;
    }
    case hdk::ir::Type::kInterval:
      switch (type->as<hdk::ir::IntervalType>()->unit()) {
        case hdk::ir::TimeUnit::kMonth:
          return std::to_string(d.bigintval) + " month(s) (year-month interval)";
        case hdk::ir::TimeUnit::kMilli:
          return std::to_string(d.bigintval) + " ms (day-time interval)";
        default:
          break;
      }
      break;
    case hdk::ir::Type::kVarChar:
    case hdk::ir::Type::kText:
      if (d.stringval == nullptr) {
        return "NULL";
      }
      return *d.stringval;
    default:
      break;
  }
  throw std::runtime_error("Internal error: invalid type " + type->toString() +
                           " in DatumToString.");
}

std::string DatumToString(Datum d, const SQLTypeInfo& ti) {
  return DatumToString(d, hdk::ir::Context::defaultCtx().fromTypeInfo(ti));
}

int64_t extract_int_type_from_datum(const Datum datum, const hdk::ir::Type* type) {
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
      return datum.boolval;
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
    case hdk::ir::Type::kExtDictionary:
      switch (type->size()) {
        case 1:
          return datum.tinyintval;
        case 2:
          return datum.smallintval;
        case 4:
          return datum.intval;
        case 8:
          return datum.bigintval;
        default:
          abort();
      }
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kInterval:
      return datum.bigintval;
    default:
      abort();
  }
}

int64_t extract_int_type_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  return extract_int_type_from_datum(datum,
                                     hdk::ir::Context::defaultCtx().fromTypeInfo(ti));
}

double extract_fp_type_from_datum(const Datum datum, const hdk::ir::Type* type) {
  switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
    case hdk::ir::FloatingPointType::kFloat:
      return datum.floatval;
    case hdk::ir::FloatingPointType::kDouble:
      return datum.doubleval;
    default:
      abort();
  }
}

double extract_fp_type_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  return extract_fp_type_from_datum(datum,
                                    hdk::ir::Context::defaultCtx().fromTypeInfo(ti));
}

SQLTypes decimal_to_int_type(const SQLTypeInfo& ti) {
  switch (ti.get_size()) {
    case 1:
      return kTINYINT;
    case 2:
      return kSMALLINT;
    case 4:
      return kINT;
    case 8:
      return kBIGINT;
    default:
      CHECK(false);
  }
  return kNULLT;
}

// Return decimal_value * 10^dscale
// where dscale = new_type_info.get_scale() - type_info.get_scale()
int64_t convert_decimal_value_to_scale(const int64_t decimal_value,
                                       const hdk::ir::Type* type,
                                       const hdk::ir::Type* new_type) {
  int const dscale = new_type->as<hdk::ir::DecimalType>()->scale() -
                     type->as<hdk::ir::DecimalType>()->scale();
  return convert_decimal_value_to_scale_internal(decimal_value, dscale);
}

int64_t convert_decimal_value_to_scale(const int64_t decimal_value,
                                       const SQLTypeInfo& type_info,
                                       const SQLTypeInfo& new_type_info) {
  return convert_decimal_value_to_scale(
      decimal_value,
      hdk::ir::Context::defaultCtx().fromTypeInfo(type_info),
      hdk::ir::Context::defaultCtx().fromTypeInfo(new_type_info));
}
