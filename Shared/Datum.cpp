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

/**
 * @file		DatumString.cpp
 * @brief		Functions to convert between strings and Datum
 *
 */

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
                                                     "POINT",
                                                     "LINESTRING",
                                                     "POLYGON",
                                                     "MULTIPOLYGON",
                                                     "TINYINT",
                                                     "GEOMETRY",
                                                     "GEOGRAPHY",
                                                     "EVAL_CONTEXT_TYPE",
                                                     "VOID",
                                                     "CURSOR",
                                                     "COLUMN",
                                                     "COLUMN_LIST",
                                                     "MULTILINESTRING",
                                                     "MULTIPOINT"};
std::string SQLTypeInfo::comp_name[kENCODING_LAST] =
    {"NONE", "FIXED", "RL", "DIFF", "DICT", "SPARSE", "COMPRESSED", "DAYS"};

namespace {
// Return decimal_value * 10^dscale
int64_t convert_decimal_value_to_scale_internal(const int64_t decimal_value,
                                                int const dscale) {
  constexpr int max_scale = sql_constants::kMaxRepresentableNumericPrecision;  // 19
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

int64_t parse_numeric(const std::string_view s, SQLTypeInfo& ti) {
  // if we are given a dimension, first parse to the maximum precision of the string
  // and then convert to the correct size
  if (ti.get_dimension() != 0) {
    SQLTypeInfo ti_string(kNUMERIC, 0, 0, false);
    return convert_decimal_value_to_scale(parse_numeric(s, ti_string), ti_string, ti);
  }
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

  // set the type info based on the literal string
  ti.set_scale(static_cast<int>(after_dot.length()));
  ti.set_dimension(static_cast<int>(before_dot_digits + ti.get_scale()));
  ti.set_notnull(false);
  if (ti.get_scale()) {
    result = convert_decimal_value_to_scale_internal(result, ti.get_scale());
  }
  result += fraction;

  return result * sign;
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

std::string toString(SQLTypeInfo const& ti, unsigned const fieldsize) {
  return ti.get_type_name() + '(' + std::to_string(fieldsize) + ')';
}

// GCC 10 does not support std::from_chars w/ double, so strtold() is used instead.
// Convert s to long double then round to integer type T.
// It's not assumed that long double is any particular size; it is to be nice to
// users who use floating point values where integers are expected. Some platforms
// may be more accommodating with larger long doubles than others.
template <typename T, typename U = long double>
T parseFloatAsInteger(std::string_view s, SQLTypeInfo const& ti) {
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
                             ti.get_type_name());
  } else if (str_begin + s.size() != str_end) {
    throw std::runtime_error(std::string("Unexpected character \"") + *str_end +
                             "\" encountered in " + ti.get_type_name() + " value " +
                             std::string(s));
  }
  value = std::round(value);
  if (!std::isfinite(value)) {
    throw std::runtime_error("Invalid conversion from \"" + std::string(s) + "\" to " +
                             ti.get_type_name());
  } else if (value < static_cast<U>(std::numeric_limits<T>::min()) ||
             static_cast<U>(std::numeric_limits<T>::max()) < value) {
    throw std::runtime_error("Integer " + std::string(s) + " is out of range for " +
                             ti.get_type_name());
  }
  return static_cast<T>(value);
}

// String ends in either "." or ".0".
inline bool hasCommonSuffix(char const* const ptr, char const* const end) {
  return *ptr == '.' && (ptr + 1 == end || (ptr[1] == '0' && ptr + 2 == end));
}

template <typename T>
T parseInteger(std::string_view s, SQLTypeInfo const& ti) {
  T retval{0};
  char const* const end = s.data() + s.size();
  auto [ptr, error_code] = std::from_chars(s.data(), end, retval);
  if (ptr != end) {
    if (error_code != std::errc() || !hasCommonSuffix(ptr, end)) {
      retval = parseFloatAsInteger<T>(s, ti);
    }
  } else if (error_code != std::errc()) {
    if (error_code == std::errc::result_out_of_range) {
      throw std::runtime_error("Integer " + std::string(s) + " is out of range for " +
                               ti.get_type_name());
    }
    throw std::runtime_error("Invalid conversion from \"" + std::string(s) + "\" to " +
                             ti.get_type_name());
  }
  // Bounds checking based on SQLTypeInfo.
  unsigned const fieldsize =
      ti.get_compression() == kENCODING_FIXED ? ti.get_comp_param() : 8 * sizeof(T);
  if (fieldsize < 8 * sizeof(T)) {
    if (maxValue<T>(fieldsize) < retval) {
      throw std::runtime_error("Integer " + std::string(s) +
                               " exceeds maximum value for " + toString(ti, fieldsize));
    } else if (ti.get_notnull()) {
      if (retval < minValue<T>(fieldsize)) {
        throw std::runtime_error("Integer " + std::string(s) +
                                 " exceeds minimum value for " + toString(ti, fieldsize));
      }
    } else {
      if (retval <= minValue<T>(fieldsize)) {
        throw std::runtime_error("Integer " + std::string(s) +
                                 " exceeds minimum value for nullable " +
                                 toString(ti, fieldsize));
      }
    }
  } else if (!ti.get_notnull() && retval == std::numeric_limits<T>::min()) {
    throw std::runtime_error("Integer " + std::string(s) +
                             " exceeds minimum value for nullable " +
                             toString(ti, fieldsize));
  }
  return retval;
}

inline SQLTypes get_type_for_datum(const SQLTypeInfo& ti) {
  SQLTypes type;
  if (ti.is_decimal()) {
    type = decimal_to_int_type(ti);
  } else if (ti.is_dict_encoded_string()) {
    type = string_dict_to_int_type(ti);
  } else {
    type = ti.get_type();
  }
  return type;
}

}  // namespace

Datum NullDatum(const SQLTypeInfo& ti) {
  Datum d;
  const auto type = get_type_for_datum(ti);
  switch (type) {
    case kBOOLEAN:
      d.boolval = inline_fixed_encoding_null_val(ti);
      break;
    case kBIGINT:
      d.bigintval = inline_fixed_encoding_null_val(ti);
      break;
    case kINT:
      d.intval = inline_fixed_encoding_null_val(ti);
      break;
    case kSMALLINT:
      d.smallintval = inline_fixed_encoding_null_val(ti);
      break;
    case kTINYINT:
      d.tinyintval = inline_fixed_encoding_null_val(ti);
      break;
    case kFLOAT:
      d.floatval = NULL_FLOAT;
      break;
    case kDOUBLE:
      d.doubleval = NULL_DOUBLE;
      break;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      d.bigintval = inline_fixed_encoding_null_val(ti);
      break;
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      throw std::runtime_error("Internal error: geometry type in NullDatum.");
    default:
      throw std::runtime_error("Internal error: invalid type in NullDatum.");
  }
  return d;
}

bool IsNullDatum(const Datum datum, const SQLTypeInfo& ti) {
  const Datum null_datum = NullDatum(ti);
  return DatumEqual(datum, null_datum, ti);
}

/*
 * @brief convert string to a datum
 */
Datum StringToDatum(const std::string_view s, SQLTypeInfo& ti) {
  Datum d;
  try {
    switch (ti.get_type()) {
      case kARRAY:
      case kCOLUMN:
      case kCOLUMN_LIST:
        break;
      case kBOOLEAN:
        if (s == "t" || s == "T" || s == "1" || to_upper(std::string(s)) == "TRUE") {
          d.boolval = true;
        } else if (s == "f" || s == "F" || s == "0" ||
                   to_upper(std::string(s)) == "FALSE") {
          d.boolval = false;
        } else {
          throw std::runtime_error("Invalid string for boolean " + std::string(s));
        }
        break;
      case kNUMERIC:
      case kDECIMAL:
        d.bigintval = parse_numeric(s, ti);
        break;
      case kBIGINT:
        d.bigintval = parseInteger<int64_t>(s, ti);
        break;
      case kINT:
        d.intval = parseInteger<int32_t>(s, ti);
        break;
      case kSMALLINT:
        d.smallintval = parseInteger<int16_t>(s, ti);
        break;
      case kTINYINT:
        d.tinyintval = parseInteger<int8_t>(s, ti);
        break;
      case kFLOAT:
        d.floatval = std::stof(std::string(s));
        break;
      case kDOUBLE:
        d.doubleval = std::stod(std::string(s));
        break;
      case kTIME:
        d.bigintval = dateTimeParse<kTIME>(s, ti.get_dimension());
        break;
      case kTIMESTAMP:
        d.bigintval = dateTimeParse<kTIMESTAMP>(s, ti.get_dimension());
        break;
      case kDATE:
        d.bigintval = dateTimeParse<kDATE>(s, ti.get_dimension());
        break;
      case kPOINT:
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        throw std::runtime_error("Internal error: geometry type in StringToDatum.");
      default:
        throw std::runtime_error("Internal error: invalid type in StringToDatum: " +
                                 ti.get_type_name());
    }
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("Invalid conversion from string to " + ti.get_type_name());
  } catch (const std::out_of_range&) {
    throw std::runtime_error("Got out of range error during conversion from string to " +
                             ti.get_type_name());
  }
  return d;
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
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
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

/*
 * @brief convert datum to string
 */
std::string DatumToString(Datum d, const SQLTypeInfo& ti) {
  constexpr size_t buf_size = 64;
  char buf[buf_size];  // Hold "2000-03-01 12:34:56.123456789" and large years.
  switch (ti.get_type()) {
    case kBOOLEAN:
      if (d.boolval) {
        return "t";
      }
      return "f";
    case kNUMERIC:
    case kDECIMAL: {
      double v = (double)d.bigintval / pow(10, ti.get_scale());
      int size = snprintf(buf, buf_size, "%*.*f", ti.get_dimension(), ti.get_scale(), v);
      CHECK_LE(0, size) << v << ' ' << ti.to_string();
      CHECK_LT(size_t(size), buf_size) << v << ' ' << ti.to_string();
      return buf;
    }
    case kINT:
      return std::to_string(d.intval);
    case kSMALLINT:
      return std::to_string(d.smallintval);
    case kTINYINT:
      return std::to_string(d.tinyintval);
    case kBIGINT:
      return std::to_string(d.bigintval);
    case kFLOAT:
      return std::to_string(d.floatval);
    case kDOUBLE:
      return std::to_string(d.doubleval);
    case kTIME: {
      size_t const len = shared::formatHMS(buf, buf_size, d.bigintval);
      CHECK_EQ(8u, len);  // 8 == strlen("HH:MM:SS")
      return buf;
    }
    case kTIMESTAMP: {
      unsigned const dim = ti.get_dimension();  // assumes dim <= 9
      size_t const len = shared::formatDateTime(buf, buf_size, d.bigintval, dim);
      CHECK_LE(19u + bool(dim) + dim, len);  // 19 = strlen("YYYY-MM-DD HH:MM:SS")
      return buf;
    }
    case kDATE: {
      size_t const len = shared::formatDate(buf, buf_size, d.bigintval);
      CHECK_LE(10u, len);  // 10 == strlen("YYYY-MM-DD")
      return buf;
    }
    case kINTERVAL_DAY_TIME:
      return std::to_string(d.bigintval) + " ms (day-time interval)";
    case kINTERVAL_YEAR_MONTH:
      return std::to_string(d.bigintval) + " month(s) (year-month interval)";
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (d.stringval == nullptr) {
        return "NULL";
      }
      return *d.stringval;
    default:
      throw std::runtime_error("Internal error: invalid type " + ti.get_type_name() +
                               " in DatumToString.");
  }
  return "";
}

int64_t extract_int_type_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  switch (type) {
    case kBOOLEAN:
      return datum.tinyintval;
    case kTINYINT:
      return datum.tinyintval;
    case kSMALLINT:
      return datum.smallintval;
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      CHECK_EQ(kENCODING_DICT, ti.get_compression());
    case kINT:
      return datum.intval;
    case kBIGINT:
      return datum.bigintval;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return datum.bigintval;
    default:
      abort();
  }
}

double extract_fp_type_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  const auto type = ti.get_type();
  switch (type) {
    case kFLOAT:
      return datum.floatval;
    case kDOUBLE:
      return datum.doubleval;
    default:
      abort();
  }
}

SQLTypes decimal_to_int_type(const SQLTypeInfo& ti) {
  return get_int_type_by_size(ti.get_size());
}

SQLTypes string_dict_to_int_type(const SQLTypeInfo& ti) {
  CHECK(ti.is_dict_encoded_string());
  switch (ti.get_size()) {
    case 1:
      return kTINYINT;
    case 2:
      return kSMALLINT;
    case 4:
      return kINT;
    default:
      UNREACHABLE() << "Unexpected string dictionary encoding size: " << ti.get_size();
  }
  return kNULLT;
}

int8_t* append_datum(int8_t* buf, const Datum& d, const SQLTypeInfo& ti) {
  SQLTypes type;
  if (ti.is_dict_encoded_string()) {
    type = string_dict_to_int_type(ti);
  } else {
    type = ti.get_type();
  }
  switch (type) {
    case kBOOLEAN:
      *(int8_t*)buf = d.boolval;
      return buf + sizeof(int8_t);
    case kNUMERIC:
    case kDECIMAL:
    case kBIGINT:
      *(int64_t*)buf = d.bigintval;
      return buf + sizeof(int64_t);
    case kINT:
      *(int32_t*)buf = d.intval;
      return buf + sizeof(int32_t);
    case kSMALLINT:
      *(int16_t*)buf = d.smallintval;
      return buf + sizeof(int16_t);
    case kTINYINT:
      *(int8_t*)buf = d.tinyintval;
      return buf + sizeof(int8_t);
    case kFLOAT:
      *(float*)buf = d.floatval;
      return buf + sizeof(float);
    case kDOUBLE:
      *(double*)buf = d.doubleval;
      return buf + sizeof(double);
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      *reinterpret_cast<int64_t*>(buf) = d.bigintval;
      return buf + sizeof(int64_t);
    default:
      UNREACHABLE() << "Unexpected type: " << type;
      return nullptr;
  }
}

// Return decimal_value * 10^dscale
// where dscale = new_type_info.get_scale() - type_info.get_scale()
int64_t convert_decimal_value_to_scale(const int64_t decimal_value,
                                       const SQLTypeInfo& type_info,
                                       const SQLTypeInfo& new_type_info) {
  int const dscale = new_type_info.get_scale() - type_info.get_scale();
  return convert_decimal_value_to_scale_internal(decimal_value, dscale);
}
