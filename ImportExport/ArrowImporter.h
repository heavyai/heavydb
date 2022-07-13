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

#ifndef ARROW_IMPORTER_H
#define ARROW_IMPORTER_H

#include <cstdlib>
#include <ctime>
#include <map>
#include <mutex>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <boost/algorithm/string.hpp>
#include <boost/variant.hpp>

#include "Shared/SqlTypesLayout.h"
#include "Shared/ThreadController.h"
#include "Shared/sqltypes.h"

using arrow::Array;
using arrow::Type;

struct ArrowImporterException : std::runtime_error {
  using std::runtime_error::runtime_error;
};

template <typename T = ArrowImporterException>
inline void arrow_throw_if(const bool cond, const std::string& message) {
  if (cond) {
    // work around race from goooogle log
    static std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);
    LOG(ERROR) << message;
    throw T(message);
  }
}

#ifdef ENABLE_IMPORT_PARQUET
#include <parquet/api/reader.h>
#include <parquet/api/writer.h>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>
#endif  // ENABLE_IMPORT_PARQUET

#include "arrow/util/decimal.h"

namespace {

using VarValue =
    boost::variant<bool, float, double, int64_t, std::string, void*, arrow::Decimal128>;

template <typename T>
using enable_if_integral = typename std::enable_if_t<std::is_integral<T>::value, T>;
template <typename T>
using enable_if_integral_not_bool =
    typename std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value,
                              T>;
template <typename T>
using enable_if_floating = typename std::enable_if_t<std::is_floating_point<T>::value, T>;

#define exprtype(expr) std::decay_t<decltype(expr)>

inline std::string error_context(const ColumnDescriptor* cd,
                                 import_export::BadRowsTracker* const bad_rows_tracker) {
  return bad_rows_tracker ? "File " + bad_rows_tracker->file_name + ", row-group " +
                                std::to_string(bad_rows_tracker->row_group) +
                                (cd ? ", column " + cd->columnName + ": " : "")
                          : std::string();
}

template <typename SrcType, typename DstType>
inline VarValue get_numeric_value(const arrow::Array& array, const int64_t idx) {
  using ArrayType = typename arrow::TypeTraits<SrcType>::ArrayType;
  return (DstType) static_cast<const ArrayType&>(array).Value(idx);
}

template <typename SrcType>
inline VarValue get_string_value(const arrow::Array& array, const int64_t idx) {
  using ArrayType = typename arrow::TypeTraits<SrcType>::ArrayType;
  return static_cast<const ArrayType&>(array).GetString(idx);
}

#define NUMERIC_CASE(tid, src_type, var_type) \
  case arrow::Type::tid:                      \
    return get_numeric_value<src_type, var_type>;
#define STRING_CASE(tid, src_type) \
  case arrow::Type::tid:           \
    return get_string_value<src_type>;

inline auto value_getter(const arrow::Array& array,
                         const ColumnDescriptor* cd,
                         import_export::BadRowsTracker* const bad_rows_tracker) {
  switch (array.type_id()) {
    NUMERIC_CASE(BOOL, arrow::BooleanType, bool)
    NUMERIC_CASE(UINT8, arrow::UInt8Type, int64_t)
    NUMERIC_CASE(UINT16, arrow::UInt16Type, int64_t)
    NUMERIC_CASE(UINT32, arrow::UInt32Type, int64_t)
    NUMERIC_CASE(UINT64, arrow::Int64Type, int64_t)
    NUMERIC_CASE(INT8, arrow::Int8Type, int64_t)
    NUMERIC_CASE(INT16, arrow::Int16Type, int64_t)
    NUMERIC_CASE(INT32, arrow::Int32Type, int64_t)
    NUMERIC_CASE(INT64, arrow::Int64Type, int64_t)
    NUMERIC_CASE(FLOAT, arrow::FloatType, float)
    NUMERIC_CASE(DOUBLE, arrow::DoubleType, double)
    NUMERIC_CASE(DATE32, arrow::Date32Type, int64_t)
    NUMERIC_CASE(DATE64, arrow::Date64Type, int64_t)
    NUMERIC_CASE(TIME64, arrow::Time64Type, int64_t)
    NUMERIC_CASE(TIME32, arrow::Time32Type, int64_t)
    NUMERIC_CASE(TIMESTAMP, arrow::TimestampType, int64_t)
    NUMERIC_CASE(DECIMAL, arrow::Decimal128Type, arrow::Decimal128)
    STRING_CASE(STRING, arrow::StringType)
    STRING_CASE(BINARY, arrow::BinaryType)
    default:
      arrow_throw_if(true,
                     error_context(cd, bad_rows_tracker) + "Parquet type " +
                         array.type()->name() + " is not supported");
      throw;
  }
}

inline void type_conversion_error(const std::string pt,
                                  const ColumnDescriptor* cd,
                                  import_export::BadRowsTracker* const bad_rows_tracker) {
  arrow_throw_if(true,
                 error_context(cd, bad_rows_tracker) +
                     "Invalid type conversion from parquet " + pt + " type to " +
                     cd->columnType.get_type_name());
}

template <typename DATA_TYPE, typename VALUE_TYPE>
inline void data_conversion_error(const VALUE_TYPE v,
                                  const ColumnDescriptor* cd,
                                  import_export::BadRowsTracker* const bad_rows_tracker) {
  arrow_throw_if(true,
                 error_context(cd, bad_rows_tracker) +
                     "Invalid data conversion from parquet value " + std::to_string(v) +
                     " to " + std::to_string(DATA_TYPE(v)));
}

inline void data_conversion_error(const std::string& v,
                                  const ColumnDescriptor* cd,
                                  import_export::BadRowsTracker* const bad_rows_tracker) {
  arrow_throw_if(true,
                 error_context(cd, bad_rows_tracker) +
                     "Invalid data conversion from parquet string '" + v + "' to " +
                     cd->columnType.get_type_name() + " column type");
}

// models the variant data buffers of TypedImportBuffer (LHS)
struct DataBufferBase {
  const ColumnDescriptor* cd;
  const arrow::Array& array;
  import_export::BadRowsTracker* const bad_rows_tracker;
  // in case of arrow-decimal to omni-decimal conversion
  // dont get/set these info on every row of arrow array
  const arrow::DataType& arrow_type;
  const int arrow_decimal_scale;
  const SQLTypeInfo old_type;
  const SQLTypeInfo new_type;
  DataBufferBase(const ColumnDescriptor* cd,
                 const arrow::Array& array,
                 import_export::BadRowsTracker* const bad_rows_tracker)
      : cd(cd)
      , array(array)
      , bad_rows_tracker(bad_rows_tracker)
      , arrow_type(*array.type())
      , arrow_decimal_scale(
            arrow_type.id() == arrow::Type::DECIMAL
                ? static_cast<const arrow::Decimal128Type&>(arrow_type).scale()
                : 0)
      , old_type(cd->columnType.get_type(),
                 cd->columnType.get_dimension(),
                 arrow_decimal_scale,
                 true)
      , new_type(cd->columnType.get_type(),
                 cd->columnType.get_dimension(),
                 cd->columnType.get_scale(),
                 true) {}
};

template <typename DATA_TYPE>
struct DataBuffer : DataBufferBase {
  std::vector<DATA_TYPE>& buffer;
  DataBuffer(const ColumnDescriptor* cd,
             const arrow::Array& array,
             std::vector<DATA_TYPE>& buffer,
             import_export::BadRowsTracker* const bad_rows_tracker)
      : DataBufferBase(cd, array, bad_rows_tracker), buffer(buffer) {}
};

constexpr int64_t kMillisecondsInSecond = 1000LL;
constexpr int64_t kMicrosecondsInSecond = 1000LL * 1000LL;
constexpr int64_t kNanosecondsinSecond = 1000LL * 1000LL * 1000LL;
constexpr int32_t kSecondsInDay = 86400;

static const std::map<std::pair<int32_t, arrow::TimeUnit::type>,
                      std::pair<SQLOps, int64_t>>
    _precision_scale_lookup{
        {{0, arrow::TimeUnit::MILLI}, {kDIVIDE, kMillisecondsInSecond}},
        {{0, arrow::TimeUnit::MICRO}, {kDIVIDE, kMicrosecondsInSecond}},
        {{0, arrow::TimeUnit::NANO}, {kDIVIDE, kNanosecondsinSecond}},
        {{3, arrow::TimeUnit::SECOND}, {kMULTIPLY, kMicrosecondsInSecond}},
        {{3, arrow::TimeUnit::MICRO}, {kDIVIDE, kMillisecondsInSecond}},
        {{3, arrow::TimeUnit::NANO}, {kDIVIDE, kMicrosecondsInSecond}},
        {{6, arrow::TimeUnit::SECOND}, {kMULTIPLY, kMicrosecondsInSecond}},
        {{6, arrow::TimeUnit::MILLI}, {kMULTIPLY, kMillisecondsInSecond}},
        {{6, arrow::TimeUnit::NANO}, {kDIVIDE, kMillisecondsInSecond}},
        {{9, arrow::TimeUnit::SECOND}, {kMULTIPLY, kNanosecondsinSecond}},
        {{9, arrow::TimeUnit::MILLI}, {kMULTIPLY, kMicrosecondsInSecond}},
        {{9, arrow::TimeUnit::MICRO}, {kMULTIPLY, kMillisecondsInSecond}}};

// models the variant values of Arrow Array (RHS)
template <typename VALUE_TYPE>
struct ArrowValueBase {
  const DataBufferBase& data;
  const VALUE_TYPE v;
  const int32_t dimension;
  ArrowValueBase(const DataBufferBase& data, const VALUE_TYPE& v)
      : data(data)
      , v(v)
      , dimension(data.cd->columnType.is_high_precision_timestamp()
                      ? data.cd->columnType.get_dimension()
                      : 0) {}
  template <bool enabled = std::is_integral<VALUE_TYPE>::value>
  int64_t resolve_time(const VALUE_TYPE& v, std::enable_if_t<enabled>* = 0) const {
    const auto& type_id = data.arrow_type.id();
    if (type_id == arrow::Type::DATE32 || type_id == arrow::Type::DATE64) {
      auto& date_type = static_cast<const arrow::DateType&>(data.arrow_type);
      switch (date_type.unit()) {
        case arrow::DateUnit::DAY:
          return v * kSecondsInDay;
        case arrow::DateUnit::MILLI:
          return v / kMillisecondsInSecond;
      }
    } else if (type_id == arrow::Type::TIME32 || type_id == arrow::Type::TIME64 ||
               type_id == arrow::Type::TIMESTAMP) {
      auto& time_type = static_cast<const arrow::TimeType&>(data.arrow_type);
      const auto result =
          _precision_scale_lookup.find(std::make_pair(dimension, time_type.unit()));
      if (result != _precision_scale_lookup.end()) {
        const auto scale = result->second;
        return scale.first == kMULTIPLY ? v * scale.second : v / scale.second;
      } else {
        return v;
      }
    }
    UNREACHABLE() << data.arrow_type << " is not a valid Arrow time or date type";
    return 0;
  }
  template <bool enabled = std::is_integral<VALUE_TYPE>::value>
  int64_t resolve_time(const VALUE_TYPE& v, std::enable_if_t<!enabled>* = 0) const {
    static_assert(enabled, "unreachable");
    return 0;
  }
};

template <typename VALUE_TYPE>
struct ArrowValue : ArrowValueBase<VALUE_TYPE> {};

template <>
struct ArrowValue<void*> : ArrowValueBase<void*> {
  using VALUE_TYPE = void*;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}

  template <typename DATA_TYPE>
  explicit operator DATA_TYPE() const {
    if constexpr (std::is_integral<DATA_TYPE>::value) {  // NOLINT
      return inline_fixed_encoding_null_val(data.cd->columnType);
    } else if constexpr (std::is_floating_point<DATA_TYPE>::value) {  // NOLINT
      return inline_fp_null_val(data.cd->columnType);
    } else if constexpr (std::is_same<DATA_TYPE, std::string>::value) {  // NOLINT
      return std::string();
    }
  }
};

template <>
struct ArrowValue<bool> : ArrowValueBase<bool> {
  using VALUE_TYPE = bool;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}

  template <typename DATA_TYPE>
  explicit operator DATA_TYPE() const {
    if constexpr (std::is_integral<DATA_TYPE>::value) {  // NOLINT
      if (!(data.cd->columnType.is_number() || data.cd->columnType.is_boolean())) {
        type_conversion_error("bool", data.cd, data.bad_rows_tracker);
      }
      return v;
    } else if constexpr (std::is_floating_point<DATA_TYPE>::value) {  // NOLINT
      return v ? 1 : 0;
    } else if constexpr (std::is_same<DATA_TYPE, std::string>::value) {  // NOLINT
      return v ? "T" : "F";
    }
  }
};

template <>
struct ArrowValue<float> : ArrowValueBase<float> {
  using VALUE_TYPE = float;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}

  template <typename DATA_TYPE>
  explicit operator DATA_TYPE() const {
    if constexpr (std::is_integral<DATA_TYPE>::value) {  // NOLINT
      const auto ti = data.cd->columnType;
      DATA_TYPE v = ti.is_decimal() ? this->v * pow(10, ti.get_scale()) : this->v;
      if (!(std::numeric_limits<DATA_TYPE>::lowest() < v &&
            v <= std::numeric_limits<DATA_TYPE>::max())) {
        data_conversion_error<DATA_TYPE>(v, data.cd, data.bad_rows_tracker);
      }
      return v;
    } else if constexpr (std::is_floating_point<DATA_TYPE>::value) {  // NOLINT
      return v;
    } else if constexpr (std::is_same<DATA_TYPE, std::string>::value) {  // NOLINT
      return std::to_string(v);
    }
  }
};

template <>
struct ArrowValue<double> : ArrowValueBase<double> {
  using VALUE_TYPE = double;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}

  template <typename DATA_TYPE>
  explicit operator DATA_TYPE() const {
    if constexpr (std::is_integral<DATA_TYPE>::value) {  // NOLINT
      const auto ti = data.cd->columnType;
      DATA_TYPE v = ti.is_decimal() ? this->v * pow(10, ti.get_scale()) : this->v;
      if (!(std::numeric_limits<DATA_TYPE>::lowest() < v &&
            v <= std::numeric_limits<DATA_TYPE>::max())) {
        data_conversion_error<DATA_TYPE>(v, data.cd, data.bad_rows_tracker);
      }
      return v;
    } else if constexpr (std::is_floating_point<DATA_TYPE>::value) {  // NOLINT
      if (std::is_same<DATA_TYPE, float>::value) {
        if (!(std::numeric_limits<float>::lowest() < v &&
              v <= std::numeric_limits<float>::max())) {
          data_conversion_error<float>(v, data.cd, data.bad_rows_tracker);
        }
      }
      return v;
    } else if constexpr (std::is_same<DATA_TYPE, std::string>::value) {  // NOLINT
      return std::to_string(v);
    }
  }
};

template <>
struct ArrowValue<int64_t> : ArrowValueBase<int64_t> {
  using VALUE_TYPE = int64_t;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}

  template <typename DATA_TYPE>
  explicit operator DATA_TYPE() const {
    if constexpr (std::is_integral<DATA_TYPE>::value) {  // NOLINT
      int64_t v = this->v;
      if (std::is_same<int64_t, DATA_TYPE>::value) {
      } else if (std::numeric_limits<DATA_TYPE>::lowest() < v &&
                 v <= std::numeric_limits<DATA_TYPE>::max()) {
      } else {
        data_conversion_error<DATA_TYPE>(v, data.cd, data.bad_rows_tracker);
      }
      if (data.cd->columnType.is_time()) {
        v = this->resolve_time(v);
      }
      return v;
    } else if constexpr (std::is_floating_point<DATA_TYPE>::value) {  // NOLINT
      return v;
    } else if constexpr (std::is_same<DATA_TYPE, std::string>::value) {  // NOLINT
      const auto& type_id = data.arrow_type.id();
      if (type_id == arrow::Type::DATE32 || type_id == arrow::Type::DATE64) {
        auto& date_type = static_cast<const arrow::DateType&>(data.arrow_type);
        SQLTypeInfo ti(kDATE);
        Datum datum;
        datum.bigintval =
            date_type.unit() == arrow::DateUnit::MILLI ? v / kMicrosecondsInSecond : v;
        return DatumToString(datum, ti);
      } else if (type_id == arrow::Type::TIME32 || type_id == arrow::Type::TIME64 ||
                 type_id == arrow::Type::TIMESTAMP) {
        auto& time_type = static_cast<const arrow::TimeType&>(data.arrow_type);
        const auto result =
            _precision_scale_lookup.find(std::make_pair(0, time_type.unit()));
        int64_t divisor{1};
        if (result != _precision_scale_lookup.end()) {
          divisor = result->second.second;
        }
        SQLTypeInfo ti(kTIMESTAMP);
        Datum datum;
        datum.bigintval = v / divisor;
        auto time_str = DatumToString(datum, ti);
        if (divisor != 1 && v % divisor) {
          time_str += "." + std::to_string(v % divisor);
        }
        return time_str;
      }
      return std::to_string(v);
    }
  }
};

template <>
struct ArrowValue<std::string> : ArrowValueBase<std::string> {
  using VALUE_TYPE = std::string;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}

  template <typename DATA_TYPE>
  explicit operator DATA_TYPE() const {
    if constexpr (std::is_same<DATA_TYPE, bool>::value) {  // NOLINT
      if (v.size() == 0) {
        return inline_int_null_value<int8_t>();
      }
      try {
        SQLTypeInfo ti(kBOOLEAN);
        auto datum = StringToDatum(v, ti);
        return datum.boolval;
      } catch (...) {
        data_conversion_error(v, data.cd, data.bad_rows_tracker);
        return false;
      }
    } else if constexpr (std::is_integral<DATA_TYPE>::value) {  // NOLINT
      if (v.size() == 0) {
        return inline_fixed_encoding_null_val(data.cd->columnType);
      }
      try {
        auto ti = data.cd->columnType;
        auto datum = StringToDatum(v, ti);
        return datum.bigintval;
      } catch (...) {
        data_conversion_error(v, data.cd, data.bad_rows_tracker);
        return 0;
      }
    } else if constexpr (std::is_floating_point<DATA_TYPE>::value) {  // NOLINT
      return atof(v.data());
    } else if constexpr (std::is_same<DATA_TYPE, std::string>::value) {  // NOLINT
      return v;
    }
  }
};

template <>
struct ArrowValue<arrow::Decimal128> : ArrowValueBase<arrow::Decimal128> {
  using VALUE_TYPE = arrow::Decimal128;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {
    // omni decimal has only 64 bits
    arrow_throw_if(!(v.high_bits() == 0 || v.high_bits() == -1),
                   error_context(data.cd, data.bad_rows_tracker) +
                       "Truncation error on Arrow Decimal128 value");
  }

  template <typename DATA_TYPE>
  explicit operator DATA_TYPE() const {
    if constexpr (std::is_integral<DATA_TYPE>::value) {  // NOLINT
      int64_t v = static_cast<int64_t>(this->v);
      if (data.cd->columnType.is_decimal()) {
        return convert_decimal_value_to_scale(v, data.old_type, data.new_type);
      }
      if (data.arrow_decimal_scale) {
        v = std::llround(v / pow(10, data.arrow_decimal_scale));
      }
      if (std::is_same<int64_t, DATA_TYPE>::value) {
      } else if (std::numeric_limits<DATA_TYPE>::lowest() < v &&
                 v <= std::numeric_limits<DATA_TYPE>::max()) {
      } else {
        data_conversion_error<DATA_TYPE>(v, data.cd, data.bad_rows_tracker);
      }
      return v;
    } else if constexpr (std::is_floating_point<DATA_TYPE>::value) {  // NOLINT
      int64_t v = static_cast<int64_t>(this->v);
      return data.arrow_decimal_scale ? v / pow(10, data.arrow_decimal_scale) : v;
    } else if constexpr (std::is_same<DATA_TYPE, std::string>::value) {  // NOLINT
      return v.ToString(data.arrow_decimal_scale);
    }
  }
};

// appends a converted RHS value to LHS data block
template <typename DATA_TYPE>
inline auto& operator<<(DataBuffer<DATA_TYPE>& data, const VarValue& var) {
  boost::apply_visitor(
      [&data](const auto& v) {
        data.buffer.push_back(DATA_TYPE(ArrowValue<exprtype(v)>(data, v)));
      },
      var);
  return data;
}

}  // namespace
#endif  // ARROW_IMPORTER_H
