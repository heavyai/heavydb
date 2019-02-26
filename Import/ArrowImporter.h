/*
 * Copyright 2019 OmniSci, Inc.
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
#include <mutex>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <boost/algorithm/string.hpp>
#include <boost/variant.hpp>

#include "Shared/SqlTypesLayout.h"
#include "Shared/ThreadController.h"
#include "Shared/sqltypes.h"

using namespace arrow;

inline void arrow_throw_if(const bool cond, const std::string& message) {
  if (cond) {
    // work around race from goooogle log
    static std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);
    LOG(ERROR) << message;
    throw std::runtime_error(message);
  }
}

#ifdef ENABLE_IMPORT_PARQUET
#include <parquet/api/reader.h>
#include <parquet/api/writer.h>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>
#endif  // ENABLE_IMPORT_PARQUET

namespace {

using VarValue = boost::variant<bool, float, double, int64_t, std::string, void*>;

template <typename T>
using enable_if_integral = typename std::enable_if_t<std::is_integral<T>::value, T>;
template <typename T>
using enable_if_integral_not_bool =
    typename std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value,
                              T>;
template <typename T>
using enable_if_floating = typename std::enable_if_t<std::is_floating_point<T>::value, T>;

#define exprtype(expr) std::decay_t<decltype(expr)>

template <typename SrcType, typename DstType>
inline VarValue get_numeric_value(const Array& array, const int64_t idx) {
  using ArrayType = typename TypeTraits<SrcType>::ArrayType;
  return (DstType) static_cast<const ArrayType&>(array).Value(idx);
}

template <typename SrcType>
inline VarValue get_string_value(const Array& array, const int64_t idx) {
  using ArrayType = typename TypeTraits<SrcType>::ArrayType;
  return static_cast<const ArrayType&>(array).GetString(idx);
}

inline std::string error_context(const ColumnDescriptor* cd,
                                 Importer_NS::BadRowsTracker* const bad_rows_tracker) {
  return bad_rows_tracker ? "File " + bad_rows_tracker->file_name + ", row-group " +
                                std::to_string(bad_rows_tracker->row_group) +
                                ", column " + cd->columnName + ": "
                          : std::string();
}

#define NUMERIC_CASE(tid, src_type, var_type) \
  case Type::tid:                             \
    return get_numeric_value<src_type, var_type>;
#define STRING_CASE(tid, src_type) \
  case Type::tid:                  \
    return get_string_value<src_type>;

inline auto value_getter(const Array& array,
                         const ColumnDescriptor* cd,
                         Importer_NS::BadRowsTracker* const bad_rows_tracker) {
  switch (array.type_id()) {
    NUMERIC_CASE(BOOL, BooleanType, bool)
    NUMERIC_CASE(UINT8, UInt8Type, int64_t)
    NUMERIC_CASE(UINT16, UInt16Type, int64_t)
    NUMERIC_CASE(UINT32, UInt32Type, int64_t)
    NUMERIC_CASE(UINT64, Int64Type, int64_t)
    NUMERIC_CASE(INT8, Int8Type, int64_t)
    NUMERIC_CASE(INT16, Int16Type, int64_t)
    NUMERIC_CASE(INT32, Int32Type, int64_t)
    NUMERIC_CASE(INT64, Int64Type, int64_t)
    NUMERIC_CASE(FLOAT, FloatType, float)
    NUMERIC_CASE(DOUBLE, DoubleType, double)
    NUMERIC_CASE(DATE32, Date32Type, int64_t)
    NUMERIC_CASE(DATE64, Date64Type, int64_t)
    NUMERIC_CASE(TIME64, Time64Type, int64_t)
    NUMERIC_CASE(TIME32, Time32Type, int64_t)
    NUMERIC_CASE(TIMESTAMP, TimestampType, int64_t)
    STRING_CASE(STRING, StringType)
    STRING_CASE(BINARY, BinaryType)
    default:
      arrow_throw_if(true,
                     error_context(cd, bad_rows_tracker) + "Parquet type " +
                         array.type()->name() + " is not supported");
      throw;
  }
}

inline void type_conversion_error(const std::string pt,
                                  const ColumnDescriptor* cd,
                                  Importer_NS::BadRowsTracker* const bad_rows_tracker) {
  arrow_throw_if(true,
                 error_context(cd, bad_rows_tracker) +
                     "Invalid type conversion from parquet " + pt + " type to " +
                     cd->columnType.get_type_name());
}

template <typename DATA_TYPE, typename VALUE_TYPE>
inline void data_conversion_error(const VALUE_TYPE v,
                                  const ColumnDescriptor* cd,
                                  Importer_NS::BadRowsTracker* const bad_rows_tracker) {
  arrow_throw_if(true,
                 error_context(cd, bad_rows_tracker) +
                     "Invalid data conversion from parquet value " + std::to_string(v) +
                     " to " + std::to_string(DATA_TYPE(v)));
}

inline void data_conversion_error(const std::string& v,
                                  const ColumnDescriptor* cd,
                                  Importer_NS::BadRowsTracker* const bad_rows_tracker) {
  arrow_throw_if(true,
                 error_context(cd, bad_rows_tracker) +
                     "Invalid data conversion from parquet string '" + v + "' to " +
                     cd->columnType.get_type_name() + " column type");
}

// models the variant data buffers of TypedImportBuffer (LHS)
struct DataBufferBase {
  const ColumnDescriptor* cd;
  const Array& array;
  const int64_t resolution;
  Importer_NS::BadRowsTracker* const bad_rows_tracker;
  DataBufferBase(const ColumnDescriptor* cd,
                 const Array& array,
                 Importer_NS::BadRowsTracker* const bad_rows_tracker)
      : cd(cd)
      , array(array)
      , resolution(pow(10, cd->columnType.get_dimension()))
      , bad_rows_tracker(bad_rows_tracker) {}
};

template <typename DATA_TYPE>
struct DataBuffer : DataBufferBase {
  std::vector<DATA_TYPE>& buffer;
  DataBuffer(const ColumnDescriptor* cd,
             const Array& array,
             std::vector<DATA_TYPE>& buffer,
             Importer_NS::BadRowsTracker* const bad_rows_tracker)
      : DataBufferBase(cd, array, bad_rows_tracker), buffer(buffer) {}
};

constexpr int64_t kMillisecondsInSecond = 1000L;
constexpr int64_t kMicrosecondsInSecond = 1000L * 1000L;
constexpr int64_t kNanosecondsinSecond = 1000L * 1000L * 1000L;
constexpr int32_t kSecondsInDay = 86400;

// models the variant values of Arrow Array (RHS)
template <typename VALUE_TYPE>
struct ArrowValueBase {
  const DataBufferBase& data;
  const VALUE_TYPE v;
  const int64_t day_divider;
  ArrowValueBase(const DataBufferBase& data, const VALUE_TYPE& v)
      : data(data)
      , v(v)
      , day_divider(data.cd->columnType.get_compression() == kENCODING_DATE_IN_DAYS
                        ? kSecondsInDay
                        : 1) {}
  template <bool enabled = std::is_integral<VALUE_TYPE>::value>
  int64_t resolve_time(const VALUE_TYPE& v, std::enable_if_t<enabled>* = 0) const {
    const auto& type = *data.array.type();
    const auto& type_id = type.id();
    int64_t tv = v * data.resolution;
    if (type_id == Type::DATE32 || type_id == Type::DATE64) {
      auto& date_type = static_cast<const DateType&>(type);
      switch (date_type.unit()) {
        case DateUnit::DAY:
          return tv * kSecondsInDay / day_divider;
        case DateUnit::MILLI:
          return tv / kMillisecondsInSecond / day_divider;
      }
    } else if (type_id == Type::TIME32 || type_id == Type::TIME64 ||
               type_id == Type::TIMESTAMP) {
      auto& time_type = static_cast<const TimeType&>(type);
      switch (time_type.unit()) {
        case TimeUnit::SECOND:
          return tv / day_divider;
        case TimeUnit::MILLI:
          return tv / kMillisecondsInSecond / day_divider;
        case TimeUnit::MICRO:
          return tv / kMicrosecondsInSecond / day_divider;
        case TimeUnit::NANO:
          return tv / kNanosecondsinSecond / day_divider;
      }
    }
    UNREACHABLE() << type << " is not a valid Arrow time or date type";
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
  template <typename DATA_TYPE, typename = enable_if_integral<DATA_TYPE>>
  explicit operator const DATA_TYPE() const {
    return inline_fixed_encoding_null_val(data.cd->columnType);
  }
  template <typename DATA_TYPE, typename = enable_if_floating<DATA_TYPE>>
  explicit operator DATA_TYPE() const {
    return inline_fp_null_val(data.cd->columnType);
  }
  explicit operator const std::string() const { return std::string(); }
};

template <>
struct ArrowValue<bool> : ArrowValueBase<bool> {
  using VALUE_TYPE = bool;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}
  template <typename DATA_TYPE, typename = enable_if_integral<DATA_TYPE>>
  explicit operator const DATA_TYPE() const {
    if (!(data.cd->columnType.is_number() || data.cd->columnType.is_boolean())) {
      type_conversion_error("bool", data.cd, data.bad_rows_tracker);
    }
    return v;
  }
  template <typename DATA_TYPE, typename = enable_if_floating<DATA_TYPE>>
  explicit operator DATA_TYPE() const {
    return v ? 1 : 0;
  }
  explicit operator const std::string() const { return v ? "T" : "F"; }
};

template <>
struct ArrowValue<float> : ArrowValueBase<float> {
  using VALUE_TYPE = float;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}
  template <typename DATA_TYPE, typename = enable_if_integral<DATA_TYPE>>
  explicit operator const DATA_TYPE() const {
    const auto ti = data.cd->columnType;
    DATA_TYPE v = ti.is_decimal() ? this->v * pow(10, ti.get_scale()) : this->v;
    if (!(std::numeric_limits<DATA_TYPE>::min() < v &&
          v <= std::numeric_limits<DATA_TYPE>::max())) {
      data_conversion_error<DATA_TYPE>(v, data.cd, data.bad_rows_tracker);
    }
    return v;
  }
  template <typename DATA_TYPE, typename = enable_if_floating<DATA_TYPE>>
  explicit operator DATA_TYPE() const {
    return v;
  }
  explicit operator const std::string() const { return std::to_string(v); }
};

template <>
struct ArrowValue<double> : ArrowValueBase<double> {
  using VALUE_TYPE = double;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}
  template <typename DATA_TYPE, typename = enable_if_integral<DATA_TYPE>>
  explicit operator const DATA_TYPE() const {
    const auto ti = data.cd->columnType;
    DATA_TYPE v = ti.is_decimal() ? this->v * pow(10, ti.get_scale()) : this->v;
    if (!(std::numeric_limits<DATA_TYPE>::min() < v &&
          v <= std::numeric_limits<DATA_TYPE>::max())) {
      data_conversion_error<DATA_TYPE>(v, data.cd, data.bad_rows_tracker);
    }
    return v;
  }
  template <typename DATA_TYPE, typename = enable_if_floating<DATA_TYPE>>
  explicit operator DATA_TYPE() const {
    if (std::is_same<DATA_TYPE, float>::value) {
      if (!(std::numeric_limits<float>::min() < v &&
            v <= std::numeric_limits<float>::max())) {
        data_conversion_error<float>(v, data.cd, data.bad_rows_tracker);
      }
    }
    return v;
  }
  explicit operator const std::string() const { return std::to_string(v); }
};

template <>
struct ArrowValue<int64_t> : ArrowValueBase<int64_t> {
  using VALUE_TYPE = int64_t;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}
  template <typename DATA_TYPE, typename = enable_if_integral<DATA_TYPE>>
  explicit operator const DATA_TYPE() const {
    DATA_TYPE v = this->v;
    if (std::is_same<int64_t, DATA_TYPE>::value) {
    } else if (std::numeric_limits<DATA_TYPE>::min() < v &&
               v <= std::numeric_limits<DATA_TYPE>::max()) {
    } else {
      data_conversion_error<DATA_TYPE>(v, data.cd, data.bad_rows_tracker);
    }
    if (data.cd->columnType.is_time()) {
      v = this->resolve_time(v);
    }
    return v;
  }
  template <typename DATA_TYPE, typename = enable_if_floating<DATA_TYPE>>
  explicit operator DATA_TYPE() const {
    return v;
  }
  explicit operator const std::string() const { return std::to_string(v); }
};

template <>
struct ArrowValue<std::string> : ArrowValueBase<std::string> {
  using VALUE_TYPE = std::string;
  ArrowValue(const DataBufferBase& data, const VALUE_TYPE& v)
      : ArrowValueBase<VALUE_TYPE>(data, v) {}
  explicit operator const bool() const {
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
  }
  template <typename DATA_TYPE, typename = enable_if_integral_not_bool<DATA_TYPE>>
  explicit operator const DATA_TYPE() const {
    if (v.size() == 0) {
      return inline_fixed_encoding_null_val(data.cd->columnType);
    }
    try {
      auto ti = data.cd->columnType;
      auto datum = StringToDatum(v, ti);
      return data.cd->columnType.is_time() ? datum.bigintval * data.resolution
                                           : datum.bigintval;
    } catch (...) {
      data_conversion_error(v, data.cd, data.bad_rows_tracker);
      return 0;
    }
  }
  template <typename DATA_TYPE, typename = enable_if_floating<DATA_TYPE>>
  explicit operator DATA_TYPE() const {
    return atof(v.data());
  }
  explicit operator const std::string() const { return v; }
};

// appends a converted RHS value to LHS data block
template <typename DATA_TYPE>
inline void operator<<(DataBuffer<DATA_TYPE>& data, const VarValue& var) {
  boost::apply_visitor(
      [&data](const auto& v) {
        data.buffer.push_back(DATA_TYPE(ArrowValue<exprtype(v)>(data, v)));
      },
      var);
}

// appends (streams) a Arrow array of values (RHS) to TypedImportBuffer (LHS)
template <typename DATA_TYPE>
size_t convert_arrow_val_to_import_buffer(
    const ColumnDescriptor* cd,
    const Array& array,
    std::vector<DATA_TYPE>& buffer,
    Importer_NS::BadRowsTracker* const bad_rows_tracker) {
  auto data =
      std::make_unique<DataBuffer<DATA_TYPE>>(cd, array, buffer, bad_rows_tracker);
  auto f = value_getter(array, cd, bad_rows_tracker);
  buffer.reserve(array.length());
  for (auto row = 0; row < array.length(); ++row) {
    try {
      *data << (array.IsNull(row) ? nullptr : f(array, row));
    } catch (...) {
      // trace bad rows of each column; otherwise rethrow.
      if (bad_rows_tracker) {
        *data << nullptr;
        std::unique_lock<std::mutex> lck(bad_rows_tracker->mutex);
        bad_rows_tracker->rows.insert(row);
      } else {
        throw;
      }
    }
  }
  // when all columns finish remove bad rows while DATA_TYPE is known
  if (bad_rows_tracker) {
    bad_rows_tracker->running_thread_controller->notify_thread_is_completed();
    if (--bad_rows_tracker->ncol > 0) {
      std::unique_lock<std::mutex> lck(bad_rows_tracker->mutex);
      bad_rows_tracker->condv.wait(
          lck, [bad_rows_tracker] { return 0 == bad_rows_tracker->ncol; });
    }
    bad_rows_tracker->condv.notify_all();
    // erase backward to minimize memory movement overhead
    for (auto rit = bad_rows_tracker->rows.crbegin();
         rit != bad_rows_tracker->rows.crend();
         ++rit) {
      buffer.erase(buffer.begin() + *rit);
    }
  }
  return buffer.size();
}

}  // namespace
#endif  // ARROW_IMPORTER_H
