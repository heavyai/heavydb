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

#pragma once

#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "DateTruncate.h"
#include "ExtractFromTime.h"
#include "TableFunctionMetadataType.h"
#include "Utils/FlatBuffer.h"

#if !(defined(__CUDACC__) || defined(NO_BOOST))
#include "../Shared/DateTimeParser.h"
#endif

/* `../` is required for UDFCompiler */
#include "../Shared/InlineNullValues.h"
#include "../Shared/funcannotations.h"

#ifndef __CUDACC__
#ifndef UDF_COMPILED
#include "../StringDictionary/StringDictionaryProxy.h"
#endif  // #ifndef UDF_COMPILED
#endif  // #ifndef __CUDACC__

// declaring CPU functions as __host__ can help catch erroneous compilation of
// these being done by the CUDA compiler at build time
#define EXTENSION_INLINE_HOST extern "C" RUNTIME_EXPORT ALWAYS_INLINE HOST
#define EXTENSION_NOINLINE_HOST extern "C" RUNTIME_EXPORT NEVER_INLINE HOST

#define EXTENSION_INLINE extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE
#define EXTENSION_NOINLINE extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE
#define TEMPLATE_INLINE ALWAYS_INLINE DEVICE
#define TEMPLATE_NOINLINE NEVER_INLINE DEVICE

EXTENSION_NOINLINE int8_t* allocate_varlen_buffer(int64_t element_count,
                                                  int64_t element_size);

/*
  Table function management functions and macros:
 */
#define FUNC_NAME (std::string(__func__).substr(0, std::string(__func__).find("__")))
// TODO: support windows path format
#define ERROR_STRING(MSG)                                                     \
  (std::string(__FILE__).substr(std::string(__FILE__).rfind("/") + 1) + ":" + \
   std::to_string(__LINE__) + " " + FUNC_NAME + ": " + MSG)                   \
      .c_str()
#define TABLE_FUNCTION_ERROR(MSG) table_function_error(ERROR_STRING(MSG))
#define ERROR_MESSAGE(MSG) error_message(ERROR_STRING(MSG))

EXTENSION_NOINLINE_HOST void set_output_array_values_total_number(
    int32_t index,
    int64_t output_array_values_total_number);
EXTENSION_NOINLINE_HOST void set_output_row_size(int64_t num_rows);
EXTENSION_NOINLINE_HOST void TableFunctionManager_set_output_array_values_total_number(
    int8_t* mgr_ptr,
    int32_t index,
    int64_t output_array_values_total_number);
EXTENSION_NOINLINE_HOST void TableFunctionManager_set_output_row_size(int8_t* mgr_ptr,
                                                                      int64_t num_rows);
EXTENSION_NOINLINE_HOST int8_t* TableFunctionManager_get_singleton();
EXTENSION_NOINLINE_HOST int32_t table_function_error(const char* message);
EXTENSION_NOINLINE_HOST int32_t TableFunctionManager_error_message(int8_t* mgr_ptr,
                                                                   const char* message);
EXTENSION_NOINLINE_HOST void TableFunctionManager_set_metadata(
    int8_t* mgr_ptr,
    const char* key,
    const uint8_t* raw_bytes,
    const size_t num_bytes,
    const TableFunctionMetadataType value_type);
EXTENSION_NOINLINE_HOST void TableFunctionManager_get_metadata(
    int8_t* mgr_ptr,
    const char* key,
    const uint8_t*& raw_bytes,
    size_t& num_bytes,
    TableFunctionMetadataType& value_type);

EXTENSION_NOINLINE_HOST int32_t TableFunctionManager_getNewDictId(int8_t* mgr_ptr);
std::string TableFunctionManager_getString(int8_t* mgr_ptr,
                                           int32_t dict_id,
                                           int32_t string_id);
EXTENSION_NOINLINE_HOST const char* TableFunctionManager_getCString(int8_t* mgr_ptr,
                                                                    int32_t dict_id,
                                                                    int32_t string_id);
EXTENSION_NOINLINE_HOST int32_t TableFunctionManager_getOrAddTransient(int8_t* mgr_ptr,
                                                                       int32_t dict_id,
                                                                       std::string str);

// https://www.fluentcpp.com/2018/04/06/strong-types-by-struct/
struct TextEncodingDict {
  int32_t value;

#ifndef __CUDACC__
  TextEncodingDict(const int32_t other) : value(other) {}
  TextEncodingDict() : value(0) {}
#endif

  operator int32_t() const { return value; }

  TextEncodingDict operator=(const int32_t other) {
    value = other;
    return *this;
  }

  DEVICE ALWAYS_INLINE bool operator==(const TextEncodingDict& other) const {
    return value == other.value;
  }

  DEVICE ALWAYS_INLINE bool operator==(const int32_t& other) const {
    return value == other;
  }

  DEVICE ALWAYS_INLINE bool operator==(const int64_t& other) const {
    return value == other;
  }

  DEVICE ALWAYS_INLINE bool operator!=(const TextEncodingDict& other) const {
    return !operator==(other);
  }
  DEVICE ALWAYS_INLINE bool operator!=(const int32_t& other) const {
    return !operator==(other);
  }

  DEVICE ALWAYS_INLINE bool operator!=(const int64_t& other) const {
    return !operator==(other);
  }

  DEVICE ALWAYS_INLINE bool operator<(const TextEncodingDict& other) const {
    return value < other.value;
  }

  DEVICE ALWAYS_INLINE bool operator<(const int32_t& other) const {
    return value < other;
  }

  DEVICE ALWAYS_INLINE bool operator<(const int64_t& other) const {
    return value < other;
  }

#ifdef HAVE_TOSTRING
  std::string toString() const {
    return ::typeName(this) + "(value=" + ::toString(value) + ")";
  }
#endif
};

template <>
DEVICE inline TextEncodingDict inline_null_value() {
#ifndef __CUDACC__
  return TextEncodingDict(inline_int_null_value<int32_t>());
#else
  TextEncodingDict null_val;
  null_val.value = inline_int_null_value<int32_t>();
  return null_val;
#endif
}

template <typename T>
struct Array {
  T* ptr;
  int64_t size;
  int8_t is_null;

  DEVICE Array(T* ptr, const int64_t size, const bool is_null = false)
      : ptr(is_null ? nullptr : ptr), size(size), is_null(is_null) {}
  DEVICE Array() : ptr(nullptr), size(0), is_null(true) {}

  DEVICE Array(const int64_t size, const bool is_null = false)
      : size(size), is_null(is_null) {
    if (!is_null) {
      ptr = reinterpret_cast<T*>(
          allocate_varlen_buffer(size, static_cast<int64_t>(sizeof(T))));
    } else {
      ptr = nullptr;
    }
  }

  DEVICE T operator()(const unsigned int index) const {
    if (index < static_cast<unsigned int>(size)) {
      return ptr[index];
    } else {
      return 0;  // see array_at
    }
  }

  DEVICE T& operator[](const unsigned int index) { return ptr[index]; }
  DEVICE const T& operator[](const unsigned int index) const { return ptr[index]; }

  DEVICE int64_t getSize() const { return size; }

  DEVICE bool isNull() const { return is_null; }

  DEVICE constexpr inline T null_value() const { return inline_null_value<T>(); }

  DEVICE bool isNull(const unsigned int index) const {
    return (is_null ? false : ptr[index] == null_value());
  }

#ifdef HAVE_TOSTRING
  std::string toString() const {
    std::string result =
        ::typeName(this) + "(ptr=" + ::toString(reinterpret_cast<void*>(ptr)) +
        ", size=" + std::to_string(size) + ", is_null=" + std::to_string(is_null) + ")[";
    for (int64_t i = 0; i < size; i++) {
      if (size > 10) {
        // show the first 8 and the last 2 values in the array:
        if (i == 8) {
          result += "..., ";
        } else if (i > 8 && i < size - 2) {
          continue;
        }
      }
      result += ::toString((*this)[i]) + ", ";
    }
    result += "]";
    return result;
  }
#endif
};

struct TextEncodingNone {
  char* ptr_;
  int64_t size_;

#ifndef __CUDACC__
  TextEncodingNone() = default;
  TextEncodingNone(const std::string& str) {
    // Note this will only be valid for the
    // lifetime of the string
    ptr_ = const_cast<char*>(str.data());
    size_ = str.length();
  }
  operator std::string() const { return std::string(ptr_, size_); }
  std::string getString() const { return std::string(ptr_, size_); }
  const char* getCString() const { return ptr_; }
#endif

  DEVICE ALWAYS_INLINE char& operator[](const unsigned int index) {
    return index < size_ ? ptr_[index] : ptr_[size_ - 1];
  }
  DEVICE ALWAYS_INLINE bool operator==(const char* rhs) const {
#ifdef __CUDACC__
    for (int i = 0; i < size_; i++) {
      if (rhs[i] == '\0' || ptr_[i] != rhs[i]) {
        return false;
      }
    }
    return rhs[size_] == '\0';
#else
    return strcmp(ptr_, rhs) == 0;
#endif
  }
  DEVICE ALWAYS_INLINE bool operator!=(const char* rhs) const {
    return !(this->operator==(rhs));
  }
  DEVICE ALWAYS_INLINE operator char*() const { return ptr_; }
  DEVICE ALWAYS_INLINE int64_t size() const { return size_; }
  DEVICE ALWAYS_INLINE bool isNull() const { return size_ == 0; }
};

struct Timestamp {
  int64_t time;

  Timestamp() = default;

  DEVICE Timestamp(int64_t timeval) : time(timeval) {}

#if !(defined(__CUDACC__) || defined(NO_BOOST))
  DEVICE Timestamp(std::string_view const str) {
    time = dateTimeParse<kTIMESTAMP>(str, 9);
  }
#endif

  DEVICE ALWAYS_INLINE const Timestamp operator+(const Timestamp& other) const {
#ifndef __CUDACC__
    if (other.time > 0) {
      if (time > (std::numeric_limits<int64_t>::max() - other.time)) {
        throw std::underflow_error("Underflow in Timestamp addition!");
      }
    } else {
      if (time < (std::numeric_limits<int64_t>::min() - other.time)) {
        throw std::overflow_error("Overflow in Timestamp addition!");
      }
    }
#endif
    return Timestamp(time + other.time);
  }

  DEVICE ALWAYS_INLINE const Timestamp operator-(const Timestamp& other) const {
#ifndef __CUDACC__
    if (other.time > 0) {
      if (time < (std::numeric_limits<int64_t>::min() + other.time)) {
        throw std::underflow_error("Underflow in Timestamp substraction!");
      }
    } else {
      if (time > (std::numeric_limits<int64_t>::max() + other.time)) {
        throw std::overflow_error("Overflow in Timestamp substraction!");
      }
    }
#endif
    return Timestamp(time - other.time);
  }

  DEVICE ALWAYS_INLINE int64_t operator/(const Timestamp& other) const {
#ifndef __CUDACC__
    if (other.time == 0) {
      throw std::runtime_error("Timestamp division by zero!");
    }
#endif
    return time / other.time;
  }

  DEVICE ALWAYS_INLINE const Timestamp operator*(const int64_t multiplier) const {
#ifndef __CUDACC__
    uint64_t overflow_test =
        static_cast<uint64_t>(time) * static_cast<uint64_t>(multiplier);
    if (time != 0 && overflow_test / time != static_cast<uint64_t>(multiplier)) {
      throw std::runtime_error("Overflow in Timestamp multiplication!");
    }
#endif
    return Timestamp(time * multiplier);
  }

  DEVICE ALWAYS_INLINE bool operator==(const Timestamp& other) const {
    return time == other.time;
  }

  DEVICE ALWAYS_INLINE bool operator!=(const Timestamp& other) const {
    return !operator==(other);
  }

  DEVICE ALWAYS_INLINE bool operator<(const Timestamp& other) const {
    return time < other.time;
  }

  DEVICE ALWAYS_INLINE Timestamp truncateToMicroseconds() const {
    return Timestamp((time / kMilliSecsPerSec) * kMilliSecsPerSec);
  }

  DEVICE ALWAYS_INLINE Timestamp truncateToMilliseconds() const {
    return Timestamp((time / kMicroSecsPerSec) * kMicroSecsPerSec);
  }

  DEVICE ALWAYS_INLINE Timestamp truncateToSeconds() const {
    return Timestamp((time / kNanoSecsPerSec) * kNanoSecsPerSec);
  }

#ifndef __CUDACC__
  DEVICE ALWAYS_INLINE Timestamp truncateToMinutes() const {
    return Timestamp(DateTruncate(dtMINUTE, (time / kNanoSecsPerSec)) * kNanoSecsPerSec);
  }

  DEVICE ALWAYS_INLINE Timestamp truncateToHours() const {
    return Timestamp(DateTruncate(dtHOUR, (time / kNanoSecsPerSec)) * kNanoSecsPerSec);
  }

  DEVICE ALWAYS_INLINE Timestamp truncateToDay() const {
    return Timestamp(DateTruncate(dtDAY, (time / kNanoSecsPerSec)) * kNanoSecsPerSec);
  }

  DEVICE ALWAYS_INLINE Timestamp truncateToMonth() const {
    return Timestamp(DateTruncate(dtMONTH, (time / kNanoSecsPerSec)) * kNanoSecsPerSec);
  }

  DEVICE ALWAYS_INLINE Timestamp truncateToYear() const {
    return Timestamp(DateTruncate(dtYEAR, (time / kNanoSecsPerSec)) * kNanoSecsPerSec);
  }

  DEVICE ALWAYS_INLINE int64_t getNanoseconds() const {
    return ExtractFromTime(kNANOSECOND, time);
  }
  // Should always be safe as we're downcasting to lower precisions
  DEVICE ALWAYS_INLINE int64_t getMicroseconds() const {
    return ExtractFromTime(kMICROSECOND, time / (kNanoSecsPerSec / kMicroSecsPerSec));
  }
  // Should always be safe as we're downcasting to lower precisions
  DEVICE ALWAYS_INLINE int64_t getMilliseconds() const {
    return ExtractFromTime(kMILLISECOND, time / (kNanoSecsPerSec / kMilliSecsPerSec));
  }
  // Should always be safe as we're downcasting to lower precisions
  DEVICE ALWAYS_INLINE int64_t getSeconds() const {
    return ExtractFromTime(kSECOND, time / kNanoSecsPerSec);
  }
  DEVICE ALWAYS_INLINE int64_t getMinutes() const {
    return ExtractFromTime(kMINUTE, time / kNanoSecsPerSec);
  }
  DEVICE ALWAYS_INLINE int64_t getHours() const {
    return ExtractFromTime(kHOUR, time / kNanoSecsPerSec);
  }
  DEVICE ALWAYS_INLINE int64_t getDay() const {
    return ExtractFromTime(kDAY, time / kNanoSecsPerSec);
  }
  DEVICE ALWAYS_INLINE int64_t getMonth() const {
    return ExtractFromTime(kMONTH, time / kNanoSecsPerSec);
  }
  DEVICE ALWAYS_INLINE int64_t getYear() const {
    return ExtractFromTime(kYEAR, time / kNanoSecsPerSec);
  }
#ifdef HAVE_TOSTRING
  std::string toString() const {
    return ::typeName(this) + "(time=" + ::toString(time) + ")";
  }
#endif
#endif
};

template <>
DEVICE inline Timestamp inline_null_value() {
  return Timestamp(inline_int_null_value<int64_t>());
}

struct GeoPoint {
  int8_t* ptr;
  int32_t sz;
  int32_t compression;
  int32_t input_srid;
  int32_t output_srid;

  DEVICE int64_t getSize() const { return sz; }

  DEVICE int32_t getCompression() const { return compression; }

  DEVICE int32_t getInputSrid() const { return input_srid; }

  DEVICE int32_t getOutputSrid() const { return output_srid; }
};

struct GeoLineString {
  int8_t* ptr;
  int32_t sz;
  int32_t compression;
  int32_t input_srid;
  int32_t output_srid;

  DEVICE int32_t getSize() const { return sz; }

  DEVICE int32_t getCompression() const { return compression; }

  DEVICE int32_t getInputSrid() const { return input_srid; }

  DEVICE int32_t getOutputSrid() const { return output_srid; }
};

struct GeoMultiLineString {
  int8_t* ptr;
  int32_t sz;
  int8_t* linestring_sizes;
  int32_t num_linestrings;
  int32_t compression;
  int32_t input_srid;
  int32_t output_srid;

  DEVICE int8_t* getCoords() const { return ptr; }

  DEVICE int32_t getCoordsSize() const { return sz; }

  DEVICE int8_t* getLineStringSizes() { return linestring_sizes; }

  DEVICE int32_t getNumLineStrings() const { return num_linestrings; }

  DEVICE int32_t getCompression() const { return compression; }

  DEVICE int32_t getInputSrid() const { return input_srid; }

  DEVICE int32_t getOutputSrid() const { return output_srid; }
};

struct GeoPolygon {
  int8_t* ptr_coords;
  int32_t coords_size;
  int8_t* ring_sizes;
  int32_t num_rings;
  int32_t compression;
  int32_t input_srid;
  int32_t output_srid;

  DEVICE int8_t* getRingSizes() { return ring_sizes; }
  DEVICE int32_t getCoordsSize() const { return coords_size; }

  DEVICE int32_t getNumRings() const { return num_rings; }

  DEVICE int32_t getCompression() const { return compression; }

  DEVICE int32_t getInputSrid() const { return input_srid; }

  DEVICE int32_t getOutputSrid() const { return output_srid; }
};

struct GeoMultiPolygon {
  int8_t* ptr_coords;
  int32_t coords_size;
  int8_t* ring_sizes;
  int32_t num_rings;
  int8_t* poly_sizes;
  int32_t num_polys;
  int32_t compression;
  int32_t input_srid;
  int32_t output_srid;

  DEVICE int8_t* getRingSizes() { return ring_sizes; }
  DEVICE int32_t getCoordsSize() const { return coords_size; }

  DEVICE int32_t getNumRings() const { return num_rings; }

  DEVICE int8_t* getPolygonSizes() { return poly_sizes; }

  DEVICE int32_t getNumPolygons() const { return num_polys; }

  DEVICE int32_t getCompression() const { return compression; }

  DEVICE int32_t getInputSrid() const { return input_srid; }

  DEVICE int32_t getOutputSrid() const { return output_srid; }
};

// There are redundant #ifndef UDF_COMPILED inside
// ifguard for StringDictionaryProxy to flag that
// if we decide to adapt C++ UDF Compiler for table
// functions, the linking issue we encountered with
// the shared_mutex include in StringDicitonaryProxy
// will need to be resolved separately.

#ifndef UDF_COMPILED

#ifdef __CUDACC__
template <typename T>
static DEVICE __constant__ T Column_null_value;
#endif

template <typename T>
struct Column {
  T* ptr_;            // row data
  int64_t num_rows_;  // row count

  DEVICE Column(T* ptr, const int64_t num_rows) : ptr_(ptr), num_rows_(num_rows) {}

#ifndef __CUDACC__
#ifndef UDF_COMPILED
  DEVICE Column(const Column& other) : ptr_(other.ptr_), num_rows_(other.num_rows_) {}
  DEVICE Column(std::vector<T>& input_vec)
      : ptr_(input_vec.data()), num_rows_(static_cast<int64_t>(input_vec.size())) {}
#endif
#endif

  DEVICE T& operator[](const unsigned int index) const {
    if (index >= num_rows_) {
#ifndef __CUDACC__
      throw std::runtime_error("column buffer index is out of range");
#else
      auto& null_value = Column_null_value<T>;
      set_null(null_value);
      return null_value;
#endif
    }
    return ptr_[index];
  }
  DEVICE inline T* getPtr() const { return ptr_; }
  DEVICE inline int64_t size() const { return num_rows_; }
  DEVICE inline void setSize(int64_t num_rows) { num_rows_ = num_rows; }

  DEVICE inline bool isNull(int64_t index) const { return is_null(ptr_[index]); }
  DEVICE inline void setNull(int64_t index) { set_null(ptr_[index]); }
  DEVICE Column<T>& operator=(const Column<T>& other) {
#ifndef __CUDACC__
    if (size() == other.size()) {
      memcpy(ptr_, &other[0], other.size() * sizeof(T));
    } else {
      throw std::runtime_error("cannot copy assign columns with different sizes");
    }
#else
    if (size() == other.size()) {
      for (unsigned int i = 0; i < size(); i++) {
        ptr_[i] = other[i];
      }
    } else {
      // TODO: set error
    }
#endif
    return *this;
  }

#ifdef HAVE_TOSTRING
  std::string toString() const {
    return ::typeName(this) + "(ptr=" + ::toString(reinterpret_cast<void*>(ptr_)) +
           ", num_rows=" + std::to_string(num_rows_) + ")";
  }
#endif
};

template <typename T>
struct Column<Array<T>> {
  // A type for a column of variable length arrays
  //
  // Internally, the varlen arrays are stored using compressed storage
  // format (see https://pearu.github.io/variable_length_arrays.html
  // for the description) using FlatBuffer tool.
  int8_t* flatbuffer_;  // contains a flat buffer of the storage, use FlatBufferManager
                        // to access it.
  int64_t num_rows_;    // total row count, the number of all varlen arrays

  Column<Array<T>>(int8_t* flatbuffer, const int64_t num_rows)
      : flatbuffer_(flatbuffer), num_rows_(num_rows) {}

  DEVICE Array<T> getItem(const int64_t index, const int64_t expected_numel = -1) const {
    FlatBufferManager m{flatbuffer_};
    int8_t* ptr;
    int64_t size;
    bool is_null;
    auto status = m.getItem(index, size, ptr, is_null);
    if (status == FlatBufferManager::Status::ItemUnspecifiedError) {
      if (expected_numel < 0) {
#ifndef __CUDACC__
        throw std::runtime_error("getItem failed: " + ::toString(status));
#endif
      }
      status = m.setItem(index,
                         nullptr,
                         expected_numel * sizeof(T),
                         nullptr);  // reserves a junk in array buffer
      if (status != FlatBufferManager::Status::Success) {
#ifndef __CUDACC__
        throw std::runtime_error("getItem failed[setItem]: " + ::toString(status));
#endif
      }
      status = m.getItem(index, size, ptr, is_null);
    }
    if (status == FlatBufferManager::Status::Success) {
      if (expected_numel >= 0 &&
          expected_numel * static_cast<int64_t>(sizeof(T)) != size) {
#ifndef __CUDACC__
        throw std::runtime_error("getItem failed: unexpected size");
#endif
      }
    } else {
#ifndef __CUDACC__
      throw std::runtime_error("getItem failed: " + ::toString(status));
#endif
    }
    Array<T> result(reinterpret_cast<T*>(ptr), size / sizeof(T), is_null);
    return result;
  }

  DEVICE inline Array<T> operator[](const unsigned int index) const {
    return getItem(static_cast<int64_t>(index));
  }

  DEVICE int64_t size() const { return num_rows_; }

  DEVICE inline bool isNull(int64_t index) const {
    FlatBufferManager m{flatbuffer_};
    bool is_null = false;
    auto status = m.isNull(index, is_null);
#ifndef __CUDACC__
    if (status != FlatBufferManager::Status::Success) {
      throw std::runtime_error("isNull failed: " + ::toString(status));
    }
#endif
    return is_null;
  }

  DEVICE inline void setNull(int64_t index) {
    FlatBufferManager m{flatbuffer_};
    auto status = m.setNull(index);
#ifndef __CUDACC__
    if (status != FlatBufferManager::Status::Success) {
      throw std::runtime_error("setNull failed: " + ::toString(status));
    }
#endif
  }

  DEVICE inline void setItem(int64_t index, const Array<T>& other) {
    FlatBufferManager m{flatbuffer_};
    FlatBufferManager::Status status;
    if (other.isNull()) {
      status = m.setNull(index);
    } else {
      status = m.setItem(index,
                         reinterpret_cast<const int8_t*>(&(other[0])),
                         other.getSize() * sizeof(T));
    }
#ifndef __CUDACC__
    if (status != FlatBufferManager::Status::Success) {
      throw std::runtime_error("setItem failed: " + ::toString(status));
    }
#endif
  }

  DEVICE inline void concatItem(int64_t index, const Array<T>& other) {
    FlatBufferManager m{flatbuffer_};
    FlatBufferManager::Status status;
    if (other.isNull()) {
      status = m.setNull(index);
    } else {
      status = m.concatItem(index,
                            reinterpret_cast<const int8_t*>(&(other[0])),
                            other.getSize() * sizeof(T));
#ifndef __CUDACC__
      if (status != FlatBufferManager::Status::Success) {
        throw std::runtime_error("concatItem failed: " + ::toString(status));
      }
#endif
    }
  }

  DEVICE inline int32_t getDictId() const {
    FlatBufferManager m{flatbuffer_};
    return m.getDTypeMetadataDictId();
  }
};

template <>
struct Column<TextEncodingDict> {
  TextEncodingDict* ptr_;  // row data
  int64_t num_rows_;       // row count
#ifndef __CUDACC__
#ifndef UDF_COMPILED
  StringDictionaryProxy* string_dict_proxy_;
  DEVICE Column(TextEncodingDict* ptr,
                const int64_t num_rows,
                StringDictionaryProxy* string_dict_proxy)
      : ptr_(ptr), num_rows_(num_rows), string_dict_proxy_(string_dict_proxy) {}
#else
  DEVICE Column(TextEncodingDict* ptr, const int64_t num_rows)
      : ptr_(ptr), num_rows_(num_rows) {}
#endif  // #ifndef UDF_COMPILED
#else
  DEVICE Column(TextEncodingDict* ptr, const int64_t num_rows)
      : ptr_(ptr), num_rows_(num_rows) {}
#endif  // #ifndef __CUDACC__

  DEVICE TextEncodingDict& operator[](const unsigned int index) const {
    if (index >= num_rows_) {
#ifndef __CUDACC__
      throw std::runtime_error("column buffer index is out of range");
#else
      static DEVICE TextEncodingDict null_value;
      set_null(null_value.value);
      return null_value;
#endif
    }
    return ptr_[index];
  }
  DEVICE inline TextEncodingDict* getPtr() const { return ptr_; }
  DEVICE inline int64_t size() const { return num_rows_; }
  DEVICE inline void setSize(int64_t num_rows) { num_rows_ = num_rows; }

  DEVICE inline bool isNull(int64_t index) const { return is_null(ptr_[index].value); }

  DEVICE inline void setNull(int64_t index) { set_null(ptr_[index].value); }

#ifndef __CUDACC__
#ifndef UDF_COMPILED
  DEVICE inline int32_t getDictId() const { return string_dict_proxy_->getDictId(); }
  DEVICE inline const std::string getString(int64_t index) const {
    return isNull(index) ? "" : string_dict_proxy_->getString(ptr_[index].value);
  }
  DEVICE inline const char* getCString(int64_t index) const {
    if (isNull(index)) {
      return nullptr;
    }
    auto [c_str, len] = string_dict_proxy_->getStringBytes(ptr_[index].value);
    return c_str;
  }
  DEVICE inline const TextEncodingDict getOrAddTransient(const std::string& str) {
    return string_dict_proxy_->getOrAddTransient(str);
  }
#endif  // #ifndef UDF_COMPILED
#endif  // #ifndef __CUDACC__

  DEVICE Column<TextEncodingDict>& operator=(const Column<TextEncodingDict>& other) {
#ifndef __CUDACC__
    if (size() == other.size()) {
      memcpy(ptr_, other.ptr_, other.size() * sizeof(TextEncodingDict));
    } else {
      throw std::runtime_error("cannot copy assign columns with different sizes");
    }
#else
    if (size() == other.size()) {
      for (unsigned int i = 0; i < size(); i++) {
        ptr_[i] = other[i];
      }
    } else {
      // TODO: set error
    }
#endif
    return *this;
  }

#ifdef HAVE_TOSTRING
  std::string toString() const {
    return ::typeName(this) + "(ptr=" + ::toString(reinterpret_cast<void*>(ptr_)) +
           ", num_rows=" + std::to_string(num_rows_) + ")";
  }
#endif
};

template <>
DEVICE inline bool Column<Timestamp>::isNull(int64_t index) const {
  return is_null(ptr_[index].time);
}

template <>
DEVICE inline void Column<Timestamp>::setNull(int64_t index) {
  set_null(ptr_[index].time);
}

template <>
CONSTEXPR DEVICE inline void set_null<Timestamp>(Timestamp& t) {
  set_null(t.time);
}

/*
  ColumnList is an ordered list of Columns.
*/
template <typename T>
struct ColumnList {
  int8_t** ptrs_;     // ptrs to columns data
  int64_t num_cols_;  // the length of columns list
  int64_t num_rows_;  // the number of rows of columns

  DEVICE ColumnList(int8_t** ptrs, const int64_t num_cols, const int64_t num_rows)
      : ptrs_(ptrs), num_cols_(num_cols), num_rows_(num_rows) {}

  DEVICE int64_t size() const { return num_rows_; }
  DEVICE int64_t numCols() const { return num_cols_; }
  DEVICE Column<T> operator[](const int index) const {
    if (index >= 0 && index < num_cols_)
      return {reinterpret_cast<T*>(ptrs_[index]), num_rows_};
    else
      return {nullptr, -1};
  }

#ifdef HAVE_TOSTRING

  std::string toString() const {
    std::string result = ::typeName(this) + "(ptrs=[";
    for (int64_t index = 0; index < num_cols_; index++) {
      result += ::toString(reinterpret_cast<void*>(ptrs_[index])) +
                (index < num_cols_ - 1 ? ", " : "");
    }
    result += "], num_cols=" + std::to_string(num_cols_) +
              ", num_rows=" + std::to_string(num_rows_) + ")";
    return result;
  }

#endif
};

template <typename T>
struct ColumnList<Array<T>> {
  int8_t** ptrs_;     // ptrs to columns data in FlatBuffer format
  int64_t num_cols_;  // the length of columns list
  int64_t num_rows_;  // the size of columns

  DEVICE int64_t size() const { return num_rows_; }
  DEVICE int64_t numCols() const { return num_cols_; }
  DEVICE Column<Array<T>> operator[](const int index) const {
    int8_t* ptr = ((index >= 0 && index < num_cols_) ? ptrs_[index] : nullptr);
    int64_t num_rows = ((index >= 0 && index < num_cols_) ? num_rows_ : -1);
    Column<Array<T>> result(ptr, num_rows);
    return result;
  }

#ifdef HAVE_TOSTRING

  std::string toString() const {
    std::string result = ::typeName(this) + "(ptrs=[";
    for (int64_t index = 0; index < num_cols_; index++) {
      result += ::toString(reinterpret_cast<void*>(ptrs_[index])) +
                (index < num_cols_ - 1 ? ", " : "");
    }
    result += "], num_cols=" + std::to_string(num_cols_) +
              ", num_rows=" + std::to_string(num_rows_) + ")";
    return result;
  }

#endif
};

template <>
struct ColumnList<TextEncodingDict> {
  int8_t** ptrs_;     // ptrs to columns data
  int64_t num_cols_;  // the length of columns list
  int64_t num_rows_;  // the size of columns
#ifndef __CUDACC__
#ifndef UDF_COMPILED
  StringDictionaryProxy** string_dict_proxies_;  // the size of columns
  DEVICE ColumnList(int8_t** ptrs,
                    const int64_t num_cols,
                    const int64_t num_rows,
                    StringDictionaryProxy** string_dict_proxies)
      : ptrs_(ptrs)
      , num_cols_(num_cols)
      , num_rows_(num_rows)
      , string_dict_proxies_(string_dict_proxies) {}
#else
  DEVICE ColumnList(int8_t** ptrs, const int64_t num_cols, const int64_t num_rows)
      : ptrs_(ptrs), num_cols_(num_cols), num_rows_(num_rows) {}
#endif  // #ifndef UDF_COMPILED
#else
  DEVICE ColumnList(int8_t** ptrs, const int64_t num_cols, const int64_t num_rows)
      : ptrs_(ptrs), num_cols_(num_cols), num_rows_(num_rows) {}
#endif  // #ifndef __CUDACC__

  DEVICE int64_t size() const { return num_rows_; }
  DEVICE int64_t numCols() const { return num_cols_; }
  DEVICE Column<TextEncodingDict> operator[](const int index) const {
    if (index >= 0 && index < num_cols_) {
      Column<TextEncodingDict> result(reinterpret_cast<TextEncodingDict*>(ptrs_[index]),
                                      num_rows_
#ifndef __CUDACC__
#ifndef UDF_COMPILED
                                      ,
                                      string_dict_proxies_[index]
#endif  // #ifndef UDF_COMPILED
#endif  // #ifndef __CUDACC__
      );
      return result;
    } else {
      Column<TextEncodingDict> result(nullptr,
                                      -1
#ifndef __CUDACC__
#ifndef UDF_COMPILED
                                      ,
                                      nullptr
#endif  // #ifndef UDF_COMPILED
#endif  // #ifndef__CUDACC__
      );
      return result;
    }
  }

#ifdef HAVE_TOSTRING

  std::string toString() const {
    std::string result = ::typeName(this) + "(ptrs=[";
    for (int64_t index = 0; index < num_cols_; index++) {
      result += ::toString(reinterpret_cast<void*>(ptrs_[index])) +
                (index < num_cols_ - 1 ? ", " : "");
    }
    result += "], num_cols=" + std::to_string(num_cols_) +
              ", num_rows=" + std::to_string(num_rows_) + ")";
    return result;
  }

#endif
};

/*
  This TableFunctionManager struct is a minimal proxy to the
  TableFunctionManager defined in TableFunctionManager.h. The
  corresponding instances share `this` but have different virtual
  tables for methods.
*/
#ifndef __CUDACC__

namespace {
template <typename T>
TableFunctionMetadataType get_metadata_type() {
  if constexpr (std::is_same<T, int8_t>::value) {
    return TableFunctionMetadataType::kInt8;
  } else if constexpr (std::is_same<T, int16_t>::value) {
    return TableFunctionMetadataType::kInt16;
  } else if constexpr (std::is_same<T, int32_t>::value) {
    return TableFunctionMetadataType::kInt32;
  } else if constexpr (std::is_same<T, int64_t>::value) {
    return TableFunctionMetadataType::kInt64;
  } else if constexpr (std::is_same<T, float>::value) {
    return TableFunctionMetadataType::kFloat;
  } else if constexpr (std::is_same<T, double>::value) {
    return TableFunctionMetadataType::kDouble;
  } else if constexpr (std::is_same<T, bool>::value) {
    return TableFunctionMetadataType::kBool;
  }
  throw std::runtime_error("Unsupported TableFunctionMetadataType");
}
}  // namespace

struct TableFunctionManager {
  static TableFunctionManager* get_singleton() {
    return reinterpret_cast<TableFunctionManager*>(TableFunctionManager_get_singleton());
  }

  void set_output_array_values_total_number(int32_t index,
                                            int64_t output_array_values_total_number) {
    TableFunctionManager_set_output_array_values_total_number(
        reinterpret_cast<int8_t*>(this), index, output_array_values_total_number);
  }

  void set_output_row_size(int64_t num_rows) {
    if (!output_allocations_disabled) {
      TableFunctionManager_set_output_row_size(reinterpret_cast<int8_t*>(this), num_rows);
    }
  }

  void disable_output_allocations() { output_allocations_disabled = true; }

  void enable_output_allocations() { output_allocations_disabled = false; }

  int32_t error_message(const char* message) {
    return TableFunctionManager_error_message(reinterpret_cast<int8_t*>(this), message);
  }

  template <typename T>
  void set_metadata(const std::string& key, const T& value) {
    TableFunctionManager_set_metadata(reinterpret_cast<int8_t*>(this),
                                      key.c_str(),
                                      reinterpret_cast<const uint8_t*>(&value),
                                      sizeof(value),
                                      get_metadata_type<T>());
  }

  template <typename T>
  void get_metadata(const std::string& key, T& value) {
    const uint8_t* raw_data{};
    size_t num_bytes{};
    TableFunctionMetadataType value_type;
    TableFunctionManager_get_metadata(
        reinterpret_cast<int8_t*>(this), key.c_str(), raw_data, num_bytes, value_type);
    if (sizeof(T) != num_bytes) {
      throw std::runtime_error("Size mismatch for Table Function Metadata '" + key + "'");
    }
    if (get_metadata_type<T>() != value_type) {
      throw std::runtime_error("Type mismatch for Table Function Metadata '" + key + "'");
    }
    std::memcpy(&value, raw_data, num_bytes);
  }
  int32_t getNewDictId() {
    return TableFunctionManager_getNewDictId(reinterpret_cast<int8_t*>(this));
  }
  std::string getString(int32_t dict_id, int32_t string_id) {
    return TableFunctionManager_getString(
        reinterpret_cast<int8_t*>(this), dict_id, string_id);
  }
  const char* getCString(int32_t dict_id, int32_t string_id) {
    return TableFunctionManager_getCString(
        reinterpret_cast<int8_t*>(this), dict_id, string_id);
  }
  int32_t getOrAddTransient(int32_t dict_id, std::string str) {
    return TableFunctionManager_getOrAddTransient(
        reinterpret_cast<int8_t*>(this), dict_id, str);
  }

#ifdef HAVE_TOSTRING
  std::string toString() const {
    std::string result = ::typeName(this) + "(";
    if (!(void*)this) {  // cast to void* to avoid warnings
      result += "UNINITIALIZED";
    }
    result += ")";
    return result;
  }
#endif  // HAVE_TOSTRING
  bool output_allocations_disabled{false};
};
#endif  // #ifndef __CUDACC__

#endif  // #ifndef UDF_COMPILED
