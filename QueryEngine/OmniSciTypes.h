/*
 * Copyright 2020 OmniSci, Inc.
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

#include <limits>
#include <stdexcept>
#include <type_traits>

/* `../` is required for UDFCompiler */
#include "../Shared/InlineNullValues.h"
#include "../Shared/funcannotations.h"

#define EXTENSION_INLINE extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE
#define EXTENSION_NOINLINE extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE
#define TEMPLATE_INLINE ALWAYS_INLINE DEVICE
#define TEMPLATE_NOINLINE NEVER_INLINE DEVICE

EXTENSION_NOINLINE int8_t* allocate_varlen_buffer(int64_t element_count,
                                                  int64_t element_size);

/*
  Table function management functions:
 */
EXTENSION_NOINLINE void set_output_row_size(int64_t num_rows);
EXTENSION_NOINLINE void TableFunctionManager_set_output_row_size(int8_t* mgr_ptr,
                                                                 int64_t num_rows);
EXTENSION_NOINLINE int8_t* TableFunctionManager_get_singleton();
EXTENSION_NOINLINE int32_t table_function_error(const char* message);
EXTENSION_NOINLINE int32_t TableFunctionManager_error_message(int8_t* mgr_ptr,
                                                              const char* message);

// https://www.fluentcpp.com/2018/04/06/strong-types-by-struct/
struct TextEncodingDict {
  int32_t value;
  operator int32_t() const { return value; }
  TextEncodingDict operator=(const int32_t other) {
    value = other;
    return *this;
  }
};

template <typename T>
struct Array {
  T* ptr;
  int64_t size;
  int8_t is_null;

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

  DEVICE int64_t getSize() const { return size; }

  DEVICE bool isNull() const { return is_null; }

  DEVICE constexpr inline T null_value() const {
    return std::is_signed<T>::value ? std::numeric_limits<T>::min()
                                    : std::numeric_limits<T>::max();
  }
};

struct TextEncodingNone {
  char* ptr_;
  int64_t size_;

#ifndef __CUDACC__
  std::string getString() const { return std::string(ptr_, size_); }
#endif

  DEVICE ALWAYS_INLINE char& operator[](const unsigned int index) {
    return index < size_ ? ptr_[index] : ptr_[size_ - 1];
  }
  DEVICE ALWAYS_INLINE operator char*() const { return ptr_; }
  DEVICE ALWAYS_INLINE int64_t size() const { return size_; }
  DEVICE ALWAYS_INLINE bool isNull() const { return size_ == 0; }
};

template <typename T>
struct Column {
  T* ptr_;        // row data
  int64_t size_;  // row count

  DEVICE T& operator[](const unsigned int index) const {
    if (index >= size_) {
#ifndef __CUDACC__
      throw std::runtime_error("column buffer index is out of range");
#else
      static DEVICE T null_value;
      if constexpr (std::is_same_v<T, TextEncodingDict>) {
        set_null(null_value.value);
      } else {
        set_null(null_value);
      }
      return null_value;
#endif
    }
    return ptr_[index];
  }
  DEVICE int64_t size() const { return size_; }

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
           ", size=" + std::to_string(size_) + ")";
  }
#endif
};

template <>
DEVICE inline bool Column<TextEncodingDict>::isNull(int64_t index) const {
  return is_null(ptr_[index].value);
}

template <>
DEVICE inline void Column<TextEncodingDict>::setNull(int64_t index) {
  set_null(ptr_[index].value);
}

/*
  ColumnList is an ordered list of Columns.
*/
template <typename T>
struct ColumnList {
  int8_t** ptrs_;     // ptrs to columns data
  int64_t num_cols_;  // the length of columns list
  int64_t size_;      // the size of columns

  DEVICE int64_t size() const { return size_; }
  DEVICE int64_t numCols() const { return num_cols_; }
  DEVICE Column<T> operator[](const int index) const {
    if (index >= 0 && index < num_cols_)
      return {reinterpret_cast<T*>(ptrs_[index]), size_};
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
              ", size=" + std::to_string(size_) + ")";
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
struct TableFunctionManager {
  static TableFunctionManager* get_singleton() {
    return reinterpret_cast<TableFunctionManager*>(TableFunctionManager_get_singleton());
  }

  void set_output_row_size(int64_t num_rows) {
    TableFunctionManager_set_output_row_size(reinterpret_cast<int8_t*>(this), num_rows);
  }

  int32_t error_message(const char* message) {
    return TableFunctionManager_error_message(reinterpret_cast<int8_t*>(this), message);
  }

#ifdef HAVE_TOSTRING

  std::string toString() const {
    std::string result = ::typeName(this) + "(";
    if (!this) {
      result += "UNINITIALIZED";
    }
    result += ")";
    return result;
  }

#endif
};
