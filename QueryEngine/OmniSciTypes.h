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
#include <type_traits>

/* `../` is required for UDFCompiler */
#include "../Shared/InlineNullValues.h"
#include "../Shared/funcannotations.h"

#define EXTENSION_INLINE extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE
#define EXTENSION_NOINLINE extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE

EXTENSION_NOINLINE int8_t* allocate_varlen_buffer(int64_t element_count,
                                                  int64_t element_size);

EXTENSION_NOINLINE void set_output_row_size(int64_t num_rows);

using TextEncodingDict = int32_t;

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

struct GeoLineString {
  int8_t* ptr;
  int64_t sz;
  int32_t compression;
  int32_t input_srid;
  int32_t output_srid;

  DEVICE int64_t getSize() const { return sz; }

  DEVICE int32_t getCompression() const { return compression; }

  DEVICE int32_t getInputSrid() const { return input_srid; }

  DEVICE int32_t getOutputSrid() const { return output_srid; }
};

struct GeoPoint {
  int8_t* ptr;
  int64_t sz;
  int32_t compression;
  int32_t input_srid;
  int32_t output_srid;

  DEVICE int64_t getSize() const { return sz; }

  DEVICE int32_t getCompression() const { return compression; }

  DEVICE int32_t getInputSrid() const { return input_srid; }

  DEVICE int32_t getOutputSrid() const { return output_srid; }
};

struct GeoPolygon {
  int8_t* ptr_coords;
  int64_t coords_size;
  int32_t* ring_sizes;
  int64_t num_rings;
  int32_t compression;
  int32_t input_srid;
  int32_t output_srid;

  DEVICE int32_t* getRingSizes() { return ring_sizes; }
  DEVICE int64_t getCoordsSize() const { return coords_size; }

  DEVICE int64_t getNumRings() const { return num_rings; }

  DEVICE int32_t getCompression() const { return compression; }

  DEVICE int32_t getInputSrid() const { return input_srid; }

  DEVICE int32_t getOutputSrid() const { return output_srid; }
};

struct GeoMultiPolygon {
  int8_t* ptr_coords;
  int64_t coords_size;
  int32_t* ring_sizes;
  int64_t num_rings;
  int32_t* poly_sizes;
  int64_t num_polys;
  int32_t compression;
  int32_t input_srid;
  int32_t output_srid;

  DEVICE int32_t* getRingSizes() { return ring_sizes; }
  DEVICE int64_t getCoordsSize() const { return coords_size; }

  DEVICE int64_t getNumRings() const { return num_rings; }

  DEVICE int32_t* getPolygonSizes() { return poly_sizes; }

  DEVICE int64_t getNumPolygons() const { return num_polys; }

  DEVICE int32_t getCompression() const { return compression; }

  DEVICE int32_t getInputSrid() const { return input_srid; }

  DEVICE int32_t getOutputSrid() const { return output_srid; }
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
      static T null_value;
      set_null(null_value);
      return null_value;
#endif
    }
    return ptr_[index];
  }
  DEVICE int64_t size() const { return size_; }

  DEVICE bool isNull(int64_t index) const { return is_null(ptr_[index]); }
  DEVICE void setNull(int64_t index) { set_null(ptr_[index]); }
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
