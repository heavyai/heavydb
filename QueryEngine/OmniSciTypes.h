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
  T* ptr;      // row data
  int64_t sz;  // row count

  DEVICE T& operator[](const unsigned int index) const { return ptr[index]; }
  DEVICE int64_t getSize() const { return sz; }
  DEVICE void setSize(int64_t size) { this->sz = size; }

  DEVICE bool isNull(int64_t index) const { return is_null(ptr[index]); }
  DEVICE void setNull(int64_t index) { set_null(ptr[index]); }

  DEVICE Column<T>& operator=(const Column<T>& other) {
    const auto& other_ptr = &other[0];
    if (other.getSize() == 0) {
      sz = 0;
    } else if (sz == other.getSize() && other_ptr != nullptr) {
#ifndef __CUDACC__
      memcpy(ptr, other_ptr, other.getSize() * sizeof(T));
#else
      assert(false);
#endif
    } else {
#ifndef __CUDACC__
      throw std::runtime_error("cannot assign columns: sizes mismatch or lack of data");
#else
      sz = -2;
#endif
    }
    return *this;
  }

#ifdef HAVE_TOSTRING

  std::string toString() const {
    return ::typeName(this) + "(ptr=" + ::toString(reinterpret_cast<void*>(ptr)) +
           ", sz=" + std::to_string(sz) + ")";
  }
#endif
};

/*
  ColumnList is an ordered list of Columns.
*/
template <typename T>
struct ColumnList {
  int8_t** ptrs;   // ptrs to columns data
  int64_t length;  // the length of columns list
  int64_t size;    // the size of columns

  DEVICE int64_t getCols() const { return length; }
  DEVICE int64_t getRows() const { return size; }

  DEVICE int64_t getLength() const { return length; }
  DEVICE int64_t getColumnSize() const { return size; }

  // Column view of a list item
  DEVICE Column<T> operator()(const int index) const {
    if (index >= 0 && index < length)
      return {reinterpret_cast<T*>(ptrs[index]), size};
    else
      return {nullptr, -1};
  }

#ifdef HAVE_TOSTRING

  std::string toString() const {
    std::string result = ::typeName(this) + "(ptrs=[";
    for (int64_t index = 0; index < length; index++) {
      result += ::toString(reinterpret_cast<void*>(ptrs[index])) +
                (index < length - 1 ? ", " : "");
    }
    result +=
        "], length=" + std::to_string(length) + ", size=" + std::to_string(size) + ")";
    return result;
  }

#endif
};
