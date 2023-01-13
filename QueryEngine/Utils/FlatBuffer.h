#pragma once

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

// clang-format off
/*
  FlatBufferManager provides a storage for a collection of buffers
  (columns of arrays, columns of strings, etc) that are collected into
  a single "flatbuffer" so that copying FlatBuffer instances becomes a
  single buffer copy operation. Flat buffers that store no pointer
  values can be straightforwardly copied in between different devices.

  FlatBuffer memory layout specification
  --------------------------------------

  The first and the last 8 bytes of the buffer contains a FlatBuffer
  storage format id (see FlatBufferFormat below) that will determine
  how the rest of the memory in the flatbuffer will be interpreted.

  The next 8 bytes contains the total size of the flatbuffer --- this
  allows flatbuffers passed around by a single pointer value without
  explicitly specifying the size of the buffer as well as checking if
  the an arbitrary memory buffer uses the flatbuffer format or not.

  At the 16 bytes, starts the data format metadata buffer that holds a
  format specific struct instance that contains various user-specified
  parameters of the storage format. The size of the data format
  metadata buffer depends on the format id.

  After the data format metadata buffer starts the data format worker
  buffer that holds various pre-computed and state parameters that
  describe the internal data layout of the format. The size of the
  data format metadata buffer depends on the format id.

  After the data format worker buffer starts the raw data buffer that
  interpretation depends on the format parameters specified above. The
  size of the raw data buffer depends on the format id and the
  user-specified parameters in the data format metadata.
  All buffers above are aligned to the 64-bit boundaries.

  In summary, the memory layout of a flatbuffer is:

    | <format id>  | <flatbuffer size> | <format metadata buffer>   | <format worker buffer>   | <raw data buffer>   | <format id>   |
    =
    |<-- 8-bytes-->|<-- 8-bytes ------>|<-- metadata buffer size -->|<-- worker buffer size -->|<-- row data size -->|<-- 8-bytes -->|
    |<------------------------ flatbuffer size ------------------------------------------------------------------------------------->|


  VarlenArray format specification
  --------------------------------

  The VarlenArray format metadata and worker buffers memory layout is
  described by VarlenArray and VarlenArrayWorker struct definitions
  below. The raw data buffer memory layout is a follows:

    | <values>                         | <compressed indices>           | <storage indices>          |
    =
    |<-- (max nof values) * 8 bytes -->|<-- (num items + 1) * 8 bytes-->|<-- (num items) * 8 bytes-->|

  GeoPoint format specification
  -----------------------------

  The GeoPoint format metadata and worker buffers memory layout is
  described by GeoPoint and GeoPointWorker struct definitions
  below. The raw data buffer memory layout is a follows:

    | <point data>                                  |
    =
    |<-- (num items) * (is_geoint ? 4 : 8) bytes -->|

  where <point data> stores points coordinates in a point-wise manner:
  X0, Y0, X1, Y1, ...

  if is_geoint is true, point coordinates are stored as integers,
  otherwise as double floating point numbers.

  GeoLineString format specification
  ----------------------------------

  The GeoLineString format metadata and worker buffers memory layout
  is described by GeoLineString and GeoLineStringWorker struct
  definitions below. The raw data buffer memory layout is a follows:

    | <point data>                                       | <compressed indices>            | <storage indices>          |
    =
    |<-- (max nof values) * (is_geoint ? 4 : 8) bytes -->|<-- (num items + 1) * 8 bytes -->|<-- (num items) * 8 bytes-->|

  where <point data> stores points coordinates in a point-wise manner:
  X0, Y0, X1, Y1, ...

  if is_geoint is true, point coordinates are stored as integers,
  otherwise as double floating point numbers.

  A line of n-th item consists of points defined by

    i0 = compressed_indices[storage_indices[n]]
    i1 = compressed_indices[storage_indices[n] + 1]
    line_string = point_data[i0:i1]

  GeoPolygon format specification
  ----------------------------------

  The GeoPolygon format metadata and worker buffers memory layout
  is described by GeoPolygon and GeoPolygonWorker struct
  definitions below. The raw data buffer memory layout is a follows:

    | <point data>                                       | <counts2>                       | <compressed indices2>               |  <compressed indices>           | <storage indices>          |
    =
    |<-- (max nof points) * (is_geoint ? 4 : 8) bytes -->|<-- (max nof rings) * 4 bytes -->|<-- (max nof rings + 1) * 8 bytes -->|<-- (num items + 1) * 8 bytes -->|<-- (num items) * 8 bytes-->|

  where

     <point data> stores points coordinates in a point-wise manner:
     X0, Y0, X1, Y1, ... If is_geoint is true, point coordinates are
     stored as integers, otherwise as double floating point numbers.

    <counts2> contains ring sizes (in points). All entires are
      non-negative.

    <compressed indices2> contains the "cumulative sum" of all ring
      sizes (in points). All entires are non-negative and sorted.

    <compressed indices> contains the "cumulative sum" of item sizes
      (in rings).  Negative entries indicate null items.

    <storage indices> defines the order of specifying items in the flat
      buffer

  Assuming that a ring consists of at least 3 points, we'll have

    <max nof rings> * 3 <= <max nof points>

  A polygon of n-th item consists of rings defined by

    i0 = compressed_indices[storage_indices[n]]
    i1 = compressed_indices[storage_indices[n] + 1]

    for i in range(i0, i1):
        j0 = compressed_indices2[i]
        j1 = compressed_indices2[i + 1]
        ring_points = values[j0:j1]

  For example, consider two polygons:

    Polygon([p0, p1 ,p2, p3], [p4, p5, p6], [p7, p8, p9])
    Polygon([r0, r1 ,r2], [r3, r4, r5], [r6, r7, r8])

  and storage_indices=[0, 1], then

    compressed_indices = [0 3 6]
    compressed_indices2 = [0 4 7 10 13 16 19]
    values = [p0, ..., p9, r0, ..., r8]

  Notice that compressed_indices2 describes the partitioning of points
  into rings and compressed_indices describes the partitioning of
  rings into polygons.

  NestedArray format specification
  --------------------------------

  NestedArray represents a storage for zero, one, two, and three
  dimensional ragged arrays. The storage format consists of sizes and
  values buffers (plus offset buffers to optimize accessing
  items). The sizes buffer stores the sizes of ragged arrays at
  various levels and the values buffer stores the values of ragged
  arrays.

  The NestedArray storage is used as a uniform storage schema for
  different types (variable-length arrays, geotypes, etc) with
  variable dimensionality. For example, a GeoMultiPolygon

    GeoMultiPolygon([
      GeoPolygon([LineString([(x000, y000), (x001, y001), ...])],
                  LineString([(x010, y010), (x011, y011), ...])],
                  ...]),
      GeoPolygon([LineString([(x100, y100), (x101, y101), ...])],
                  LineString([(x110, y110), (x111, y111), ...])],
                  ...]),
      ...
    ])

  is represented as a three dimensional ragged array where the sizes
  buffer contains the number of polygons in the multi-polygon, all the
  numbers of linestrings in polygons, all the numbers of points in
  linestrings, and finally, the values buffer contains all the
  coordinates. Note that a "value" is defined as a point with two
  coordinates.

  The current implementation of NestedArray supports dimensionalities
  up to 3 but the format can be extended to arbitrary dimensions.

  NestedArray API
  ---------------

  To compute flatbuffer size required to represent a nested array with
  the given dimensionsinality, total items count, estimated total
  sizes and values counts, value type, and user data buffer size,
  use::

    int64_t compute_flatbuffer_size(dimensions,
                                    total_items_count,
                                    total_sizes_count,
                                    total_values_count,
                                    value_type,
                                    user_data_size)

  To initialize the provided buffer for nested array format, use::

    Status .initialize(dimensions,
                       total_items_count,
                       total_sizes_count,
                       total_values_count,
                       value_type,
                       null_value_ptr,
                       user_data_ptr, user_data_size)

  To test if the provided buffer contains an initialized FlatBuffer::

    bool isFlatBuffer(buffer)

  To get the size of an initialized FlatBuffer::

    int64_t getBufferSize(buffer)
    int64_t .getBufferSize()

  To get the size of the values buffer::

    size_t .getValuesBufferSize()

  To get the size of a value::

    size_t .getValueSize()

  To get the number of specified values::

    size_t .getValuesCount()

  To get the dimensionality of a nested array::

    size_t .getDimensions()

  To get various buffers::

    int8_t* .get_user_data_buffer()
    int8_t* .get_values_buffer()
    sizes_t* .get_sizes_buffer()
    offsets_t* .get_values_offsets()
    offsets_t* .get_sizes_offsets()
    int8_t* .getNullValuePtr()

  To test if the provided buffer contains null value::

    bool .containsNullValue()

  To get the item and subitems of a nested array::

    template <typename CT>
    Status .getItem(index,
                    vector<CT>& values,
                    vector<int32_t>& sizes,
                    vector<int32_t>& sizes_of_sizes,
                    bool& is_null)                           # ndims == 3

    Status .getItem(index,
                    subindex,
                    vector<CT>& values,
                    vector<int32_t>& sizes,
                    bool& is_null)                           # ndims == 3

    Status .getItem(index,
                    subindex,
                    subsubindex,
                    vector<CT>& values,
                    bool& is_null)                           # ndims == 3

    Status .getItem(index,
                    int32_t& nof_values,
                    int8_t*& values,
                    int32_t& nof_sizes,
                    int8_t*& sizes,
                    int32_t& nof_sizes_of_sizes,
                    int8_t*& sizes_of_sizes,
                    bool& is_null)                           # ndims == 3

    Status .getItem(index,
                    subindex,
                    int32_t& nof_values,
                    int8_t*& values,
                    int32_t& nof_sizes,
                    int8_t*& sizes,
                    bool& is_null)                            # ndims == 3

    Status .getItem(index,
                    subindex,
                    subsubindex,
                    int32_t& nof_values,
                    int8_t*& values,
                    int32_t& nof_sizes,
                    int8_t*& sizes,
                    bool& is_null)                            # ndims == 3

  To get the item or subitem lengths::

    Status .getLength(index, size_t& length)                        # ndims == 3
    Status .getLength(index, subindex, size_t& length)              # ndims == 3
    Status .getLength(index, subindex, subsubindex, size_t& length) # ndims == 3

  To set an item of a nested array array::

    Status .setItem(index, vector<double>& arr)               # ndims == 1

    Status .setItem(index, vector<vector<double>>& arr)       # ndims == 2

    template <typename CT>
    Status .setItem(index, vector<vector<vector<CT>>>& arr)   # ndims == 3

    template <typename CT>
    Status .setItem(index,
                    vector<CT>
                    vector<int32_t>& sizes,
                    vector<int32_t>& sizes_of_sizes)          # ndims == 3

    Status setItem(const int64_t index,
                   int8_t* values_buf,
                   size_t values_buf_size,
                   int32_t* sizes_buf,
                   int32_t nof_sizes,
                   int32_t* sizes_of_sizes_buf,
                   int32_t nof_sizes_of_sizes)                # ndims == 3

  To test if item is NULL::

    Status isNull(index, bool& is_null)

  FlatBuffer usage
  ----------------

  FlatBuffer implements various methods for accessing its content for
  retrieving or storing data. These methods usually are provided as
  pairs of safe and unsafe methods. Safe methods validate method
  inputs and return success or error status depending on the
  validation results. Unsafe methods (that names have "NoValidation"
  suffix) performs no validation on the inputs and (almost) always
  return the success status. Use unsafe methods (for efficency) only
  when one can prove that the inputs are always valid (e.g. indices
  are in the expected range, buffer memory allocation is sufficient
  for storing data, etc). Otherwise, use safe methods that will lead
  to predictable and non-server-crashing behaviour in the case of
  possibly invalid input data.
*/
// clang-format on

#ifdef FLATBUFFER_ERROR_ABORTS
#include "../../Shared/toString.h"
#define RETURN_ERROR(exc) \
  {                       \
    PRINT(exc);           \
    abort();              \
    return (exc);         \
  }
#else
#define RETURN_ERROR(exc) return (exc)
#endif

#include <float.h>
#ifdef HAVE_TOSTRING
#include <ostream>
#include <string.h>
#endif
#include <vector>

#include "../../Shared/funcannotations.h"

#define FLATBUFFER_UNREACHABLE() \
  { abort(); }

// Notice that the format value is used to recognize if a memory
// buffer uses some flat buffer format or not. To minimize chances for
// false positive test results, use a non-trival integer value when
// introducing new formats.
enum FlatBufferFormat {
  VarlenArrayFormatId = 0x7661726c65634152,    // hex repr of 'varlenAR'
  GeoPointFormatId = 0x67656f706f696e74,       // hex repr of 'geopoint'
  GeoLineStringFormatId = 0x676c696e65737472,  // hex repr of 'glinestr'
  GeoPolygonFormatId = 0x67706f6c79676f6e,     // hex repr of 'gpolygon'
  // GeoMultiPointFormatId      = 0x47656f706f696e74,  // hex repr of 'Geopoint'
  // GeoMultiLineStringFormatId = 0x476c696e65737472,  // hex repr of 'Glinestr'
  NestedArrayFormatId = 0x6e65737465644152  // hex repr of 'nestedAR'
};

inline int64_t _align_to_int64(int64_t addr) {
  addr += sizeof(int64_t) - 1;
  return (int64_t)(((uint64_t)addr >> 3) << 3);
}

struct FlatBufferManager {
  enum ValueType {
    Bool8,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    PointInt32,
    PointFloat64
  };

#ifdef HAVE_TOSTRING
  static std::string toString(const ValueType& type);
#endif

  static size_t get_size(ValueType type) {
    switch (type) {
      case Bool8:
      case Int8:
      case UInt8:
        return 1;
      case Int16:
      case UInt16:
        return 2;
      case Int32:
      case UInt32:
      case Float32:
        return 4;
      case Int64:
      case UInt64:
      case Float64:
      case PointInt32:
        return 8;
      case PointFloat64:
        return 16;
    }
    FLATBUFFER_UNREACHABLE();
    return 0;
  }

  /*
    sizes_t is the type of a container size. Here we use int32_t
    because Geospatial uses it as the type for the vector of ring and
    polygon sizes.

    offsets_t is the type of offsets that is used to locate
    sub-buffers within the FlatBuffer main buffer. Because NULL items
    are encoded as negative offset values, the offsets type must be a
    signed type. Hence, we define offsets_t as int64_t.
   */

  typedef int32_t sizes_t;
  typedef int64_t offsets_t;

#define FLATBUFFER_SIZES_T_VALUE_TYPE Int32
#define FLATBUFFER_OFFSETS_T_VALUE_TYPE UInt64

  struct BaseWorker {
    FlatBufferFormat format_id;
    offsets_t flatbuffer_size;
    offsets_t format_metadata_offset;  // the offset of the data format metadata buffer
    offsets_t format_worker_offset;    // the offset of the data format worker buffer
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "format_id=" + std::to_string(format_id);
      result += ",\n    flatbuffer_size=" + std::to_string(flatbuffer_size);
      result += ",\n    format_metadata_offset=" + std::to_string(format_metadata_offset);
      result += ",\n    format_worker_offset=" + std::to_string(format_worker_offset);
      result += "}";
      return result;
    }
#endif
  };

#define VARLENARRAY_NOFPARAMS 2

  enum VarlenArrayParamValues {
    VarlenArrayParamDictId = 0,
    VarlenArrayParamDbId = 1,
    VarlenArrayParamsCount = 2
  };

  struct VarlenArray {
    int64_t total_items_count;               // the total number of items
    int64_t max_nof_values;                  // the maximum number of values in all items
    size_t dtype_size;                       // the size of item dtype in bytes
    int32_t params[VarlenArrayParamsCount];  // dtype parameters (e.g. [dict_id, db_id])
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "total_items_count=" + std::to_string(total_items_count);
      result += ", max_nof_values=" + std::to_string(max_nof_values);
      result += ", dtype_size=" + std::to_string(dtype_size);
      result += ", params=[";
      for (int i = 0; i < VarlenArrayParamsCount; i++) {
        result += (i > 0 ? ", " : " ");
        result += std::to_string(params[i]);
      }
      result += "]}";
      return result;
    }
#endif
  };

  struct VarlenArrayWorker {
    int64_t items_count;    // the number of specified items
    int64_t values_offset;  // the offset of values buffer within the flatbuffer memory
    int64_t compressed_indices_offset;  // the offset of compressed_indices buffer
    int64_t storage_indices_offset;     // the offset of storage_indices buffer
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "items_count=" + std::to_string(items_count);
      result += ", values_offset=" + std::to_string(values_offset);
      result +=
          ", compressed_indices_offset=" + std::to_string(compressed_indices_offset);
      result += ", storage_indices_offset=" + std::to_string(storage_indices_offset);
      result += "}";
      return result;
    }
#endif
  };

  struct GeoPoint {
    int64_t total_items_count;  // the total number of items
    int32_t input_srid;
    int32_t output_srid;
    bool is_geoint;
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "total_items_count=" + std::to_string(total_items_count);
      result += ", input_srid=" + std::to_string(output_srid);
      result += ", output_srid=" + std::to_string(output_srid);
      result += ", is_geoint=" + std::to_string(is_geoint);
      result += "}";
      return result;
    }
#endif
  };

  struct GeoPointWorker {
    int64_t values_offset;  // the offset of values buffer within the flatbuffer memory
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "values_offset=" + std::to_string(values_offset);
      result += "}";
      return result;
    }
#endif
  };

  struct GeoLineString {
    int64_t total_items_count;  // the total number of items
    int64_t max_nof_values;     // the maximum number of points in all items
    int32_t input_srid;
    int32_t output_srid;
    bool is_geoint;
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "total_items_count=" + std::to_string(total_items_count);
      result += ", max_nof_values=" + std::to_string(max_nof_values);
      result += ", input_srid=" + std::to_string(output_srid);
      result += ", output_srid=" + std::to_string(output_srid);
      result += ", is_geoint=" + std::to_string(is_geoint);
      result += "}";
      return result;
    }
#endif
  };

  struct GeoLineStringWorker {
    int64_t items_count;  // the number of specified items
    int64_t values_offset;
    int64_t compressed_indices_offset;
    int64_t storage_indices_offset;
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "items_count=" + std::to_string(items_count);
      result += ", values_offset=" + std::to_string(values_offset);
      result +=
          ", compressed_indices_offset=" + std::to_string(compressed_indices_offset);
      result += ", storage_indices_offset=" + std::to_string(storage_indices_offset);
      result += "}";
      return result;
    }
#endif
  };

  struct GeoPolygon {
    int64_t total_items_count;  // the total number of items
    int64_t max_nof_values;     // the maximum number of points in all items
    int64_t max_nof_rings;      // the maximum number of rings in all items
    int32_t input_srid;
    int32_t output_srid;
    bool is_geoint;
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "total_items_count=" + std::to_string(total_items_count);
      result += ", max_nof_values=" + std::to_string(max_nof_values);
      result += ", max_nof_rings=" + std::to_string(max_nof_rings);
      result += ", input_srid=" + std::to_string(output_srid);
      result += ", output_srid=" + std::to_string(output_srid);
      result += ", is_geoint=" + std::to_string(is_geoint);
      result += "}";
      return result;
    }
#endif
  };

  struct GeoPolygonWorker {
    int64_t items_count;   // the number of specified items
    int64_t items2_count;  // the number of specified rings
    int64_t values_offset;
    int64_t counts2_offset;
    int64_t compressed_indices2_offset;
    int64_t compressed_indices_offset;
    int64_t storage_indices_offset;

#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "items_count=" + std::to_string(items_count);
      result += ", items2_count=" + std::to_string(items2_count);
      result += ", values_offset=" + std::to_string(values_offset);
      result += ", counts2_offset=" + std::to_string(counts2_offset);
      result +=
          ", compressed_indices2_offset=" + std::to_string(compressed_indices2_offset);
      result +=
          ", compressed_indices_offset=" + std::to_string(compressed_indices_offset);
      result += ", storage_indices_offset=" + std::to_string(storage_indices_offset);
      result += "}";
      return result;
    }
#endif
  };

  struct NestedArrayWorker {
    int64_t specified_items_count;
    // all offsets are in bytes
    offsets_t storage_indices_offset;
    offsets_t sizes_offsets_offset;
    offsets_t values_offsets_offset;
    offsets_t sizes_buffer_offset;
    offsets_t values_buffer_offset;
    offsets_t user_data_buffer_offset;
    size_t value_size;
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "specified_items_count=" + std::to_string(specified_items_count);
      result += ",\n    storage_indices_offset=" + std::to_string(storage_indices_offset);
      result += ",\n    sizes_offsets_offset=" + std::to_string(sizes_offsets_offset);
      result += ",\n    values_offsets_offset=" + std::to_string(values_offsets_offset);
      result += ",\n    sizes_buffer_offset=" + std::to_string(sizes_buffer_offset);
      result += ",\n    values_buffer_offset=" + std::to_string(values_buffer_offset);
      result +=
          ",\n    user_data_buffer_offset=" + std::to_string(user_data_buffer_offset);
      result += ",\n    value_size=" + std::to_string(value_size);
      result += "}";
      return result;
    }
#endif
  };

  struct NestedArray {
    size_t dimensions;
    int64_t total_items_count;
    int64_t total_sizes_count;
    int64_t total_values_count;
    ValueType value_type;
    size_t user_data_size;
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "dimensions=" + std::to_string(dimensions);
      result += ",\n    total_items_count=" + std::to_string(total_items_count);
      result += ",\n    total_sizes_count=" + std::to_string(total_sizes_count);
      result += ",\n    total_values_count=" + std::to_string(total_values_count);
      result += ",\n    value_type=" + FlatBufferManager::toString(value_type);
      result += ",\n    user_data_size=" + std::to_string(user_data_size);
      result += "}";
      return result;
    }
#endif
  };

  enum Status {
    Success = 0,
    IndexError,
    SubIndexError,
    SizeError,
    FlatbufferSizeError,
    ItemAlreadySpecifiedError,
    ItemUnspecifiedError,
    UnexpectedNullItemError,
    ValuesBufferTooSmallError,
    SizesBufferTooSmallError,
    CompressedIndices2BufferTooSmallError,
    MemoryError,
    NotImplementedError,
    NotSupportedFormatError,
    InvalidUserDataError,
    DimensionalityError,
    TypeError,
    UserDataError,
    InconsistentSizesError,
    UnknownFormatError
  };

  // FlatBuffer main buffer. It is the only member of the FlatBuffer struct.
  int8_t* buffer;

  // Check if a buffer contains FlatBuffer formatted data
  HOST DEVICE static bool isFlatBuffer(const void* buffer) {
    if (buffer) {
      // warning: assume that buffer size is at least 8 bytes
      const auto* base = reinterpret_cast<const BaseWorker*>(buffer);
      FlatBufferFormat header_format = base->format_id;
      switch (header_format) {
        case NestedArrayFormatId:
        case VarlenArrayFormatId:
        case GeoPointFormatId:
        case GeoLineStringFormatId:
        case GeoPolygonFormatId: {
          int64_t flatbuffer_size = base->flatbuffer_size;
          if (flatbuffer_size > 0) {
            FlatBufferFormat footer_format = static_cast<FlatBufferFormat>(
                ((int64_t*)buffer)[flatbuffer_size / sizeof(int64_t) - 1]);
            return footer_format == header_format;
          }
          break;
        }
        default:
          break;
      }
    }
    return false;
  }

  // Return the allocation size of the the FlatBuffer storage, in bytes
  // TODO: return size_t value, 0 when not a flat buffer
  static int64_t getBufferSize(const void* buffer) {
    if (isFlatBuffer(buffer)) {
      return reinterpret_cast<const BaseWorker*>(buffer)->flatbuffer_size;
    } else {
      return -1;
    }
  }

  // Return the allocation size of the the FlatBuffer storage, in bytes
  // TODO: int64_t -> size_t
  inline int64_t getBufferSize() const {
    return reinterpret_cast<const BaseWorker*>(buffer)->flatbuffer_size;
  }

  inline bool isNestedArray() const { return format() == NestedArrayFormatId; }

  inline size_t getValueSize() const { return getNestedArrayWorker()->value_size; }

  inline size_t getValuesBufferSize() const {
    const auto* metadata = getNestedArrayMetadata();
    const auto* worker = getNestedArrayWorker();
    return worker->value_size * metadata->total_values_count;
  }

  inline size_t getValuesCount() const {
    const auto* worker = getNestedArrayWorker();
    const auto* values_offsets = get_values_offsets();
    const auto storage_index = worker->specified_items_count;
    const auto values_offset = values_offsets[storage_index];
    if (values_offset < 0) {
      return -(values_offset + 1);
    }
    return values_offset;
  }

  // Return the format of FlatBuffer
  HOST DEVICE inline FlatBufferFormat format() const {
    const auto* base = reinterpret_cast<const BaseWorker*>(buffer);
    return base->format_id;
  }

  // Return the number of items
  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t itemsCount() const {
    switch (format()) {
      case VarlenArrayFormatId:
        return getVarlenArrayMetadata()->total_items_count;
      case GeoPointFormatId:
        return getGeoPointMetadata()->total_items_count;
      case GeoLineStringFormatId:
        return getGeoLineStringMetadata()->total_items_count;
      case GeoPolygonFormatId:
        return getGeoPolygonMetadata()->total_items_count;
      case NestedArrayFormatId:
        return getNestedArrayMetadata()->total_items_count;
      default:
        break;
    }
    return -1;  // invalid value
  }

  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t items2Count() const {
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        break;
      case GeoPolygonFormatId:
        return getGeoPolygonMetadata()->max_nof_rings;
      default:
        break;
    }
    return -1;  // invalid value
  }

  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t valueByteSize() const {
    switch (format()) {
      case VarlenArrayFormatId:
        return getVarlenArrayMetadata()->dtype_size;
      case GeoPointFormatId:
        return 2 * (getGeoPointMetadata()->is_geoint ? sizeof(int32_t) : sizeof(double));
      case GeoLineStringFormatId:
        return 2 *
               (getGeoLineStringMetadata()->is_geoint ? sizeof(int32_t) : sizeof(double));
      case GeoPolygonFormatId:
        return 2 *
               (getGeoPolygonMetadata()->is_geoint ? sizeof(int32_t) : sizeof(double));
      default:
        break;
    }
    return -1;
  }

  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t dtypeSize() const {  // TODO: use valueByteSize instead
    switch (format()) {
      case VarlenArrayFormatId:
        return getVarlenArrayMetadata()->dtype_size;
      case GeoPointFormatId:
        return 2 * (getGeoPointMetadata()->is_geoint ? sizeof(int32_t) : sizeof(double));
      case GeoLineStringFormatId:
        return 2 *
               (getGeoLineStringMetadata()->is_geoint ? sizeof(int32_t) : sizeof(double));
      case GeoPolygonFormatId:
        return 2 *
               (getGeoPolygonMetadata()->is_geoint ? sizeof(int32_t) : sizeof(double));
      default:
        break;
    }
    return -1;
  }

  // VarlenArray support:

  // To be deprecated in favor of NestedArray format
  static int64_t compute_flatbuffer_size(FlatBufferFormat format_id,
                                         const int8_t* format_metadata_ptr) {
    int64_t flatbuffer_size = _align_to_int64(sizeof(FlatBufferManager::BaseWorker));
    switch (format_id) {
      case VarlenArrayFormatId: {
        const auto format_metadata =
            reinterpret_cast<const VarlenArray*>(format_metadata_ptr);
        flatbuffer_size += _align_to_int64(sizeof(VarlenArray));
        flatbuffer_size += _align_to_int64(sizeof(VarlenArrayWorker));
        flatbuffer_size +=
            _align_to_int64(format_metadata->dtype_size *
                            format_metadata->max_nof_values);  // values buffer size
        flatbuffer_size +=
            _align_to_int64(sizeof(int64_t) * (format_metadata->total_items_count +
                                               1));  // compressed_indices buffer size
        flatbuffer_size += _align_to_int64(
            sizeof(int64_t) *
            (format_metadata->total_items_count));  // storage_indices buffer size
        break;
      }
      case GeoPointFormatId: {
        const auto format_metadata =
            reinterpret_cast<const GeoPoint*>(format_metadata_ptr);
        flatbuffer_size += _align_to_int64(sizeof(GeoPoint));
        flatbuffer_size += _align_to_int64(sizeof(GeoPointWorker));
        const auto itemsize =
            2 * (format_metadata->is_geoint ? sizeof(int32_t) : sizeof(double));
        flatbuffer_size += _align_to_int64(
            itemsize * format_metadata->total_items_count);  // values buffer size
        break;
      }
      case GeoLineStringFormatId: {
        const auto format_metadata =
            reinterpret_cast<const GeoLineString*>(format_metadata_ptr);
        flatbuffer_size += _align_to_int64(sizeof(GeoLineString));
        flatbuffer_size += _align_to_int64(sizeof(GeoLineStringWorker));
        const auto itemsize =
            2 * (format_metadata->is_geoint ? sizeof(int32_t) : sizeof(double));
        flatbuffer_size += _align_to_int64(
            itemsize * format_metadata->max_nof_values);  // values buffer size
        flatbuffer_size +=
            _align_to_int64(sizeof(int64_t) * (format_metadata->total_items_count +
                                               1));  // compressed_indices buffer size
        flatbuffer_size += _align_to_int64(
            sizeof(int64_t) *
            (format_metadata->total_items_count));  // storage_indices buffer size
        break;
      }
      case GeoPolygonFormatId: {
        const auto format_metadata =
            reinterpret_cast<const GeoPolygon*>(format_metadata_ptr);
        flatbuffer_size += _align_to_int64(sizeof(GeoPolygon));
        flatbuffer_size += _align_to_int64(sizeof(GeoPolygonWorker));
        const auto itemsize =
            2 * (format_metadata->is_geoint ? sizeof(int32_t) : sizeof(double));
        flatbuffer_size += _align_to_int64(
            itemsize * format_metadata->max_nof_values);  // values buffer size
        flatbuffer_size += _align_to_int64(
            sizeof(int32_t) * (format_metadata->max_nof_rings));  // counts2 buffer size
        flatbuffer_size +=
            _align_to_int64(sizeof(int64_t) * (format_metadata->max_nof_rings +
                                               1));  // compressed_indices2 buffer size
        flatbuffer_size +=
            _align_to_int64(sizeof(int64_t) * (format_metadata->total_items_count +
                                               1));  // compressed_indices buffer size
        flatbuffer_size += _align_to_int64(
            sizeof(int64_t) *
            (format_metadata->total_items_count));  // storage_indices buffer size
        break;
      }
      default:
        FLATBUFFER_UNREACHABLE();
    }
    flatbuffer_size += _align_to_int64(sizeof(int64_t));  // footer format id
    return flatbuffer_size;
  }

  HOST DEVICE inline BaseWorker* getBaseWorker() {
    return reinterpret_cast<FlatBufferManager::BaseWorker*>(buffer);
  }
  HOST DEVICE inline const BaseWorker* getBaseWorker() const {
    return reinterpret_cast<const BaseWorker*>(buffer);
  }

#define FLATBUFFER_MANAGER_FORMAT_TOOLS(TYPENAME)                                    \
  HOST DEVICE inline TYPENAME##Worker* get##TYPENAME##Worker() {                     \
    auto* base = getBaseWorker();                                                    \
    return reinterpret_cast<TYPENAME##Worker*>(buffer + base->format_worker_offset); \
  }                                                                                  \
  HOST DEVICE inline TYPENAME* get##TYPENAME##Metadata() {                           \
    auto* base = getBaseWorker();                                                    \
    return reinterpret_cast<TYPENAME*>(buffer + base->format_metadata_offset);       \
  }                                                                                  \
  HOST DEVICE inline const TYPENAME##Worker* get##TYPENAME##Worker() const {         \
    const auto* base = getBaseWorker();                                              \
    return reinterpret_cast<const TYPENAME##Worker*>(buffer +                        \
                                                     base->format_worker_offset);    \
  }                                                                                  \
  HOST DEVICE inline const TYPENAME* get##TYPENAME##Metadata() const {               \
    const auto* base = getBaseWorker();                                              \
    return reinterpret_cast<const TYPENAME*>(buffer + base->format_metadata_offset); \
  }

  // To be deprecated in favor of NestedArray format
  FLATBUFFER_MANAGER_FORMAT_TOOLS(VarlenArray);
  FLATBUFFER_MANAGER_FORMAT_TOOLS(GeoPoint);
  FLATBUFFER_MANAGER_FORMAT_TOOLS(GeoLineString);
  FLATBUFFER_MANAGER_FORMAT_TOOLS(GeoPolygon);
  /*
  HOST DEVICE inline NestedArrayWorker* getNestedArrayWorker() {
    auto* base = getBaseWorker();
    return reinterpret_cast<NestedArrayWorker*>(buffer + base->format_worker_offset);
  }
  HOST DEVICE inline const NestedArrayWorker* getNestedArrayWorker() const {
    const auto* base = getBaseWorker();
    return reinterpret_cast<const NestedArrayWorker*>(buffer +
                                                      base->format_worker_offset);
  }
  */
#define FLATBUFFER_MANAGER_FORMAT_TOOLS_NEW(TYPENAME)                                 \
  HOST DEVICE inline NestedArrayWorker* get##TYPENAME##Worker() {                     \
    auto* base = getBaseWorker();                                                     \
    return reinterpret_cast<NestedArrayWorker*>(buffer + base->format_worker_offset); \
  }                                                                                   \
  HOST DEVICE inline TYPENAME* get##TYPENAME##Metadata() {                            \
    auto* base = getBaseWorker();                                                     \
    return reinterpret_cast<TYPENAME*>(buffer + base->format_metadata_offset);        \
  }                                                                                   \
  HOST DEVICE inline const NestedArrayWorker* get##TYPENAME##Worker() const {         \
    const auto* base = getBaseWorker();                                               \
    return reinterpret_cast<const NestedArrayWorker*>(buffer +                        \
                                                      base->format_worker_offset);    \
  }                                                                                   \
  HOST DEVICE inline const TYPENAME* get##TYPENAME##Metadata() const {                \
    const auto* base = getBaseWorker();                                               \
    return reinterpret_cast<const TYPENAME*>(buffer + base->format_metadata_offset);  \
  }

  FLATBUFFER_MANAGER_FORMAT_TOOLS(NestedArray);

#undef FLATBUFFER_MANAGER_FORMAT_TOOLS
#undef FLATBUFFER_MANAGER_FORMAT_TOOLS_NEW

#define FLATBUFFER_MANAGER_SET_OFFSET(OBJ, NAME, SIZE)                   \
  offset = OBJ->NAME##_offset = offset + _align_to_int64(previous_size); \
  previous_size = SIZE;

  static int64_t compute_flatbuffer_size(int64_t dimensions,
                                         int64_t total_items_count,
                                         int64_t total_sizes_count,
                                         int64_t total_values_count,
                                         ValueType value_type,
                                         size_t user_data_size) {
    size_t value_size = get_size(value_type);
    offsets_t flatbuffer_size = _align_to_int64(sizeof(FlatBufferManager::BaseWorker));
    flatbuffer_size += _align_to_int64(sizeof(NestedArray));
    flatbuffer_size += _align_to_int64(sizeof(NestedArrayWorker));
    flatbuffer_size +=
        _align_to_int64(value_size * (total_values_count + 1));  // values buffer
    flatbuffer_size +=
        _align_to_int64(sizeof(sizes_t) * total_sizes_count);  // sizes buffer
    flatbuffer_size +=
        _align_to_int64(sizeof(offsets_t) * (total_items_count + 1));  // values offsets
    flatbuffer_size += _align_to_int64(
        sizeof(offsets_t) * (total_items_count * dimensions + 1));  // sizes offsets
    flatbuffer_size += _align_to_int64(
        sizeof(sizes_t) * total_items_count);  // storage indices, must use signed type
    flatbuffer_size += _align_to_int64(user_data_size);   // user data
    flatbuffer_size += _align_to_int64(sizeof(int64_t));  // format id
    return flatbuffer_size;
  }

  Status initialize(FlatBufferFormat format_id,  // TODO: eliminate format_id or add it to
                                                 // compute_flatbuffer_size
                    int64_t dimensions,
                    int64_t total_items_count,
                    int64_t total_sizes_count,
                    int64_t total_values_count,
                    ValueType value_type,
                    const int8_t* null_value_ptr,
                    const int8_t* user_data_ptr,
                    size_t user_data_size) {
    auto* base = getBaseWorker();
    base->format_id = format_id;
    size_t value_size = get_size(value_type);
    base->flatbuffer_size = compute_flatbuffer_size(dimensions,
                                                    total_items_count,
                                                    total_sizes_count,
                                                    total_values_count,
                                                    value_type,
                                                    user_data_size);
    offsets_t offset = 0;
    size_t previous_size = sizeof(FlatBufferManager::BaseWorker);
    FLATBUFFER_MANAGER_SET_OFFSET(base, format_metadata, sizeof(NestedArray));
    FLATBUFFER_MANAGER_SET_OFFSET(base, format_worker, sizeof(NestedArrayWorker));

    auto* metadata = getNestedArrayMetadata();
    metadata->dimensions = dimensions;
    metadata->total_items_count = total_items_count;
    metadata->total_sizes_count = total_sizes_count;
    metadata->total_values_count = total_values_count;
    metadata->value_type = value_type;
    metadata->user_data_size = user_data_size;

    auto* worker = getNestedArrayWorker();
    worker->specified_items_count = 0;
    worker->value_size = value_size;

    FLATBUFFER_MANAGER_SET_OFFSET(
        worker, values_buffer, value_size * (total_values_count + 1));
    FLATBUFFER_MANAGER_SET_OFFSET(
        worker, sizes_buffer, sizeof(sizes_t) * total_sizes_count);
    FLATBUFFER_MANAGER_SET_OFFSET(
        worker, values_offsets, sizeof(offsets_t) * (total_items_count + 1));
    FLATBUFFER_MANAGER_SET_OFFSET(
        worker, sizes_offsets, sizeof(offsets_t) * (total_items_count * dimensions + 1));
    FLATBUFFER_MANAGER_SET_OFFSET(
        worker, storage_indices, sizeof(sizes_t) * total_items_count);
    FLATBUFFER_MANAGER_SET_OFFSET(worker, user_data_buffer, user_data_size);

    if (base->flatbuffer_size !=
        offset + _align_to_int64(previous_size) + _align_to_int64(sizeof(int64_t))) {
      RETURN_ERROR(FlatbufferSizeError);
    }

    offsets_t* values_offsets = get_values_offsets();
    offsets_t* sizes_offsets = get_sizes_offsets();
    values_offsets[0] = 0;
    sizes_offsets[0] = 0;
    sizes_t* storage_indices = get_storage_indices_new();
    for (int i = 0; i < total_items_count; i++) {
      storage_indices[i] = -1;
    }

    // the last value in values_buffer stores a null value:
    int8_t* null_value_buffer = get_values_buffer() + value_size * total_values_count;
    if (null_value_ptr != nullptr) {
      if (memcpy(null_value_buffer, null_value_ptr, value_size) == nullptr) {
        RETURN_ERROR(MemoryError);
      }
    } else {
      if (memset(null_value_buffer, 0, value_size) == nullptr) {
        RETURN_ERROR(MemoryError);
      }
    }

    if (user_data_size > 0 && user_data_ptr != nullptr) {
      int8_t* user_data_buffer = get_user_data_buffer();
      if (memcpy(user_data_buffer, user_data_ptr, user_data_size) == nullptr) {
        RETURN_ERROR(MemoryError);
      }
    }

    ((int64_t*)buffer)[base->flatbuffer_size / sizeof(int64_t) - 1] =
        static_cast<int64_t>(format_id);

    if (isFlatBuffer(buffer)) {
      return Success;
    }
    RETURN_ERROR(UnknownFormatError);
  }

  // To be deprecated in favor of NestedArray format
  void initialize(FlatBufferFormat format_id, const int8_t* format_metadata_ptr) {
    auto* base = getBaseWorker();
    base->format_id = format_id;
    base->flatbuffer_size = compute_flatbuffer_size(format_id, format_metadata_ptr);
    base->format_metadata_offset = _align_to_int64(sizeof(FlatBufferManager::BaseWorker));
    switch (format_id) {
      case NestedArrayFormatId:
        FLATBUFFER_UNREACHABLE();
        break;
      case VarlenArrayFormatId: {
        base->format_worker_offset =
            base->format_metadata_offset +
            _align_to_int64(sizeof(FlatBufferManager::VarlenArray));

        const auto* format_metadata =
            reinterpret_cast<const VarlenArray*>(format_metadata_ptr);
        auto* this_metadata = getVarlenArrayMetadata();
        this_metadata->total_items_count = format_metadata->total_items_count;
        this_metadata->max_nof_values = format_metadata->max_nof_values;
        this_metadata->dtype_size = format_metadata->dtype_size;
        for (int i = 0; i < VarlenArrayParamsCount; i++) {
          this_metadata->params[i] = format_metadata->params[i];
        }

        auto* this_worker = getVarlenArrayWorker();
        this_worker->items_count = 0;
        this_worker->values_offset =
            base->format_worker_offset + _align_to_int64(sizeof(VarlenArrayWorker));
        this_worker->compressed_indices_offset =
            this_worker->values_offset +
            _align_to_int64(this_metadata->dtype_size * this_metadata->max_nof_values);
        ;
        this_worker->storage_indices_offset =
            this_worker->compressed_indices_offset +
            _align_to_int64(sizeof(int64_t) * (this_metadata->total_items_count + 1));

        int64_t* compressed_indices =
            reinterpret_cast<int64_t*>(buffer + this_worker->compressed_indices_offset);
        int64_t* storage_indices =
            reinterpret_cast<int64_t*>(buffer + this_worker->storage_indices_offset);
        for (int i = 0; i < this_metadata->total_items_count; i++) {
          compressed_indices[i] = 0;
          storage_indices[i] = -1;
        }
        compressed_indices[this_metadata->total_items_count] = 0;
        break;
      }
      case GeoPointFormatId: {
        base->format_worker_offset =
            base->format_metadata_offset + _align_to_int64(sizeof(GeoPoint));

        const auto* format_metadata =
            reinterpret_cast<const GeoPoint*>(format_metadata_ptr);
        auto* this_metadata = getGeoPointMetadata();
        this_metadata->total_items_count = format_metadata->total_items_count;
        this_metadata->input_srid = format_metadata->input_srid;
        this_metadata->output_srid = format_metadata->output_srid;
        this_metadata->is_geoint = format_metadata->is_geoint;

        auto* this_worker = getGeoPointWorker();
        this_worker->values_offset =
            base->format_worker_offset + _align_to_int64(sizeof(GeoPointWorker));
        break;
      }
      case GeoLineStringFormatId: {
        base->format_worker_offset =
            base->format_metadata_offset + _align_to_int64(sizeof(GeoLineString));

        const auto* format_metadata =
            reinterpret_cast<const GeoLineString*>(format_metadata_ptr);
        auto* this_metadata = getGeoLineStringMetadata();
        this_metadata->total_items_count = format_metadata->total_items_count;
        this_metadata->max_nof_values = format_metadata->max_nof_values;
        this_metadata->input_srid = format_metadata->input_srid;
        this_metadata->output_srid = format_metadata->output_srid;
        this_metadata->is_geoint = format_metadata->is_geoint;

        const auto itemsize =
            2 * (this_metadata->is_geoint ? sizeof(int32_t) : sizeof(double));

        auto* this_worker = getGeoLineStringWorker();
        this_worker->items_count = 0;
        this_worker->values_offset =
            base->format_worker_offset + _align_to_int64(sizeof(GeoLineStringWorker));
        this_worker->compressed_indices_offset =
            this_worker->values_offset +
            _align_to_int64(itemsize * this_metadata->max_nof_values);
        ;
        this_worker->storage_indices_offset =
            this_worker->compressed_indices_offset +
            _align_to_int64(sizeof(int64_t) * (this_metadata->total_items_count + 1));

        int64_t* compressed_indices =
            reinterpret_cast<int64_t*>(buffer + this_worker->compressed_indices_offset);
        int64_t* storage_indices =
            reinterpret_cast<int64_t*>(buffer + this_worker->storage_indices_offset);
        for (int i = 0; i < this_metadata->total_items_count; i++) {
          compressed_indices[i] = 0;
          storage_indices[i] = -1;
        }
        compressed_indices[this_metadata->total_items_count] = 0;
        break;
      }
      case GeoPolygonFormatId: {
        base->format_worker_offset =
            base->format_metadata_offset + _align_to_int64(sizeof(GeoPolygon));

        const auto* format_metadata =
            reinterpret_cast<const GeoPolygon*>(format_metadata_ptr);
        auto* this_metadata = getGeoPolygonMetadata();
        this_metadata->total_items_count = format_metadata->total_items_count;
        this_metadata->max_nof_values = format_metadata->max_nof_values;
        this_metadata->max_nof_rings = format_metadata->max_nof_rings;
        this_metadata->input_srid = format_metadata->input_srid;
        this_metadata->output_srid = format_metadata->output_srid;
        this_metadata->is_geoint = format_metadata->is_geoint;

        const auto itemsize =
            2 * (this_metadata->is_geoint ? sizeof(int32_t) : sizeof(double));

        auto* this_worker = getGeoPolygonWorker();
        this_worker->items_count = 0;
        this_worker->items2_count = 0;
        this_worker->values_offset =
            base->format_worker_offset + _align_to_int64(sizeof(GeoPolygonWorker));

        this_worker->counts2_offset =
            this_worker->values_offset +
            _align_to_int64(itemsize * this_metadata->max_nof_values);

        this_worker->compressed_indices2_offset =
            this_worker->counts2_offset +
            _align_to_int64(sizeof(int32_t) * this_metadata->max_nof_rings);

        this_worker->compressed_indices_offset =
            this_worker->compressed_indices2_offset +
            _align_to_int64(sizeof(int64_t) * (this_metadata->max_nof_rings + 1));
        ;
        this_worker->storage_indices_offset =
            this_worker->compressed_indices_offset +
            _align_to_int64(sizeof(int64_t) * (this_metadata->total_items_count + 1));

        int32_t* counts2 = get_counts2();
        int64_t* compressed_indices2 = get_compressed_indices2();
        int64_t* compressed_indices = get_compressed_indices();
        int64_t* storage_indices = get_storage_indices();
        for (int i = 0; i < this_metadata->max_nof_rings; i++) {
          counts2[i] = 0;
          compressed_indices2[i] = 0;
        }
        compressed_indices2[this_metadata->max_nof_rings] = 0;
        for (int i = 0; i < this_metadata->total_items_count; i++) {
          compressed_indices[i] = 0;
          storage_indices[i] = -1;
        }
        compressed_indices[this_metadata->total_items_count] = 0;
        break;
      }
    }
    ((int64_t*)buffer)[base->flatbuffer_size / sizeof(int64_t) - 1] =
        static_cast<int64_t>(format_id);
  }

  // Low-level API

  inline size_t getDimensions() const {
    if (isNestedArray()) {
      return getNestedArrayMetadata()->dimensions;
    }
    FLATBUFFER_UNREACHABLE();
    return 0;
  }

  // Return the upper bound to the total number of points in all items
  // To be deprecated in favor of NestedArray format
  inline int64_t get_max_nof_values() const {
    switch (format()) {
      case VarlenArrayFormatId:
        return getVarlenArrayMetadata()->max_nof_values;
      case GeoPointFormatId:
        return getGeoPointMetadata()->total_items_count;
      case GeoLineStringFormatId:
        return getGeoLineStringMetadata()->max_nof_values;
      case GeoPolygonFormatId:
        return getGeoPolygonMetadata()->max_nof_values;
      default:
        break;
    }
    return -1;
  }

  // Return the total number of values in all specified items
  // To be deprecated in favor of NestedArray format
  inline int64_t get_nof_values() const {
    switch (format()) {
      case NestedArrayFormatId: {
        FLATBUFFER_UNREACHABLE();
        break;
      }
      case GeoPolygonFormatId: {
        const int64_t storage_count2 = get_storage_count2();
        const int64_t* compressed_indices2 = get_compressed_indices2();
        return compressed_indices2[storage_count2];
      }
      default: {
        const int64_t storage_count = get_storage_count();
        const int64_t* compressed_indices = get_compressed_indices();
        return compressed_indices[storage_count];
      }
    }
  }

  // Return the number of specified items
  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t& get_storage_count() {
    switch (format()) {
      case VarlenArrayFormatId:
        return getVarlenArrayWorker()->items_count;
      case GeoPointFormatId:
        return getGeoPointMetadata()->total_items_count;
      case GeoLineStringFormatId:
        return getGeoLineStringWorker()->items_count;
      case GeoPolygonFormatId:
        return getGeoPolygonWorker()->items_count;
      default:
        break;
    }
    static int64_t dummy_storage_count = -1;
    return dummy_storage_count;
  }

  // To be deprecated in favor of NestedArray format
  inline const int64_t& get_storage_count() const {
    switch (format()) {
      case VarlenArrayFormatId:
        return getVarlenArrayWorker()->items_count;
      case GeoPointFormatId:
        return getGeoPointMetadata()->total_items_count;
      case GeoLineStringFormatId:
        return getGeoLineStringWorker()->items_count;
      case GeoPolygonFormatId:
        return getGeoPolygonWorker()->items_count;
      default:
        break;
    }
    static int64_t dummy = -1;
    return dummy;
  }

  // Return the number of specified blocks
  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t& get_storage_count2() {
    switch (format()) {
      case GeoPolygonFormatId:
        return getGeoPolygonWorker()->items2_count;
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
      default:
        break;
    }
    static int64_t dummy_storage_count = -1;
    return dummy_storage_count;
  }

  // To be deprecated in favor of NestedArray format
  inline const int64_t& get_storage_count2() const {
    switch (format()) {
      case GeoPolygonFormatId:
        return getGeoPolygonWorker()->items2_count;
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
      default:
        break;
    }
    static int64_t dummy_storage_count = -1;
    return dummy_storage_count;
  }

  // Return the size of values buffer in bytes
  // To be deprecated in favor of NestedArray format
  inline int64_t get_values_buffer_size() const {
    switch (format()) {
      case VarlenArrayFormatId: {
        const auto* worker = getVarlenArrayWorker();
        return worker->compressed_indices_offset - worker->values_offset;
      }
      case GeoPointFormatId: {
        const auto* metadata = getGeoPointMetadata();
        const auto itemsize =
            2 * (metadata->is_geoint ? sizeof(int32_t) : sizeof(double));
        return _align_to_int64(itemsize * metadata->total_items_count);
      }
      case GeoLineStringFormatId: {
        const auto* worker = getGeoLineStringWorker();
        return worker->compressed_indices_offset - worker->values_offset;
      }
      case GeoPolygonFormatId: {
        const auto* worker = getGeoPolygonWorker();
        return worker->compressed_indices2_offset - worker->values_offset;
      }
      default:
        break;
    }
    static int64_t dummy = -1;
    return dummy;
  }

  // Return the size of compressed_indices2 buffer in bytes
  // To be deprecated in favor of NestedArray format
  inline int64_t get_compressed_indices2_buffer_size() const {
    switch (format()) {
      case GeoPolygonFormatId: {
        const auto* worker = getGeoPolygonWorker();
        return worker->compressed_indices_offset - worker->compressed_indices2_offset;
      }
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
      default:
        break;
    }
    static int64_t dummy = -1;
    return dummy;
  }

  // Return the pointer to values buffer
  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int8_t* get_values() {
    int64_t offset = 0;
    switch (format()) {
      case VarlenArrayFormatId:
        offset = getVarlenArrayWorker()->values_offset;
        break;
      case GeoPointFormatId:
        offset = getGeoPointWorker()->values_offset;
        break;
      case GeoLineStringFormatId:
        offset = getGeoLineStringWorker()->values_offset;
        break;
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->values_offset;
        break;
      default:
        return nullptr;
    }
    return buffer + offset;
  }

  // To be deprecated in favor of NestedArray format
  inline const int8_t* get_values() const {
    int64_t offset = 0;
    switch (format()) {
      case VarlenArrayFormatId:
        offset = getVarlenArrayWorker()->values_offset;
        break;
      case GeoPointFormatId:
        offset = getGeoPointWorker()->values_offset;
        break;
      case GeoLineStringFormatId:
        offset = getGeoLineStringWorker()->values_offset;
        break;
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->values_offset;
        break;
      default:
        return nullptr;
    }
    return buffer + offset;
  }

  // Return the pointer to counts2 buffer
  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int32_t* get_counts2() {
    int64_t offset = 0;
    switch (format()) {
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->counts2_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int32_t*>(buffer + offset);
  }

  // To be deprecated in favor of NestedArray format
  inline const int32_t* get_counts2() const {
    int64_t offset = 0;
    switch (format()) {
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->counts2_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int32_t*>(buffer + offset);
  }

  // Return the pointer to compressed indices2 buffer
  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t* get_compressed_indices2() {
    int64_t offset = 0;
    switch (format()) {
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->compressed_indices2_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int64_t*>(buffer + offset);
  }

  // To be deprecated in favor of NestedArray format
  inline const int64_t* get_compressed_indices2() const {
    int64_t offset = 0;
    switch (format()) {
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->compressed_indices2_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int64_t*>(buffer + offset);
  }

  // Return the pointer to compressed indices buffer
  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t* get_compressed_indices() {
    int64_t offset = 0;
    switch (format()) {
      case VarlenArrayFormatId:
        offset = getVarlenArrayWorker()->compressed_indices_offset;
        break;
      case GeoLineStringFormatId:
        offset = getGeoLineStringWorker()->compressed_indices_offset;
        break;
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->compressed_indices_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int64_t*>(buffer + offset);
  }

  // To be deprecated in favor of NestedArray format
  inline const int64_t* get_compressed_indices() const {
    int64_t offset = 0;
    switch (format()) {
      case VarlenArrayFormatId:
        offset = getVarlenArrayWorker()->compressed_indices_offset;
        break;
      case GeoLineStringFormatId:
        offset = getGeoLineStringWorker()->compressed_indices_offset;
        break;
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->compressed_indices_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int64_t*>(buffer + offset);
  }

#define FLATBUFFER_GET_BUFFER_METHODS(BUFFERNAME, BUFFERTYPE) \
  HOST DEVICE inline BUFFERTYPE* get_##BUFFERNAME() {         \
    int64_t offset = 0;                                       \
    switch (format()) {                                       \
      case NestedArrayFormatId:                               \
        offset = getNestedArrayWorker()->BUFFERNAME##_offset; \
        break;                                                \
      default:                                                \
        return nullptr;                                       \
    }                                                         \
    return reinterpret_cast<BUFFERTYPE*>(buffer + offset);    \
  }                                                           \
  inline const BUFFERTYPE* get_##BUFFERNAME() const {         \
    int64_t offset = 0;                                       \
    switch (format()) {                                       \
      case NestedArrayFormatId:                               \
        offset = getNestedArrayWorker()->BUFFERNAME##_offset; \
        break;                                                \
      default:                                                \
        return nullptr;                                       \
    }                                                         \
    return reinterpret_cast<BUFFERTYPE*>(buffer + offset);    \
  }

  FLATBUFFER_GET_BUFFER_METHODS(user_data_buffer, int8_t);
  FLATBUFFER_GET_BUFFER_METHODS(values_buffer, int8_t);
  FLATBUFFER_GET_BUFFER_METHODS(sizes_buffer, sizes_t);
  FLATBUFFER_GET_BUFFER_METHODS(values_offsets, offsets_t);
  FLATBUFFER_GET_BUFFER_METHODS(sizes_offsets, offsets_t);

#undef FLATBUFFER_GET_BUFFER_METHODS

  inline const int8_t* getNullValuePtr() const {
    if (isNestedArray()) {
      return get_values_buffer() + getValuesBufferSize();
    }
    return nullptr;
  }

  inline bool containsNullValue(const int8_t* value_ptr) const {
    const int8_t* null_value_ptr = getNullValuePtr();
    if (null_value_ptr != nullptr) {
      switch (getValueSize()) {
        case 1:
          return *null_value_ptr == *value_ptr;
        case 2:
          return *reinterpret_cast<const int16_t*>(null_value_ptr) ==
                 *reinterpret_cast<const int16_t*>(value_ptr);
        case 4:
          return *reinterpret_cast<const int32_t*>(null_value_ptr) ==
                 *reinterpret_cast<const int32_t*>(value_ptr);
        case 8:
          return *reinterpret_cast<const int64_t*>(null_value_ptr) ==
                 *reinterpret_cast<const int64_t*>(value_ptr);
        case 16:
          return (*reinterpret_cast<const int64_t*>(null_value_ptr) ==
                      *reinterpret_cast<const int64_t*>(value_ptr) &&
                  *(reinterpret_cast<const int64_t*>(null_value_ptr) + 1) ==
                      *(reinterpret_cast<const int64_t*>(value_ptr) + 1));
        default:
          break;
      }
    }
    return false;
  }

  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t* get_storage_indices() {
    int64_t offset = 0;
    switch (format()) {
      case VarlenArrayFormatId:
        offset = getVarlenArrayWorker()->storage_indices_offset;
        break;
      case GeoLineStringFormatId:
        offset = getGeoLineStringWorker()->storage_indices_offset;
        break;
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->storage_indices_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int64_t*>(buffer + offset);
  }

  // To be deprecated in favor of NestedArray format
  inline const int64_t* get_storage_indices() const {
    int64_t offset = 0;
    switch (format()) {
      case VarlenArrayFormatId:
        offset = getVarlenArrayWorker()->storage_indices_offset;
        break;
      case GeoLineStringFormatId:
        offset = getGeoLineStringWorker()->storage_indices_offset;
        break;
      case GeoPolygonFormatId:
        offset = getGeoPolygonWorker()->storage_indices_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int64_t*>(buffer + offset);
  }

  // TODO: rename to get_storage_indices
  HOST DEVICE inline sizes_t* get_storage_indices_new() {
    offsets_t offset = 0;
    switch (format()) {
      case NestedArrayFormatId:
        offset = getNestedArrayWorker()->storage_indices_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<sizes_t*>(buffer + offset);
  }

  inline const sizes_t* get_storage_indices_new() const {
    offsets_t offset = 0;
    switch (format()) {
      case NestedArrayFormatId:
        offset = getNestedArrayWorker()->storage_indices_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<sizes_t*>(buffer + offset);
  }

  Status getItemPrepare(const int64_t index, const size_t ndims) const {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    if (format() != NestedArrayFormatId) {
      RETURN_ERROR(NotSupportedFormatError);
    }
    if (getDimensions() != ndims) {
      RETURN_ERROR(DimensionalityError);
    }
    return Success;
  }

  inline sizes_t get_storage_index(const int64_t index) const {
    return get_storage_indices_new()[index];
  }

  // High-level API

  Status getLength(const int64_t index, size_t& length) const {
    const size_t ndims = getDimensions();
    Status status = getItemPrepare(index, ndims);
    if (status != Success) {
      return status;
    }
    const auto storage_index = get_storage_index(index);
    if (ndims == 3) {
      const auto* values_offsets = get_values_offsets();
      const auto values_offset = values_offsets[storage_index];
      if (values_offset < 0) {
        length = 0;
        return Success;
      }
      const auto* sizes_offsets = get_sizes_offsets();
      const auto* sizes_buffer = get_sizes_buffer();
      const auto sizes_offset = sizes_offsets[storage_index * ndims];
      length = sizes_buffer[sizes_offset];
    } else {
      RETURN_ERROR(NotImplementedError);
    }

    return Success;
  }

  Status getLength(const int64_t index, const int64_t subindex, size_t& length) const {
    const size_t ndims = getDimensions();
    Status status = getItemPrepare(index, ndims);
    if (status != Success) {
      return status;
    }
    const auto storage_index = get_storage_index(index);
    if (ndims == 3) {
      const auto* values_offsets = get_values_offsets();
      const auto values_offset = values_offsets[storage_index];
      if (values_offset < 0) {
        RETURN_ERROR(IndexError);
      }
      const auto* sizes_offsets = get_sizes_offsets();
      const auto* sizes_buffer = get_sizes_buffer();
      const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
      length = (sizes_buffer + sizes2_offset)[subindex];
    } else {
      RETURN_ERROR(NotImplementedError);
    }
    return Success;
  }

  Status getLength(const int64_t index,
                   const int64_t subindex,
                   const int64_t subsubindex,
                   size_t& length) const {
    const size_t ndims = getDimensions();
    Status status = getItemPrepare(index, ndims);
    if (status != Success) {
      return status;
    }
    const auto storage_index = get_storage_index(index);
    if (ndims == 3) {
      const auto* values_offsets = get_values_offsets();
      const auto values_offset = values_offsets[storage_index];
      if (values_offset < 0) {
        RETURN_ERROR(IndexError);
      }
      const auto* sizes_offsets = get_sizes_offsets();
      const auto* sizes_buffer = get_sizes_buffer();
      const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
      const auto sizes3_offset = sizes_offsets[storage_index * ndims + 2];
      offsets_t soffset = 0;
      for (int64_t i = 0; i < subindex; i++) {
        soffset += (sizes_buffer + sizes2_offset)[i];
      }
      length = (sizes_buffer + sizes3_offset + soffset)[subsubindex];
    } else {
      RETURN_ERROR(NotImplementedError);
    }
    return Success;
  }

  // TODO: parametrize sizes type
  template <typename CT>
  Status getItem(const int64_t index,
                 std::vector<CT>& values,
                 std::vector<int32_t>& sizes,
                 std::vector<int32_t>& sizes_of_sizes,
                 bool& is_null) {
    if constexpr (!std::is_same<CT, uint8_t>::value) {
      if constexpr (std::is_same<CT, double>::value) {
        const auto* metadata = getNestedArrayMetadata();
        if (metadata->value_type != PointFloat64) {
          RETURN_ERROR(TypeError);
        }
      } else if constexpr (std::is_same<CT, int32_t>::value) {
        const auto* metadata = getNestedArrayMetadata();
        if (metadata->value_type != PointInt32) {
          RETURN_ERROR(TypeError);
        }
      } else {
        RETURN_ERROR(NotImplementedError);
      }
    }
    int32_t nof_values;
    int8_t* values_ptr;
    int32_t nof_sizes;
    int8_t* sizes_ptr;
    int32_t nof_sizes_of_sizes;
    int8_t* sizes_of_sizes_ptr;

    Status status = getItem(index,
                            nof_values,
                            values_ptr,
                            nof_sizes,
                            sizes_ptr,
                            nof_sizes_of_sizes,
                            sizes_of_sizes_ptr,
                            is_null);

    if (status != Success) {
      return status;
    }
    if (is_null) {
      return Success;
    }
    const auto valuesize = getValueSize();
    const auto values_count = nof_values * valuesize / sizeof(CT);
    values.reserve(values_count);
    values.insert(values.end(),
                  reinterpret_cast<CT*>(values_ptr),
                  reinterpret_cast<CT*>(values_ptr) + values_count);

    sizes.reserve(nof_sizes);
    sizes.insert(sizes.end(),
                 reinterpret_cast<sizes_t*>(sizes_ptr),
                 reinterpret_cast<sizes_t*>(sizes_ptr + nof_sizes * sizeof(sizes_t)));

    sizes_of_sizes.reserve(nof_sizes_of_sizes);
    sizes_of_sizes.insert(sizes_of_sizes.end(),
                          reinterpret_cast<sizes_t*>(sizes_of_sizes_ptr),
                          reinterpret_cast<sizes_t*>(
                              sizes_of_sizes_ptr + nof_sizes_of_sizes * sizeof(sizes_t)));

    return Success;
  }

  template <typename CT>
  Status getItem(const int64_t index,
                 const int64_t subindex,
                 std::vector<CT>& values,
                 std::vector<int32_t>& sizes,
                 bool& is_null) {
    if constexpr (!std::is_same<CT, uint8_t>::value) {
      if constexpr (std::is_same<CT, double>::value) {
        const auto* metadata = getNestedArrayMetadata();
        if (metadata->value_type != PointFloat64) {
          RETURN_ERROR(TypeError);
        }
      } else if constexpr (std::is_same<CT, int32_t>::value) {
        const auto* metadata = getNestedArrayMetadata();
        if (metadata->value_type != PointInt32) {
          RETURN_ERROR(TypeError);
        }
      } else {
        RETURN_ERROR(NotImplementedError);
      }
    }
    int32_t nof_values;
    int8_t* values_ptr;
    int32_t nof_sizes;
    int8_t* sizes_ptr;

    Status status =
        getItem(index, subindex, nof_values, values_ptr, nof_sizes, sizes_ptr, is_null);

    if (status != Success) {
      return status;
    }
    if (is_null) {
      return Success;
    }
    const auto valuesize = getValueSize();
    const auto values_count = nof_values * valuesize / sizeof(CT);
    values.reserve(values_count);
    values.insert(values.end(),
                  reinterpret_cast<CT*>(values_ptr),
                  reinterpret_cast<CT*>(values_ptr) + values_count);

    sizes.reserve(nof_sizes);
    sizes.insert(sizes.end(),
                 reinterpret_cast<sizes_t*>(sizes_ptr),
                 reinterpret_cast<sizes_t*>(sizes_ptr + nof_sizes * sizeof(sizes_t)));

    return Success;
  }

  template <typename CT>
  Status getItem(const int64_t index,
                 const int64_t subindex,
                 const int64_t subsubindex,
                 std::vector<CT>& values,
                 bool& is_null) {
    if constexpr (!std::is_same<CT, uint8_t>::value) {
      if constexpr (std::is_same<CT, double>::value) {
        const auto* metadata = getNestedArrayMetadata();
        if (metadata->value_type != PointFloat64) {
          RETURN_ERROR(TypeError);
        }
      } else if constexpr (std::is_same<CT, int32_t>::value) {
        const auto* metadata = getNestedArrayMetadata();
        if (metadata->value_type != PointInt32) {
          RETURN_ERROR(TypeError);
        }
      } else {
        RETURN_ERROR(NotImplementedError);
      }
    }
    int32_t nof_values;
    int8_t* values_ptr;

    Status status =
        getItem(index, subindex, subsubindex, nof_values, values_ptr, is_null);
    if (status != Success) {
      return status;
    }
    if (is_null) {
      return Success;
    }
    const auto valuesize = getValueSize();
    const auto values_count = nof_values * valuesize / sizeof(CT);
    values.reserve(values_count);
    values.insert(values.end(),
                  reinterpret_cast<CT*>(values_ptr),
                  reinterpret_cast<CT*>(values_ptr) + values_count);
    return Success;
  }

  Status getItem(const int64_t index,
                 const int64_t subindex,
                 int32_t& nof_values,
                 int8_t*& values,
                 int32_t& nof_sizes,
                 int8_t*& sizes,
                 bool& is_null) {
    const size_t ndims = 3;
    Status status = getItemPrepare(index, ndims);
    if (status != Success) {
      return status;
    }
    const auto storage_index = get_storage_index(index);
    auto* values_offsets = get_values_offsets();
    auto values_offset = values_offsets[storage_index];
    if (values_offset < 0) {
      is_null = true;
      nof_values = 0;
      nof_sizes = 0;
      values = nullptr;
      sizes = nullptr;
      return Success;
    }

    const auto* sizes_offsets = get_sizes_offsets();
    auto* sizes_buffer = get_sizes_buffer();
    auto* values_buffer = get_values_buffer();
    const auto valuesize = getValueSize();
    const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
    const auto sizes3_offset = sizes_offsets[storage_index * ndims + 2];

    is_null = false;
    offsets_t soffset = 0;
    for (int64_t i = 0; i < subindex; i++) {
      soffset += (sizes_buffer + sizes2_offset)[i];
    }
    nof_sizes = (sizes_buffer + sizes2_offset)[subindex];
    values = values_buffer + (values_offset + soffset) * valuesize;
    sizes = reinterpret_cast<int8_t*>(sizes_buffer + sizes3_offset + soffset);
    nof_values = 0;
    for (int64_t i = 0; i < nof_sizes; i++) {
      nof_values += sizes[i];
    }
    return Success;
  }

  Status getItem(const int64_t index,
                 const int64_t subindex,
                 const int64_t subsubindex,
                 int32_t& nof_values,
                 int8_t*& values,
                 bool& is_null) {
    const size_t ndims = 3;
    Status status = getItemPrepare(index, ndims);
    if (status != Success) {
      return status;
    }
    const auto storage_index = get_storage_index(index);
    auto* values_offsets = get_values_offsets();
    auto values_offset = values_offsets[storage_index];
    if (values_offset < 0) {
      is_null = true;
      nof_values = 0;
      values = nullptr;
      return Success;
    }

    const auto* sizes_offsets = get_sizes_offsets();
    auto* sizes_buffer = get_sizes_buffer();
    auto* values_buffer = get_values_buffer();
    const auto valuesize = getValueSize();
    const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
    const auto sizes3_offset = sizes_offsets[storage_index * ndims + 2];

    is_null = false;
    offsets_t soffset = 0;
    for (int64_t i = 0; i < subindex; i++) {
      soffset += (sizes_buffer + sizes2_offset)[i];
    }
    offsets_t soffset2 = 0;
    for (int64_t i = 0; i < subsubindex; i++) {
      soffset2 += (sizes_buffer + sizes3_offset + soffset)[i];
    }
    values = values_buffer + (values_offset + soffset + soffset2) * valuesize;
    nof_values = (sizes_buffer + sizes3_offset + soffset)[subsubindex];
    return Success;
  }

  Status getItem(const int64_t index,
                 int32_t& nof_values,
                 int8_t*& values,
                 int32_t& nof_sizes,
                 int8_t*& sizes,
                 int32_t& nof_sizes_of_sizes,
                 int8_t*& sizes_of_sizes,
                 bool& is_null) {
    const size_t ndims = 3;
    Status status = getItemPrepare(index, ndims);
    if (status != Success) {
      return status;
    }
    const auto storage_index = get_storage_index(index);

    auto* values_offsets = get_values_offsets();
    auto values_offset = values_offsets[storage_index];
    if (values_offset < 0) {
      is_null = true;
      nof_values = 0;
      nof_sizes = 0;
      nof_sizes_of_sizes = 0;
      values = nullptr;
      sizes = nullptr;
      sizes_of_sizes = nullptr;
      return Success;
    }

    const auto* sizes_offsets = get_sizes_offsets();
    auto* sizes_buffer = get_sizes_buffer();
    auto* values_buffer = get_values_buffer();
    const auto valuesize = getValueSize();
    const auto sizes_offset = sizes_offsets[storage_index * ndims];
    const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
    const auto sizes3_offset = sizes_offsets[storage_index * ndims + 2];
    const auto next_values_offset = values_offsets[storage_index + 1];

    is_null = false;
    if (next_values_offset < 0) {
      nof_values = -(next_values_offset + 1) - values_offset;
    } else {
      nof_values = next_values_offset - values_offset;
    }
    values = values_buffer + values_offset * valuesize;

    nof_sizes_of_sizes = sizes_buffer[sizes_offset];
    sizes_of_sizes = reinterpret_cast<int8_t*>(sizes_buffer + sizes2_offset);
    sizes = reinterpret_cast<int8_t*>(sizes_buffer + sizes3_offset);
    nof_sizes = 0;
    for (int32_t i = 0; i < nof_sizes_of_sizes; i++) {
      nof_sizes += (sizes_buffer + sizes2_offset)[i];
    }
    return Success;
  }

  Status setItemPrepare(const int64_t index, const size_t ndims) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    if (format() != NestedArrayFormatId) {
      RETURN_ERROR(NotSupportedFormatError);
    }
    if (getDimensions() != ndims) {
      RETURN_ERROR(DimensionalityError);
    }
    auto* storage_indices = get_storage_indices_new();
    if (storage_indices[index] >= 0) {
      RETURN_ERROR(ItemAlreadySpecifiedError);
    }
    auto* worker = getNestedArrayWorker();
    auto storage_index = worker->specified_items_count;
    storage_indices[index] = storage_index;
    worker->specified_items_count++;
    return Success;
  }

  // TODO: rename to setNull
  Status setNullNew(int64_t index) {
    const size_t ndims = getDimensions();
    Status status = setItemPrepare(index, ndims);
    if (status != Success) {
      RETURN_ERROR(status);
    }
    const auto storage_index = get_storage_index(index);

    auto* values_offsets = get_values_offsets();
    auto* sizes_offsets = get_sizes_offsets();

    const auto values_offset = values_offsets[storage_index];
    const auto sizes_offset = sizes_offsets[storage_index * ndims];
    auto* sizes_buffer = get_sizes_buffer();

    sizes_buffer[sizes_offset] = 0;
    for (size_t i = 0; i < ndims; i++) {
      sizes_offsets[storage_index * ndims + i + 1] = sizes_offset + 1;
    }
    values_offsets[storage_index] = -(values_offset + 1);
    values_offsets[storage_index + 1] = values_offset;
    return Success;
  }

  Status setItem(const int64_t index, std::vector<double>& arr) {
    const size_t ndims = 1;
    Status status = setItemPrepare(index, ndims);
    if (status != Success) {
      return status;
    }
    const auto storage_index = get_storage_index(index);

    auto* values_offsets = get_values_offsets();
    auto* sizes_offsets = get_sizes_offsets();
    auto* sizes_buffer = get_sizes_buffer();
    auto* values_buffer = get_values_buffer();
    const auto* metadata = getNestedArrayMetadata();
    const auto valuesize = getValueSize();

    auto values_offset = values_offsets[storage_index];
    const auto sizes_offset = sizes_offsets[storage_index * ndims];

    sizes_t sz = (arr.size() * sizeof(double)) / valuesize;
    sizes_buffer[sizes_offset] = sz;

    if (values_offset + sz > metadata->total_values_count) {
      RETURN_ERROR(ValuesBufferTooSmallError);
    }
    if (memcpy(values_buffer + values_offset * valuesize, arr.data(), sz * valuesize) ==
        nullptr) {
      RETURN_ERROR(MemoryError);
    }
    values_offset += sz;

    sizes_offsets[storage_index * ndims + 1] = sizes_offset + 1;
    values_offsets[storage_index + 1] = values_offset;
    return Success;
  }

  Status setItem(const int64_t index, const std::vector<std::vector<double>>& item) {
    const size_t ndims = 2;
    Status status = setItemPrepare(index, ndims);
    if (status != Success) {
      return status;
    }
    const auto storage_index = get_storage_index(index);

    auto* values_offsets = get_values_offsets();
    auto* sizes_offsets = get_sizes_offsets();
    auto* sizes_buffer = get_sizes_buffer();
    auto* values_buffer = get_values_buffer();
    const auto* metadata = getNestedArrayMetadata();
    const auto valuesize = getValueSize();
    const sizes_t size = item.size();

    auto values_offset = values_offsets[storage_index];
    const auto sizes_offset = sizes_offsets[storage_index * ndims];
    const auto sizes2_offset = sizes_offset + 1;
    if (sizes2_offset + size > metadata->total_sizes_count) {
      RETURN_ERROR(SizesBufferTooSmallError);
    }
    sizes_buffer[sizes_offset] = size;
    for (sizes_t i = 0; i < size; i++) {
      std::vector<double> arr = item[i];
      sizes_t sz = (arr.size() * sizeof(double)) / valuesize;
      sizes_buffer[sizes2_offset + i] = sz;
      if (values_offset + sz > metadata->total_values_count) {
        RETURN_ERROR(ValuesBufferTooSmallError);
      }
      if (memcpy(values_buffer + values_offset * valuesize, arr.data(), sz * valuesize) ==
          nullptr) {
        RETURN_ERROR(MemoryError);
      }
      values_offset += sz;
    }
    sizes_offsets[storage_index * ndims + 1] = sizes2_offset;
    sizes_offsets[storage_index * ndims + 2] = sizes2_offset + size;
    values_offsets[storage_index + 1] = values_offset;
    return Success;
  }

  template <typename CT>
  Status setItem(const int64_t index,
                 const std::vector<std::vector<std::vector<CT>>>& item) {
    const size_t ndims = 3;
    Status status = setItemPrepare(index, ndims);
    if (status != Success) {
      return status;
    }
    const auto storage_index = get_storage_index(index);

    auto* values_offsets = get_values_offsets();
    auto* sizes_offsets = get_sizes_offsets();
    auto* sizes_buffer = get_sizes_buffer();
    auto* values_buffer = get_values_buffer();
    const auto* metadata = getNestedArrayMetadata();
    const auto valuesize = getValueSize();
    const sizes_t size = item.size();

    auto values_offset = values_offsets[storage_index];
    const auto sizes_offset = sizes_offsets[storage_index * ndims];
    const auto sizes2_offset = sizes_offset + 1;
    const auto sizes3_offset = sizes2_offset + size;
    sizes_t i3 = 0;
    sizes_buffer[sizes_offset] = size;

    for (sizes_t i = 0; i < size; i++) {
      const std::vector<std::vector<CT>>& item2 = item[i];
      sizes_t size2 = item2.size();
      if (sizes3_offset + i3 + size2 > metadata->total_sizes_count) {
        RETURN_ERROR(SizesBufferTooSmallError);
      }
      sizes_buffer[sizes2_offset + i] = size2;
      for (sizes_t j = 0; j < size2; j++) {
        const std::vector<CT>& arr = item2[j];
        sizes_t sz = (arr.size() * sizeof(CT)) / valuesize;
        sizes_buffer[sizes3_offset + i3] = sz;
        i3 += 1;
        if (values_offset + sz > metadata->total_values_count) {
          RETURN_ERROR(ValuesBufferTooSmallError);
        }
        if (memcpy(values_buffer + values_offset * valuesize,
                   arr.data(),
                   sz * valuesize) == nullptr) {
          RETURN_ERROR(MemoryError);
        }
        values_offset += sz;
      }
    }
    sizes_offsets[storage_index * ndims + 1] = sizes2_offset;
    sizes_offsets[storage_index * ndims + 2] = sizes3_offset;
    sizes_offsets[storage_index * ndims + 3] = sizes3_offset + i3;
    values_offsets[storage_index + 1] = values_offset;
    return Success;
  }

  template <typename CT>
  Status setItem(const int64_t index,
                 const std::vector<CT>& values,
                 const std::vector<int32_t>& sizes,
                 const std::vector<int32_t>& sizes_of_sizes) {
    const auto* metadata = getNestedArrayMetadata();
    if constexpr (!std::is_same<CT, uint8_t>::value) {
      if constexpr (std::is_same<CT, double>::value) {
        if (metadata->value_type != PointFloat64) {
          RETURN_ERROR(TypeError);
        }
      } else if constexpr (std::is_same<CT, int32_t>::value) {
        if (metadata->value_type != PointInt32) {
          RETURN_ERROR(TypeError);
        }
      } else {
        RETURN_ERROR(NotImplementedError);
      }
    }
    return setItem(index,
                   reinterpret_cast<const int8_t*>(values.data()),
                   values.size() * sizeof(CT),
                   sizes.data(),
                   sizes.size(),
                   sizes_of_sizes.data(),
                   sizes_of_sizes.size());
  }

  Status setItem(const int64_t index,
                 const int8_t* values_buf,
                 const size_t values_buf_size,  // in bytes
                 const int32_t* sizes_buf,
                 const int32_t nof_sizes,
                 const int32_t* sizes_of_sizes_buf,
                 const int32_t nof_sizes_of_sizes) {
    const size_t ndims = 3;
    Status status = setItemPrepare(index, ndims);
    if (status != Success) {
      RETURN_ERROR(status);
    }
    const auto* metadata = getNestedArrayMetadata();
    const auto storage_index = get_storage_index(index);

    auto* values_offsets = get_values_offsets();
    auto* sizes_offsets = get_sizes_offsets();
    auto* sizes_buffer = get_sizes_buffer();
    auto* values_buffer = get_values_buffer();
    const auto valuesize = getValueSize();
    auto values_offset = values_offsets[storage_index];
    const auto sizes_offset = sizes_offsets[storage_index * ndims];
    const auto sizes2_offset = sizes_offset + 1;
    const auto sizes3_offset = sizes2_offset + nof_sizes_of_sizes;
    if (sizes_offset + 1 + nof_sizes_of_sizes + nof_sizes > metadata->total_sizes_count) {
      RETURN_ERROR(SizesBufferTooSmallError);
    }
    sizes_t sum_sizes_of_sizes = 0;
    for (sizes_t i = 0; i < nof_sizes_of_sizes; i++) {
      sum_sizes_of_sizes += sizes_of_sizes_buf[i];
    }
    if (sum_sizes_of_sizes != nof_sizes) {
      RETURN_ERROR(InconsistentSizesError);
    }
    sizes_t sum_sizes = 0;
    for (sizes_t i = 0; i < nof_sizes; i++) {
      sum_sizes += sizes_buf[i];
    }
    sizes_t values_count = values_buf_size / valuesize;
    if (sum_sizes != values_count) {
      RETURN_ERROR(InconsistentSizesError);
    }
    if (values_offset + values_count > metadata->total_values_count) {
      RETURN_ERROR(ValuesBufferTooSmallError);
    }
    sizes_buffer[sizes_offset] = nof_sizes_of_sizes;
    if (memcpy(sizes_buffer + sizes2_offset,
               sizes_of_sizes_buf,
               nof_sizes_of_sizes * sizeof(int32_t)) == nullptr) {
      RETURN_ERROR(MemoryError);
    }
    if (memcpy(sizes_buffer + sizes3_offset, sizes_buf, nof_sizes * sizeof(int32_t)) ==
        nullptr) {
      RETURN_ERROR(MemoryError);
    }
    if (memcpy(values_buffer + values_offset * valuesize, values_buf, values_buf_size) ==
        nullptr) {
      RETURN_ERROR(MemoryError);
    }
    sizes_offsets[storage_index * ndims + 1] = sizes2_offset;
    sizes_offsets[storage_index * ndims + 2] = sizes3_offset;
    sizes_offsets[storage_index * ndims + 3] = sizes3_offset + sum_sizes_of_sizes;
    values_offsets[storage_index + 1] = values_offset + values_count;
    return Success;
  }

  // Set a new item with index and size (in bytes) and initialize its
  // elements from source buffer. The item values will be
  // uninitialized when source buffer is nullptr. If dest != nullptr
  // then the item's buffer pointer will be stored in *dest.
  // To be deprecated in favor of NestedArray format
  Status setItem(const int64_t index,
                 const int8_t* src,
                 const int64_t size,
                 int8_t** dest = nullptr) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId: {
        int64_t& storage_count = get_storage_count();
        int64_t* compressed_indices = get_compressed_indices();
        int64_t* storage_indices = get_storage_indices();
        const int64_t itemsize = dtypeSize();
        if (size % itemsize != 0) {
          return SizeError;  // size must be multiple of itemsize. Perhaps size is not in
          // bytes?
        }
        if (storage_indices[index] >= 0) {
          RETURN_ERROR(ItemAlreadySpecifiedError);
        }
        const int64_t cindex = compressed_indices[storage_count];
        const int64_t values_buffer_size = get_values_buffer_size();
        const int64_t csize = cindex * itemsize;
        if (csize + size > values_buffer_size) {
          RETURN_ERROR(ValuesBufferTooSmallError);
        }
        break;
      }
      case GeoPointFormatId: {
        const int64_t itemsize = dtypeSize();
        if (size != itemsize) {
          RETURN_ERROR(SizeError);
        }
        break;
      }
      case GeoPolygonFormatId: {
        const int64_t itemsize = dtypeSize();
        const int32_t counts = size / itemsize;
        return setItemCountsAndData(index, &counts, 1, src, dest);
      }
      default:
        RETURN_ERROR(UnknownFormatError);
    }
    return setItemNoValidation(index, src, size, dest);
  }

  // Same as setItem but performs no input validation
  // To be deprecated in favor of NestedArray format
  Status setItemNoValidation(const int64_t index,
                             const int8_t* src,
                             const int64_t size,
                             int8_t** dest) {
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId: {
        int64_t& storage_count = get_storage_count();
        int64_t* storage_indices = get_storage_indices();
        int64_t* compressed_indices = get_compressed_indices();
        int8_t* values = get_values();
        const int64_t itemsize = dtypeSize();
        const int64_t values_count = size / itemsize;
        const int64_t cindex = compressed_indices[storage_count];
        const int64_t csize = cindex * itemsize;
        storage_indices[index] = storage_count;
        compressed_indices[storage_count + 1] = cindex + values_count;
        if (size > 0 && src != nullptr && memcpy(values + csize, src, size) == nullptr) {
          RETURN_ERROR(MemoryError);
        }
        if (dest != nullptr) {
          *dest = values + csize;
        }
        storage_count++;
        break;
      }
      case GeoPointFormatId: {
        int8_t* values = get_values();
        const int64_t itemsize = dtypeSize();
        const int64_t csize = index * itemsize;
        if (src != nullptr && memcpy(values + csize, src, size) == nullptr) {
          RETURN_ERROR(MemoryError);
        }
        if (dest != nullptr) {
          *dest = values + csize;
        }
        break;
      }
      case GeoPolygonFormatId: {
        const int64_t itemsize = dtypeSize();
        const int32_t counts = size / itemsize;
        return setItemCountsAndDataNoValidation(index, &counts, 1, src, dest);
      }
      default:
        RETURN_ERROR(UnknownFormatError);
    }

    return Success;
  }

  // To be deprecated in favor of NestedArray format
  Status setItemCountsAndData(const int64_t index,
                              const int32_t* counts,
                              const int64_t nof_counts,
                              const int8_t* src,
                              int8_t** dest = nullptr) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        RETURN_ERROR(NotSupportedFormatError);
      case GeoPolygonFormatId: {
        const int64_t& storage_count = get_storage_count();
        const int64_t& storage_count2 = get_storage_count2();
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t* compressed_indices2 = get_compressed_indices2();
        const int64_t* storage_indices = get_storage_indices();
        const int64_t valuesize = dtypeSize();
        if (storage_indices[index] >= 0) {
          RETURN_ERROR(ItemAlreadySpecifiedError);
        }

        const int64_t compressed_indices2_buffer_size =
            get_compressed_indices2_buffer_size();
        if (compressed_indices[storage_count] + nof_counts >
            compressed_indices2_buffer_size) {
          RETURN_ERROR(CompressedIndices2BufferTooSmallError);
        }

        const int64_t offset = compressed_indices2[storage_count2] * valuesize;
        int64_t size = 0;
        for (int i = 0; i < nof_counts; i++) {
          size += valuesize * counts[i];
        }
        const int64_t values_buffer_size = get_values_buffer_size();
        if (offset + size > values_buffer_size) {
          RETURN_ERROR(ValuesBufferTooSmallError);
        }
        break;
      }
      default:
        RETURN_ERROR(UnknownFormatError);
    }
    return setItemCountsAndDataNoValidation(index, counts, nof_counts, src, dest);
  }

  // To be deprecated in favor of NestedArray format
  // Same as setItem but performs no input validation
  Status setItemCountsAndDataNoValidation(
      const int64_t index,
      const int32_t* counts,     // counts of points in rings
      const int64_t nof_counts,  // nof rings
      const int8_t* src,         // coordinates of points
      int8_t** dest) {
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        RETURN_ERROR(NotSupportedFormatError);
      case GeoPolygonFormatId: {
        int64_t& storage_count = get_storage_count();
        int64_t& storage_count2 = get_storage_count2();
        int64_t* storage_indices = get_storage_indices();
        int64_t* compressed_indices = get_compressed_indices();
        int32_t* counts2 = get_counts2();
        int64_t* compressed_indices2 = get_compressed_indices2();
        int8_t* values = get_values();
        const int64_t valuesize = dtypeSize();
        storage_indices[index] = storage_count;

        compressed_indices[storage_count + 1] =
            compressed_indices[storage_count] + nof_counts;

        int64_t cindex2 = compressed_indices2[storage_count2];
        const int64_t offset = cindex2 * valuesize;
        int64_t size = 0;
        for (int i = 0; i < nof_counts; i++) {
          size += valuesize * counts[i];
          cindex2 += counts[i];
          counts2[storage_count2] = counts[i];
          storage_count2++;
          compressed_indices2[storage_count2] = cindex2;
        }
        if (size > 0 && src != nullptr && memcpy(values + offset, src, size) == nullptr) {
          RETURN_ERROR(MemoryError);
        }
        if (dest != nullptr) {
          *dest = values + offset;
        }
        storage_count++;
        break;
      }
      default:
        RETURN_ERROR(UnknownFormatError);
    }

    return Success;
  }

  // To be deprecated in favor of NestedArray format
  Status setSubItem(const int64_t index,
                    const int64_t subindex,
                    const int8_t* src,
                    const int64_t size,
                    int8_t** dest = nullptr) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        return NotSupportedFormatError;
      case GeoPolygonFormatId: {
        const int64_t* storage_indices = get_storage_indices();
        const int64_t storage_index = storage_indices[index];
        if (storage_index < 0) {
          RETURN_ERROR(ItemUnspecifiedError);
        }
        int64_t* compressed_indices = get_compressed_indices();
        const int64_t cindex = compressed_indices[storage_index];
        if (cindex < 0) {
          if (size > 0) {
            RETURN_ERROR(UnexpectedNullItemError);
          }
        } else {
          const int64_t next_cindex = compressed_indices[storage_index + 1];
          const int64_t nof_counts =
              (next_cindex < 0 ? -(next_cindex + 1) - cindex : next_cindex - cindex);
          if (subindex < 0 || subindex >= nof_counts) {
            return SubIndexError;
          }
          int64_t* compressed_indices2 = get_compressed_indices2();
          const int64_t valuesize = dtypeSize();
          const int64_t cindex2 = compressed_indices2[cindex + subindex];
          const int64_t next_cindex2 = compressed_indices2[cindex + subindex + 1];
          const int64_t expected_size = (next_cindex2 - cindex2) * valuesize;
          if (expected_size != size) {
            RETURN_ERROR(SizeError);
          }
        }
        break;
      }
      default:
        RETURN_ERROR(UnknownFormatError);
    }
    return setSubItemNoValidation(index, subindex, src, size, dest);
  }

  // To be deprecated in favor of NestedArray format
  Status setSubItemNoValidation(const int64_t index,
                                const int64_t subindex,
                                const int8_t* src,
                                const int64_t size,
                                int8_t** dest) {
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        RETURN_ERROR(NotSupportedFormatError);
      case GeoPolygonFormatId: {
        const int64_t* storage_indices = get_storage_indices();
        const int64_t storage_index = storage_indices[index];
        int64_t* compressed_indices = get_compressed_indices();
        const int64_t cindex = compressed_indices[storage_index];
        int8_t* values = get_values();
        int64_t* compressed_indices2 = get_compressed_indices2();
        const int64_t valuesize = dtypeSize();
        const int64_t cindex2 = compressed_indices2[cindex + subindex];
        const int64_t offset = cindex2 * valuesize;
        if (size > 0 && src != nullptr && memcpy(values + offset, src, size) == nullptr) {
          RETURN_ERROR(MemoryError);
        }
        if (dest != nullptr) {
          *dest = values + offset;
        }
        break;
      }
      default:
        RETURN_ERROR(UnknownFormatError);
    }
    return Success;
  }

  // Set a new item with index and size but without initializing item
  // elements. The buffer pointer of the new item will be stored in
  // *dest if dest != nullptr. Inputs are not validated!
  Status setEmptyItemNoValidation(int64_t index, int64_t size, int8_t** dest) {
    return setItemNoValidation(index, nullptr, size, dest);
  }

  // To be deprecated in favor of NestedArray format
  Status concatItem(int64_t index, const int8_t* src, int64_t size) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId: {
        int64_t next_storage_count = get_storage_count();
        int64_t storage_count = next_storage_count - 1;
        int64_t* compressed_indices = get_compressed_indices();
        int64_t* storage_indices = get_storage_indices();
        int8_t* values = get_values();
        int64_t itemsize = dtypeSize();
        int64_t storage_index = storage_indices[index];
        if (storage_index == -1) {  // unspecified, so setting the item
          return setItem(index, src, size, nullptr);
        }
        if (size % itemsize != 0) {
          RETURN_ERROR(SizeError);
        }
        if (storage_index != storage_count) {
          RETURN_ERROR(IndexError);  // index does not correspond to the last set
                                     // item, only the last item can be
                                     // concatenated
        }
        if (compressed_indices[storage_index] < 0) {
          RETURN_ERROR(NotImplementedError);  // todo: support concat to null when last
        }
        int64_t values_count =
            compressed_indices[next_storage_count] - compressed_indices[storage_index];
        int64_t extra_values_count = size / itemsize;
        compressed_indices[next_storage_count] += extra_values_count;
        int8_t* ptr = values + compressed_indices[storage_index] * itemsize;
        if (size > 0 && src != nullptr &&
            memcpy(ptr + values_count * itemsize, src, size) == nullptr) {
          RETURN_ERROR(MemoryError);
        }
        return Success;
      }
      default:;
    }
    RETURN_ERROR(UnknownFormatError);
  }

  // Set item with index as a null item
  // To be deprecated in favor of NestedArray format
  Status setNull(int64_t index) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId:
      case GeoPolygonFormatId: {
        int64_t* storage_indices = get_storage_indices();
        if (storage_indices[index] >= 0) {
          return ItemAlreadySpecifiedError;
        }
        return setNullNoValidation(index);
      }
      case GeoPointFormatId: {
        return setNullNoValidation(index);
      }
      case NestedArrayFormatId: {
        return setNullNew(index);
      }
      default:
        break;
    }
    RETURN_ERROR(UnknownFormatError);
  }

  // Same as setNull but performs no input validation
  // To be deprecated in favor of NestedArray format
  Status setNullNoValidation(int64_t index) {
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId:
      case GeoPolygonFormatId: {
        int64_t& storage_count = get_storage_count();
        int64_t* storage_indices = get_storage_indices();
        int64_t* compressed_indices = get_compressed_indices();
        const int64_t cindex = compressed_indices[storage_count];
        storage_indices[index] = storage_count;
        compressed_indices[storage_count] = -(cindex + 1);
        compressed_indices[storage_count + 1] = cindex;
        storage_count++;
        break;
      }
      case GeoPointFormatId: {
        int8_t* values = get_values();
        int64_t itemsize = dtypeSize();
        const auto* metadata = getGeoPointMetadata();
        if (metadata->is_geoint) {
          // per Geospatial/CompressionRuntime.h:is_null_point_longitude_geoint32
          *reinterpret_cast<uint32_t*>(values + index * itemsize) = 0x80000000U;
          *reinterpret_cast<uint32_t*>(values + index * itemsize + sizeof(int32_t)) =
              0x80000000U;
        } else {
          // per Shared/InlineNullValues.h:NULL_ARRAY_DOUBLE
          *reinterpret_cast<double*>(values + index * itemsize) = 2 * DBL_MIN;
          *reinterpret_cast<double*>(values + index * itemsize + sizeof(double)) =
              2 * DBL_MIN;
        }
        break;
      }
      default:
        RETURN_ERROR(UnknownFormatError);
    }
    return Success;
  }

  // Check if the item is unspecified or null.
  Status isNull(int64_t index, bool& is_null) const {
    if (isNestedArray()) {
      const size_t ndims = getDimensions();
      Status status = getItemPrepare(index, ndims);
      if (status != Success) {
        return status;
      }
      const auto storage_index = get_storage_index(index);
      const auto* values_offsets = get_values_offsets();
      const auto values_offset = values_offsets[storage_index];
      is_null = values_offset < 0;
      return Success;
    }
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    // To be deprecated in favor of NestedArray format:
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId:
      case GeoPolygonFormatId: {
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t* storage_indices = get_storage_indices();
        const int64_t storage_index = storage_indices[index];
        if (storage_index < 0) {
          RETURN_ERROR(ItemUnspecifiedError);
        }
        is_null = (compressed_indices[storage_index] < 0);
        return Success;
      }
      case GeoPointFormatId: {
        const int8_t* values = get_values();
        const auto* metadata = getGeoPointMetadata();
        int64_t itemsize = dtypeSize();
        if (metadata->is_geoint) {
          // per Geospatial/CompressionRuntime.h:is_null_point_longitude_geoint32
          is_null = (*reinterpret_cast<const uint32_t*>(values + index * itemsize)) ==
                    0x80000000U;
        } else {
          // per Shared/InlineNullValues.h:NULL_ARRAY_DOUBLE
          is_null = (*reinterpret_cast<const double*>(values + index * itemsize)) ==
                    2 * DBL_MIN;
        }
        return Success;
      }
      default:
        break;
    }
    RETURN_ERROR(UnknownFormatError);
  }

  // Get item at index by storing its size (in bytes), values buffer,
  // and nullity information to the corresponding pointer
  // arguments.
  // To be deprecated in favor of NestedArray format
  HOST DEVICE Status getItem(int64_t index, int64_t& size, int8_t*& dest, bool& is_null) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId: {
        int8_t* values = get_values();
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t* storage_indices = get_storage_indices();
        const int64_t storage_index = storage_indices[index];
        if (storage_index < 0) {
          return ItemUnspecifiedError;
        }
        const int64_t cindex = compressed_indices[storage_index];
        if (cindex < 0) {
          // null varlen array
          size = 0;
          dest = nullptr;
          is_null = true;
        } else {
          const int64_t dtypesize = dtypeSize();
          const int64_t next_cindex = compressed_indices[storage_index + 1];
          const int64_t length =
              (next_cindex < 0 ? -(next_cindex + 1) - cindex : next_cindex - cindex);
          size = length * dtypesize;
          dest = values + cindex * dtypesize;
          is_null = false;
        }
        return Success;
      }
      case GeoPointFormatId: {
        int8_t* values = get_values();
        int64_t itemsize = dtypeSize();
        size = itemsize;
        dest = values + index * itemsize;
        is_null = false;
        return Success;
      }
      case GeoPolygonFormatId:
        break;
      default:
        break;
    }
    RETURN_ERROR(UnknownFormatError);
  }

  // To be deprecated in favor of NestedArray format
  HOST DEVICE Status getItem(int64_t index, size_t& size, int8_t*& dest, bool& is_null) {
    int64_t sz{0};
    Status status = getItem(index, sz, dest, is_null);
    size = sz;
    return status;
  }

  // To be deprecated in favor of NestedArray format
  HOST DEVICE Status getItem2(int64_t index,
                              int64_t*& cumcounts,
                              int64_t& nof_counts,
                              int8_t*& dest,
                              bool& is_null) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        break;
      case GeoPolygonFormatId: {
        const int64_t* storage_indices = get_storage_indices();
        int64_t* compressed_indices = get_compressed_indices();
        int64_t* compressed_indices2 = get_compressed_indices2();
        int8_t* values = get_values();

        const int64_t storage_index = storage_indices[index];
        if (storage_index < 0) {
          RETURN_ERROR(ItemUnspecifiedError);
        }
        const int64_t cindex = compressed_indices[storage_index];
        if (cindex < 0) {
          cumcounts = nullptr;
          nof_counts = 0;
          dest = nullptr;
          is_null = true;
        } else {
          const int64_t next_cindex = compressed_indices[storage_index + 1];
          const int64_t valuesize = dtypeSize();
          const int64_t cindex2 = compressed_indices2[cindex];
          nof_counts =
              (next_cindex < 0 ? -(next_cindex + 1) - cindex : next_cindex - cindex);
          cumcounts = compressed_indices2 + cindex;
          dest = values + cindex2 * valuesize;
          is_null = false;
        }
        return Success;
      }
      default:
        break;
    }
    RETURN_ERROR(UnknownFormatError);
  }

  // To be deprecated in favor of NestedArray format
  HOST DEVICE Status getItemCountsAndData(const int64_t index,
                                          int32_t*& counts,
                                          int64_t& nof_counts,
                                          int8_t*& dest,
                                          int64_t& size,
                                          bool& is_null) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        break;
      case GeoPolygonFormatId: {
        const int64_t* storage_indices = get_storage_indices();
        int64_t* compressed_indices = get_compressed_indices();
        const int64_t* compressed_indices2 = get_compressed_indices2();
        int8_t* values = get_values();

        const int64_t storage_index = storage_indices[index];
        if (storage_index < 0) {
          RETURN_ERROR(ItemUnspecifiedError);
        }
        const int64_t cindex = compressed_indices[storage_index];
        if (cindex < 0) {
          counts = nullptr;
          nof_counts = 0;
          dest = nullptr;
          size = 0;
          is_null = true;
        } else {
          const int64_t next_cindex = compressed_indices[storage_index + 1];
          const int64_t valuesize = dtypeSize();
          const int64_t cindex2 = compressed_indices2[cindex];
          nof_counts =
              (next_cindex < 0 ? -(next_cindex + 1) - cindex : next_cindex - cindex);
          const int64_t* cumcounts = compressed_indices2 + cindex;
          counts = get_counts2() + cindex;
          dest = values + cindex2 * valuesize;
          size = (cumcounts[nof_counts] - cumcounts[0]) * valuesize;
          is_null = false;
        }
        return Success;
      }
      default:
        break;
    }
    RETURN_ERROR(UnknownFormatError);
  }

  // To be deprecated in favor of NestedArray format
  Status getItemLength(const int64_t index, int64_t& length) const {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case GeoPointFormatId:
        break;
      case VarlenArrayFormatId:
      case GeoLineStringFormatId:
      case GeoPolygonFormatId: {
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t* storage_indices = get_storage_indices();
        const int64_t storage_index = storage_indices[index];
        if (storage_index < 0) {
          RETURN_ERROR(ItemUnspecifiedError);
        }
        const int64_t cindex = compressed_indices[storage_index];
        if (cindex < 0) {
          length = 0;
        } else {
          const int64_t next_cindex = compressed_indices[storage_index + 1];
          length = (next_cindex < 0 ? -(next_cindex + 1) - cindex : next_cindex - cindex);
        }
        return Success;
      }
      default:
        break;
    }
    RETURN_ERROR(UnknownFormatError);
  }

  // To be deprecated in favor of NestedArray format
  Status getSubItemLength(const int64_t index,
                          const int64_t subindex,
                          int64_t& length) const {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        break;
      case GeoPolygonFormatId: {
        const int64_t* storage_indices = get_storage_indices();
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t storage_index = storage_indices[index];
        if (storage_index < 0) {
          RETURN_ERROR(ItemUnspecifiedError);
        }
        const int64_t cindex = compressed_indices[storage_index];
        if (cindex < 0) {
          length = 0;
        } else {
          const int64_t next_cindex = compressed_indices[storage_index + 1];
          const int64_t nof_counts =
              (next_cindex < 0 ? -(next_cindex + 1) - cindex : next_cindex - cindex);
          if (subindex < 0 || subindex >= nof_counts) {
            return SubIndexError;
          }
          const int64_t* compressed_indices2 = get_compressed_indices2();
          const int64_t cindex2 = compressed_indices2[cindex + subindex];
          const int64_t next_cindex2 = compressed_indices2[cindex + subindex + 1];
          length = next_cindex2 - cindex2;
        }
        return Success;
      }
      default:
        break;
    }
    RETURN_ERROR(UnknownFormatError);
  }

  // Get a subitem data of an item, e.g. a linestring within a polygon
  // To be deprecated in favor of NestedArray format
  HOST DEVICE Status getSubItem(int64_t index,
                                int64_t subindex,
                                int64_t& size,
                                int8_t*& dest,
                                bool& is_null) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        RETURN_ERROR(NotSupportedFormatError);
      case GeoPolygonFormatId: {
        const int64_t* storage_indices = get_storage_indices();
        const int64_t storage_index = storage_indices[index];
        if (storage_index < 0) {
          RETURN_ERROR(ItemUnspecifiedError);
        }
        int64_t* compressed_indices = get_compressed_indices();
        const int64_t cindex = compressed_indices[storage_index];
        if (cindex < 0) {
          dest = nullptr;
          size = 0;
          is_null = true;
        } else {
          const int64_t next_cindex = compressed_indices[storage_index + 1];
          const int64_t nof_counts =
              (next_cindex < 0 ? -(next_cindex + 1) - cindex : next_cindex - cindex);
          if (subindex < 0 || subindex >= nof_counts) {
            return SubIndexError;
          }
          int8_t* values = get_values();
          int64_t* compressed_indices2 = get_compressed_indices2();
          const int64_t valuesize = dtypeSize();
          const int64_t cindex2 = compressed_indices2[cindex + subindex];
          const int64_t next_cindex2 = compressed_indices2[cindex + subindex + 1];
          dest = values + cindex2 * valuesize;
          size = (next_cindex2 - cindex2) * valuesize;
          is_null = false;
        }
        return Success;
      }
      default:
        break;
    }
    RETURN_ERROR(UnknownFormatError);
  }

#ifdef HAVE_TOSTRING
#define HAVE_FLATBUFFER_TOSTRING
  std::string bufferToString(const int8_t* buffer,
                             const size_t size,
                             ValueType value_type) const {
    size_t value_size = get_size(value_type);
    size_t count = size / value_size;
    std::string result = "";
    for (size_t i = 0; i < count; i++) {
      if (i > 0) {
        result += ", ";
      }
      switch (value_type) {
        case Bool8:
          result += (buffer[i] ? "true" : "false");
          break;
        case Int8:
          result += std::to_string(buffer[i]);
          break;
        case Int16:
          result += std::to_string(reinterpret_cast<const int16_t*>(buffer)[i]);
          break;
        case Int32:
          result += std::to_string(reinterpret_cast<const int32_t*>(buffer)[i]);
          break;
        case Int64:
          result += std::to_string(reinterpret_cast<const int64_t*>(buffer)[i]);
          break;
        case UInt8:
          result += std::to_string(reinterpret_cast<const uint8_t*>(buffer)[i]);
          break;
        case UInt16:
          result += std::to_string(reinterpret_cast<const uint16_t*>(buffer)[i]);
          break;
        case UInt32:
          result += std::to_string(reinterpret_cast<const uint32_t*>(buffer)[i]);
          break;
        case UInt64:
          result += std::to_string(reinterpret_cast<const uint64_t*>(buffer)[i]);
          break;
        case Float32:
          result += std::to_string(reinterpret_cast<const float*>(buffer)[i]);
          break;
        case Float64:
          result += std::to_string(reinterpret_cast<const double*>(buffer)[i]);
          break;
        case PointInt32:
          result += "(";
          if (containsNullValue(buffer + 2 * i * sizeof(int32_t))) {
            result += "NULL";
          } else {
            result += std::to_string(reinterpret_cast<const int32_t*>(buffer)[2 * i]);
            result += ", ";
            result += std::to_string(reinterpret_cast<const int32_t*>(buffer)[2 * i + 1]);
          }
          result += ")";
          break;
        case PointFloat64:
          result += "(";
          if (containsNullValue(buffer + 2 * i * sizeof(double))) {
            result += "NULL";
          } else {
            result += std::to_string(reinterpret_cast<const double*>(buffer)[2 * i]);
            result += ", ";
            result += std::to_string(reinterpret_cast<const double*>(buffer)[2 * i + 1]);
          }
          result += ")";
          break;
      }
    }
    return result;
  }

  std::string toString() const {
    if (buffer == nullptr) {
      return ::typeName(this) + "[UNINITIALIZED]";
    }
    std::string result = typeName(this) + "@" + ::toString((void*)buffer) + "(";
    result += "" + getBaseWorker()->toString();

    if (isNestedArray()) {
      const auto* metadata = getNestedArrayMetadata();
      const auto* worker = getNestedArrayWorker();
      result += ",\n  " + metadata->toString();
      result += ",\n  " + worker->toString();
      result += ",\n  values_buffer=[" +
                bufferToString(
                    get_values_buffer(), getValuesBufferSize(), metadata->value_type) +
                "]";
      result += ",\n  sizes_buffer=[" +
                bufferToString(
                    reinterpret_cast<const int8_t*>(get_sizes_buffer()),
                    metadata->total_sizes_count * get_size(FLATBUFFER_SIZES_T_VALUE_TYPE),
                    FLATBUFFER_SIZES_T_VALUE_TYPE) +
                "]";
      result += ",\n  values_offsets=[" +
                bufferToString(reinterpret_cast<const int8_t*>(get_values_offsets()),
                               (metadata->total_items_count + 1) *
                                   get_size(FLATBUFFER_OFFSETS_T_VALUE_TYPE),
                               FLATBUFFER_OFFSETS_T_VALUE_TYPE) +
                "]";
      result += ",\n  sizes_offsets=[" +
                bufferToString(reinterpret_cast<const int8_t*>(get_sizes_offsets()),
                               (metadata->total_items_count * metadata->dimensions + 1) *
                                   get_size(FLATBUFFER_OFFSETS_T_VALUE_TYPE),
                               FLATBUFFER_OFFSETS_T_VALUE_TYPE) +
                "]";
      result += ",\n  storage_indices=[" +
                bufferToString(
                    reinterpret_cast<const int8_t*>(get_storage_indices_new()),
                    metadata->total_items_count * get_size(FLATBUFFER_SIZES_T_VALUE_TYPE),
                    FLATBUFFER_SIZES_T_VALUE_TYPE) +
                "]";
      result += ",\n  user_data_buffer=[" +
                bufferToString(get_user_data_buffer(), metadata->user_data_size, Int8) +
                "]";
      result += ")";
      return result;
    }

    // To be deprecated in favor of NestedArray format:
    const FlatBufferFormat fmt = format();

    std::cout << "fmt=" << static_cast<int64_t>(fmt) << ", " << sizeof(fmt) << std::endl;
    switch (fmt) {
      case VarlenArrayFormatId: {
        result += ", " + getVarlenArrayMetadata()->toString();
        result += ", " + getVarlenArrayWorker()->toString();
        break;
      }
      case GeoLineStringFormatId: {
        result += ", " + getGeoLineStringMetadata()->toString();
        result += ", " + getGeoLineStringWorker()->toString();
        break;
      }
      case GeoPointFormatId: {
        result += ", " + getGeoPointMetadata()->toString();
        result += ", " + getGeoPointWorker()->toString();
        break;
      }
      case GeoPolygonFormatId: {
        result += ", " + getGeoPolygonMetadata()->toString();
        result += ", " + getGeoPolygonWorker()->toString();
        break;
      }
      default:
        break;
    }

    switch (fmt) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId:
      case GeoPolygonFormatId: {
        result += ", values=";
        const int64_t numvalues = get_nof_values();
        const int64_t itemsize = dtypeSize();
        switch (itemsize) {
          case 1: {
            const int8_t* values_buf = get_values();
            std::vector<int8_t> values(values_buf, values_buf + numvalues);
            result += ::toString(values);
          } break;
          case 2: {
            const int16_t* values_buf = reinterpret_cast<const int16_t*>(get_values());
            std::vector<int16_t> values(values_buf, values_buf + numvalues);
            result += ::toString(values);
          } break;
          case 4: {
            const int32_t* values_buf = reinterpret_cast<const int32_t*>(get_values());
            std::vector<int32_t> values(values_buf, values_buf + numvalues);
            result += ::toString(values);
          } break;
          case 8: {
            if (fmt == GeoLineStringFormatId || fmt == GeoPolygonFormatId) {
              const int32_t* values_buf = reinterpret_cast<const int32_t*>(get_values());
              std::vector<int32_t> values(values_buf, values_buf + numvalues * 2);
              result += ::toString(values);
            } else {
              const int64_t* values_buf = reinterpret_cast<const int64_t*>(get_values());
              std::vector<int64_t> values(values_buf, values_buf + numvalues);
              result += ::toString(values);
            }
          } break;
          case 16: {
            if (fmt == GeoLineStringFormatId || fmt == GeoPolygonFormatId) {
              const double* values_buf = reinterpret_cast<const double*>(get_values());
              std::vector<double> values(values_buf, values_buf + numvalues * 2);
              result += ::toString(values);
              break;
            }
          }
          default:
            result += "[UNEXPECTED ITEMSIZE:" + std::to_string(itemsize) + "]";
        }
        if (fmt == GeoPolygonFormatId) {
          const int64_t numitems2 = items2Count();
          const int64_t* compressed_indices2_buf = get_compressed_indices2();
          std::vector<int64_t> compressed_indices2(
              compressed_indices2_buf, compressed_indices2_buf + numitems2 + 1);
          result += ", compressed_indices2=" + ::toString(compressed_indices2);

          const int32_t* counts2_buf = get_counts2();
          std::vector<int32_t> counts2(counts2_buf, counts2_buf + numitems2);
          result += ", counts2=" + ::toString(counts2);
        }

        const int64_t numitems = itemsCount();
        const int64_t* compressed_indices_buf = get_compressed_indices();
        std::vector<int64_t> compressed_indices(compressed_indices_buf,
                                                compressed_indices_buf + numitems + 1);
        result += ", compressed_indices=" + ::toString(compressed_indices);

        const int64_t* storage_indices_buf = get_storage_indices();
        std::vector<int64_t> storage_indices(storage_indices_buf,
                                             storage_indices_buf + numitems);
        result += ", storage_indices=" + ::toString(storage_indices);
        return result + ")";
      }
      case GeoPointFormatId: {
        const auto* metadata = getGeoPointMetadata();
        result += ", point data=";
        int64_t numitems = itemsCount();
        if (metadata->is_geoint) {
          const int32_t* values_buf = reinterpret_cast<const int32_t*>(get_values());
          std::vector<int32_t> values(values_buf, values_buf + numitems * 2);
          result += ::toString(values);
        } else {
          const double* values_buf = reinterpret_cast<const double*>(get_values());
          std::vector<double> values(values_buf, values_buf + numitems * 2);
          result += ::toString(values);
        }
        return result + ")";
      }
      default:
        break;
    }
    return ::typeName(this) + "[UNKNOWN FORMAT]";
  }
#endif
};

#ifdef HAVE_TOSTRING
inline std::ostream& operator<<(std::ostream& os,
                                FlatBufferManager::ValueType const type) {
  switch (type) {
    case FlatBufferManager::Bool8:
      os << "Bool8";
      break;
    case FlatBufferManager::Int8:
      os << "Int8";
      break;
    case FlatBufferManager::Int16:
      os << "Int16";
      break;
    case FlatBufferManager::Int32:
      os << "Int32";
      break;
    case FlatBufferManager::Int64:
      os << "Int64";
      break;
    case FlatBufferManager::UInt8:
      os << "UInt8";
      break;
    case FlatBufferManager::UInt16:
      os << "UInt16";
      break;
    case FlatBufferManager::UInt32:
      os << "UInt32";
      break;
    case FlatBufferManager::UInt64:
      os << "UInt64";
      break;
    case FlatBufferManager::Float32:
      os << "Float32";
      break;
    case FlatBufferManager::Float64:
      os << "Float64";
      break;
    case FlatBufferManager::PointInt32:
      os << "PointInt32";
      break;
    case FlatBufferManager::PointFloat64:
      os << "PointFloat64";
      break;
  }
  return os;
}

inline std::string FlatBufferManager::toString(const FlatBufferManager::ValueType& type) {
  std::ostringstream ss;
  ss << type;
  return ss.str();
}

inline std::string toString(const FlatBufferManager::ValueType& type) {
  std::ostringstream ss;
  ss << type;
  return ss.str();
}

inline std::ostream& operator<<(std::ostream& os,
                                FlatBufferManager::Status const status) {
  switch (status) {
    case FlatBufferManager::Success:
      os << "Success";
      break;
    case FlatBufferManager::IndexError:
      os << "IndexError";
      break;
    case FlatBufferManager::SubIndexError:
      os << "SubIndexError";
      break;
    case FlatBufferManager::SizeError:
      os << "SizeError";
      break;
    case FlatBufferManager::FlatbufferSizeError:
      os << "FlatbufferSizeError";
      break;
    case FlatBufferManager::ItemAlreadySpecifiedError:
      os << "ItemAlreadySpecifiedError";
      break;
    case FlatBufferManager::ItemUnspecifiedError:
      os << "ItemUnspecifiedError";
      break;
    case FlatBufferManager::UnexpectedNullItemError:
      os << "UnexpectedNullItemError";
      break;
    case FlatBufferManager::ValuesBufferTooSmallError:
      os << "ValuesBufferTooSmallError";
      break;
    case FlatBufferManager::SizesBufferTooSmallError:
      os << "SizesBufferTooSmallError";
      break;
    case FlatBufferManager::CompressedIndices2BufferTooSmallError:
      os << "CompressedIndices2BufferTooSmallError";
      break;
    case FlatBufferManager::MemoryError:
      os << "MemoryError";
      break;
    case FlatBufferManager::UnknownFormatError:
      os << "UnknownFormatError";
      break;
    case FlatBufferManager::NotSupportedFormatError:
      os << "NotSupportedFormatError";
      break;
    case FlatBufferManager::NotImplementedError:
      os << "NotImplementedError";
      break;
    case FlatBufferManager::InvalidUserDataError:
      os << "InvalidUserDataError";
      break;
    case FlatBufferManager::DimensionalityError:
      os << "DimensionalityError";
      break;
    case FlatBufferManager::UserDataError:
      os << "UserDataError";
      break;
    case FlatBufferManager::TypeError:
      os << "TypeError";
      break;
    case FlatBufferManager::InconsistentSizesError:
      os << "InconsistentSizesError";
      break;
    default:
      os << "[Unknown FlatBufferManager::Status value]";
  }
  return os;
}

inline std::string toString(const FlatBufferManager::Status& status) {
  std::ostringstream ss;
  ss << status;
  return ss.str();
}
#endif

#undef RETURN_ERROR
