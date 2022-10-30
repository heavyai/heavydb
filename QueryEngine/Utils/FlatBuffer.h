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

    | <point data>                                       | <compressed indices2>               |  <compressed indices>           | <storage indices>          |
    =
    |<-- (max nof points) * (is_geoint ? 4 : 8) bytes -->|<-- (max nof rings + 1) * 4 bytes -->|<-- (num items + 1) * 8 bytes -->|<-- (num items) * 8 bytes-->|

  where

     <point data> stores points coordinates in a point-wise manner:
     X0, Y0, X1, Y1, ... If is_geoint is true, point coordinates are
     stored as integers, otherwise as double floating point numbers.

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

#include <float.h>
#ifdef HAVE_TOSTRING
#include <ostream>
#include <string.h>
#endif

#include "../../Shared/funcannotations.h"

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
  // GeoMultiPolygonFormatId    = 0x47706f6c79676f6e,  // hex repr of 'Gpolygon'
};

inline int64_t _align_to_int64(int64_t addr) {
  addr += sizeof(int64_t) - 1;
  return (int64_t)(((uint64_t)addr >> 3) << 3);
}

struct FlatBufferManager {
  struct BaseWorker {
    int64_t format_id;
    int64_t flatbuffer_size;
    int64_t format_metadata_offset;  // the offset of the data format metadata buffer
    int64_t format_worker_offset;    // the offset of the data format worker buffer
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "format_id=" + std::to_string(format_id);
      result += ", flatbuffer_size=" + std::to_string(flatbuffer_size);
      result += ", format_metadata_offset=" + std::to_string(format_metadata_offset);
      result += ", format_worker_offset=" + std::to_string(format_worker_offset);
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
      for (int i; i < VarlenArrayParamsCount; i++) {
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
    int64_t compressed_indices2_offset;
    int64_t compressed_indices_offset;
    int64_t storage_indices_offset;

#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "items_count=" + std::to_string(items_count);
      result += ", items2_count=" + std::to_string(items2_count);
      result += ", values_offset=" + std::to_string(values_offset);
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

  enum Status {
    Success = 0,
    IndexError,
    SubIndexError,
    SizeError,
    ItemAlreadySpecifiedError,
    ItemUnspecifiedError,
    UnexpectedNullItemError,
    ValuesBufferTooSmallError,
    CompressedIndices2BufferTooSmallError,
    MemoryError,
    NotImplementedError,
    NotSupportedFormatError,
    UnknownFormatError
  };

  int8_t* buffer;

  // Check if a buffer contains FlatBuffer formatted data
  HOST DEVICE static bool isFlatBuffer(const void* buffer) {
    if (buffer) {
      // warning: assume that buffer size is at least 8 bytes
      const auto* base = reinterpret_cast<const BaseWorker*>(buffer);
      FlatBufferFormat header_format = static_cast<FlatBufferFormat>(base->format_id);
      switch (header_format) {
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
        } break;
        default:;
      }
    }
    return false;
  }

  // Return the allocation size of the the FlatBuffer storage, in bytes
  static int64_t getBufferSize(const void* buffer) {
    if (isFlatBuffer(buffer)) {
      return reinterpret_cast<const BaseWorker*>(buffer)->flatbuffer_size;
    } else {
      return -1;
    }
  }

  // Return the allocation size of the the FlatBuffer storage, in bytes
  inline int64_t getBufferSize() const {
    return reinterpret_cast<const BaseWorker*>(buffer)->flatbuffer_size;
  }

  // Return the format of FlatBuffer
  HOST DEVICE inline FlatBufferFormat format() const {
    const auto* base = reinterpret_cast<const BaseWorker*>(buffer);
    return static_cast<FlatBufferFormat>(base->format_id);
  }

  // Return the number of items
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
    }
    return -1;  // invalid value
  }

  HOST DEVICE inline int64_t items2Count() const {
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        break;
      case GeoPolygonFormatId:
        return getGeoPolygonMetadata()->max_nof_rings;
    }
    return -1;  // invalid value
  }

  HOST DEVICE inline int64_t dtypeSize() const {
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
    }
    return -1;
  }

  // VarlenArray support:

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
      default:;
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

  FLATBUFFER_MANAGER_FORMAT_TOOLS(VarlenArray);
  FLATBUFFER_MANAGER_FORMAT_TOOLS(GeoPoint);
  FLATBUFFER_MANAGER_FORMAT_TOOLS(GeoLineString);
  FLATBUFFER_MANAGER_FORMAT_TOOLS(GeoPolygon);

#undef FLATBUFFER_MANAGER_FORMAT_TOOLS

  void initialize(FlatBufferFormat format_id, const int8_t* format_metadata_ptr) {
    auto* base = getBaseWorker();
    base->format_id = format_id;
    base->flatbuffer_size = compute_flatbuffer_size(format_id, format_metadata_ptr);
    base->format_metadata_offset = _align_to_int64(sizeof(FlatBufferManager::BaseWorker));
    switch (format_id) {
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

        this_worker->compressed_indices2_offset =
            this_worker->values_offset +
            _align_to_int64(itemsize * this_metadata->max_nof_values);

        this_worker->compressed_indices_offset =
            this_worker->compressed_indices2_offset +
            _align_to_int64(sizeof(int64_t) * (this_metadata->max_nof_rings + 1));
        ;
        this_worker->storage_indices_offset =
            this_worker->compressed_indices_offset +
            _align_to_int64(sizeof(int64_t) * (this_metadata->total_items_count + 1));

        int64_t* compressed_indices2 = get_compressed_indices2();
        int64_t* compressed_indices = get_compressed_indices();
        int64_t* storage_indices = get_storage_indices();
        for (int i = 0; i <= this_metadata->max_nof_rings; i++) {
          compressed_indices2[i] = 0;
        }
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

  // Return the upper bound to the total number of points in all items
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
    }
    return -1;
  }

  // Return the total number of values in all specified items
  inline int64_t get_nof_values() const {
    switch (format()) {
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
    }
    static int64_t dummy_storage_count = -1;
    return dummy_storage_count;
  }

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
    }
    static int64_t dummy = -1;
    return dummy;
  }

  // Return the number of specified blocks
  HOST DEVICE inline int64_t& get_storage_count2() {
    switch (format()) {
      case GeoPolygonFormatId:
        return getGeoPolygonWorker()->items2_count;
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        break;
    }
    static int64_t dummy_storage_count = -1;
    return dummy_storage_count;
  }

  inline const int64_t& get_storage_count2() const {
    switch (format()) {
      case GeoPolygonFormatId:
        return getGeoPolygonWorker()->items2_count;
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        break;
    }
    static int64_t dummy_storage_count = -1;
    return dummy_storage_count;
  }

  // Return the size of values buffer in bytes
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
    }
    static int64_t dummy = -1;
    return dummy;
  }

  // Return the size of compressed_indices2 buffer in bytes
  inline int64_t get_compressed_indices2_buffer_size() const {
    switch (format()) {
      case GeoPolygonFormatId: {
        const auto* worker = getGeoPolygonWorker();
        return worker->compressed_indices_offset - worker->compressed_indices2_offset;
      }
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        break;
    }
    static int64_t dummy = -1;
    return dummy;
  }

  // Return the pointer to values buffer
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

  // Return the pointer to compressed indices2 buffer
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

  // Return the pointer to storage indices buffer
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

  // High-level API

  // Set a new item with index and size (in bytes) and initialize its
  // elements from source buffer. The item values will be
  // uninitialized when source buffer is nullptr. If dest != nullptr
  // then the item's buffer pointer will be stored in *dest.
  Status setItem(const int64_t index,
                 const int8_t* src,
                 const int64_t size,
                 int8_t** dest = nullptr) {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
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
          return ItemAlreadySpecifiedError;
        }
        const int64_t cindex = compressed_indices[storage_count];
        const int64_t values_buffer_size = get_values_buffer_size();
        const int64_t csize = cindex * itemsize;
        if (csize + size > values_buffer_size) {
          return ValuesBufferTooSmallError;
        }
        break;
      }
      case GeoPointFormatId: {
        const int64_t itemsize = dtypeSize();
        if (size != itemsize) {
          return SizeError;
        }
        break;
      }
      case GeoPolygonFormatId: {
        const int64_t itemsize = dtypeSize();
        const int64_t counts = size / itemsize;
        return setItem2(index, src, &counts, 1, dest);
      }
      default:
        return UnknownFormatError;
    }
    return setItemNoValidation(index, src, size, dest);
  }

  // Same as setItem but performs no input validation
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
          return MemoryError;
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
          return MemoryError;
        }
        if (dest != nullptr) {
          *dest = values + csize;
        }
        break;
      }
      case GeoPolygonFormatId: {
        const int64_t itemsize = dtypeSize();
        const int64_t counts = size / itemsize;
        return setItem2NoValidation(index, src, &counts, 1, dest);
      }
      default:
        return UnknownFormatError;
    }

    return Success;
  }

  Status setItem2(int64_t index,
                  const int8_t* src,
                  const int64_t* counts,
                  int64_t nof_counts,
                  int8_t** dest = nullptr) {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        return NotSupportedFormatError;
      case GeoPolygonFormatId: {
        const int64_t& storage_count = get_storage_count();
        const int64_t& storage_count2 = get_storage_count2();
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t* compressed_indices2 = get_compressed_indices2();
        const int64_t* storage_indices = get_storage_indices();
        const int64_t valuesize = dtypeSize();
        if (storage_indices[index] >= 0) {
          return ItemAlreadySpecifiedError;
        }

        const int64_t compressed_indices2_buffer_size =
            get_compressed_indices2_buffer_size();
        if (compressed_indices[storage_count] + nof_counts >
            compressed_indices2_buffer_size) {
          return CompressedIndices2BufferTooSmallError;
        }

        const int64_t offset = compressed_indices2[storage_count2] * valuesize;
        int64_t size = 0;
        for (int i = 0; i < nof_counts; i++) {
          size += valuesize * counts[i];
        }
        const int64_t values_buffer_size = get_values_buffer_size();
        if (offset + size > values_buffer_size) {
          return ValuesBufferTooSmallError;
        }
        break;
      }
      default:
        return UnknownFormatError;
    }
    return setItem2NoValidation(index, src, counts, nof_counts, dest);
  }

  // Same as setItem but performs no input validation
  Status setItem2NoValidation(int64_t index,
                              const int8_t* src,      // coordinates of points
                              const int64_t* counts,  // counts of points in rings
                              int64_t nof_counts,     // nof rings
                              int8_t** dest) {
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        return NotSupportedFormatError;
      case GeoPolygonFormatId: {
        int64_t& storage_count = get_storage_count();
        int64_t& storage_count2 = get_storage_count2();
        int64_t* storage_indices = get_storage_indices();
        int64_t* compressed_indices = get_compressed_indices();
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
          storage_count2++;
          compressed_indices2[storage_count2] = cindex2;
        }
        if (size > 0 && src != nullptr && memcpy(values + offset, src, size) == nullptr) {
          return MemoryError;
        }
        if (dest != nullptr) {
          *dest = values + offset;
        }
        storage_count++;
        break;
      }
      default:
        return UnknownFormatError;
    }

    return Success;
  }

  Status setSubItem(const int64_t index,
                    const int64_t subindex,
                    const int8_t* src,
                    const int64_t size,
                    int8_t** dest = nullptr) {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
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
          return ItemUnspecifiedError;
        }
        int64_t* compressed_indices = get_compressed_indices();
        const int64_t cindex = compressed_indices[storage_index];
        if (cindex < 0) {
          if (size > 0) {
            return UnexpectedNullItemError;
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
            return SizeError;
          }
        }
        break;
      }
      default:
        return UnknownFormatError;
    }
    return setSubItemNoValidation(index, subindex, src, size, dest);
  }

  Status setSubItemNoValidation(const int64_t index,
                                const int64_t subindex,
                                const int8_t* src,
                                const int64_t size,
                                int8_t** dest) {
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoPointFormatId:
      case GeoLineStringFormatId:
        return NotSupportedFormatError;
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
          return MemoryError;
        }
        if (dest != nullptr) {
          *dest = values + offset;
        }
        break;
      }
      default:
        return UnknownFormatError;
    }
    return Success;
  }

  // Set a new item with index and size but without initializing item
  // elements. The buffer pointer of the new item will be stored in
  // *dest if dest != nullptr. Inputs are not validated!
  Status setEmptyItemNoValidation(int64_t index, int64_t size, int8_t** dest) {
    return setItemNoValidation(index, nullptr, size, dest);
  }

  Status concatItem(int64_t index, const int8_t* src, int64_t size) {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
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
          return SizeError;
        }
        if (storage_index != storage_count) {
          return IndexError;  // index does not correspond to the last set
                              // item, only the last item can be
                              // concatenated
        }
        if (compressed_indices[storage_index] < 0) {
          return NotImplementedError;  // todo: support concat to null when last
        }
        int64_t values_count =
            compressed_indices[next_storage_count] - compressed_indices[storage_index];
        int64_t extra_values_count = size / itemsize;
        compressed_indices[next_storage_count] += extra_values_count;
        int8_t* ptr = values + compressed_indices[storage_index] * itemsize;
        if (size > 0 && src != nullptr &&
            memcpy(ptr + values_count * itemsize, src, size) == nullptr) {
          return MemoryError;
        }
        return Success;
      }
      default:;
    }
    return UnknownFormatError;
  }

  // Set item with index as a null item
  Status setNull(int64_t index) {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
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
    }
    return UnknownFormatError;
  }

  // Same as setNull but performs no input validation
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
        return UnknownFormatError;
    }
    return Success;
  }

  // Check if the item is unspecified or null.
  Status isNull(int64_t index, bool& is_null) const {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
    }
    switch (format()) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId:
      case GeoPolygonFormatId: {
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t* storage_indices = get_storage_indices();
        const int64_t storage_index = storage_indices[index];
        if (storage_index < 0) {
          return ItemUnspecifiedError;
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
    }
    return UnknownFormatError;
  }

  // Get item at index by storing its size (in bytes), values buffer,
  // and nullity information to the corresponding pointer
  // arguments.
  HOST DEVICE Status getItem(int64_t index, int64_t& size, int8_t*& dest, bool& is_null) {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
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
    }
    return UnknownFormatError;
  }

  HOST DEVICE Status getItem(int64_t index, size_t& size, int8_t*& dest, bool& is_null) {
    int64_t sz{0};
    Status status = getItem(index, sz, dest, is_null);
    size = sz;
    return status;
  }

  HOST DEVICE Status getItem2(int64_t index,
                              int64_t*& cumcounts,
                              int64_t& nof_counts,
                              int8_t*& dest,
                              bool& is_null) {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
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
          return ItemUnspecifiedError;
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
    }
    return UnknownFormatError;
  }

  Status getItemLength(const int64_t index, int64_t& length) const {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
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
          return ItemUnspecifiedError;
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
    }
    return UnknownFormatError;
  }

  Status getSubItemLength(const int64_t index,
                          const int64_t subindex,
                          int64_t& length) const {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
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
          return ItemUnspecifiedError;
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
    }
    return UnknownFormatError;
  }

  // Get a subitem data of an item, e.g. a linestring within a polygon
  HOST DEVICE Status getSubItem(int64_t index,
                                int64_t subindex,
                                int64_t& size,
                                int8_t*& dest,
                                bool& is_null) {
    if (index < 0 || index >= itemsCount()) {
      return IndexError;
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
          return ItemUnspecifiedError;
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
    }
    return UnknownFormatError;
  }

#ifdef HAVE_TOSTRING
#define HAVE_FLATBUFFER_TOSTRING
  std::string toString() const {
    if (buffer == nullptr) {
      return ::typeName(this) + "[UNINITIALIZED]";
    }
    std::string result = typeName(this) + "(";
    result += "" + getBaseWorker()->toString();

    const auto fmt = format();
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
    }
    switch (fmt) {
      case VarlenArrayFormatId:
      case GeoLineStringFormatId: {
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
                const int32_t* values_buf =
                    reinterpret_cast<const int32_t*>(get_values());
                std::vector<int32_t> values(values_buf, values_buf + numvalues * 2);
                result += ::toString(values);
              } else {
                const int64_t* values_buf =
                    reinterpret_cast<const int64_t*>(get_values());
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
        default:;
      }
        return ::typeName(this) + "[UNKNOWN FORMAT]";
    }
  }
#endif
};

#ifdef HAVE_TOSTRING
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
