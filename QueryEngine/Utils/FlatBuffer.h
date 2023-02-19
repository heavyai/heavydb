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

    int64_t compute_flatbuffer_size(ndims,
                                    total_items_count,
                                    total_sizes_count,
                                    total_values_count,
                                    value_type,
                                    user_data_size)

  To initialize the provided buffer for nested array format, use::

    Status .initialize(ndims,
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

    size_t .getNDims()

  To get various buffers::

    int8_t* .get_user_data_buffer()
    int8_t* .get_values_buffer()
    sizes_t* .get_sizes_buffer()
    offsets_t* .get_values_offsets()
    offsets_t* .get_sizes_offsets()
    int8_t* .getNullValuePtr()

  To test if the provided buffer contains null value::

    bool .containsNullValue()

  To get the item (and subitems) of a nested array::

    template <size_t NDIM>
    Status getItemWorker(const int64_t index[NDIM],
                        const size_t n,
                        int8_t*& values,
                        int32_t& nof_values,
                        int32_t* sizes_buffers[NDIM],
                        int32_t sizes_lengths[NDIM],
                        int32_t& nof_sizes,
                        bool& is_null)

    template <size_t NDIM>
    Status getItem(const int64_t index, NestedArrayItem<NDIM>& result)

    template <size_t NDIM>
    Status getItem(const int64_t index[NDIM], const size_t n, NestedArrayItem<NDIM>& result)

    template <typename CT>
    Status getItem(const int64_t index,
                   std::vector<CT>& values,
                   std::vector<int32_t>& sizes,
                   bool& is_null)

    template <typename CT>
    Status getItem(const int64_t index,
                   std::vector<CT>& values,
                   std::vector<int32_t>& sizes,
                   std::vector<int32_t>& sizes_of_sizes,
                   bool& is_null)



  To set the item of a nested array::

    template <size_t NDIM, bool check_sizes=true>
    Status setItemWorker(const int64_t index,
                         const int8_t* values,
                         const int32_t nof_values,
                         const int32_t* const sizes_buffers[NDIM],
                         const int32_t sizes_lengths[NDIM],
                         const int32_t nof_sizes)

    template <size_t NDIM=0, bool check_sizes=true>
    Status setItem(const int64_t index,
                   const int8_t* values_buf,
                   const int32_t nof_values)

    template <size_t NDIM=1, bool check_sizes=true>
    Status setItem(const int64_t index,
                   const int8_t* values_buf,
                   const int32_t nof_values,
                   const int32_t* sizes_buf,
                   const int32_t nof_sizes)

    template <size_t NDIM=2, bool check_sizes=true>
    Status setItem(const int64_t index,
                   const int8_t* values_buf,
                   const int32_t nof_values,
                   const int32_t* sizes_buf,
                   const int32_t nof_sizes,
                   const int32_t* sizes_of_sizes_buf,
                   const int32_t nof_sizes_of_sizes)

    template <typename CT, size_t NDIM=0>
    Status setItem(const int64_t index, const std::vector<CT>& arr)

    template <typename CT, size_t NDIM=1, bool check_sizes=false>
    Status setItem(const int64_t index, const std::vector<std::vector<CT>>& item)

    template <typename CT, size_t NDIM=2, bool check_sizes=false>
    Status setItem(const int64_t index,
                   const std::vector<std::vector<std::vector<CT>>>& item)

    template <typename CT, size_t NDIM=1, bool check_sizes=true>
    Status setItem(const int64_t index,
                   const std::vector<CT>& values,
                   const std::vector<int32_t>& sizes)

    template <typename CT, size_t NDIM=2, bool check_sizes=true>
    Status setItem(const int64_t index,
                   const std::vector<CT>& values,
                   const std::vector<int32_t>& sizes,
                   const std::vector<int32_t>& sizes_of_sizes)

    Status setNull(int64_t index)

  To test if the item is NULL::

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
  VarlenArrayFormatId = 0x7661726c65634152,  // hex repr of 'varlenAR'
  GeoPointFormatId = 0x67656f706f696e74,     // hex repr of 'geopoint'
  // GeoLineStringFormatId = 0x676c696e65737472,  // hex repr of 'glinestr'
  // GeoPolygonFormatId = 0x67706f6c79676f6e,     // hex repr of 'gpolygon'
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
    size_t ndims;
    int64_t total_items_count;
    int64_t total_sizes_count;
    int64_t total_values_count;
    ValueType value_type;
    size_t user_data_size;
#ifdef HAVE_TOSTRING
    std::string toString() const {
      std::string result = ::typeName(this) + "{";
      result += "ndims=" + std::to_string(ndims);
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
        case GeoPointFormatId: {
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
  // TODO?: return size_t value, 0 when not a flat buffer
  static int64_t getBufferSize(const void* buffer) {
    if (isFlatBuffer(buffer)) {
      return reinterpret_cast<const BaseWorker*>(buffer)->flatbuffer_size;
    } else {
      return -1;
    }
  }

  // Return the allocation size of the the FlatBuffer storage, in bytes
  // TODO?: int64_t -> size_t
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
  HOST DEVICE inline int64_t itemsCount() const {
    switch (format()) {
      case VarlenArrayFormatId:
        return getVarlenArrayMetadata()->total_items_count;
      case GeoPointFormatId:
        return getGeoPointMetadata()->total_items_count;
      case NestedArrayFormatId:
        return getNestedArrayMetadata()->total_items_count;
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

  static int64_t computeBufferSizeNestedArray(int64_t ndims,
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
    flatbuffer_size += _align_to_int64(sizeof(offsets_t) *
                                       (total_items_count * ndims + 1));  // sizes offsets
    flatbuffer_size += _align_to_int64(
        sizeof(sizes_t) * total_items_count);  // storage indices, must use signed type
    flatbuffer_size += _align_to_int64(user_data_size);   // user data
    flatbuffer_size += _align_to_int64(sizeof(int64_t));  // format id
    return flatbuffer_size;
  }

  Status initializeNestedArray(int64_t ndims,
                               int64_t total_items_count,
                               int64_t total_sizes_count,
                               int64_t total_values_count,
                               ValueType value_type,
                               const int8_t* null_value_ptr,
                               const int8_t* user_data_ptr,
                               size_t user_data_size) {
    auto* base = getBaseWorker();
    base->format_id = NestedArrayFormatId;
    size_t value_size = get_size(value_type);
    base->flatbuffer_size = computeBufferSizeNestedArray(ndims,
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
    metadata->ndims = ndims;
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
        worker, sizes_offsets, sizeof(offsets_t) * (total_items_count * ndims + 1));
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
    sizes_t* storage_indices = get_storage_indices();
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
        static_cast<int64_t>(base->format_id);

    if (isFlatBuffer(buffer)) {
      // make sure that initialization leads to a valid FlatBuffer
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
    }
    ((int64_t*)buffer)[base->flatbuffer_size / sizeof(int64_t) - 1] =
        static_cast<int64_t>(format_id);
  }

  // Low-level API

  inline size_t getNDims() const {
    if (isNestedArray()) {
      return getNestedArrayMetadata()->ndims;
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
      default:
        break;
    }
    static int64_t dummy = -1;
    return dummy;
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
      default:
        return nullptr;
    }
    return buffer + offset;
  }

  // Return the pointer to compressed indices buffer
  // To be deprecated in favor of NestedArray format
  HOST DEVICE inline int64_t* get_compressed_indices() {
    int64_t offset = 0;
    switch (format()) {
      case VarlenArrayFormatId:
        offset = getVarlenArrayWorker()->compressed_indices_offset;
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
  HOST DEVICE inline int64_t* get_storage_indices_old() {
    int64_t offset = 0;
    switch (format()) {
      case VarlenArrayFormatId:
        offset = getVarlenArrayWorker()->storage_indices_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int64_t*>(buffer + offset);
  }

  // To be deprecated in favor of NestedArray format
  inline const int64_t* get_storage_indices_old() const {
    int64_t offset = 0;
    switch (format()) {
      case VarlenArrayFormatId:
        offset = getVarlenArrayWorker()->storage_indices_offset;
        break;
      default:
        return nullptr;
    }
    return reinterpret_cast<int64_t*>(buffer + offset);
  }

  HOST DEVICE inline sizes_t* get_storage_indices() {
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

  inline const sizes_t* get_storage_indices() const {
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

  inline sizes_t get_storage_index(const int64_t index) const {
    return get_storage_indices()[index];
  }

  // High-level API

  // This getLength method is a worker method of accessing the
  // flatbuffer content.
  template <size_t NDIM>
  Status getLength(const int64_t index[NDIM], const size_t n, size_t& length) const {
    if (!isNestedArray()) {
      RETURN_ERROR(NotSupportedFormatError);
    }
    const size_t ndims = getNDims();
    if (n == 0) {
      length = itemsCount();
      return Success;
    }
    if (n > ndims + 1) {
      RETURN_ERROR(DimensionalityError);
    }
    const auto storage_index = get_storage_index(index[0]);
    const auto* values_offsets = get_values_offsets();
    const auto values_offset = values_offsets[storage_index];
    if (values_offset < 0) {  // NULL item
      length = 0;
      return Success;
    }
    const auto* sizes_offsets = get_sizes_offsets();
    const auto* sizes_buffer = get_sizes_buffer();
    const auto sizes_offset = sizes_offsets[storage_index * ndims];
    switch (n) {
      case 1: {
        length = sizes_buffer[sizes_offset];
      } break;
      case 2: {
        const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
        if (index[1] < 0 || index[1] >= sizes_buffer[sizes_offset]) {
          RETURN_ERROR(SubIndexError);
        }
        length = sizes_buffer[sizes2_offset + index[1]];
      } break;
      case 3: {
        const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
        const auto sizes3_offset = sizes_offsets[storage_index * ndims + 2];
        if (index[1] < 0 || index[1] >= sizes_buffer[sizes_offset]) {
          RETURN_ERROR(SubIndexError);
        }
        if (index[2] < 0 || index[2] >= sizes_buffer[sizes2_offset + index[1]]) {
          RETURN_ERROR(SubIndexError);
        }
        offsets_t soffset = 0;
        for (int64_t i = 0; i < index[1]; i++) {
          soffset += sizes_buffer[sizes2_offset + i];
        }
        length = sizes_buffer[sizes3_offset + soffset + index[2]];
      } break;
      default:
        RETURN_ERROR(NotImplementedError);
        break;
    }
    return Success;
  }

  // This getItem method is a worker method of accessing the
  // flatbuffer content.
  template <size_t NDIM>
  Status getItemWorker(const int64_t index[NDIM],
                       const size_t n,
                       int8_t*& values,
                       int32_t& nof_values,
                       int32_t* sizes_buffers[NDIM],
                       int32_t sizes_lengths[NDIM],
                       int32_t& nof_sizes,
                       bool& is_null) {
    values = nullptr;
    nof_values = 0;
    nof_sizes = 0;
    is_null = true;

    if (format() != NestedArrayFormatId) {
      RETURN_ERROR(NotSupportedFormatError);
    }

    const size_t ndims = getNDims();
    if (n <= 0 || n > ndims + 1) {
      RETURN_ERROR(DimensionalityError);
    }
    // clang-format off
    /*
      multipolygon (ndims == 3):

        n == 0 means return a column of multipolygons: flatbuffer, getLenght returns
          itemsCount()

        n == 1 means return a multipolygon: values, sizes(=sizes_buffers[1]),
          sizes_of_sizes(=sizes_buffers[0]), getLength returns
          len(sizes_of_sizes)(=sizes_lengths[0])

        n == 2 means return a polygon: values, sizes, getLength
          returns len(sizes)

        n == 3 means return a linestring: values, getLength returns
          len(values)

        n == 4 means return a point: value, getLength returns 0 [NOTIMPL]

      polygon/multilinestring (ndims == 2):

        n == 0 means return a column of polygons/multilinestring:
          flatbuffer, getLenght returns itemsCount()

        n == 1 means return a polygon/multilinestring: values, sizes,
          getLength returns len(sizes)

        n == 2 means return a linestring: values, getLength
          returns len(values)

        n == 3 means return a point: value, getLength returns 0 [NOTIMPL]

      linestring/multipoint (ndims == 1):

        n == 0 means return a column of linestring/multipoint:
          flatbuffer, getLenght returns itemsCount()

        n == 1 means return a linestring: values, getLength returns
          len(values)

        n == 2 means return a point: value, getLength returns 0 [NOTIMPL]

    */
    // clang-format off
    const auto storage_index = get_storage_index(index[0]);
    const auto* values_offsets = get_values_offsets();
    const auto values_offset = values_offsets[storage_index];
    if (values_offset < 0) {
      return Success;
    }
    is_null = false;
    const auto* sizes_offsets = get_sizes_offsets();
    auto* sizes_buffer = get_sizes_buffer();
    auto* values_buffer = get_values_buffer();
    const auto valuesize = getValueSize();
    const auto next_values_offset = values_offsets[storage_index + 1];

    const auto sizes_offset = sizes_offsets[storage_index * ndims];
    nof_sizes = ndims - n;
    switch (n) {
      case 1: {
        if (next_values_offset < 0) {
          nof_values = -(next_values_offset + 1) - values_offset;
        } else {
          nof_values = next_values_offset - values_offset;
        }
        values = values_buffer + values_offset * valuesize;
        switch (ndims) {
          case 3: {
            const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
            const auto sizes3_offset = sizes_offsets[storage_index * ndims + 2];
            sizes_buffers[0] = sizes_buffer + sizes2_offset;
            sizes_buffers[1] = sizes_buffer + sizes3_offset;
            sizes_lengths[0] = sizes_buffer[sizes_offset];
            sizes_lengths[1] = sizes_offsets[storage_index * ndims + 3] - sizes3_offset;
            break;
          }
          case 2: {
            const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
            sizes_buffers[0] = sizes_buffer + sizes2_offset;
            sizes_lengths[0] = sizes_buffer[sizes_offset];
            break;
          }
          case 1:
            break;
          default:
            FLATBUFFER_UNREACHABLE();
            break;
        }
        break;
      }
      case 2: {
        const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
        if (index[1] < 0 || index[1] >= sizes_buffer[sizes_offset]) {
          RETURN_ERROR(SubIndexError);
        }
        offsets_t soffset = 0;
        for (int64_t i = 0; i < index[1]; i++) {
          soffset += sizes_buffer[sizes2_offset + i];
        }
        values = values_buffer + (values_offset + soffset) * valuesize;
        switch (ndims) {
          case 3: {
            const auto sizes3_offset = sizes_offsets[storage_index * ndims + 2];
            const sizes_t nsizes = sizes_buffer[sizes2_offset + index[1]];
            auto sizes_buf = sizes_buffer + sizes3_offset + soffset;
            sizes_buffers[0] = sizes_buf;
            sizes_lengths[0] = nsizes;
            nof_values = 0;
            for (int64_t i = 0; i < nsizes; i++) {
              nof_values += sizes_buf[i];
            }
            break;
          }
          case 2: {
            nof_values = sizes_buffer[sizes2_offset + index[1]];
            break;
          }
          default:
            FLATBUFFER_UNREACHABLE();
            break;
        }
        break;
      }
      case 3: {
        if (ndims != 3) {
          RETURN_ERROR(NotImplementedError);
        }
        const auto sizes2_offset = sizes_offsets[storage_index * ndims + 1];
        const auto sizes3_offset = sizes_offsets[storage_index * ndims + 2];
        if (index[1] < 0 || index[1] >= sizes_buffer[sizes_offset]) {
          RETURN_ERROR(SubIndexError);
        }
        if (index[2] < 0 || index[2] >= sizes_buffer[sizes2_offset + index[1]]) {
          RETURN_ERROR(SubIndexError);
        }

        int64_t i3 = 0;
        int64_t soffset = 0;
        int64_t voffset = 0;
        for (int64_t i = 0; i < index[1]; i++) {
          auto size2 = sizes_buffer[sizes2_offset + i];
          soffset += size2;
          for (int64_t j = 0; j < size2; j++) {
            voffset += sizes_buffer[sizes3_offset + i3];
            i3++;
          }
        }
        for (int64_t j = 0; j < index[2]; j++) {
          voffset += sizes_buffer[sizes3_offset + i3];
          i3++;
        }
        values = values_buffer + (values_offset + voffset) * valuesize;
        nof_values = sizes_buffer[sizes3_offset + soffset + index[2]];
        break;
      }
      default:
        RETURN_ERROR(NotImplementedError);
        break;
    }
    return Success;
  }

  template <size_t NDIM>
  struct NestedArrayItem {
    int8_t* values;
    int32_t nof_values;
    int32_t* sizes_buffers[NDIM];
    int32_t sizes_lengths[NDIM];
    int32_t nof_sizes;
    bool is_null;
  };

  template <size_t NDIM>
  Status getItem(const int64_t index, NestedArrayItem<NDIM>& result) {
    const int64_t index_[NDIM] = {index};
    return getItem<NDIM>(index_, 1, result);
  }

  template <size_t NDIM>
  Status getItem(const int64_t index[NDIM], const size_t n, NestedArrayItem<NDIM>& result) {
    return getItemWorker<NDIM>(index, n,
                         result.values,
                         result.nof_values,
                         result.sizes_buffers,
                         result.sizes_lengths,
                         result.nof_sizes,
                         result.is_null);
  }

  // This setItem method is a worker method of initializing the
  // flatbuffer content. It can be called once per index value.
  template <size_t NDIM, bool check_sizes=true>
  Status setItemWorker(const int64_t index,
                 const int8_t* values,
                 const int32_t nof_values,
                 const int32_t* const sizes_buffers[NDIM],
                 const int32_t sizes_lengths[NDIM],
                 const int32_t nof_sizes) {
    if (format() != NestedArrayFormatId) {
      RETURN_ERROR(NotSupportedFormatError);
    }
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    const int32_t ndims = getNDims();
    if (nof_sizes + 1 != ndims) {
      RETURN_ERROR(DimensionalityError);
    }

    auto* storage_indices = get_storage_indices();
    if (storage_indices[index] >= 0) {
      RETURN_ERROR(ItemAlreadySpecifiedError);
    }
    auto* worker = getNestedArrayWorker();
    const auto storage_index = worker->specified_items_count;
    storage_indices[index] = storage_index;
    worker->specified_items_count++;

    auto* values_offsets = get_values_offsets();
    auto* sizes_offsets = get_sizes_offsets();
    auto* sizes_buffer = get_sizes_buffer();
    auto* values_buffer = get_values_buffer();
    const auto* metadata = getNestedArrayMetadata();
    const auto valuesize = getValueSize();

    auto values_offset = values_offsets[storage_index];
    const auto sizes_offset = sizes_offsets[storage_index * ndims];
    if (values_offset + nof_values > metadata->total_values_count) {
      RETURN_ERROR(ValuesBufferTooSmallError);
    }

    switch (ndims) {
      case 1: {
        sizes_buffer[sizes_offset] = nof_values;
        sizes_offsets[storage_index * ndims + 1] = sizes_offset + 1;
      } break;
      case 2: {
        const auto sizes2_offset = sizes_offset + 1;
        if (sizes2_offset + sizes_lengths[0] > metadata->total_sizes_count) {
          RETURN_ERROR(SizesBufferTooSmallError);
        }
        sizes_buffer[sizes_offset] = sizes_lengths[0];
        if constexpr (check_sizes) {
          // check consistency of sizes and nof_values
          int32_t sum_of_sizes = 0;
          for (int32_t i=0; i < sizes_lengths[0]; i++) {
            sum_of_sizes += sizes_buffers[0][i];
          }
          if (nof_values != sum_of_sizes) {
            RETURN_ERROR(InconsistentSizesError);
          }
        }
        if (memcpy(sizes_buffer + sizes2_offset,
                   sizes_buffers[0],
                   sizes_lengths[0] * sizeof(sizes_t)) == nullptr) {
          RETURN_ERROR(MemoryError);
        }
        sizes_offsets[storage_index * ndims + 1] = sizes2_offset;
        sizes_offsets[storage_index * ndims + 2] = sizes2_offset + sizes_lengths[0];
      } break;
      case 3: {
        const auto sizes2_offset = sizes_offset + 1;
        const auto sizes3_offset = sizes2_offset + sizes_lengths[0];
        if (sizes2_offset + sizes_lengths[0] + sizes_lengths[1] >
            metadata->total_sizes_count) {
          RETURN_ERROR(SizesBufferTooSmallError);
        }
        sizes_buffer[sizes_offset] = sizes_lengths[0];
        if constexpr (check_sizes) {
          // check consistency of sizes of sizes and nof_sizes
          int32_t sum_of_sizes_of_sizes = 0;
          for (int32_t i=0; i < sizes_lengths[0]; i++) {
            sum_of_sizes_of_sizes += sizes_buffers[0][i];
          }
          if (sizes_lengths[1] != sum_of_sizes_of_sizes) {
            RETURN_ERROR(InconsistentSizesError);
          }
        }
        if (memcpy(sizes_buffer + sizes2_offset,
                   sizes_buffers[0],
                   sizes_lengths[0] * sizeof(sizes_t)) == nullptr) {
          RETURN_ERROR(MemoryError);
        }
        if constexpr (check_sizes) {
          // check consistency of sizes and nof_values
          int32_t sum_of_sizes = 0;
          for (int32_t i=0; i < sizes_lengths[1]; i++) {
            sum_of_sizes += sizes_buffers[1][i];
          }
          if (nof_values != sum_of_sizes) {
            RETURN_ERROR(InconsistentSizesError);
          }
        }
        if (memcpy(sizes_buffer + sizes3_offset,
                   sizes_buffers[1],
                   sizes_lengths[1] * sizeof(sizes_t)) == nullptr) {
          RETURN_ERROR(MemoryError);
        }
        sizes_offsets[storage_index * ndims + 1] = sizes2_offset;
        sizes_offsets[storage_index * ndims + 2] = sizes3_offset;
        sizes_offsets[storage_index * ndims + 3] = sizes3_offset + sizes_lengths[1];
      } break;
      default:
        FLATBUFFER_UNREACHABLE();
        break;
    }
    if (values != nullptr) {
      if (memcpy(values_buffer + values_offset * valuesize,
                 values,
                 nof_values * valuesize) == nullptr) {
        RETURN_ERROR(MemoryError);
      }
    }
    values_offsets[storage_index + 1] = values_offset + nof_values;
    return Success;
  }

  template <size_t NDIM=0, bool check_sizes=true>
  Status setItem(const int64_t index,
                 const int8_t* values_buf,
                 const int32_t nof_values) {
    const int32_t* const sizes_buffers[1] = {nullptr};
    int32_t sizes_lengths[1] = {0};
    return setItemWorker<1, check_sizes>(index,
                                            values_buf,
                                            nof_values,
                                            sizes_buffers,
                                            sizes_lengths,
                                            0);
  }

  template <size_t NDIM=1, bool check_sizes=true>
  Status setItem(const int64_t index,
                 const int8_t* values_buf,
                 const int32_t nof_values,
                 const int32_t* sizes_buf,
                 const int32_t nof_sizes) {
    const int32_t* const sizes_buffers[NDIM] = {sizes_buf};
    int32_t sizes_lengths[NDIM] = {nof_sizes};
    return setItemWorker<NDIM, check_sizes>(index,
                                            values_buf,
                                            nof_values,
                                            sizes_buffers,
                                            sizes_lengths,
                                            static_cast<int32_t>(NDIM));
  }

  template <size_t NDIM=2, bool check_sizes=true>
  Status setItem(const int64_t index,
                 const int8_t* values_buf,
                 const int32_t nof_values,
                 const int32_t* sizes_buf,
                 const int32_t nof_sizes,
                 const int32_t* sizes_of_sizes_buf,
                 const int32_t nof_sizes_of_sizes) {
    const int32_t* const sizes_buffers[NDIM] = {sizes_of_sizes_buf, sizes_buf};
    int32_t sizes_lengths[NDIM] = {nof_sizes_of_sizes, nof_sizes};
    return setItemWorker<NDIM, check_sizes>(index,
                                            values_buf,
                                            nof_values,
                                            sizes_buffers,
                                            sizes_lengths,
                                            static_cast<int32_t>(NDIM));
  }

  template <typename CT>
  Status getItem(const int64_t index,
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
    NestedArrayItem<2> item;
    Status status = getItem(index, item);
    if (status != Success) {
      return status;
    }
    if (item.is_null) {
      return Success;
    }
    if (item.nof_sizes != 1) {
      RETURN_ERROR(InconsistentSizesError);
    }
    const auto valuesize = getValueSize();
    const auto values_count = item.nof_values * valuesize / sizeof(CT);
    values.reserve(values_count);
    values.insert(values.end(),
                  reinterpret_cast<CT*>(item.values),
                  reinterpret_cast<CT*>(item.values) + values_count);

    sizes.reserve(item.sizes_lengths[0]);
    sizes.insert(sizes.end(),
                 reinterpret_cast<sizes_t*>(item.sizes_buffers[0]),
                 reinterpret_cast<sizes_t*>(item.sizes_buffers[0] + item.sizes_lengths[0] * sizeof(int32_t)));
  return Success;
  }

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
    NestedArrayItem<3> item;
    Status status = getItem(index, item);
    if (status != Success) {
      return status;
    }
    if (item.is_null) {
      return Success;
    }
    if (item.nof_sizes != 2) {
      RETURN_ERROR(InconsistentSizesError);
    }
    const auto valuesize = getValueSize();
    const auto values_count = item.nof_values * valuesize / sizeof(CT);
    values.reserve(values_count);
    values.insert(values.end(),
                  reinterpret_cast<CT*>(item.values),
                  reinterpret_cast<CT*>(item.values) + values_count);

    sizes.reserve(item.sizes_lengths[1]);
    sizes.insert(sizes.end(),
                 reinterpret_cast<sizes_t*>(item.sizes_buffers[1]),
                 reinterpret_cast<sizes_t*>(item.sizes_buffers[1] + item.sizes_lengths[1] * sizeof(int32_t)));

    sizes_of_sizes.reserve(item.sizes_lengths[0]);
    sizes_of_sizes.insert(sizes_of_sizes.end(),
                          reinterpret_cast<sizes_t*>(item.sizes_buffers[0]),
                          reinterpret_cast<sizes_t*>(item.sizes_buffers[0] + item.sizes_lengths[0] * sizeof(int32_t)));
  return Success;

  }

  template <typename CT, size_t NDIM=0>
  Status setItem(const int64_t index, const std::vector<CT>& arr) {
    if (getNDims() != NDIM + 1) {
      RETURN_ERROR(DimensionalityError);
    }
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
    const auto valuesize = getValueSize();
    auto sz = (arr.size() * sizeof(CT)) / valuesize;
    const int32_t* const sizes_buffers[1] = {nullptr};
    int32_t sizes_lengths[1] = {0};
    return setItemWorker<1, false>(index,
                                reinterpret_cast<const int8_t*>(arr.data()),
                                sz,
                                sizes_buffers,
                                sizes_lengths,
                                0);
  }

  template <typename CT, size_t NDIM=1, bool check_sizes=false>
  Status setItem(const int64_t index, const std::vector<std::vector<CT>>& item) {
    const auto valuesize = getValueSize();
    std::vector<int32_t> sizes;
    sizes.reserve(item.size());
    int32_t nof_values = 0;
    size_t nof_elements = 0;
    for (const auto& subitem: item) {
      const auto sz = (subitem.size() * sizeof(CT)) / valuesize;
      sizes.push_back(sz);
      nof_values += sz;
      nof_elements += subitem.size();
    }
    std::vector<CT> flatitem;
    flatitem.reserve(nof_elements);
    for (const auto& subitem: item) {
      flatitem.insert(flatitem.end(), subitem.begin(), subitem.end());
    }
    return setItem<CT, NDIM, check_sizes>(index, flatitem, sizes);
  }

  template <typename CT, size_t NDIM=2, bool check_sizes=false>
  Status setItem(const int64_t index,
                 const std::vector<std::vector<std::vector<CT>>>& item) {
    const auto valuesize = getValueSize();
    std::vector<int32_t> sizes_of_sizes;
    std::vector<int32_t> sizes;
    std::vector<CT> flatitem;
    sizes_of_sizes.reserve(item.size());
    size_t nof_sizes_of_sizes = 0;
    for (const auto& subitem: item) {
      sizes_of_sizes.push_back(subitem.size());
      nof_sizes_of_sizes += subitem.size();
    }
    sizes.reserve(nof_sizes_of_sizes);
    int32_t nof_values = 0;
    size_t nof_elements = 0;
    for (const auto& subitem: item) {
      for (const auto& subitem1: subitem) {
        const auto sz = (subitem1.size() * sizeof(CT)) / valuesize;
        sizes.push_back(sz);
        nof_values += sz;
        nof_elements += subitem1.size();
      }
    }
    flatitem.reserve(nof_elements);
    for (const auto& subitem: item) {
      for (const auto& subitem1: subitem) {
        flatitem.insert(flatitem.end(), subitem1.begin(), subitem1.end());
      }
    }
    return setItem<CT, NDIM, check_sizes>(index, flatitem, sizes, sizes_of_sizes);
  }

  template <typename CT, size_t NDIM=1, bool check_sizes=true>
  Status setItem(const int64_t index,
                 const std::vector<CT>& values,
                 const std::vector<int32_t>& sizes) {
    if (getNDims() != NDIM + 1) {
      RETURN_ERROR(DimensionalityError);
    }
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
    const auto valuesize = getValueSize();
    const int32_t nof_values = (values.size() * sizeof(CT)) / valuesize;
    return setItem<NDIM, check_sizes>(index,
                                      reinterpret_cast<const int8_t*>(values.data()),
                                      nof_values,
                                      sizes.data(),
                                      sizes.size());
  }

  template <typename CT, size_t NDIM=2, bool check_sizes=true>
  Status setItem(const int64_t index,
                 const std::vector<CT>& values,
                 const std::vector<int32_t>& sizes,
                 const std::vector<int32_t>& sizes_of_sizes) {
    if (getNDims() != NDIM + 1) {
      RETURN_ERROR(DimensionalityError);
    }
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
    const auto valuesize = getValueSize();
    const auto nof_values = (values.size() * sizeof(CT)) / valuesize;
    return setItem<NDIM, check_sizes>(index,
                                      reinterpret_cast<const int8_t*>(values.data()),
                                      nof_values,
                                      sizes.data(),
                                      sizes.size(),
                                      sizes_of_sizes.data(),
                                      sizes_of_sizes.size()
                                      );
  }

  // Set a new item with index and size (in bytes) and initialize its
  // elements from source buffer. The item values will be
  // uninitialized when source buffer is nullptr. If dest != nullptr
  // then the item's buffer pointer will be stored in *dest.
  // To be deprecated in favor of NestedArray format
  Status setItemOld(const int64_t index,
                 const int8_t* src,
                 const int64_t size,
                 int8_t** dest = nullptr) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case VarlenArrayFormatId: {
        int64_t& storage_count = get_storage_count();
        int64_t* compressed_indices = get_compressed_indices();
        int64_t* storage_indices = get_storage_indices_old();
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
      case VarlenArrayFormatId: {
        int64_t& storage_count = get_storage_count();
        int64_t* storage_indices = get_storage_indices_old();
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
      case VarlenArrayFormatId: {
        int64_t next_storage_count = get_storage_count();
        int64_t storage_count = next_storage_count - 1;
        int64_t* compressed_indices = get_compressed_indices();
        int64_t* storage_indices = get_storage_indices_old();
        int8_t* values = get_values();
        int64_t itemsize = dtypeSize();
        int64_t storage_index = storage_indices[index];
        if (storage_index == -1) {  // unspecified, so setting the item
          return setItemOld(index, src, size, nullptr);
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
  Status setNull(int64_t index) {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    if (isNestedArray()) {
      auto* storage_indices = get_storage_indices();
      if (storage_indices[index] >= 0) {
        RETURN_ERROR(ItemAlreadySpecifiedError);
      }
      auto* worker = getNestedArrayWorker();
      const auto storage_index = worker->specified_items_count;
      worker->specified_items_count++;
      storage_indices[index] = storage_index;
      const size_t ndims = getNDims();
      auto* sizes_buffer = get_sizes_buffer();
      auto* values_offsets = get_values_offsets();
      auto* sizes_offsets = get_sizes_offsets();
      const auto values_offset = values_offsets[storage_index];
      const auto sizes_offset = sizes_offsets[storage_index * ndims];
      sizes_buffer[sizes_offset] = 0;
      for (size_t i = 0; i < ndims; i++) {
        sizes_offsets[storage_index * ndims + i + 1] = sizes_offset + 1;
      }
      values_offsets[storage_index] = -(values_offset + 1);
      values_offsets[storage_index + 1] = values_offset;
      return Success;
    }
    // To be deprecated in favor of NestedArray format:
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }

    switch (format()) {
      case VarlenArrayFormatId: {
        int64_t* storage_indices = get_storage_indices_old();
        if (storage_indices[index] >= 0) {
          return ItemAlreadySpecifiedError;
        }
        return setNullNoValidation(index);
      }
      case GeoPointFormatId: {
        return setNullNoValidation(index);
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
      case VarlenArrayFormatId: {
        int64_t& storage_count = get_storage_count();
        int64_t* storage_indices = get_storage_indices_old();
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
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    if (isNestedArray()) {
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
      case VarlenArrayFormatId: {
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t* storage_indices = get_storage_indices_old();
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
      case VarlenArrayFormatId: {
        int8_t* values = get_values();
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t* storage_indices = get_storage_indices_old();
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
  Status getItemLength(const int64_t index, int64_t& length) const {
    if (index < 0 || index >= itemsCount()) {
      RETURN_ERROR(IndexError);
    }
    switch (format()) {
      case GeoPointFormatId:
        break;
      case VarlenArrayFormatId: {
        const int64_t* compressed_indices = get_compressed_indices();
        const int64_t* storage_indices = get_storage_indices_old();
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
                               (metadata->total_items_count * metadata->ndims + 1) *
                                   get_size(FLATBUFFER_OFFSETS_T_VALUE_TYPE),
                               FLATBUFFER_OFFSETS_T_VALUE_TYPE) +
                "]";
      result += ",\n  storage_indices=[" +
                bufferToString(
                    reinterpret_cast<const int8_t*>(get_storage_indices()),
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
      case GeoPointFormatId: {
        result += ", " + getGeoPointMetadata()->toString();
        result += ", " + getGeoPointWorker()->toString();
        break;
      }
      default:
        break;
    }

    switch (fmt) {
      case VarlenArrayFormatId: {
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
            const int64_t* values_buf = reinterpret_cast<const int64_t*>(get_values());
            std::vector<int64_t> values(values_buf, values_buf + numvalues);
            result += ::toString(values);
          } break;
          default:
            result += "[UNEXPECTED ITEMSIZE:" + std::to_string(itemsize) + "]";
        }

        const int64_t numitems = itemsCount();
        const int64_t* compressed_indices_buf = get_compressed_indices();
        std::vector<int64_t> compressed_indices(compressed_indices_buf,
                                                compressed_indices_buf + numitems + 1);
        result += ", compressed_indices=" + ::toString(compressed_indices);

        const int64_t* storage_indices_buf = get_storage_indices_old();
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
