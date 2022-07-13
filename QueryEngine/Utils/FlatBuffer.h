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
  a single flat buffer so that copying FlatBuffer instances becomes a
  single buffer copy operation. Flat buffers that store no pointer
  values can be straightforwardly copied in between different devices.

  FlatBuffer memory layout specification
  --------------------------------------

  The first 8 bytes of the buffer contains a FlatBuffer storage format
  id (see FlatBufferFormat below) that will determine how the rest of
  the bytes in the flat buffer will be interpreted. The next 8 bytes
  of the buffer contains the total size of the flat buffer --- this
  allows flat buffers passed around by a single pointer value without
  explicitly specifying the size of the buffer.

  The memory layout of a flatbuffer is (using units of 8 byte):

    | <format id>  | <flatbuffer size> | <data> ...                           | <format id>   |
    =
    |<-- 8-bytes-->|<-- 8-bytes ------>|<-- flatbuffer size minus 24 bytes -->|<-- 8-bytes -->|
    |<------------------------ flatbuffer size ---------------------------------------------->|

  where flatbuffer size is specified in bytes.

  VarlenArray format specification
  --------------------------------

  If a FlatBuffer instance uses VarlenArray format (format id is
  0x766c61) then the <data> part of the FlatBuffer is defined as follows:

    | <items count>  | <dtype size> | <max nof values> | <num specified items> | <offsets>    | <varlen array data>                |
    =
    |<--- 8-bytes -->|<--8-bytes -->|<-- 8-bytes ----->|<-- 8-bytes ---------->|<--24-bytes-->|<--flatbuffer size minus 80 bytes-->|
    |<------------------------- flatbuffer size minus 24 bytes ------------------------------------------------------------------->|

  where

    <items count> is the number of items (e.g. varlen arrays) that the
      flat buffer instance is holding. In the context of columns of
      arrays, the items count is the number of rows in a column. This
      is a user-specified parameter.

    <dtype size> is the byte size of a single element in an item
      (e.g. the size of varlen array element). For example, if dtype
      is double or int64 then dtype size is 8, for float or int32 it
      is 4, etc. Notice that flat buffer does not store the actual
      dtype of elements as it is unneeded from the perspective
      managing the memory of a flat buffer instance. We may reconsider
      this if needed.  User-specified parameter.

    <max nof values> is the maximum total number of elements in all
      items that the flat buffer instance can hold. The value defines
      the size of values buffer. User-specified parameter.

    <num specified items> is the number of items that has initial
      value 0 and that is incremented by one on each setItem or
      setNull call. The flat buffer is completely filled with items
      when <num specified items> becomes equal to <items count>. Used
      internally.

    <offsets> are precomputed offsets of data buffers. Used internally.

      | <values offset> | <compressed_indices offset> | <storage indices offset> |
      |<-- 8-bytes ---->|<-- 8-bytes ---------------->|<-- 8-bytes ------------->|

    <varlen array data> is

      | <values>                        | <compressed indices>           | <storage indices>          |
      =
      |<- (max nof values) * 8 bytes -> |<-- (num items + 1) * 8 bytes-->|<-- (num items) * 8 bytes-->|
      |<------------------------ flatbuffer size minus 56 bytes ------------------------------------->|

  and

  - values stores the elements of all items (e.g. the values of all
    varlen arrays). Item elements are contiguous within an item,
    however, the storage order of items can be arbitrary.

  - compressed indices contains the "cumulative sum" of storage
    indices.  Negative entries indicate null items.

  - storage indices defines the order of specifying items in the flat
    buffer.

  For the detailed description of values, compressed_indices, and
  storage_indices, as well as how empty arrays and null arrays are
  represented, see https://pearu.github.io/variable_length_arrays.html .

  FlatBuffer usage
  ----------------

  FlatBuffer implements various methods for accessing its content for
  retriving or storing data. These methods usually are provided as
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

#ifdef HAVE_TOSTRING
#include <ostream>
#endif

// Notice that the format value is used to recognize if a memory
// buffer uses some flat buffer format or not. To minimize chances for
// false positive test results, use a non-trival integer value when
// introducing new formats.
enum FlatBufferFormat {
  VarlenArray = 0x7661726c65634152,  // hex repr of 'varlenAR'
};

inline int64_t _align_to_int64(int64_t addr) {
  addr += sizeof(int64_t) - 1;
  return (int64_t)(((uint64_t)addr >> 3) << 3);
}

struct FlatBufferManager {
  enum Status {
    Success = 0,
    IndexError,
    SizeError,
    ItemExistsError,
    ItemUnspecifiedError,
    ValuesBufferTooSmallError,
    MemoryError,
    NotImplemnentedError,
    UnknownFormatError
  };

  enum VarlenArrayHeader {
    FORMAT_ID = 0,    // storage format id
    FLATBUFFER_SIZE,  // in bytes
    ITEMS_COUNT,      /* the number of items
                         (e.g. the number of
                         rows in a column) */
    DTYPE_SIZE,       /* the size of dtype of
                         the item elements, in
                         bytes */
    MAX_NOF_VALUES,   /* the upper bound to the
                         total number of values
                         in all items */
    STORAGE_COUNT,    /* the number of specified
                         items, incremented by
                         one on each
                         setNull/setItem call */
    VALUES_OFFSET,
    COMPRESSED_INDICES_OFFSET,
    STORAGE_INDICES_OFFSET,
    EOFHEADER
  };

  int8_t* buffer;

  // Check if a buffer contains FlatBuffer formatted data
  static bool isFlatBuffer(const void* buffer) {
    if (buffer) {
      // warning: assume that buffer size is at least 8 bytes
      FlatBufferFormat header_format =
          static_cast<FlatBufferFormat>(((int64_t*)buffer)[VarlenArrayHeader::FORMAT_ID]);
      if (header_format == FlatBufferFormat::VarlenArray) {
        int64_t flatbuffer_size = ((int64_t*)buffer)[VarlenArrayHeader::FLATBUFFER_SIZE];
        if (flatbuffer_size > 0) {
          FlatBufferFormat footer_format = static_cast<FlatBufferFormat>(
              ((int64_t*)buffer)[flatbuffer_size / sizeof(int64_t) - 1]);
          return footer_format == header_format;
        }
      }
    }
    return false;
  }

  // Return the format of FlatBuffer
  FlatBufferFormat format() const {
    return static_cast<FlatBufferFormat>(
        ((int64_t*)buffer)[VarlenArrayHeader::FORMAT_ID]);
  }

  // Return the allocation size of the the FlatBuffer storage, in bytes
  int64_t flatbufferSize() const {
    return ((int64_t*)buffer)[VarlenArrayHeader::FLATBUFFER_SIZE];
  }

  // VarlenArray support:
  static int64_t get_VarlenArray_flatbuffer_size(int64_t items_count,
                                                 int64_t max_nof_values,
                                                 int64_t dtype_size) {
    int64_t VarlenArray_buffer_header_size =
        VarlenArrayHeader::EOFHEADER * sizeof(int64_t);
    int64_t values_buffer_size = _align_to_int64(dtype_size * max_nof_values);
    return (VarlenArray_buffer_header_size  // see above
            + values_buffer_size            // size of values buffer, aligned to int64
            + (items_count + 1              // size of compressed_indices buffer
               + items_count                // size of storage_indices buffer
               + 1                          // footer format id
               ) * sizeof(int64_t));
  }

  // Initialize FlatBuffer for VarlenArray storage
  void initializeVarlenArray(int64_t items_count,
                             int64_t max_nof_values,
                             int64_t dtype_size) {
    const int64_t VarlenArray_buffer_header_size =
        VarlenArrayHeader::EOFHEADER * sizeof(int64_t);
    const int64_t values_buffer_size = _align_to_int64(dtype_size * max_nof_values);
    const int64_t compressed_indices_buffer_size = (items_count + 1) * sizeof(int64_t);
    const int64_t flatbuffer_size =
        get_VarlenArray_flatbuffer_size(items_count, max_nof_values, dtype_size);
    ((int64_t*)buffer)[VarlenArrayHeader::FORMAT_ID] =
        static_cast<int64_t>(FlatBufferFormat::VarlenArray);
    ((int64_t*)buffer)[VarlenArrayHeader::FLATBUFFER_SIZE] = flatbuffer_size;
    ((int64_t*)buffer)[VarlenArrayHeader::ITEMS_COUNT] = items_count;
    ((int64_t*)buffer)[VarlenArrayHeader::MAX_NOF_VALUES] = max_nof_values;
    ((int64_t*)buffer)[VarlenArrayHeader::DTYPE_SIZE] = dtype_size;
    ((int64_t*)buffer)[VarlenArrayHeader::STORAGE_COUNT] = 0;
    ((int64_t*)buffer)[VarlenArrayHeader::VALUES_OFFSET] = VarlenArray_buffer_header_size;
    ((int64_t*)buffer)[VarlenArrayHeader::COMPRESSED_INDICES_OFFSET] =
        ((const int64_t*)buffer)[VarlenArrayHeader::VALUES_OFFSET] + values_buffer_size;
    ((int64_t*)buffer)[VarlenArrayHeader::STORAGE_INDICES_OFFSET] =
        ((const int64_t*)buffer)[VarlenArrayHeader::COMPRESSED_INDICES_OFFSET] +
        compressed_indices_buffer_size;

    int64_t* compressed_indices = VarlenArray_compressed_indices();
    int64_t* storage_indices = VarlenArray_storage_indices();

    for (int i = 0; i < items_count; i++) {
      compressed_indices[i] = 0;
      storage_indices[i] = -1;
    }
    compressed_indices[items_count] = 0;
    // store footer format id in the last 8-bytes of the flatbuffer:
    ((int64_t*)buffer)[flatbuffer_size / sizeof(int64_t) - 1] =
        static_cast<int64_t>(FlatBufferFormat::VarlenArray);
  }

  // Return the number of items
  inline int64_t itemsCount() const {
    if (format() == FlatBufferFormat::VarlenArray) {
      return ((int64_t*)buffer)[VarlenArrayHeader::ITEMS_COUNT];
    }
    return -1;  // invalid value
  }

  // Return the size of dtype of the item elements, in bytes
  inline int64_t dtypeSize() const {
    return ((int64_t*)buffer)[VarlenArrayHeader::DTYPE_SIZE];
  }

  // Return the upper bound to the total number of values in all items
  inline int64_t VarlenArray_max_nof_values() const {
    return ((int64_t*)buffer)[VarlenArrayHeader::MAX_NOF_VALUES];
  }

  // Return the size of values buffer in bytes
  inline int64_t VarlenArray_values_buffer_size() const {
    return _align_to_int64(dtypeSize() * VarlenArray_max_nof_values());
  }

  // Return the number of specified items
  inline int64_t& VarlenArray_storage_count() {
    return ((int64_t*)buffer)[VarlenArrayHeader::STORAGE_COUNT];
  }

  // Return the pointer to values buffer
  inline int8_t* VarlenArray_values() {
    return buffer + ((const int64_t*)buffer)[VarlenArrayHeader::VALUES_OFFSET];
  }

  inline const int8_t* VarlenArray_values() const {
    return buffer + ((const int64_t*)buffer)[VarlenArrayHeader::VALUES_OFFSET];
  }

  // Return the pointer to compressed indices buffer
  inline int64_t* VarlenArray_compressed_indices() {
    return reinterpret_cast<int64_t*>(
        buffer + ((const int64_t*)buffer)[VarlenArrayHeader::COMPRESSED_INDICES_OFFSET]);
  }
  inline const int64_t* VarlenArray_compressed_indices() const {
    return reinterpret_cast<const int64_t*>(
        buffer + ((const int64_t*)buffer)[VarlenArrayHeader::COMPRESSED_INDICES_OFFSET]);
  }

  // Return the pointer to storage indices buffer
  inline int64_t* VarlenArray_storage_indices() {
    return reinterpret_cast<int64_t*>(
        buffer + ((const int64_t*)buffer)[VarlenArrayHeader::STORAGE_INDICES_OFFSET]);
  }
  inline const int64_t* VarlenArray_storage_indices() const {
    return reinterpret_cast<const int64_t*>(
        buffer + ((const int64_t*)buffer)[VarlenArrayHeader::STORAGE_INDICES_OFFSET]);
  }

  // Set a new item with index and size (in bytes) and initialize its
  // elements from source buffer. The item values will be
  // uninitialized when source buffer is nullptr. If dest != nullptr
  // then the item's buffer pointer will be stored in *dest.
  Status setItem(int64_t index,
                 const int8_t* src,
                 int64_t size,
                 int8_t** dest = nullptr) {
    if (format() == FlatBufferFormat::VarlenArray) {
      if (index < 0 || index >= itemsCount()) {
        return IndexError;
      }
      int64_t& storage_count = VarlenArray_storage_count();
      int64_t* compressed_indices = VarlenArray_compressed_indices();
      int64_t* storage_indices = VarlenArray_storage_indices();
      const int64_t itemsize = dtypeSize();
      if (size % itemsize != 0) {
        return SizeError;  // size must be multiple of itemsize. Perhaps size is not in
                           // bytes?
      }
      if (storage_indices[index] >= 0) {
        return ItemExistsError;
      }
      const int64_t cindex = compressed_indices[storage_count];
      const int64_t values_buffer_size = VarlenArray_values_buffer_size();
      const int64_t csize = cindex * itemsize;
      if (csize + size > values_buffer_size) {
        return ValuesBufferTooSmallError;
      }
      return setItemNoValidation(index, src, size, dest);
    }
    return UnknownFormatError;
  }

  // Same as setItem but performs no input validation
  Status setItemNoValidation(int64_t index,
                             const int8_t* src,
                             int64_t size,
                             int8_t** dest) {
    int64_t& storage_count = VarlenArray_storage_count();
    int64_t* storage_indices = VarlenArray_storage_indices();
    int64_t* compressed_indices = VarlenArray_compressed_indices();
    int8_t* values = VarlenArray_values();
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
    return Success;
  }

  // Set a new item with index and size but without initializing item
  // elements. The buffer pointer of the new item will be stored in
  // *dest if dest != nullptr. Inputs are not validated!
  Status setEmptyItemNoValidation(int64_t index, int64_t size, int8_t** dest) {
    return setItemNoValidation(index, nullptr, size, dest);
  }

  Status concatItem(int64_t index, const int8_t* src, int64_t size) {
    if (format() == FlatBufferFormat::VarlenArray) {
      if (index < 0 || index >= itemsCount()) {
        return IndexError;
      }
      int64_t next_storage_count = VarlenArray_storage_count();
      int64_t storage_count = next_storage_count - 1;
      int64_t* compressed_indices = VarlenArray_compressed_indices();
      int64_t* storage_indices = VarlenArray_storage_indices();
      int8_t* values = VarlenArray_values();
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
        return NotImplemnentedError;  // todo: support concat to null when last
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
    return UnknownFormatError;
  }

  // Set item with index as a null item
  Status setNull(int64_t index) {
    if (format() == FlatBufferFormat::VarlenArray) {
      if (index < 0 || index >= itemsCount()) {
        return IndexError;
      }
      int64_t* storage_indices = VarlenArray_storage_indices();
      if (storage_indices[index] >= 0) {
        return ItemExistsError;
      }
      return setNullNoValidation(index);
    }
    return UnknownFormatError;
  }

  // Same as setNull but performs no input validation
  Status setNullNoValidation(int64_t index) {
    int64_t& storage_count = VarlenArray_storage_count();
    int64_t* storage_indices = VarlenArray_storage_indices();
    int64_t* compressed_indices = VarlenArray_compressed_indices();
    const int64_t cindex = compressed_indices[storage_count];
    storage_indices[index] = storage_count;
    compressed_indices[storage_count] = -(cindex + 1);
    compressed_indices[storage_count + 1] = cindex;
    storage_count++;
    return Success;
  }

  // Check if the item is unspecified or null.
  Status isNull(int64_t index, bool& is_null) const {
    if (format() == FlatBufferFormat::VarlenArray) {
      if (index < 0 || index >= itemsCount()) {
        return IndexError;
      }
      const int64_t* compressed_indices = VarlenArray_compressed_indices();
      const int64_t* storage_indices = VarlenArray_storage_indices();
      const int64_t storage_index = storage_indices[index];
      if (storage_index < 0) {
        return ItemUnspecifiedError;
      }
      is_null = (compressed_indices[storage_index] < 0);
      return Success;
    }
    return UnknownFormatError;
  }

  // Get item at index by storing its size (in bytes), values buffer,
  // and nullity information to the corresponding pointer
  // arguments. Return 0 on success.
  Status getItem(int64_t index, int64_t& size, int8_t*& dest, bool& is_null) {
    if (format() == FlatBufferFormat::VarlenArray) {
      if (index < 0 || index >= itemsCount()) {
        return IndexError;
      }
      int8_t* values = VarlenArray_values();
      const int64_t* compressed_indices = VarlenArray_compressed_indices();
      const int64_t* storage_indices = VarlenArray_storage_indices();
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
    } else {
      return UnknownFormatError;
    }
  }

#ifdef HAVE_TOSTRING
  std::string toString() const {
    if (buffer == nullptr) {
      return ::typeName(this) + "[UNINITIALIZED]";
    }
    switch (format()) {
      case FlatBufferFormat::VarlenArray: {
        const int64_t* buf = reinterpret_cast<const int64_t*>(buffer);
        std::vector<int64_t> v(buf, buf + flatbufferSize() / sizeof(int64_t));
        return ::typeName(this) + ::toString(v);
      }
      default:;
    }
    return ::typeName(this) + "[UNKNOWN FORMAT]";
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
    case FlatBufferManager::SizeError:
      os << "SizeError";
      break;
    case FlatBufferManager::ItemExistsError:
      os << "ItemExistsError";
      break;
    case FlatBufferManager::ItemUnspecifiedError:
      os << "UnknownFormatError";
      break;
    case FlatBufferManager::ValuesBufferTooSmallError:
      os << "ValuesBufferTooSmallError";
      break;
    case FlatBufferManager::MemoryError:
      os << "MemoryError";
      break;
    case FlatBufferManager::UnknownFormatError:
      os << "UnknownFormatError";
      break;
    case FlatBufferManager::NotImplemnentedError:
      os << "NotImplemnentedError";
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
