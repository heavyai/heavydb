/*
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

#include "ArrowStorageUtils.h"

// TODO: use <Shared/threading.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

#include <iostream>

using namespace std::string_literals;

namespace {

template <
    typename V,
    std::enable_if_t<!std::is_same_v<V, bool> && std::is_integral<V>::value, int> = 0>
inline V inline_null_value() {
  return inline_int_null_value<V>();
}

template <typename V, std::enable_if_t<std::is_same_v<V, bool>, int> = 0>
inline int8_t inline_null_value() {
  return inline_int_null_value<int8_t>();
}

template <typename V, std::enable_if_t<std::is_floating_point<V>::value, int> = 0>
inline V inline_null_value() {
  return inline_fp_null_value<V>();
}

template <
    typename V,
    std::enable_if_t<!std::is_same_v<V, bool> && std::is_integral<V>::value, int> = 0>
inline V inline_null_array_value() {
  return inline_int_null_array_value<V>();
}

template <typename V, std::enable_if_t<std::is_same_v<V, bool>, int> = 0>
inline int8_t inline_null_array_value() {
  return inline_int_null_array_value<int8_t>();
}

template <typename V, std::enable_if_t<std::is_floating_point<V>::value, int> = 0>
inline V inline_null_array_value() {
  return inline_fp_null_array_value<V>();
}

void convertBoolBitmapBufferWithNulls(int8_t* dst,
                                      const uint8_t* src,
                                      const uint8_t* bitmap,
                                      size_t offs,
                                      size_t length,
                                      int8_t null_value) {
  size_t start_full_byte = (offs + 7) / 8;
  size_t end_full_byte = (offs + length) / 8;
  size_t head_bits = (offs % 8) ? std::min(8 - (offs % 8), length) : 0;
  size_t tail_bits = head_bits == length ? 0 : (offs + length) % 8;
  size_t dst_offs = 0;

  for (size_t i = 0; i < head_bits; ++i) {
    auto is_null = (~bitmap[offs / 8] >> (offs % 8 + i)) & 1;
    auto val = (src[offs / 8] >> (offs % 8 + i)) & 1;
    dst[dst_offs++] = is_null ? null_value : val;
  }

  for (size_t bitmap_idx = start_full_byte; bitmap_idx < end_full_byte; ++bitmap_idx) {
    auto source = src[bitmap_idx];
    auto inversed_bitmap = ~bitmap[bitmap_idx];
    for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
      auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
      auto val = (source >> bitmap_offset) & 1;
      dst[dst_offs++] = is_null ? null_value : val;
    }
  }

  for (size_t i = 0; i < tail_bits; ++i) {
    auto is_null = (~bitmap[end_full_byte] >> i) & 1;
    auto val = (src[end_full_byte] >> i) & 1;
    dst[dst_offs++] = is_null ? null_value : val;
  }
}

// offs - an offset is source buffer it bits
// length - a number of bits to convert
void convertBoolBitmapBufferWithoutNulls(int8_t* dst,
                                         const uint8_t* src,
                                         size_t offs,
                                         size_t length) {
  size_t start_full_byte = (offs + 7) / 8;
  size_t end_full_byte = (offs + length) / 8;
  size_t head_bits = (offs % 8) ? std::min(8 - (offs % 8), length) : 0;
  size_t tail_bits = head_bits == length ? 0 : (offs + length) % 8;
  size_t dst_offs = 0;

  for (size_t i = 0; i < head_bits; ++i) {
    dst[dst_offs++] = (src[offs / 8] >> (offs % 8 + i)) & 1;
  }

  for (size_t bitmap_idx = start_full_byte; bitmap_idx < end_full_byte; ++bitmap_idx) {
    auto source = src[bitmap_idx];
    for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
      dst[dst_offs++] = (source >> bitmap_offset) & 1;
    }
  }

  for (size_t i = 0; i < tail_bits; ++i) {
    dst[dst_offs++] = (src[end_full_byte] >> i) & 1;
  }
}

template <typename T>
void copyArrayDataReplacingNulls(T* dst,
                                 std::shared_ptr<arrow::Array> arr,
                                 size_t offs,
                                 size_t length) {
  auto src = reinterpret_cast<const T*>(arr->data()->buffers[1]->data());
  auto null_value = inline_null_value<T>();

  if (arr->null_count() == arr->length()) {
    if constexpr (std::is_same_v<T, bool>) {
      std::fill(reinterpret_cast<int8_t*>(dst),
                reinterpret_cast<int8_t*>(dst + length),
                null_value);
    } else {
      std::fill(dst, dst + length, null_value);
    }
  } else if (arr->null_count() == 0) {
    if constexpr (std::is_same_v<T, bool>) {
      convertBoolBitmapBufferWithoutNulls(reinterpret_cast<int8_t*>(dst),
                                          reinterpret_cast<const uint8_t*>(src),
                                          offs,
                                          length);
    } else {
      std::copy(src + offs, src + offs + length, dst);
    }
  } else {
    const uint8_t* bitmap_data = arr->null_bitmap_data();
    if constexpr (std::is_same_v<T, bool>) {
      convertBoolBitmapBufferWithNulls(reinterpret_cast<int8_t*>(dst),
                                       reinterpret_cast<const uint8_t*>(src),
                                       bitmap_data,
                                       offs,
                                       length,
                                       null_value);
    } else {
      size_t start_full_byte = (offs + 7) / 8;
      size_t end_full_byte = (offs + length) / 8;
      size_t head_bits = (offs % 8) ? std::min(8 - (offs % 8), length) : 0;
      size_t tail_bits = head_bits == length ? 0 : (offs + length) % 8;
      size_t dst_offs = 0;
      size_t src_offs = offs;

      for (size_t i = 0; i < head_bits; ++i) {
        auto is_null = (~bitmap_data[offs / 8] >> (offs % 8 + i)) & 1;
        auto val = src[src_offs++];
        dst[dst_offs++] = is_null ? null_value : val;
      }

      for (size_t bitmap_idx = start_full_byte; bitmap_idx < end_full_byte;
           ++bitmap_idx) {
        auto inversed_bitmap = ~bitmap_data[bitmap_idx];
        for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
          auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
          auto val = src[src_offs++];
          dst[dst_offs++] = is_null ? null_value : val;
        }
      }

      for (size_t i = 0; i < tail_bits; ++i) {
        auto is_null = (~bitmap_data[end_full_byte] >> i) & 1;
        auto val = src[src_offs++];
        dst[dst_offs++] = is_null ? null_value : val;
      }
    }
  }
}

template <typename T>
void copyArrayDataReplacingNulls(T* dst, std::shared_ptr<arrow::Array> arr) {
  copyArrayDataReplacingNulls(dst, arr, 0, arr->length());
}

template <typename T>
std::shared_ptr<arrow::ChunkedArray> replaceNullValuesImpl(
    std::shared_ptr<arrow::ChunkedArray> arr) {
  if (!std::is_same_v<T, bool> && arr->null_count() == 0) {
    // for boolean columns we still need to convert bitmaps to array
    return arr;
  }

  auto resultBuf = arrow::AllocateBuffer(sizeof(T) * arr->length()).ValueOrDie();
  auto resultData = reinterpret_cast<T*>(resultBuf->mutable_data());

  tbb::parallel_for(tbb::blocked_range<int>(0, arr->num_chunks()),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int c = r.begin(); c != r.end(); ++c) {
                        size_t offset = 0;
                        for (int i = 0; i < c; i++) {
                          offset += arr->chunk(i)->length();
                        }
                        copyArrayDataReplacingNulls<T>(resultData + offset,
                                                       arr->chunk(c));
                      }
                    });

  std::shared_ptr<arrow::Array> array;
  if constexpr (std::is_same_v<T, bool>) {
    array = std::make_shared<arrow::Int8Array>(arr->length(), std::move(resultBuf));
  } else {
    array = std::make_shared<arrow::PrimitiveArray>(
        arr->type(), arr->length(), std::move(resultBuf));
  }

  return std::make_shared<arrow::ChunkedArray>(array);
}

// With current parser we used timestamp convertor for time type.
// It sets 1899-12-31 date (-2209075200 secs from epoch) for parsed values
// and therefore values should be adjusted.
void copyTimestampToTimeReplacingNulls(int64_t* dst, std::shared_ptr<arrow::Array> arr) {
  auto src = reinterpret_cast<const int64_t*>(arr->data()->buffers[1]->data());
  auto null_value = inline_null_value<int64_t>();
  auto length = arr->length();

  if (arr->null_count() == length) {
    std::fill(dst, dst + length, null_value);
  } else if (arr->null_count() == 0) {
    std::transform(src, src + length, dst, [](int64_t v) { return v + 2209075200; });
  } else {
    const uint8_t* bitmap_data = arr->null_bitmap_data();

    size_t end_full_byte = length / 8;
    size_t tail_bits = length % 8;
    size_t offs = 0;

    for (size_t bitmap_idx = 0; bitmap_idx < end_full_byte; ++bitmap_idx) {
      auto inversed_bitmap = ~bitmap_data[bitmap_idx];
      for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
        auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
        auto val = src[offs] + 2209075200;
        dst[offs] = is_null ? null_value : val;
        ++offs;
      }
    }

    for (size_t i = 0; i < tail_bits; ++i) {
      auto is_null = (~bitmap_data[end_full_byte] >> i) & 1;
      auto val = src[offs] + 2209075200;
      dst[offs] = is_null ? null_value : val;
      ++offs;
    }
  }
}

std::shared_ptr<arrow::ChunkedArray> convertTimestampToTimeReplacingNulls(
    std::shared_ptr<arrow::ChunkedArray> arr) {
  auto resultBuf = arrow::AllocateBuffer(sizeof(int64_t) * arr->length()).ValueOrDie();
  auto resultData = reinterpret_cast<int64_t*>(resultBuf->mutable_data());

  tbb::parallel_for(tbb::blocked_range<int>(0, arr->num_chunks()),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int c = r.begin(); c != r.end(); ++c) {
                        size_t offset = 0;
                        for (int i = 0; i < c; i++) {
                          offset += arr->chunk(i)->length();
                        }
                        copyTimestampToTimeReplacingNulls(resultData + offset,
                                                          arr->chunk(c));
                      }
                    });

  auto array = std::make_shared<arrow::Int64Array>(arr->length(), std::move(resultBuf));
  return std::make_shared<arrow::ChunkedArray>(array);
}

template <typename ArrowIntType, typename ResultIntType>
void copyDateReplacingNulls(ResultIntType* dst, std::shared_ptr<arrow::Array> arr) {
  auto src = reinterpret_cast<const ArrowIntType*>(arr->data()->buffers[1]->data());
  auto null_value = inline_null_value<ResultIntType>();
  auto length = arr->length();

  if (arr->null_count() == length) {
    std::fill(dst, dst + length, null_value);
  } else if (arr->null_count() == 0) {
    std::transform(src, src + length, dst, [](ArrowIntType v) {
      return static_cast<ResultIntType>(v);
    });
  } else {
    const uint8_t* bitmap_data = arr->null_bitmap_data();

    size_t end_full_byte = length / 8;
    size_t tail_bits = length % 8;
    size_t offs = 0;

    for (size_t bitmap_idx = 0; bitmap_idx < end_full_byte; ++bitmap_idx) {
      auto inversed_bitmap = ~bitmap_data[bitmap_idx];
      for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
        auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
        auto val = static_cast<ArrowIntType>(src[offs]);
        dst[offs] = is_null ? null_value : val;
        ++offs;
      }
    }

    for (size_t i = 0; i < tail_bits; ++i) {
      auto is_null = (~bitmap_data[end_full_byte] >> i) & 1;
      auto val = static_cast<ArrowIntType>(src[offs]);
      dst[offs] = is_null ? null_value : val;
      ++offs;
    }
  }
}

template <typename ArrowIntType, typename ResultIntType>
std::shared_ptr<arrow::ChunkedArray> convertDateReplacingNulls(
    std::shared_ptr<arrow::ChunkedArray> arr) {
  auto resultBuf =
      arrow::AllocateBuffer(sizeof(ResultIntType) * arr->length()).ValueOrDie();
  auto resultData = reinterpret_cast<ResultIntType*>(resultBuf->mutable_data());

  tbb::parallel_for(tbb::blocked_range<int>(0, arr->num_chunks()),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int c = r.begin(); c != r.end(); ++c) {
                        size_t offset = 0;
                        for (int i = 0; i < c; i++) {
                          offset += arr->chunk(i)->length();
                        }
                        copyDateReplacingNulls<ArrowIntType, ResultIntType>(
                            resultData + offset, arr->chunk(c));
                      }
                    });

  using ResultArrowType = typename arrow::CTypeTraits<ResultIntType>::ArrowType;
  using ArrayType = typename arrow::TypeTraits<ResultArrowType>::ArrayType;

  auto array = std::make_shared<ArrayType>(arr->length(), std::move(resultBuf));
  return std::make_shared<arrow::ChunkedArray>(array);
}

template <typename T>
std::shared_ptr<arrow::ChunkedArray> replaceNullValuesVarlenArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr) {
  size_t elems_count = 0;
  int32_t arrays_count = 0;
  std::vector<int32_t> out_elem_offsets;
  std::vector<int32_t> out_offset_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  out_offset_offsets.reserve(arr->num_chunks());

  for (auto& chunk : arr->chunks()) {
    auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(chunk);
    CHECK(chunk_list);

    out_offset_offsets.push_back(arrays_count);
    out_elem_offsets.push_back(static_cast<int32_t>(elems_count));

    auto offset_data = chunk_list->data()->GetValues<uint32_t>(1);
    arrays_count += chunk->length();
    elems_count += offset_data[chunk->length()] - offset_data[0];
  }

  if (elems_count > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("Input arrow array is too big for conversion.");
  }

  auto offset_buf =
      arrow::AllocateBuffer(sizeof(int32_t) * arr->length() + 1).ValueOrDie();
  auto offset_ptr = reinterpret_cast<int32_t*>(offset_buf->mutable_data());

  auto elem_buf = arrow::AllocateBuffer(sizeof(T) * elems_count).ValueOrDie();
  auto elem_ptr = reinterpret_cast<T*>(elem_buf->mutable_data());

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr->num_chunks()),
      [&](const tbb::blocked_range<int>& r) {
        for (int c = r.begin(); c != r.end(); ++c) {
          auto elem_offset = out_elem_offsets[c];
          auto offset_offset = out_offset_offsets[c];
          auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
          auto elem_array = chunk_list->values();

          auto offset_data = chunk_list->data()->GetValues<uint32_t>(1);
          if (chunk_list->null_count() == 0) {
            auto first_elem_offset = offset_data[0];
            auto elems_to_copy = offset_data[chunk_list->length()] - first_elem_offset;
            copyArrayDataReplacingNulls<T>(
                elem_ptr + elem_offset, elem_array, first_elem_offset, elems_to_copy);
            std::transform(offset_data,
                           offset_data + chunk_list->length(),
                           offset_ptr + offset_offset,
                           [offs = elem_offset - first_elem_offset](uint32_t val) {
                             return (val + offs) * sizeof(T);
                           });
          } else {
            bool use_negative_offset = false;
            for (int64_t i = 0; i < chunk_list->length(); ++i) {
              offset_ptr[offset_offset++] = use_negative_offset ? -elem_offset * sizeof(T)
                                                                : elem_offset * sizeof(T);
              if (chunk_list->IsNull(i)) {
                use_negative_offset = true;
              } else {
                use_negative_offset = false;
                auto elems_to_copy = offset_data[i + 1] - offset_data[i];
                copyArrayDataReplacingNulls<T>(
                    elem_ptr + elem_offset, elem_array, offset_data[i], elems_to_copy);
                elem_offset += elems_to_copy;
              }
            }
          }
        }
      });
  auto last_chunk = arr->chunk(arr->num_chunks() - 1);
  offset_ptr[arr->length()] = static_cast<int32_t>(
      last_chunk->IsNull(last_chunk->length() - 1) ? -elems_count * sizeof(T)
                                                   : elems_count * sizeof(T));

  std::shared_ptr<arrow::Array> elem_array;
  auto list_type = arr->type();
  if constexpr (std::is_same_v<T, bool>) {
    elem_array = std::make_shared<arrow::Int8Array>(elems_count, std::move(elem_buf));
    list_type = arrow::list(arrow::int8());
  } else {
    using ElemsArrowType = typename arrow::CTypeTraits<T>::ArrowType;
    using ElemsArrayType = typename arrow::TypeTraits<ElemsArrowType>::ArrayType;
    elem_array = std::make_shared<ElemsArrayType>(elems_count, std::move(elem_buf));
  }

  auto list_array = std::make_shared<arrow::ListArray>(
      list_type, arr->length(), std::move(offset_buf), elem_array);
  return std::make_shared<arrow::ChunkedArray>(list_array);
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValuesVarlenStringArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr,
    StringDictionary* dict) {
  size_t elems_count = 0;
  int32_t arrays_count = 0;
  std::vector<int32_t> out_elem_offsets;
  std::vector<int32_t> out_offset_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  out_offset_offsets.reserve(arr->num_chunks());

  for (auto& chunk : arr->chunks()) {
    auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(chunk);
    CHECK(chunk_list);

    out_offset_offsets.push_back(arrays_count);
    out_elem_offsets.push_back(static_cast<int32_t>(elems_count));

    auto offset_data = chunk_list->data()->GetValues<uint32_t>(1);
    arrays_count += chunk->length();
    elems_count += offset_data[chunk->length()] - offset_data[0];
  }

  if (elems_count > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("Input arrow array is too big for conversion.");
  }

  auto offset_buf =
      arrow::AllocateBuffer(sizeof(int32_t) * arr->length() + 1).ValueOrDie();
  auto offset_ptr = reinterpret_cast<int32_t*>(offset_buf->mutable_data());

  auto elem_buf = arrow::AllocateBuffer(sizeof(int32_t) * elems_count).ValueOrDie();
  auto elem_ptr = reinterpret_cast<int32_t*>(elem_buf->mutable_data());

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr->num_chunks()),
      [&](const tbb::blocked_range<int>& r) {
        for (int c = r.begin(); c != r.end(); ++c) {
          auto elem_offset = out_elem_offsets[c];
          auto offset_offset = out_offset_offsets[c];
          auto chunk_array = std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
          auto elem_array =
              std::dynamic_pointer_cast<arrow::StringArray>(chunk_array->values());
          CHECK(elem_array);

          bool use_negative_offset = false;
          for (int64_t i = 0; i < chunk_array->length(); ++i) {
            offset_ptr[offset_offset++] = use_negative_offset
                                              ? -elem_offset * sizeof(int32_t)
                                              : elem_offset * sizeof(int32_t);
            if (chunk_array->IsNull(i)) {
              use_negative_offset = true;
            } else {
              use_negative_offset = false;
              auto offs = chunk_array->value_offset(i);
              auto len = chunk_array->value_length(i);
              for (int j = 0; j < len; ++j) {
                auto view = elem_array->GetView(offs + j);
                elem_ptr[elem_offset++] =
                    dict->getOrAdd(std::string_view(view.data(), view.length()));
              }
            }
          }
        }
      });
  auto last_chunk = arr->chunk(arr->num_chunks() - 1);
  offset_ptr[arr->length()] = static_cast<int32_t>(
      last_chunk->IsNull(last_chunk->length() - 1) ? -elems_count * sizeof(int32_t)
                                                   : elems_count * sizeof(int32_t));

  auto elem_array = std::make_shared<arrow::Int32Array>(elems_count, std::move(elem_buf));
  auto list_array = std::make_shared<arrow::ListArray>(
      arrow::list(arrow::int32()), arr->length(), std::move(offset_buf), elem_array);
  return std::make_shared<arrow::ChunkedArray>(list_array);
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValuesVarlenArray(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type,
    StringDictionary* dict) {
  auto list_type = std::dynamic_pointer_cast<arrow::ListType>(arr->type());
  if (!list_type) {
    throw std::runtime_error("Unsupported large Arrow list:: "s + list_type->ToString());
  }

  auto elem_type = type.get_elem_type();
  if (elem_type.is_integer() || is_datetime(elem_type.get_type())) {
    switch (elem_type.get_size()) {
      case 1:
        return replaceNullValuesVarlenArrayImpl<int8_t>(arr);
      case 2:
        return replaceNullValuesVarlenArrayImpl<int16_t>(arr);
      case 4:
        return replaceNullValuesVarlenArrayImpl<int32_t>(arr);
      case 8:
        return replaceNullValuesVarlenArrayImpl<int64_t>(arr);
      default:
        throw std::runtime_error("Unsupported integer or datetime array: "s +
                                 type.toString());
    }
  } else if (elem_type.is_fp()) {
    switch (elem_type.get_size()) {
      case 4:
        return replaceNullValuesVarlenArrayImpl<float>(arr);
      case 8:
        return replaceNullValuesVarlenArrayImpl<double>(arr);
    }
  } else if (elem_type.is_boolean()) {
    return replaceNullValuesVarlenArrayImpl<bool>(arr);
  } else if (elem_type.is_dict_encoded_string()) {
    CHECK_EQ(elem_type.get_size(), 4);
    return replaceNullValuesVarlenStringArrayImpl(arr, dict);
  }

  throw std::runtime_error("Unsupported varlen array: "s + type.toString());
  return nullptr;
}

template <typename T>
std::shared_ptr<arrow::ChunkedArray> replaceNullValuesFixedSizeArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr,
    int list_size) {
  int64_t total_length = 0;
  std::vector<size_t> out_elem_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  for (auto& chunk : arr->chunks()) {
    out_elem_offsets.push_back(total_length);
    total_length += chunk->length() * list_size;
  }

  auto elem_buf = arrow::AllocateBuffer(sizeof(T) * total_length).ValueOrDie();
  auto elem_ptr = reinterpret_cast<T*>(elem_buf->mutable_data());

  auto null_array_value = inline_null_array_value<T>();
  auto null_value = inline_null_value<T>();

  tbb::parallel_for(tbb::blocked_range<int>(0, arr->num_chunks()),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int c = r.begin(); c != r.end(); ++c) {
                        auto chunk_array =
                            std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
                        auto elem_array = chunk_array->values();

                        auto dst_ptr = elem_ptr + out_elem_offsets[c];
                        for (int64_t i = 0; i < chunk_array->length(); ++i) {
                          if (chunk_array->IsNull(i)) {
                            dst_ptr[0] = null_array_value;
                            for (int j = 1; j < list_size; ++j) {
                              dst_ptr[j] = null_value;
                            }
                          } else {
                            // We add NULL elements if input array is too short and cut
                            // too long arrays.
                            auto offs = chunk_array->value_offset(i);
                            auto len = std::min(chunk_array->value_length(i), list_size);
                            copyArrayDataReplacingNulls(dst_ptr, elem_array, offs, len);
                            for (int j = len; j < list_size; ++j) {
                              dst_ptr[j] = null_value;
                            }
                          }
                          dst_ptr += list_size;
                        }
                      }
                    });

  std::shared_ptr<arrow::Array> elem_array;
  if constexpr (std::is_same_v<T, bool>) {
    elem_array = std::make_shared<arrow::Int8Array>(total_length, std::move(elem_buf));
  } else {
    using ElemsArrowType = typename arrow::CTypeTraits<T>::ArrowType;
    using ElemsArrayType = typename arrow::TypeTraits<ElemsArrowType>::ArrayType;
    elem_array = std::make_shared<ElemsArrayType>(total_length, std::move(elem_buf));
  }

  return std::make_shared<arrow::ChunkedArray>(elem_array);
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValuesFixedSizeStringArrayImpl(
    std::shared_ptr<arrow::ChunkedArray> arr,
    int list_size,
    StringDictionary* dict) {
  int64_t total_length = 0;
  std::vector<size_t> out_elem_offsets;
  out_elem_offsets.reserve(arr->num_chunks());
  for (auto& chunk : arr->chunks()) {
    out_elem_offsets.push_back(total_length);
    total_length += chunk->length() * list_size;
  }

  auto list_type = std::dynamic_pointer_cast<arrow::ListType>(arr->type());
  CHECK(list_type);
  auto elem_type = list_type->value_type();
  if (elem_type->id() != arrow::Type::STRING) {
    throw std::runtime_error(
        "Dictionary encoded string arrays are not supported in Arrow import: "s +
        list_type->ToString());
  }

  auto elem_buf = arrow::AllocateBuffer(sizeof(int32_t) * total_length).ValueOrDie();
  auto elem_ptr = reinterpret_cast<int32_t*>(elem_buf->mutable_data());

  auto null_array_value = inline_null_array_value<int32_t>();
  auto null_value = inline_null_value<int32_t>();

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr->num_chunks()),
      [&](const tbb::blocked_range<int>& r) {
        for (int c = r.begin(); c != r.end(); ++c) {
          auto chunk_array = std::dynamic_pointer_cast<arrow::ListArray>(arr->chunk(c));
          auto elem_array =
              std::dynamic_pointer_cast<arrow::StringArray>(chunk_array->values());
          CHECK(elem_array);

          auto dst_ptr = elem_ptr + out_elem_offsets[c];
          for (int64_t i = 0; i < chunk_array->length(); ++i) {
            if (chunk_array->IsNull(i)) {
              dst_ptr[0] = null_array_value;
              for (int j = 1; j < list_size; ++j) {
                dst_ptr[j] = null_value;
              }
            } else {
              // We add NULL elements if input array is too short and cut
              // too long arrays.
              auto offs = chunk_array->value_offset(i);
              auto len = std::min(chunk_array->value_length(i), list_size);
              for (int j = 0; j < len; ++j) {
                auto view = elem_array->GetView(offs + j);
                dst_ptr[j] = dict->getOrAdd(std::string_view(view.data(), view.length()));
              }
              for (int j = len; j < list_size; ++j) {
                dst_ptr[j] = null_value;
              }
            }
            dst_ptr += list_size;
          }
        }
      });

  using ElemsArrowType = typename arrow::CTypeTraits<int32_t>::ArrowType;
  using ElemsArrayType = typename arrow::TypeTraits<ElemsArrowType>::ArrayType;

  auto elem_array = std::make_shared<ElemsArrayType>(total_length, std::move(elem_buf));
  return std::make_shared<arrow::ChunkedArray>(elem_array);
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValuesFixedSizeArray(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type,
    StringDictionary* dict) {
  auto elem_type = type.get_elem_type();
  int list_size = type.get_size() / elem_type.get_size();
  if (elem_type.is_integer() || is_datetime(elem_type.get_type())) {
    switch (elem_type.get_size()) {
      case 1:
        return replaceNullValuesFixedSizeArrayImpl<int8_t>(arr, list_size);
      case 2:
        return replaceNullValuesFixedSizeArrayImpl<int16_t>(arr, list_size);
      case 4:
        return replaceNullValuesFixedSizeArrayImpl<int32_t>(arr, list_size);
      case 8:
        return replaceNullValuesFixedSizeArrayImpl<int64_t>(arr, list_size);
      default:
        throw std::runtime_error("Unsupported integer or datetime array: "s +
                                 type.toString());
    }
  } else if (elem_type.is_fp()) {
    switch (elem_type.get_size()) {
      case 4:
        return replaceNullValuesFixedSizeArrayImpl<float>(arr, list_size);
      case 8:
        return replaceNullValuesFixedSizeArrayImpl<double>(arr, list_size);
    }
  } else if (elem_type.is_boolean()) {
    return replaceNullValuesFixedSizeArrayImpl<bool>(arr, list_size);
  } else if (elem_type.is_dict_encoded_string()) {
    CHECK_EQ(elem_type.get_size(), 4);
    return replaceNullValuesFixedSizeStringArrayImpl(arr, list_size, dict);
  }

  throw std::runtime_error("Unsupported fixed size array: "s + type.toString());
  return nullptr;
}

template <typename IntType, typename ChunkType>
std::shared_ptr<arrow::ChunkedArray> convertDecimalToInteger(
    std::shared_ptr<arrow::ChunkedArray> arr_col_chunked_array) {
  size_t column_size = 0;
  std::vector<int> offsets(arr_col_chunked_array->num_chunks());
  for (int i = 0; i < arr_col_chunked_array->num_chunks(); i++) {
    offsets[i] = column_size;
    column_size += arr_col_chunked_array->chunk(i)->length();
  }

  std::shared_ptr<arrow::Buffer> result_buffer;
  auto res = arrow::AllocateBuffer(column_size * sizeof(IntType));
  CHECK(res.ok());
  result_buffer = std::move(res).ValueOrDie();

  IntType* buffer_data = reinterpret_cast<IntType*>(result_buffer->mutable_data());
  tbb::parallel_for(
      tbb::blocked_range(0, arr_col_chunked_array->num_chunks()),
      [buffer_data, &offsets, arr_col_chunked_array](auto& range) {
        for (int chunk_idx = range.begin(); chunk_idx < range.end(); chunk_idx++) {
          auto offset = offsets[chunk_idx];
          IntType* chunk_buffer = buffer_data + offset;

          auto decimalArray = std::static_pointer_cast<arrow::Decimal128Array>(
              arr_col_chunked_array->chunk(chunk_idx));
          auto empty =
              arr_col_chunked_array->null_count() == arr_col_chunked_array->length();
          for (int i = 0; i < decimalArray->length(); i++) {
            if (empty || decimalArray->null_count() == decimalArray->length() ||
                decimalArray->IsNull(i)) {
              chunk_buffer[i] = inline_null_value<IntType>();
            } else {
              arrow::Decimal128 val(decimalArray->GetValue(i));
              chunk_buffer[i] =
                  static_cast<int64_t>(val);  // arrow can cast only to int64_t
            }
          }
        }
      });
  auto array = std::make_shared<ChunkType>(column_size, result_buffer);
  return std::make_shared<arrow::ChunkedArray>(array);
}

}  // anonymous namespace

// TODO: this overlaps with getArrowType() from ArrowResultSetConverter.cpp but with few
// differences in kTEXT and kDATE
std::shared_ptr<arrow::DataType> getArrowImportType(const SQLTypeInfo type) {
  using namespace arrow;
  auto ktype = type.get_type();
  if (IS_INTEGER(ktype)) {
    switch (type.get_size()) {
      case 1:
        return int8();
      case 2:
        return int16();
      case 4:
        return int32();
      case 8:
        return int64();
      default:
        CHECK(false);
    }
  }
  switch (ktype) {
    case kBOOLEAN:
      return arrow::boolean();
    case kFLOAT:
      return float32();
    case kDOUBLE:
      return float64();
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      return utf8();
    case kDECIMAL:
    case kNUMERIC:
      return decimal(type.get_precision(), type.get_scale());
    case kTIME:
      return timestamp(TimeUnit::SECOND);
    case kDATE:
#ifdef HAVE_CUDA
      return arrow::date64();
#else
      return arrow::date32();
#endif
    case kTIMESTAMP:
      switch (type.get_precision()) {
        case 0:
          return timestamp(TimeUnit::SECOND);
        case 3:
          return timestamp(TimeUnit::MILLI);
        case 6:
          return timestamp(TimeUnit::MICRO);
        case 9:
          return timestamp(TimeUnit::NANO);
        default:
          throw std::runtime_error("Unsupported timestamp precision for Arrow: " +
                                   std::to_string(type.get_precision()));
      }
    case kARRAY:
      if (type.is_string_array()) {
        return list(utf8());
      } else if (type.is_fixlen_array()) {
        // Arraw JSON parser doesn't support conversion to fixed size lists.
        // So we use variable length lists in Arrow and then do the conversion.
        return list(getArrowImportType(type.get_elem_type()));
      } else {
        CHECK(type.is_varlen_array());
        return list(getArrowImportType(type.get_elem_type()));
      }
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    default:
      throw std::runtime_error(type.get_type_name() + " is not supported in Arrow.");
  }
  return nullptr;
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValues(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type,
    StringDictionary* dict) {
  if (type.get_type() == kTIME) {
    if (type.get_size() != 8) {
      throw std::runtime_error("Unsupported time type for Arrow import: "s +
                               type.toString());
    }
    return convertTimestampToTimeReplacingNulls(arr);
  }
  if (type.get_type() == kDATE) {
    switch (type.get_size()) {
      case 2:
        return convertDateReplacingNulls<int32_t, int16_t>(arr);
      case 4:
        return replaceNullValuesImpl<int32_t>(arr);
      case 8:
        return convertDateReplacingNulls<int32_t, int64_t>(arr);
      default:
        throw std::runtime_error("Unsupported date type for Arrow import: "s +
                                 type.toString());
    }
  } else if (type.is_integer() || is_datetime(type.get_type())) {
    switch (type.get_size()) {
      case 1:
        return replaceNullValuesImpl<int8_t>(arr);
      case 2:
        return replaceNullValuesImpl<int16_t>(arr);
      case 4:
        return replaceNullValuesImpl<int32_t>(arr);
      case 8:
        return replaceNullValuesImpl<int64_t>(arr);
      default:
        throw std::runtime_error("Unsupported integer/datetime type for Arrow import: "s +
                                 type.toString());
    }
  } else if (type.is_fp()) {
    switch (type.get_size()) {
      case 4:
        return replaceNullValuesImpl<float>(arr);
      case 8:
        return replaceNullValuesImpl<double>(arr);
    }
  } else if (type.is_boolean()) {
    return replaceNullValuesImpl<bool>(arr);
  } else if (type.is_fixlen_array()) {
    return replaceNullValuesFixedSizeArray(arr, type, dict);
  } else if (type.is_varlen_array()) {
    return replaceNullValuesVarlenArray(arr, type, dict);
  }
  CHECK(false) << "Unexpected type: " << type.toString();
  return nullptr;
}

std::shared_ptr<arrow::ChunkedArray> convertDecimalToInteger(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type) {
  CHECK(type.get_type() == kDECIMAL || type.get_type() == kNUMERIC);
  switch (type.get_size()) {
    case 2:
      return convertDecimalToInteger<int16_t, arrow::Int16Array>(arr);
    case 4:
      return convertDecimalToInteger<int32_t, arrow::Int32Array>(arr);
    case 8:
      return convertDecimalToInteger<int64_t, arrow::Int64Array>(arr);
    default:
      // TODO: throw unsupported decimal type exception
      CHECK(false) << "Unsupported decimal type: " << type.toString();
      break;
  }
  return nullptr;
}

SQLTypeInfo getOmnisciType(const arrow::DataType& type) {
  using namespace arrow;
  switch (type.id()) {
    case Type::INT8:
      return SQLTypeInfo(kTINYINT, false);
    case Type::INT16:
      return SQLTypeInfo(kSMALLINT, false);
    case Type::INT32:
      return SQLTypeInfo(kINT, false);
    case Type::INT64:
      return SQLTypeInfo(kBIGINT, false);
    case Type::BOOL:
      return SQLTypeInfo(kBOOLEAN, false);
    case Type::FLOAT:
      return SQLTypeInfo(kFLOAT, false);
    case Type::DATE32:
    case Type::DATE64:
      return SQLTypeInfo(kDATE, false);
    case Type::DOUBLE:
      return SQLTypeInfo(kDOUBLE, false);
      // uncomment when arrow 2.0 will be released and modin support for dictionary
      // types in read_csv would be implemented

      // case Type::DICTIONARY: {
      //   auto type = SQLTypeInfo(kTEXT, false, kENCODING_DICT);
      //   // this is needed because createTable forces type.size to be equal to
      //   // comp_param / 8, no matter what type.size you set here
      //   type.set_comp_param(sizeof(uint32_t) * 8);
      //   return type;
      // }
      // case Type::STRING:
      //   return SQLTypeInfo(kTEXT, false, kENCODING_NONE);

    case Type::STRING: {
      auto type = SQLTypeInfo(kTEXT, false, kENCODING_DICT);
      // this is needed because createTable forces type.size to be equal to
      // comp_param / 8, no matter what type.size you set here
      type.set_comp_param(sizeof(uint32_t) * 8);
      return type;
    }
    case Type::DECIMAL: {
      const auto& decimal_type = static_cast<const arrow::DecimalType&>(type);
      return SQLTypeInfo(kDECIMAL, decimal_type.precision(), decimal_type.scale(), false);
    }
    case Type::TIME32:
      return SQLTypeInfo(kTIME, false);
    case Type::TIMESTAMP:
      switch (static_cast<const arrow::TimestampType&>(type).unit()) {
        case TimeUnit::SECOND:
          return SQLTypeInfo(kTIMESTAMP, 0, 0);
        case TimeUnit::MILLI:
          return SQLTypeInfo(kTIMESTAMP, 3, 0);
        case TimeUnit::MICRO:
          return SQLTypeInfo(kTIMESTAMP, 6, 0);
        case TimeUnit::NANO:
          return SQLTypeInfo(kTIMESTAMP, 9, 0);
      }
    default:
      throw std::runtime_error(type.ToString() + " is not yet supported.");
  }
}

namespace {

template <typename IndexType>
std::shared_ptr<arrow::ChunkedArray> createDictionaryEncodedColumn(
    StringDictionary* dict,
    std::shared_ptr<arrow::ChunkedArray> arr) {
  // calculate offsets for every fragment in bulk
  size_t bulk_size = 0;
  std::vector<int> offsets(arr->num_chunks());
  for (int i = 0; i < arr->num_chunks(); i++) {
    offsets[i] = bulk_size;
    bulk_size += arr->chunk(i)->length();
  }

  std::vector<std::string_view> bulk(bulk_size);

  tbb::parallel_for(tbb::blocked_range<int>(0, arr->num_chunks()),
                    [&bulk, &arr, &offsets](const tbb::blocked_range<int>& r) {
                      for (int i = r.begin(); i < r.end(); i++) {
                        auto chunk =
                            std::static_pointer_cast<arrow::StringArray>(arr->chunk(i));
                        auto offset = offsets[i];
                        for (int j = 0; j < chunk->length(); j++) {
                          auto view = chunk->GetView(j);
                          bulk[offset + j] = std::string_view(view.data(), view.length());
                        }
                      }
                    });

  std::shared_ptr<arrow::Buffer> indices_buf;
  auto res = arrow::AllocateBuffer(bulk_size * sizeof(int32_t));
  CHECK(res.ok());
  indices_buf = std::move(res).ValueOrDie();
  auto raw_data = reinterpret_cast<int*>(indices_buf->mutable_data());
  dict->getOrAddBulk(bulk, raw_data);

  if constexpr (std::is_same_v<IndexType, uint32_t>) {
    auto array = std::make_shared<arrow::Int32Array>(bulk_size, indices_buf);
    return std::make_shared<arrow::ChunkedArray>(array);
  } else {
    // We have to convert to a narrower index type. Indexes which don't fit
    // the type are replaced with invalid id.
    static_assert(sizeof(IndexType) < sizeof(int32_t));
    static_assert(std::is_unsigned_v<IndexType>);
    auto converted_res = arrow::AllocateBuffer(bulk_size * sizeof(IndexType));
    CHECK(converted_res.ok());
    std::shared_ptr<arrow::Buffer> converted_indices_buf =
        std::move(converted_res).ValueOrDie();
    auto raw_converted_data =
        reinterpret_cast<IndexType*>(converted_indices_buf->mutable_data());
    for (size_t i = 0; i < bulk_size; ++i) {
      if (raw_data[i] > static_cast<int32_t>(std::numeric_limits<IndexType>::max()) ||
          raw_data[i] == inline_null_value<int>()) {
        raw_converted_data[i] = static_cast<IndexType>(StringDictionary::INVALID_STR_ID);
      } else {
        raw_converted_data[i] = static_cast<IndexType>(raw_data[i]);
      }
    }

    using IndexArrowType = typename arrow::CTypeTraits<IndexType>::ArrowType;
    using ArrayType = typename arrow::TypeTraits<IndexArrowType>::ArrayType;

    auto array = std::make_shared<ArrayType>(bulk_size, converted_indices_buf);
    return std::make_shared<arrow::ChunkedArray>(array);
  }
}

}  // anonymous namespace

std::shared_ptr<arrow::ChunkedArray> createDictionaryEncodedColumn(
    StringDictionary* dict,
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type) {
  switch (type.get_size()) {
    case 4:
      return createDictionaryEncodedColumn<uint32_t>(dict, arr);
    case 2:
      return createDictionaryEncodedColumn<uint16_t>(dict, arr);
    case 1:
      return createDictionaryEncodedColumn<uint8_t>(dict, arr);
    default:
      throw std::runtime_error(
          "Unsupported OmniSci dictionary for Arrow strings import: "s + type.toString());
  }
  return nullptr;
}

std::shared_ptr<arrow::ChunkedArray> convertArrowDictionary(
    StringDictionary* dict,
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type) {
  if (type.get_size() != 4) {
    throw std::runtime_error(
        "Unsupported OmniSci dictionary for Arrow dictionary import: "s +
        type.toString());
  }
  // TODO: allocate one big array and split it by fragments as it is done in
  // createDictionaryEncodedColumn
  std::vector<std::shared_ptr<arrow::Array>> converted_chunks;
  for (auto& chunk : arr->chunks()) {
    auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(chunk);
    auto values = std::static_pointer_cast<arrow::StringArray>(dict_array->dictionary());
    std::vector<std::string_view> strings(values->length());
    for (int i = 0; i < values->length(); i++) {
      auto view = values->GetView(i);
      strings[i] = std::string_view(view.data(), view.length());
    }
    auto arrow_indices =
        std::static_pointer_cast<arrow::Int32Array>(dict_array->indices());
    std::vector<int> indices_mapping(values->length());
    dict->getOrAddBulk(strings, indices_mapping.data());

    // create new arrow chunk with remapped indices
    std::shared_ptr<arrow::Buffer> dict_indices_buf;
    auto res = arrow::AllocateBuffer(arrow_indices->length() * sizeof(int32_t));
    CHECK(res.ok());
    dict_indices_buf = std::move(res).ValueOrDie();
    auto raw_data = reinterpret_cast<int32_t*>(dict_indices_buf->mutable_data());

    for (int i = 0; i < arrow_indices->length(); i++) {
      raw_data[i] = indices_mapping[arrow_indices->Value(i)];
    }

    converted_chunks.push_back(
        std::make_shared<arrow::Int32Array>(arrow_indices->length(), dict_indices_buf));
  }
  return std::make_shared<arrow::ChunkedArray>(converted_chunks);
}
