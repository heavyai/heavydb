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

void convertBoolBitmapBufferWithNulls(int8_t* dst,
                                      const uint8_t* src,
                                      const uint8_t* bitmap,
                                      int64_t length,
                                      int8_t null_value) {
  for (int64_t bitmap_idx = 0; bitmap_idx < length / 8; ++bitmap_idx) {
    auto source = src[bitmap_idx];
    auto dest = dst + bitmap_idx * 8;
    auto inversed_bitmap = ~bitmap[bitmap_idx];
    for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
      auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
      auto val = (source >> bitmap_offset) & 1;
      dest[bitmap_offset] = is_null ? null_value : val;
    }
  }

  for (int64_t j = (length / 8) * 8; j < length; ++j) {
    auto is_null = (~bitmap[length / 8] >> (j % 8)) & 1;
    auto val = (src[length / 8] >> (j % 8)) & 1;
    dst[j] = is_null ? null_value : val;
  }
}

void convertBoolBitmapBufferWithoutNulls(int8_t* dst,
                                         const uint8_t* src,
                                         int64_t length) {
  for (int64_t bitmap_idx = 0; bitmap_idx < length / 8; ++bitmap_idx) {
    auto source = src[bitmap_idx];
    auto dest = dst + bitmap_idx * 8;
    for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
      dest[bitmap_offset] = (source >> bitmap_offset) & 1;
    }
  }

  for (int64_t j = (length / 8) * 8; j < length; ++j) {
    dst[j] = (src[length / 8] >> (j % 8)) & 1;
  }
}

template <typename T>
std::shared_ptr<arrow::ChunkedArray> replaceNullValuesImpl(
    std::shared_ptr<arrow::ChunkedArray> arr) {
  if (!std::is_same_v<T, bool> && arr->null_count() == 0) {
    // for boolean columns we still need to convert bitmaps to array
    return arr;
  }

  auto null_value = inline_null_value<T>();

  auto resultBuf = arrow::AllocateBuffer(sizeof(T) * arr->length()).ValueOrDie();
  auto resultData = reinterpret_cast<T*>(resultBuf->mutable_data());

  tbb::parallel_for(
      tbb::blocked_range<int>(0, arr->num_chunks()),
      [&](const tbb::blocked_range<int>& r) {
        for (int c = r.begin(); c != r.end(); ++c) {
          size_t offset = 0;
          for (int i = 0; i < c; i++) {
            offset += arr->chunk(i)->length();
          }
          auto resWithOffset = resultData + offset;

          auto chunk = arr->chunk(c);

          if (chunk->null_count() == chunk->length()) {
            std::fill(resWithOffset, resWithOffset + chunk->length(), null_value);
            continue;
          }

          auto chunkData = reinterpret_cast<const T*>(chunk->data()->buffers[1]->data());

          const uint8_t* bitmap_data = chunk->null_bitmap_data();
          const int64_t length = chunk->length();

          if (chunk->null_count() == 0) {
            if constexpr (std::is_same_v<T, bool>) {
              convertBoolBitmapBufferWithoutNulls(
                  reinterpret_cast<int8_t*>(resWithOffset),
                  reinterpret_cast<const uint8_t*>(chunkData),
                  length);
            } else {
              std::copy(chunkData, chunkData + chunk->length(), resWithOffset);
            }
            continue;
          }

          if constexpr (std::is_same_v<T, bool>) {
            convertBoolBitmapBufferWithNulls(reinterpret_cast<int8_t*>(resWithOffset),
                                             reinterpret_cast<const uint8_t*>(chunkData),
                                             bitmap_data,
                                             length,
                                             null_value);
          } else {
            for (int64_t bitmap_idx = 0; bitmap_idx < length / 8; ++bitmap_idx) {
              auto source = chunkData + bitmap_idx * 8;
              auto dest = resWithOffset + bitmap_idx * 8;
              auto inversed_bitmap = ~bitmap_data[bitmap_idx];
              for (int8_t bitmap_offset = 0; bitmap_offset < 8; ++bitmap_offset) {
                auto is_null = (inversed_bitmap >> bitmap_offset) & 1;
                auto val = is_null ? null_value : source[bitmap_offset];
                dest[bitmap_offset] = val;
              }
            }

            for (int64_t j = length / 8 * 8; j < length; ++j) {
              auto is_null = (~bitmap_data[length / 8] >> (j % 8)) & 1;
              auto val = is_null ? null_value : chunkData[j];
              resWithOffset[j] = val;
            }
          }
        }
      });

  using ArrowType = typename arrow::CTypeTraits<T>::ArrowType;
  using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

  auto array = std::make_shared<ArrayType>(arr->length(), std::move(resultBuf));
  return std::make_shared<arrow::ChunkedArray>(array);
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
      return time32(TimeUnit::SECOND);
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
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    default:
      throw std::runtime_error(type.get_type_name() + " is not supported in Arrow.");
  }
  return nullptr;
}

std::shared_ptr<arrow::ChunkedArray> replaceNullValues(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type) {
  if (type.is_integer() || is_datetime(type.get_type())) {
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
        // TODO: throw unsupported integer type exception
        CHECK(false);
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
      // uncomment when arrow 2.0 will be released and modin support for dictionary types
      // in read_csv would be implemented

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
  auto array = std::make_shared<arrow::Int32Array>(bulk_size, indices_buf);
  return std::make_shared<arrow::ChunkedArray>(array);
}

std::shared_ptr<arrow::ChunkedArray> convertArrowDictionary(
    StringDictionary* dict,
    std::shared_ptr<arrow::ChunkedArray> arr) {
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
