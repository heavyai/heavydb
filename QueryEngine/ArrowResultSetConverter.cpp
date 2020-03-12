/*
 * Copyright 2019 OmniSci, Inc.
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

#include "../Shared/DateConverters.h"
#include "ArrowResultSet.h"
#include "Execute.h"

#include <sys/shm.h>
#include <sys/types.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "arrow/api.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/api.h"

#include "ArrowUtil.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <future>

#define ARROW_RECORDBATCH_MAKE arrow::RecordBatch::Make

#define APPENDVALUES AppendValues

using namespace arrow;

namespace {

inline SQLTypes get_dict_index_type(const SQLTypeInfo& ti) {
  CHECK(ti.is_dict_encoded_string());
  switch (ti.get_size()) {
    case 1:
      return kTINYINT;
    case 2:
      return kSMALLINT;
    case 4:
      return kINT;
    case 8:
      return kBIGINT;
    default:
      CHECK(false);
  }
  return ti.get_type();
}

inline SQLTypeInfo get_dict_index_type_info(const SQLTypeInfo& ti) {
  CHECK(ti.is_dict_encoded_string());
  switch (ti.get_size()) {
    case 1:
      return SQLTypeInfo(kTINYINT, ti.get_notnull());
    case 2:
      return SQLTypeInfo(kSMALLINT, ti.get_notnull());
    case 4:
      return SQLTypeInfo(kINT, ti.get_notnull());
    case 8:
      return SQLTypeInfo(kBIGINT, ti.get_notnull());
    default:
      CHECK(false);
  }
  return ti;
}

inline SQLTypes get_physical_type(const SQLTypeInfo& ti) {
  auto logical_type = ti.get_type();
  if (IS_INTEGER(logical_type)) {
    switch (ti.get_size()) {
      case 1:
        return kTINYINT;
      case 2:
        return kSMALLINT;
      case 4:
        return kINT;
      case 8:
        return kBIGINT;
      default:
        CHECK(false);
    }
  }
  return logical_type;
}

template <typename TYPE, typename C_TYPE>
void create_or_append_value(const ScalarTargetValue& val_cty,
                            std::shared_ptr<ValueArray>& values,
                            const size_t max_size) {
  auto pval_cty = boost::get<C_TYPE>(&val_cty);
  CHECK(pval_cty);
  auto val_ty = static_cast<TYPE>(*pval_cty);
  if (!values) {
    values = std::make_shared<ValueArray>(std::vector<TYPE>());
    boost::get<std::vector<TYPE>>(*values).reserve(max_size);
  }
  CHECK(values);
  auto values_ty = boost::get<std::vector<TYPE>>(values.get());
  CHECK(values_ty);
  values_ty->push_back(val_ty);
}

template <typename TYPE>
void create_or_append_validity(const ScalarTargetValue& value,
                               const SQLTypeInfo& col_type,
                               std::shared_ptr<std::vector<bool>>& null_bitmap,
                               const size_t max_size) {
  if (col_type.get_notnull()) {
    CHECK(!null_bitmap);
    return;
  }
  auto pvalue = boost::get<TYPE>(&value);
  CHECK(pvalue);
  bool is_valid = false;
  if (col_type.is_boolean()) {
    is_valid = inline_int_null_val(col_type) != static_cast<int8_t>(*pvalue);
  } else if (col_type.is_dict_encoded_string()) {
    is_valid = inline_int_null_val(col_type) != static_cast<int32_t>(*pvalue);
  } else if (col_type.is_integer() || col_type.is_time()) {
    is_valid = inline_int_null_val(col_type) != static_cast<int64_t>(*pvalue);
  } else if (col_type.is_fp()) {
    is_valid = inline_fp_null_val(col_type) != static_cast<double>(*pvalue);
  } else {
    UNREACHABLE();
  }

  if (!null_bitmap) {
    null_bitmap = std::make_shared<std::vector<bool>>();
    null_bitmap->reserve(max_size);
  }
  CHECK(null_bitmap);
  null_bitmap->push_back(is_valid);
}

template <typename TYPE, typename enable = void>
class null_type {};

template <typename TYPE>
struct null_type<TYPE, std::enable_if_t<std::is_integral<TYPE>::value>> {
  using type = typename std::make_signed<TYPE>::type;
  static constexpr type value = inline_int_null_value<type>();
};

template <typename TYPE>
struct null_type<TYPE, std::enable_if_t<std::is_floating_point<TYPE>::value>> {
  using type = TYPE;
  static constexpr type value = inline_fp_null_value<type>();
};

template <typename TYPE>
using null_type_t = typename null_type<TYPE>::type;

template <typename C_TYPE, typename ARROW_TYPE = typename CTypeTraits<C_TYPE>::ArrowType>
void convert_column(ResultSetPtr result,
                    size_t col,
                    std::unique_ptr<int8_t[]>& values,
                    std::unique_ptr<uint8_t[]>& is_valid,
                    size_t entry_count,
                    std::shared_ptr<Array>& out) {
  CHECK(sizeof(C_TYPE) == result->getColType(col).get_size());
  CHECK(!values);
  CHECK(!is_valid);

  const int8_t* data_ptr;
  if (result->isZeroCopyColumnarConversionPossible(col)) {
    data_ptr = result->getColumnarBuffer(col);
  } else {
    values.reset(new int8_t[entry_count * sizeof(C_TYPE)]);
    result->copyColumnIntoBuffer(col, values.get(), entry_count * sizeof(C_TYPE));
    data_ptr = values.get();
  }

  int64_t null_count = 0;
  is_valid.reset(new uint8_t[(entry_count + 7) / 8]);

  const null_type_t<C_TYPE>* vals =
      reinterpret_cast<const null_type_t<C_TYPE>*>(data_ptr);
  null_type_t<C_TYPE> null_val = null_type<C_TYPE>::value;

  size_t unroll_count = entry_count & 0xFFFFFFFFFFFFFFF8ULL;
  for (size_t i = 0; i < unroll_count; i += 8) {
    uint8_t valid_byte = 0;
    uint8_t valid;
    valid = vals[i + 0] != null_val;
    valid_byte |= valid << 0;
    null_count += !valid;
    valid = vals[i + 1] != null_val;
    valid_byte |= valid << 1;
    null_count += !valid;
    valid = vals[i + 2] != null_val;
    valid_byte |= valid << 2;
    null_count += !valid;
    valid = vals[i + 3] != null_val;
    valid_byte |= valid << 3;
    null_count += !valid;
    valid = vals[i + 4] != null_val;
    valid_byte |= valid << 4;
    null_count += !valid;
    valid = vals[i + 5] != null_val;
    valid_byte |= valid << 5;
    null_count += !valid;
    valid = vals[i + 6] != null_val;
    valid_byte |= valid << 6;
    null_count += !valid;
    valid = vals[i + 7] != null_val;
    valid_byte |= valid << 7;
    null_count += !valid;
    is_valid[i >> 3] = valid_byte;
  }
  if (unroll_count != entry_count) {
    uint8_t valid_byte = 0;
    for (size_t i = unroll_count; i < entry_count; ++i) {
      bool valid = vals[i] != null_val;
      valid_byte |= valid << (i & 7);
      null_count += !valid;
    }
    is_valid[unroll_count >> 3] = valid_byte;
  }

  if (!null_count)
    is_valid.reset();

  // TODO: support date/time + scaling
  // TODO: support booleans
  // TODO: support strings (dictionaries)
  std::shared_ptr<Buffer> data(new Buffer(reinterpret_cast<const uint8_t*>(data_ptr),
                                          entry_count * sizeof(C_TYPE)));
  if (null_count) {
    std::shared_ptr<Buffer> null_bitmap(
        new Buffer(is_valid.get(), (entry_count + 7) / 8));
    out.reset(new NumericArray<ARROW_TYPE>(entry_count, data, null_bitmap, null_count));
  } else {
    out.reset(new NumericArray<ARROW_TYPE>(entry_count, data));
  }
}

std::pair<key_t, void*> get_shm(size_t shmsz) {
  if (!shmsz) {
    return std::make_pair(IPC_PRIVATE, nullptr);
  }
  // Generate a new key for a shared memory segment. Keys to shared memory segments
  // are OS global, so we need to try a new key if we encounter a collision. It seems
  // incremental keygen would be deterministically worst-case. If we use a hash
  // (like djb2) + nonce, we could still get collisions if multiple clients specify
  // the same nonce, so using rand() in lieu of a better approach
  // TODO(ptaylor): Is this common? Are these assumptions true?
  auto key = static_cast<key_t>(rand());
  int shmid = -1;
  // IPC_CREAT - indicates we want to create a new segment for this key if it doesn't
  // exist IPC_EXCL - ensures failure if a segment already exists for this key
  while ((shmid = shmget(key, shmsz, IPC_CREAT | IPC_EXCL | 0666)) < 0) {
    // If shmget fails and errno is one of these four values, try a new key.
    // TODO(ptaylor): is checking for the last three values really necessary? Checking
    // them by default to be safe. EEXIST - a shared memory segment is already associated
    // with this key EACCES - a shared memory segment is already associated with this key,
    // but we don't have permission to access it EINVAL - a shared memory segment is
    // already associated with this key, but the size is less than shmsz ENOENT -
    // IPC_CREAT was not set in shmflg and no shared memory segment associated with key
    // was found
    if (!(errno & (EEXIST | EACCES | EINVAL | ENOENT))) {
      throw std::runtime_error("failed to create a shared memory");
    }
    key = static_cast<key_t>(rand());
  }
  // get a pointer to the shared memory segment
  auto ipc_ptr = shmat(shmid, NULL, 0);
  if (reinterpret_cast<int64_t>(ipc_ptr) == -1) {
    throw std::runtime_error("failed to attach a shared memory");
  }

  return std::make_pair(key, ipc_ptr);
}

std::pair<key_t, std::shared_ptr<Buffer>> get_shm_buffer(size_t size) {
  auto [key, ipc_ptr] = get_shm(size);
  std::shared_ptr<Buffer> buffer(new MutableBuffer(static_cast<uint8_t*>(ipc_ptr), size));
  return std::make_pair<key_t, std::shared_ptr<Buffer>>(std::move(key),
                                                        std::move(buffer));
}

}  // namespace

namespace arrow {

key_t get_and_copy_to_shm(const std::shared_ptr<Buffer>& data) {
  auto [key, ipc_ptr] = get_shm(data->size());
  // copy the arrow records buffer to shared memory
  // TODO(ptaylor): I'm sure it's possible to tell Arrow's RecordBatchStreamWriter to
  // write directly to the shared memory segment as a sink
  memcpy(ipc_ptr, data->data(), data->size());
  // detach from the shared memory segment
  shmdt(ipc_ptr);
  return key;
}

}  // namespace arrow

// WARN(ptaylor): users are responsible for detaching and removing shared memory segments,
// e.g.,
//   int shmid = shmget(...);
//   auto ipc_ptr = shmat(shmid, ...);
//   ...
//   shmdt(ipc_ptr);
//   shmctl(shmid, IPC_RMID, 0);
// WARN(miyu): users are responsible to free all device copies, e.g.,
//   cudaIpcMemHandle_t mem_handle = ...
//   void* dev_ptr;
//   cudaIpcOpenMemHandle(&dev_ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess);
//   ...
//   cudaIpcCloseMemHandle(dev_ptr);
//   cudaFree(dev_ptr);
//
// TODO(miyu): verify if the server still needs to free its own copies after last uses
ArrowResult ArrowResultSetConverter::getArrowResultImpl() const {
  auto timer = DEBUG_TIMER(__func__);
  const auto serialized_arrow_output = getSerializedArrowOutput();
  const auto& serialized_schema = serialized_arrow_output.schema;
  const auto& serialized_records = serialized_arrow_output.records;
  key_t record_key = serialized_arrow_output.records_shm_key;
  key_t schema_key;

  {
    auto timer = DEBUG_TIMER("get and copy schema to shm");
    schema_key = arrow::get_and_copy_to_shm(serialized_schema);
  }
  CHECK(schema_key != IPC_PRIVATE);
  std::vector<char> schema_handle_buffer(sizeof(key_t), 0);
  memcpy(&schema_handle_buffer[0],
         reinterpret_cast<const unsigned char*>(&schema_key),
         sizeof(key_t));
  if (device_type_ == ExecutorDeviceType::CPU) {
    if (record_key == IPC_PRIVATE) {
      auto timer = DEBUG_TIMER("get and copy records to shm");
      record_key = arrow::get_and_copy_to_shm(serialized_records);
    } else {
      shmdt(serialized_records->data());
    }
    std::vector<char> record_handle_buffer(sizeof(key_t), 0);
    memcpy(&record_handle_buffer[0],
           reinterpret_cast<const unsigned char*>(&record_key),
           sizeof(key_t));

    return {schema_handle_buffer,
            serialized_schema->size(),
            record_handle_buffer,
            serialized_records->size(),
            nullptr};
  }
#ifdef HAVE_CUDA
  if (serialized_records->size()) {
    CHECK(data_mgr_);
    const auto cuda_mgr = data_mgr_->getCudaMgr();
    CHECK(cuda_mgr);
    auto dev_ptr = reinterpret_cast<CUdeviceptr>(
        cuda_mgr->allocateDeviceMem(serialized_records->size(), device_id_));
    CUipcMemHandle record_handle;
    cuIpcGetMemHandle(&record_handle, dev_ptr);
    cuda_mgr->copyHostToDevice(
        reinterpret_cast<int8_t*>(dev_ptr),
        reinterpret_cast<const int8_t*>(serialized_records->data()),
        serialized_records->size(),
        device_id_);
    std::vector<char> record_handle_buffer(sizeof(record_handle), 0);
    memcpy(&record_handle_buffer[0],
           reinterpret_cast<unsigned char*>(&record_handle),
           sizeof(CUipcMemHandle));
    return {schema_handle_buffer,
            serialized_schema->size(),
            record_handle_buffer,
            serialized_records->size(),
            reinterpret_cast<int8_t*>(dev_ptr)};
  }
#endif
  return {schema_handle_buffer, serialized_schema->size(), {}, 0, nullptr};
}

ArrowResultSetConverter::SerializedArrowOutput
ArrowResultSetConverter::getSerializedArrowOutput() const {
  auto timer = DEBUG_TIMER(__func__);
  arrow::ipc::DictionaryMemo dict_memo;
  std::shared_ptr<arrow::RecordBatch> arrow_copy = convertToArrow(dict_memo);
  std::shared_ptr<arrow::Buffer> serialized_records, serialized_schema;
  key_t records_shm_key = IPC_PRIVATE;

  {
    auto timer = DEBUG_TIMER("serialize schema");
    ARROW_THROW_NOT_OK(arrow::ipc::SerializeSchema(
        *arrow_copy->schema(), arrow::default_memory_pool(), &serialized_schema));
  }

  if (arrow_copy->num_rows()) {
    {
      auto timer = DEBUG_TIMER("validate batch");
      ARROW_THROW_NOT_OK(arrow_copy->Validate());
    }
    {
      auto timer = DEBUG_TIMER("serialize records");
      int64_t size = 0;
      ARROW_THROW_NOT_OK(ipc::GetRecordBatchSize(*arrow_copy, &size));
      std::tie(records_shm_key, serialized_records) = get_shm_buffer(size);

      io::FixedSizeBufferWriter stream(serialized_records);
      ARROW_THROW_NOT_OK(
          ipc::SerializeRecordBatch(*arrow_copy, arrow::default_memory_pool(), &stream));
    }
  } else {
    ARROW_THROW_NOT_OK(arrow::AllocateBuffer(0, &serialized_records));
  }
  return {serialized_schema, serialized_records, records_shm_key};
}

std::shared_ptr<arrow::RecordBatch> ArrowResultSetConverter::convertToArrow(
    arrow::ipc::DictionaryMemo& memo) const {
  auto timer = DEBUG_TIMER(__func__);
  const auto col_count = results_->colCount();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  CHECK(col_names_.empty() || col_names_.size() == col_count);
  for (size_t i = 0; i < col_count; ++i) {
    const auto ti = results_->getColType(i);
    std::shared_ptr<arrow::Array> dict;
    if (ti.is_dict_encoded_string()) {
      const int dict_id = ti.get_comp_param();
      if (memo.HasDictionaryId(dict_id)) {
        ARROW_THROW_NOT_OK(memo.GetDictionary(dict_id, &dict));
      } else {
        auto str_list = results_->getStringDictionaryPayloadCopy(dict_id);

        arrow::StringBuilder builder;
        // TODO(andrewseidl): replace with AppendValues() once Arrow 0.7.1 support is
        // fully deprecated
        for (const std::string& val : *str_list) {
          ARROW_THROW_NOT_OK(builder.Append(val));
        }
        ARROW_THROW_NOT_OK(builder.Finish(&dict));
        ARROW_THROW_NOT_OK(memo.AddDictionary(dict_id, dict));
      }
    }
    fields.push_back(makeField(col_names_.empty() ? "" : col_names_[i], ti, dict));
  }
  return getArrowBatch(arrow::schema(fields));
}

std::shared_ptr<arrow::RecordBatch> ArrowResultSetConverter::getArrowBatch(
    const std::shared_ptr<arrow::Schema>& schema) const {
  std::vector<std::shared_ptr<arrow::Array>> result_columns;

  const size_t entry_count = top_n_ < 0
                                 ? results_->entryCount()
                                 : std::min(size_t(top_n_), results_->entryCount());
  if (!entry_count) {
    return ARROW_RECORDBATCH_MAKE(schema, 0, result_columns);
  }
  const auto col_count = results_->colCount();
  size_t row_count = 0;

  result_columns.resize(col_count);
  std::vector<ColumnBuilder> builders(col_count);

  // Create array builders
  for (size_t i = 0; i < col_count; ++i) {
    initializeColumnBuilder(builders[i], results_->getColType(i), schema->field(i));
  }

  // TODO(miyu): speed up for columnar buffers
  auto fetch = [&](std::vector<std::shared_ptr<ValueArray>>& value_seg,
                   std::vector<std::shared_ptr<std::vector<bool>>>& null_bitmap_seg,
                   const std::vector<bool>& non_lazy_cols,
                   const size_t start_entry,
                   const size_t end_entry) -> size_t {
    CHECK_EQ(value_seg.size(), col_count);
    CHECK_EQ(null_bitmap_seg.size(), col_count);
    const auto entry_count = end_entry - start_entry;
    size_t seg_row_count = 0;
    for (size_t i = start_entry; i < end_entry; ++i) {
      auto row = results_->getRowAtNoTranslations(i, non_lazy_cols);
      if (row.empty()) {
        continue;
      }
      ++seg_row_count;
      for (size_t j = 0; j < col_count; ++j) {
        if (!non_lazy_cols.empty() && non_lazy_cols[j]) {
          continue;
        }

        auto scalar_value = boost::get<ScalarTargetValue>(&row[j]);
        // TODO(miyu): support more types other than scalar.
        CHECK(scalar_value);
        const auto& column = builders[j];
        switch (column.physical_type) {
          case kBOOLEAN:
            create_or_append_value<bool, int64_t>(
                *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kTINYINT:
            create_or_append_value<int8_t, int64_t>(
                *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kSMALLINT:
            create_or_append_value<int16_t, int64_t>(
                *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kINT:
            create_or_append_value<int32_t, int64_t>(
                *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kBIGINT:
            create_or_append_value<int64_t, int64_t>(
                *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kFLOAT:
            create_or_append_value<float, float>(
                *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<float>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kDOUBLE:
            create_or_append_value<double, double>(
                *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<double>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kTIME:
            create_or_append_value<int32_t, int64_t>(
                *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kDATE:
            device_type_ == ExecutorDeviceType::GPU
                ? create_or_append_value<int64_t, int64_t>(
                      *scalar_value, value_seg[j], entry_count)
                : create_or_append_value<int32_t, int64_t>(
                      *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kTIMESTAMP:
            create_or_append_value<int64_t, int64_t>(
                *scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(
                *scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          default:
            // TODO(miyu): support more scalar types.
            throw std::runtime_error(column.col_type.get_type_name() +
                                     " is not supported in Arrow result sets.");
        }
      }
    }
    return seg_row_count;
  };

  auto convert_columns = [&](std::vector<std::unique_ptr<int8_t[]>>& values,
                             std::vector<std::unique_ptr<uint8_t[]>>& is_valid,
                             std::vector<std::shared_ptr<arrow::Array>>& result,
                             const std::vector<bool>& non_lazy_cols,
                             const size_t start_col,
                             const size_t end_col) {
    for (size_t col = start_col; col < end_col; ++col) {
      if (!non_lazy_cols.empty() && !non_lazy_cols[col]) {
        continue;
      }

      const auto& column = builders[col];
      switch (column.physical_type) {
        // TODO: support booleans
        // case kBOOLEAN:
        // convert_column<uint8_t, BooleanType>(
        //    results_, col, values[col], is_valid[col], entry_count, result[col]);
        // break;
        case kTINYINT:
          convert_column<int8_t>(
              results_, col, values[col], is_valid[col], entry_count, result[col]);
          break;
        case kSMALLINT:
          convert_column<int16_t>(
              results_, col, values[col], is_valid[col], entry_count, result[col]);
          break;
        case kINT:
          convert_column<int32_t>(
              results_, col, values[col], is_valid[col], entry_count, result[col]);
          break;
        case kBIGINT:
          convert_column<int64_t>(
              results_, col, values[col], is_valid[col], entry_count, result[col]);
          break;
        case kFLOAT:
          convert_column<float>(
              results_, col, values[col], is_valid[col], entry_count, result[col]);
          break;
        case kDOUBLE:
          convert_column<double>(
              results_, col, values[col], is_valid[col], entry_count, result[col]);
          break;
        // case kTIME:
        //  convert_column<int32_t>(
        //      results_, col, values[col], is_valid[col], entry_count, result[col]);
        //  break;
        // case kDATE:
        //  device_type_ == ExecutorDeviceType::GPU
        //      ? convert_column<int64_t>(
        //            results_, col, values[col], is_valid[col], entry_count, result[col])
        //      : convert_column<int32_t>(
        //            results_, col, values[col], is_valid[col], entry_count,
        //            result[col]);
        //  break;
        // case kTIMESTAMP:
        //  convert_column<int64_t>(
        //      results_, col, values[col], is_valid[col], entry_count, result[col]);
        //  break;
        default:
          throw std::runtime_error(column.col_type.get_type_name() +
                                   " is not supported in Arrow column converter.");
      }
    }
  };

  std::vector<std::shared_ptr<ValueArray>> column_values(col_count, nullptr);
  std::vector<std::shared_ptr<std::vector<bool>>> null_bitmaps(col_count, nullptr);
  const bool multithreaded = entry_count > 10000 && !results_->isTruncated();
  bool use_columnar_converter = results_->isDirectColumnarConversionPossible() &&
                                entry_count == results_->entryCount();
  std::vector<bool> non_lazy_cols;
  if (use_columnar_converter) {
    auto timer = DEBUG_TIMER("columnar converter");
    std::vector<size_t> non_lazy_col_pos;
    size_t non_lazy_col_count = 0;
    const auto& lazy_fetch_info = results_->getLazyFetchInfo();

    non_lazy_cols.reserve(col_count);
    non_lazy_col_pos.reserve(col_count);
    for (size_t i = 0; i < col_count; ++i) {
      bool is_lazy =
          lazy_fetch_info.empty() ? false : lazy_fetch_info[i].is_lazily_fetched;
      // Currently column converter cannot handle some data types.
      // Treat them as lazy for a while.
      switch (builders[i].physical_type) {
        case kBOOLEAN:
        case kTIME:
        case kDATE:
        case kTIMESTAMP:
          is_lazy = true;
          break;
        default:
          break;
      }
      if (builders[i].field->type()->id() == Type::DICTIONARY) {
        is_lazy = true;
      }
      non_lazy_cols.emplace_back(!is_lazy);
      if (!is_lazy) {
        ++non_lazy_col_count;
        non_lazy_col_pos.emplace_back(i);
      }
    }
    if (non_lazy_col_count == col_count) {
      non_lazy_cols.clear();
      non_lazy_col_pos.clear();
    } else {
      non_lazy_col_pos.emplace_back(col_count);
    }

    values_.resize(col_count);
    is_valid_.resize(col_count);
    std::vector<std::future<void>> child_threads;
    size_t num_threads =
        std::min(multithreaded ? (size_t)cpu_threads() : (size_t)1, non_lazy_col_count);

    size_t start_col = 0;
    size_t end_col = 0;
    for (size_t i = 0; i < num_threads; ++i) {
      start_col = end_col;
      end_col = (i + 1) * non_lazy_col_count / num_threads;
      size_t phys_start_col =
          non_lazy_col_pos.empty() ? start_col : non_lazy_col_pos[start_col];
      size_t phys_end_col =
          non_lazy_col_pos.empty() ? end_col : non_lazy_col_pos[end_col];
      child_threads.push_back(std::async(std::launch::async,
                                         convert_columns,
                                         std::ref(values_),
                                         std::ref(is_valid_),
                                         std::ref(result_columns),
                                         non_lazy_cols,
                                         phys_start_col,
                                         phys_end_col));
    }
    for (auto& child : child_threads) {
      child.get();
    }
    row_count = entry_count;
  }
  if (!use_columnar_converter || !non_lazy_cols.empty()) {
    auto timer = DEBUG_TIMER("row converter");
    row_count = 0;
    if (multithreaded) {
      const size_t cpu_count = cpu_threads();
      std::vector<std::future<size_t>> child_threads;
      std::vector<std::vector<std::shared_ptr<ValueArray>>> column_value_segs(
          cpu_count, std::vector<std::shared_ptr<ValueArray>>(col_count, nullptr));
      std::vector<std::vector<std::shared_ptr<std::vector<bool>>>> null_bitmap_segs(
          cpu_count, std::vector<std::shared_ptr<std::vector<bool>>>(col_count, nullptr));
      const auto stride = (entry_count + cpu_count - 1) / cpu_count;
      for (size_t i = 0, start_entry = 0; start_entry < entry_count;
           ++i, start_entry += stride) {
        const auto end_entry = std::min(entry_count, start_entry + stride);
        child_threads.push_back(std::async(std::launch::async,
                                           fetch,
                                           std::ref(column_value_segs[i]),
                                           std::ref(null_bitmap_segs[i]),
                                           non_lazy_cols,
                                           start_entry,
                                           end_entry));
      }
      for (auto& child : child_threads) {
        row_count += child.get();
      }
      {
        auto timer = DEBUG_TIMER("append rows to arrow");
        for (int i = 0; i < schema->num_fields(); ++i) {
          if (!non_lazy_cols.empty() && non_lazy_cols[i]) {
            continue;
          }

          reserveColumnBuilderSize(builders[i], row_count);
          for (size_t j = 0; j < cpu_count; ++j) {
            if (!column_value_segs[j][i]) {
              continue;
            }
            append(builders[i], *column_value_segs[j][i], null_bitmap_segs[j][i]);
          }
        }
      }
    } else {
      row_count =
          fetch(column_values, null_bitmaps, non_lazy_cols, size_t(0), entry_count);
      {
        auto timer = DEBUG_TIMER("append rows to arrow");
        for (int i = 0; i < schema->num_fields(); ++i) {
          if (!non_lazy_cols.empty() && non_lazy_cols[i]) {
            continue;
          }

          reserveColumnBuilderSize(builders[i], row_count);
          append(builders[i], *column_values[i], null_bitmaps[i]);
        }
      }
    }

    {
      auto timer = DEBUG_TIMER("finish builders");
      for (size_t i = 0; i < col_count; ++i) {
        if (!non_lazy_cols.empty() && non_lazy_cols[i]) {
          continue;
        }

        result_columns[i] = finishColumnBuilder(builders[i]);
      }
    }
  }

  return ARROW_RECORDBATCH_MAKE(schema, row_count, result_columns);
}

std::shared_ptr<arrow::Field> ArrowResultSetConverter::makeField(
    const std::string name,
    const SQLTypeInfo& target_type,
    const std::shared_ptr<arrow::Array>& dictionary) const {
  return arrow::field(
      name, getArrowType(target_type, dictionary), !target_type.get_notnull());
}

std::shared_ptr<arrow::DataType> ArrowResultSetConverter::getArrowType(
    const SQLTypeInfo& mapd_type,
    const std::shared_ptr<arrow::Array>& dict_values) const {
  switch (get_physical_type(mapd_type)) {
    case kBOOLEAN:
      return boolean();
    case kTINYINT:
      return int8();
    case kSMALLINT:
      return int16();
    case kINT:
      return int32();
    case kBIGINT:
      return int64();
    case kFLOAT:
      return float32();
    case kDOUBLE:
      return float64();
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      if (mapd_type.is_dict_encoded_string()) {
        CHECK(dict_values);
        const auto index_type =
            getArrowType(get_dict_index_type_info(mapd_type), nullptr);
        return dictionary(index_type, dict_values);
      }
      return utf8();
    case kDECIMAL:
    case kNUMERIC:
      return decimal(mapd_type.get_precision(), mapd_type.get_scale());
    case kTIME:
      return time32(TimeUnit::SECOND);
    case kDATE:
      // TODO(wamsi) : Remove date64() once date32() support is added in cuDF. date32()
      // Currently support for date32() is missing in cuDF.Hence, if client requests for
      // date on GPU, return date64() for the time being, till support is added.
      return device_type_ == ExecutorDeviceType::GPU ? date64() : date32();
    case kTIMESTAMP:
      switch (mapd_type.get_precision()) {
        case 0:
          return timestamp(TimeUnit::SECOND);
        case 3:
          return timestamp(TimeUnit::MILLI);
        case 6:
          return timestamp(TimeUnit::MICRO);
        case 9:
          return timestamp(TimeUnit::NANO);
        default:
          throw std::runtime_error(
              "Unsupported timestamp precision for Arrow result sets: " +
              std::to_string(mapd_type.get_precision()));
      }
    case kARRAY:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    default:
      throw std::runtime_error(mapd_type.get_type_name() +
                               " is not supported in Arrow result sets.");
  }
  return nullptr;
}

void ArrowResultSet::deallocateArrowResultBuffer(
    const ArrowResult& result,
    const ExecutorDeviceType device_type,
    const size_t device_id,
    std::shared_ptr<Data_Namespace::DataMgr>& data_mgr) {
  // Remove shared memory on sysmem
  CHECK_EQ(sizeof(key_t), result.sm_handle.size());
  const key_t& schema_key = *(key_t*)(&result.sm_handle[0]);
  auto shm_id = shmget(schema_key, result.sm_size, 0666);
  if (shm_id < 0) {
    throw std::runtime_error(
        "failed to get an valid shm ID w/ given shm key of the schema");
  }
  if (-1 == shmctl(shm_id, IPC_RMID, 0)) {
    throw std::runtime_error("failed to deallocate Arrow schema on errorno(" +
                             std::to_string(errno) + ")");
  }

  if (device_type == ExecutorDeviceType::CPU) {
    CHECK_EQ(sizeof(key_t), result.df_handle.size());
    const key_t& df_key = *(key_t*)(&result.df_handle[0]);
    auto shm_id = shmget(df_key, result.df_size, 0666);
    if (shm_id < 0) {
      throw std::runtime_error(
          "failed to get an valid shm ID w/ given shm key of the data");
    }
    if (-1 == shmctl(shm_id, IPC_RMID, 0)) {
      throw std::runtime_error("failed to deallocate Arrow data frame");
    }
    return;
  }

  CHECK(device_type == ExecutorDeviceType::GPU);
  if (!result.df_dev_ptr) {
    throw std::runtime_error("null pointer to data frame on device");
  }

  data_mgr->getCudaMgr()->freeDeviceMem(result.df_dev_ptr);
}

void ArrowResultSetConverter::initializeColumnBuilder(
    ColumnBuilder& column_builder,
    const SQLTypeInfo& col_type,
    const std::shared_ptr<arrow::Field>& field) const {
  column_builder.field = field;
  column_builder.col_type = col_type;
  column_builder.physical_type = col_type.is_dict_encoded_string()
                                     ? get_dict_index_type(col_type)
                                     : get_physical_type(col_type);

  auto value_type = field->type();
  if (value_type->id() == Type::DICTIONARY) {
    value_type = static_cast<const DictionaryType&>(*value_type).index_type();
  }
  ARROW_THROW_NOT_OK(
      arrow::MakeBuilder(default_memory_pool(), value_type, &column_builder.builder));
}

void ArrowResultSetConverter::reserveColumnBuilderSize(ColumnBuilder& column_builder,
                                                       const size_t row_count) const {
  ARROW_THROW_NOT_OK(column_builder.builder->Reserve(static_cast<int64_t>(row_count)));
}

std::shared_ptr<arrow::Array> ArrowResultSetConverter::finishColumnBuilder(
    ColumnBuilder& column_builder) const {
  std::shared_ptr<Array> values;
  ARROW_THROW_NOT_OK(column_builder.builder->Finish(&values));
  if (column_builder.field->type()->id() == Type::DICTIONARY) {
    return std::make_shared<DictionaryArray>(column_builder.field->type(), values);
  } else {
    return values;
  }
}

template <typename BuilderType, typename C_TYPE>
void ArrowResultSetConverter::appendToColumnBuilder(
    ColumnBuilder& column_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) const {
  std::vector<C_TYPE> vals = boost::get<std::vector<C_TYPE>>(values);

  if (scale_epoch_values<BuilderType>()) {
    auto scale_sec_to_millisec = [](auto seconds) { return seconds * kMilliSecsPerSec; };
    auto scale_values = [&](auto epoch) {
      return std::is_same<BuilderType, Date32Builder>::value
                 ? DateConverters::get_epoch_days_from_seconds(epoch)
                 : scale_sec_to_millisec(epoch);
    };
    std::transform(vals.begin(), vals.end(), vals.begin(), scale_values);
  }

  auto typed_builder = static_cast<BuilderType*>(column_builder.builder.get());
  if (column_builder.field->nullable()) {
    CHECK(is_valid.get());
    ARROW_THROW_NOT_OK(typed_builder->APPENDVALUES(vals, *is_valid));
  } else {
    ARROW_THROW_NOT_OK(typed_builder->APPENDVALUES(vals));
  }
}

template <typename BuilderType, typename C_TYPE>
void ArrowResultSetConverter::appendToColumnBuilder(ColumnBuilder& column_builder,
                                                    int8_t* data,
                                                    size_t entry_count,
                                                    const uint8_t* is_valid) const {
  C_TYPE* vals = reinterpret_cast<C_TYPE*>(data);

  if (scale_epoch_values<BuilderType>()) {
    auto scale_sec_to_millisec = [](auto seconds) { return seconds * kMilliSecsPerSec; };
    auto scale_values = [&](auto epoch) {
      return std::is_same<BuilderType, Date32Builder>::value
                 ? DateConverters::get_epoch_days_from_seconds(epoch)
                 : scale_sec_to_millisec(epoch);
    };
    std::transform(vals, vals + entry_count, vals, scale_values);
  }

  auto typed_builder = static_cast<BuilderType*>(column_builder.builder.get());
  if (is_valid) {
    CHECK(column_builder.field->nullable());
    ARROW_THROW_NOT_OK(typed_builder->APPENDVALUES(vals, entry_count, is_valid));
  } else {
    ARROW_THROW_NOT_OK(typed_builder->APPENDVALUES(vals, entry_count));
  }
}

void ArrowResultSetConverter::append(
    ColumnBuilder& column_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) const {
  switch (column_builder.physical_type) {
    case kBOOLEAN:
      appendToColumnBuilder<BooleanBuilder, bool>(column_builder, values, is_valid);
      break;
    case kTINYINT:
      appendToColumnBuilder<Int8Builder, int8_t>(column_builder, values, is_valid);
      break;
    case kSMALLINT:
      appendToColumnBuilder<Int16Builder, int16_t>(column_builder, values, is_valid);
      break;
    case kINT:
      appendToColumnBuilder<Int32Builder, int32_t>(column_builder, values, is_valid);
      break;
    case kBIGINT:
      appendToColumnBuilder<Int64Builder, int64_t>(column_builder, values, is_valid);
      break;
    case kFLOAT:
      appendToColumnBuilder<FloatBuilder, float>(column_builder, values, is_valid);
      break;
    case kDOUBLE:
      appendToColumnBuilder<DoubleBuilder, double>(column_builder, values, is_valid);
      break;
    case kTIME:
      appendToColumnBuilder<Time32Builder, int32_t>(column_builder, values, is_valid);
      break;
    case kTIMESTAMP:
      appendToColumnBuilder<TimestampBuilder, int64_t>(column_builder, values, is_valid);
      break;
    case kDATE:
      device_type_ == ExecutorDeviceType::GPU
          ? appendToColumnBuilder<Date64Builder, int64_t>(
                column_builder, values, is_valid)
          : appendToColumnBuilder<Date32Builder, int32_t>(
                column_builder, values, is_valid);
      break;
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
    default:
      // TODO(miyu): support more scalar types.
      throw std::runtime_error(column_builder.col_type.get_type_name() +
                               " is not supported in Arrow result sets.");
  }
}

void ArrowResultSetConverter::append(ColumnBuilder& column_builder,
                                     int8_t* data,
                                     size_t entry_count,
                                     const uint8_t* is_valid) const {
  switch (column_builder.physical_type) {
    case kBOOLEAN:
      appendToColumnBuilder<BooleanBuilder, uint8_t>(
          column_builder, data, entry_count, is_valid);
      break;
    case kTINYINT:
      appendToColumnBuilder<Int8Builder, int8_t>(
          column_builder, data, entry_count, is_valid);
      break;
    case kSMALLINT:
      appendToColumnBuilder<Int16Builder, int16_t>(
          column_builder, data, entry_count, is_valid);
      break;
    case kINT:
      appendToColumnBuilder<Int32Builder, int32_t>(
          column_builder, data, entry_count, is_valid);
      break;
    case kBIGINT:
      appendToColumnBuilder<Int64Builder, int64_t>(
          column_builder, data, entry_count, is_valid);
      break;
    case kFLOAT:
      appendToColumnBuilder<FloatBuilder, float>(
          column_builder, data, entry_count, is_valid);
      break;
    case kDOUBLE:
      appendToColumnBuilder<DoubleBuilder, double>(
          column_builder, data, entry_count, is_valid);
      break;
    case kTIME:
      appendToColumnBuilder<Time32Builder, int32_t>(
          column_builder, data, entry_count, is_valid);
      break;
    case kTIMESTAMP:
      appendToColumnBuilder<TimestampBuilder, int64_t>(
          column_builder, data, entry_count, is_valid);
      break;
    case kDATE:
      device_type_ == ExecutorDeviceType::GPU
          ? appendToColumnBuilder<Date64Builder, int64_t>(
                column_builder, data, entry_count, is_valid)
          : appendToColumnBuilder<Date32Builder, int32_t>(
                column_builder, data, entry_count, is_valid);
      break;
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
    default:
      // TODO(miyu): support more scalar types.
      throw std::runtime_error(column_builder.col_type.get_type_name() +
                               " is not supported in Arrow result sets.");
  }
}

// helpers for debugging

#ifdef ENABLE_ARROW_DEBUG
void print_serialized_schema(const uint8_t* data, const size_t length) {
  io::BufferReader reader(std::make_shared<arrow::Buffer>(data, length));
  std::shared_ptr<Schema> schema;
  ARROW_THROW_NOT_OK(ipc::ReadSchema(&reader, &schema));

  std::cout << "Arrow Schema: " << std::endl;
  const PrettyPrintOptions options{0};
  ARROW_THROW_NOT_OK(PrettyPrint(*(schema.get()), options, &std::cout));
}

void print_serialized_records(const uint8_t* data,
                              const size_t length,
                              const std::shared_ptr<Schema>& schema) {
  if (data == nullptr || !length) {
    std::cout << "No row found" << std::endl;
    return;
  }
  std::shared_ptr<RecordBatch> batch;

  io::BufferReader buffer_reader(std::make_shared<arrow::Buffer>(data, length));
  ARROW_THROW_NOT_OK(ipc::ReadRecordBatch(schema, &buffer_reader, &batch));
}
#endif
