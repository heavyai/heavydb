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

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <string>

#include "arrow/api.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/api.h"

#include "Shared/ArrowUtil.h"

#ifdef HAVE_CUDA
#include <arrow/gpu/cuda_api.h>
#include <cuda.h>
#endif  // HAVE_CUDA

#define ARROW_RECORDBATCH_MAKE arrow::RecordBatch::Make

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

template <typename TYPE, typename VALUE_ARRAY_TYPE>
void create_or_append_value(const ScalarTargetValue& val_cty,
                            std::shared_ptr<ValueArray>& values,
                            const size_t max_size) {
  auto pval_cty = boost::get<VALUE_ARRAY_TYPE>(&val_cty);
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

//! Serialize an Arrow result to IPC memory. Users are responsible for freeing all CPU IPC
//! buffers using deallocateArrowResultBuffer. GPU buffers will become owned by the caller
//! upon deserialization, and will be automatically freed when they go out of scope.
ArrowResult ArrowResultSetConverter::getArrowResult() const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<arrow::RecordBatch> record_batch = convertToArrow();

  if (device_type_ == ExecutorDeviceType::CPU) {
    auto timer = DEBUG_TIMER("serialize batch to shared memory");
    std::shared_ptr<Buffer> serialized_records;
    std::shared_ptr<Buffer> serialized_schema;
    std::vector<char> schema_handle_buffer;
    std::vector<char> record_handle_buffer(sizeof(key_t), 0);
    int64_t total_size = 0;
    int64_t records_size = 0;
    int64_t schema_size = 0;
    key_t records_shm_key = IPC_PRIVATE;
    ipc::DictionaryMemo memo;
    auto options = ipc::IpcOptions::Defaults();
    auto dict_stream = arrow::io::BufferOutputStream::Create(1024).ValueOrDie();

    ARROW_THROW_NOT_OK(CollectDictionaries(*record_batch, &memo));
    for (auto& pair : memo.id_to_dictionary()) {
      ipc::internal::IpcPayload payload;
      int64_t dictionary_id = pair.first;
      const auto& dictionary = pair.second;

      ARROW_THROW_NOT_OK(GetDictionaryPayload(
          dictionary_id, dictionary, options, default_memory_pool(), &payload));
      int32_t metadata_length = 0;
      WriteIpcPayload(payload, options, dict_stream.get(), &metadata_length);
    }
    auto serialized_dict = dict_stream->Finish().ValueOrDie();
    auto dict_size = serialized_dict->size();

    ARROW_THROW_NOT_OK(ipc::SerializeSchema(
        *record_batch->schema(), nullptr, default_memory_pool(), &serialized_schema));
    schema_size = serialized_schema->size();

    ARROW_THROW_NOT_OK(ipc::GetRecordBatchSize(*record_batch, &records_size));
    total_size = schema_size + dict_size + records_size;
    std::tie(records_shm_key, serialized_records) = get_shm_buffer(total_size);

    memcpy(serialized_records->mutable_data(),
           serialized_schema->data(),
           (size_t)schema_size);
    memcpy(serialized_records->mutable_data() + schema_size,
           serialized_dict->data(),
           (size_t)dict_size);

    io::FixedSizeBufferWriter stream(
        SliceMutableBuffer(serialized_records, schema_size + dict_size));
    ARROW_THROW_NOT_OK(
        ipc::SerializeRecordBatch(*record_batch, arrow::default_memory_pool(), &stream));
    memcpy(&record_handle_buffer[0],
           reinterpret_cast<const unsigned char*>(&records_shm_key),
           sizeof(key_t));

    return {schema_handle_buffer,
            0,
            record_handle_buffer,
            serialized_records->size(),
            std::string{""}};
  }
#ifdef HAVE_CUDA
  CHECK(device_type_ == ExecutorDeviceType::GPU);

  // Copy the schema to the schema handle
  auto out_stream_result = arrow::io::BufferOutputStream::Create(1024);
  ARROW_THROW_NOT_OK(out_stream_result.status());
  auto out_stream = std::move(out_stream_result).ValueOrDie();

  arrow::ipc::DictionaryMemo current_memo;
  arrow::ipc::DictionaryMemo serialized_memo;

  arrow::ipc::internal::IpcPayload schema_payload;
  ARROW_THROW_NOT_OK(
      arrow::ipc::internal::GetSchemaPayload(*record_batch->schema(),
                                             arrow::ipc::IpcOptions::Defaults(),
                                             &serialized_memo,
                                             &schema_payload));
  int32_t schema_payload_length = 0;
  ARROW_THROW_NOT_OK(
      arrow::ipc::internal::WriteIpcPayload(schema_payload,
                                            arrow::ipc::IpcOptions::Defaults(),
                                            out_stream.get(),
                                            &schema_payload_length));

  ARROW_THROW_NOT_OK(CollectDictionaries(*record_batch, &current_memo));

  // now try a dictionary
  std::shared_ptr<arrow::Schema> dummy_schema;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dict_batches;
  for (int i = 0; i < record_batch->schema()->num_fields(); i++) {
    auto field = record_batch->schema()->field(i);
    if (field->type()->id() == arrow::Type::DICTIONARY) {
      int64_t dict_id = -1;
      ARROW_THROW_NOT_OK(current_memo.GetId(*field, &dict_id));
      CHECK_GE(dict_id, 0);
      std::shared_ptr<Array> dict;
      ARROW_THROW_NOT_OK(current_memo.GetDictionary(dict_id, &dict));
      CHECK(dict);

      if (!dummy_schema) {
        auto dummy_field = std::make_shared<arrow::Field>("", dict->type());
        dummy_schema = std::make_shared<arrow::Schema>(
            std::vector<std::shared_ptr<arrow::Field>>{dummy_field});
      }
      dict_batches.emplace_back(
          arrow::RecordBatch::Make(dummy_schema, dict->length(), {dict}));
    }
  }

  if (!dict_batches.empty()) {
    ARROW_THROW_NOT_OK(arrow::ipc::WriteRecordBatchStream(
        dict_batches, ipc::IpcOptions::Defaults(), out_stream.get()));
  }

  auto complete_ipc_stream = out_stream->Finish();
  ARROW_THROW_NOT_OK(complete_ipc_stream.status());
  auto serialized_records = std::move(complete_ipc_stream).ValueOrDie();

  const auto record_key = arrow::get_and_copy_to_shm(serialized_records);
  std::vector<char> schema_record_key_buffer(sizeof(key_t), 0);
  memcpy(&schema_record_key_buffer[0],
         reinterpret_cast<const unsigned char*>(&record_key),
         sizeof(key_t));

  arrow::cuda::CudaDeviceManager* manager;
  ARROW_THROW_NOT_OK(arrow::cuda::CudaDeviceManager::GetInstance(&manager));
  std::shared_ptr<arrow::cuda::CudaContext> context;
  ARROW_THROW_NOT_OK(manager->GetContext(device_id_, &context));

  std::shared_ptr<arrow::cuda::CudaBuffer> device_serialized;
  ARROW_THROW_NOT_OK(
      SerializeRecordBatch(*record_batch, context.get(), &device_serialized));

  std::shared_ptr<arrow::cuda::CudaIpcMemHandle> cuda_handle;
  ARROW_THROW_NOT_OK(device_serialized->ExportForIpc(&cuda_handle));

  std::shared_ptr<arrow::Buffer> serialized_cuda_handle;
  ARROW_THROW_NOT_OK(
      cuda_handle->Serialize(arrow::default_memory_pool(), &serialized_cuda_handle));

  std::vector<char> record_handle_buffer(serialized_cuda_handle->size(), 0);
  memcpy(&record_handle_buffer[0],
         serialized_cuda_handle->data(),
         serialized_cuda_handle->size());

  return {schema_record_key_buffer,
          serialized_records->size(),
          record_handle_buffer,
          serialized_cuda_handle->size(),
          serialized_cuda_handle->ToString()};
#else
  UNREACHABLE();
  return {std::vector<char>{}, 0, std::vector<char>{}, 0, ""};
#endif
}

ArrowResultSetConverter::SerializedArrowOutput
ArrowResultSetConverter::getSerializedArrowOutput(
    arrow::ipc::DictionaryMemo* memo) const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<arrow::RecordBatch> arrow_copy = convertToArrow();
  std::shared_ptr<arrow::Buffer> serialized_records, serialized_schema;

  ARROW_THROW_NOT_OK(arrow::ipc::SerializeSchema(
      *arrow_copy->schema(), memo, arrow::default_memory_pool(), &serialized_schema));

  ARROW_THROW_NOT_OK(CollectDictionaries(*arrow_copy, memo));

  if (arrow_copy->num_rows()) {
    auto timer = DEBUG_TIMER("serialize records");
    ARROW_THROW_NOT_OK(arrow_copy->Validate());
    ARROW_THROW_NOT_OK(arrow::ipc::SerializeRecordBatch(
        *arrow_copy, arrow::default_memory_pool(), &serialized_records));
  } else {
    ARROW_THROW_NOT_OK(arrow::AllocateBuffer(0, &serialized_records));
  }
  return {serialized_schema, serialized_records};
}

std::shared_ptr<arrow::RecordBatch> ArrowResultSetConverter::convertToArrow() const {
  const auto col_count = results_->colCount();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  CHECK(col_names_.empty() || col_names_.size() == col_count);
  for (size_t i = 0; i < col_count; ++i) {
    const auto ti = results_->getColType(i);
    fields.push_back(makeField(col_names_.empty() ? "" : col_names_[i], ti));
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

  std::vector<ColumnBuilder> builders(col_count);

  // Create array builders
  for (size_t i = 0; i < col_count; ++i) {
    initializeColumnBuilder(builders[i], results_->getColType(i), schema->field(i));
  }

  // TODO(miyu): speed up for columnar buffers
  auto fetch = [&](std::vector<std::shared_ptr<ValueArray>>& value_seg,
                   std::vector<std::shared_ptr<std::vector<bool>>>& null_bitmap_seg,
                   const size_t start_entry,
                   const size_t end_entry) -> size_t {
    CHECK_EQ(value_seg.size(), col_count);
    CHECK_EQ(null_bitmap_seg.size(), col_count);
    const auto entry_count = end_entry - start_entry;
    size_t seg_row_count = 0;
    for (size_t i = start_entry; i < end_entry; ++i) {
      auto row = results_->getRowAtNoTranslations(i);
      if (row.empty()) {
        continue;
      }
      ++seg_row_count;
      for (size_t j = 0; j < col_count; ++j) {
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

  std::vector<std::shared_ptr<ValueArray>> column_values(col_count, nullptr);
  std::vector<std::shared_ptr<std::vector<bool>>> null_bitmaps(col_count, nullptr);
  const bool multithreaded = entry_count > 10000 && !results_->isTruncated();
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
                                         start_entry,
                                         end_entry));
    }
    for (auto& child : child_threads) {
      row_count += child.get();
    }
    for (int i = 0; i < schema->num_fields(); ++i) {
      for (size_t j = 0; j < cpu_count; ++j) {
        if (!column_value_segs[j][i]) {
          continue;
        }
        append(builders[i], *column_value_segs[j][i], null_bitmap_segs[j][i]);
      }
    }
  } else {
    row_count = fetch(column_values, null_bitmaps, size_t(0), entry_count);
    for (int i = 0; i < schema->num_fields(); ++i) {
      append(builders[i], *column_values[i], null_bitmaps[i]);
    }
  }

  for (size_t i = 0; i < col_count; ++i) {
    result_columns.push_back(finishColumnBuilder(builders[i]));
  }
  return ARROW_RECORDBATCH_MAKE(schema, row_count, result_columns);
}

namespace {

std::shared_ptr<arrow::DataType> get_arrow_type(const SQLTypeInfo& sql_type,
                                                const ExecutorDeviceType device_type) {
  switch (get_physical_type(sql_type)) {
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
      if (sql_type.is_dict_encoded_string()) {
        auto value_type = std::make_shared<StringType>();
        return dictionary(int32(), value_type, false);
      }
      return utf8();
    case kDECIMAL:
    case kNUMERIC:
      return decimal(sql_type.get_precision(), sql_type.get_scale());
    case kTIME:
      return time32(TimeUnit::SECOND);
    case kDATE:
      // TODO(wamsi) : Remove date64() once date32() support is added in cuDF. date32()
      // Currently support for date32() is missing in cuDF.Hence, if client requests for
      // date on GPU, return date64() for the time being, till support is added.
      return device_type == ExecutorDeviceType::GPU ? date64() : date32();
    case kTIMESTAMP:
      switch (sql_type.get_precision()) {
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
              std::to_string(sql_type.get_precision()));
      }
    case kARRAY:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    default:
      throw std::runtime_error(sql_type.get_type_name() +
                               " is not supported in Arrow result sets.");
  }
  return nullptr;
}

}  // namespace

std::shared_ptr<arrow::Field> ArrowResultSetConverter::makeField(
    const std::string name,
    const SQLTypeInfo& target_type) const {
  return arrow::field(
      name, get_arrow_type(target_type, device_type_), !target_type.get_notnull());
}

void ArrowResultSet::deallocateArrowResultBuffer(
    const ArrowResult& result,
    const ExecutorDeviceType device_type,
    const size_t device_id,
    std::shared_ptr<Data_Namespace::DataMgr>& data_mgr) {
  // CPU buffers skip the sm handle, serializing the entire RecordBatch to df.
  // Remove shared memory on sysmem
  if (!result.sm_handle.empty()) {
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
  }
  // CUDA buffers become owned by the caller, and will automatically be freed
  // TODO: What if the client never takes ownership of the result? we may want to
  // establish a check to see if the GPU buffer still exists, and then free it.
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
  if (col_type.is_dict_encoded_string()) {
    column_builder.builder.reset(new StringDictionary32Builder());
    // add values to the builder
    const int dict_id = col_type.get_comp_param();
    auto str_list = results_->getStringDictionaryPayloadCopy(dict_id);

    arrow::StringBuilder str_array_builder;
    ARROW_THROW_NOT_OK(str_array_builder.AppendValues(*str_list));
    std::shared_ptr<StringArray> string_array;
    ARROW_THROW_NOT_OK(str_array_builder.Finish(&string_array));

    auto dict_builder =
        dynamic_cast<arrow::StringDictionary32Builder*>(column_builder.builder.get());
    CHECK(dict_builder);

    ARROW_THROW_NOT_OK(dict_builder->InsertMemoValues(*string_array));
  } else {
    ARROW_THROW_NOT_OK(
        arrow::MakeBuilder(default_memory_pool(), value_type, &column_builder.builder));
  }
}

std::shared_ptr<arrow::Array> ArrowResultSetConverter::finishColumnBuilder(
    ColumnBuilder& column_builder) const {
  std::shared_ptr<Array> values;
  ARROW_THROW_NOT_OK(column_builder.builder->Finish(&values));
  return values;
}

namespace {

template <typename BUILDER_TYPE, typename VALUE_ARRAY_TYPE>
void appendToColumnBuilder(ArrowResultSetConverter::ColumnBuilder& column_builder,
                           const ValueArray& values,
                           const std::shared_ptr<std::vector<bool>>& is_valid) {
  static_assert(!std::is_same<BUILDER_TYPE, arrow::StringDictionary32Builder>::value,
                "Dictionary encoded string builder requires function specialization.");

  std::vector<VALUE_ARRAY_TYPE> vals = boost::get<std::vector<VALUE_ARRAY_TYPE>>(values);

  if (scale_epoch_values<BUILDER_TYPE>()) {
    auto scale_sec_to_millisec = [](auto seconds) { return seconds * kMilliSecsPerSec; };
    auto scale_values = [&](auto epoch) {
      return std::is_same<BUILDER_TYPE, Date32Builder>::value
                 ? DateConverters::get_epoch_days_from_seconds(epoch)
                 : scale_sec_to_millisec(epoch);
    };
    std::transform(vals.begin(), vals.end(), vals.begin(), scale_values);
  }

  auto typed_builder = dynamic_cast<BUILDER_TYPE*>(column_builder.builder.get());
  CHECK(typed_builder);
  if (column_builder.field->nullable()) {
    CHECK(is_valid.get());
    ARROW_THROW_NOT_OK(typed_builder->AppendValues(vals, *is_valid));
  } else {
    ARROW_THROW_NOT_OK(typed_builder->AppendValues(vals));
  }
}

template <>
void appendToColumnBuilder<arrow::StringDictionary32Builder, int32_t>(
    ArrowResultSetConverter::ColumnBuilder& column_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) {
  auto typed_builder =
      dynamic_cast<arrow::StringDictionary32Builder*>(column_builder.builder.get());
  CHECK(typed_builder);

  std::vector<int32_t> vals = boost::get<std::vector<int32_t>>(values);

  if (column_builder.field->nullable()) {
    CHECK(is_valid.get());
    // TODO(adb): Generate this instead of the boolean bitmap
    std::vector<uint8_t> transformed_bitmap;
    transformed_bitmap.reserve(is_valid->size());
    std::for_each(
        is_valid->begin(), is_valid->end(), [&transformed_bitmap](const bool is_valid) {
          transformed_bitmap.push_back(is_valid ? 1 : 0);
        });

    ARROW_THROW_NOT_OK(typed_builder->AppendIndices(
        vals.data(), static_cast<int64_t>(vals.size()), transformed_bitmap.data()));
  } else {
    ARROW_THROW_NOT_OK(
        typed_builder->AppendIndices(vals.data(), static_cast<int64_t>(vals.size())));
  }
}

}  // namespace

void ArrowResultSetConverter::append(
    ColumnBuilder& column_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) const {
  if (column_builder.col_type.is_dict_encoded_string()) {
    CHECK_EQ(column_builder.physical_type,
             kINT);  // assume all dicts use none-encoded type for now
    appendToColumnBuilder<StringDictionary32Builder, int32_t>(
        column_builder, values, is_valid);
    return;
  }
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
