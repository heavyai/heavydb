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

#include "../Shared/DateConverters.h"
#include "ArrowResultSet.h"
#include "Execute.h"
#include "arrow/ipc/dictionary.h"
#include "arrow/ipc/options.h"

#ifndef _MSC_VER
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#else
// IPC shared memory not yet supported on windows
using key_t = size_t;
#define IPC_PRIVATE 0
#endif

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

#define ARROW_CONVERTER_DEBUG true

#define ARROW_LOG(category) \
  VLOG(1) << "[Arrow]"      \
          << "[" << category "] "

namespace {

/* We can create Arrow buffers which refer memory owned by ResultSet.
   For safe memory access we should keep a ResultSetPtr to keep
   data live while buffer lives. Use this custom buffer for that. */
class ResultSetBuffer : public arrow::Buffer {
 public:
  ResultSetBuffer(const uint8_t* buf, size_t size, ResultSetPtr rs)
      : arrow::Buffer(buf, size), _rs(rs) {}

 private:
  ResultSetPtr _rs;
};

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
  if (!values) {
    values = std::make_shared<ValueArray>(std::vector<TYPE>());
    boost::get<std::vector<TYPE>>(*values).reserve(max_size);
  }
  CHECK(values);
  auto values_ty = boost::get<std::vector<TYPE>>(values.get());
  CHECK(values_ty);

  auto pval_cty = boost::get<VALUE_ARRAY_TYPE>(&val_cty);
  CHECK(pval_cty);
  if constexpr (std::is_same_v<VALUE_ARRAY_TYPE, NullableString>) {
    if (auto str = boost::get<std::string>(pval_cty)) {
      values_ty->push_back(*str);
    } else {
      values_ty->push_back("");
    }
  } else {
    auto val_ty = static_cast<TYPE>(*pval_cty);
    values_ty->push_back(val_ty);
  }
}

template <typename TYPE, typename VALUE_ARRAY_TYPE>
void create_or_append_value(const ArrayTargetValue& val_ctys,
                            std::shared_ptr<ValueArray>& values,
                            const size_t max_size) {
  if (!values) {
    values = std::make_shared<ValueArray>(Vec2<TYPE>());
    boost::get<Vec2<TYPE>>(*values).reserve(max_size);
  }
  CHECK(values);

  Vec2<TYPE>* values_ty = boost::get<Vec2<TYPE>>(values.get());
  CHECK(values_ty);

  values_ty->emplace_back(std::vector<TYPE>{});

  if (val_ctys) {
    for (auto val_cty : val_ctys.value()) {
      auto pval_cty = boost::get<VALUE_ARRAY_TYPE>(&val_cty);
      CHECK(pval_cty);
      values_ty->back().emplace_back(static_cast<TYPE>(*pval_cty));
    }
  }
}

void create_or_append_validity(const ArrayTargetValue& value,
                               const SQLTypeInfo& col_type,
                               std::shared_ptr<std::vector<bool>>& null_bitmap,
                               const size_t max_size) {
  if (col_type.get_notnull()) {
    CHECK(!null_bitmap);
    return;
  }

  if (!null_bitmap) {
    null_bitmap = std::make_shared<std::vector<bool>>();
    null_bitmap->reserve(max_size);
  }
  CHECK(null_bitmap);
  null_bitmap->push_back(value ? true : false);
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
  if constexpr (std::is_same_v<TYPE, NullableString>) {
    is_valid = boost::get<std::string>(pvalue) != nullptr;
  } else {
    if (col_type.is_boolean()) {
      is_valid = inline_int_null_val(col_type) != static_cast<int8_t>(*pvalue);
    } else if (col_type.is_dict_encoded_string()) {
      is_valid = inline_int_null_val(col_type) != static_cast<int32_t>(*pvalue);
    } else if (col_type.is_integer() || col_type.is_time() || col_type.is_decimal()) {
      is_valid = inline_int_null_val(col_type) != static_cast<int64_t>(*pvalue);
    } else if (col_type.is_fp()) {
      is_valid = inline_fp_null_val(col_type) != static_cast<double>(*pvalue);
    } else {
      UNREACHABLE();
    }
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

template <typename C_TYPE,
          typename ARROW_TYPE = typename arrow::CTypeTraits<C_TYPE>::ArrowType>
void convert_column(ResultSetPtr result,
                    size_t col,
                    size_t entry_count,
                    std::shared_ptr<arrow::Array>& out) {
  CHECK(sizeof(C_TYPE) == result->getColType(col).get_size());

  std::shared_ptr<arrow::Buffer> values;
  std::shared_ptr<arrow::Buffer> is_valid;
  const int64_t buf_size = entry_count * sizeof(C_TYPE);
  if (result->isZeroCopyColumnarConversionPossible(col)) {
    values.reset(new ResultSetBuffer(
        reinterpret_cast<const uint8_t*>(result->getColumnarBuffer(col)),
        buf_size,
        result));
  } else {
    auto res = arrow::AllocateBuffer(buf_size);
    CHECK(res.ok());
    values = std::move(res).ValueOrDie();
    result->copyColumnIntoBuffer(
        col, reinterpret_cast<int8_t*>(values->mutable_data()), buf_size);
  }

  int64_t null_count = 0;
  auto res = arrow::AllocateBuffer((entry_count + 7) / 8);
  CHECK(res.ok());
  is_valid = std::move(res).ValueOrDie();

  auto is_valid_data = is_valid->mutable_data();

  const null_type_t<C_TYPE>* vals =
      reinterpret_cast<const null_type_t<C_TYPE>*>(values->data());
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
    is_valid_data[i >> 3] = valid_byte;
  }
  if (unroll_count != entry_count) {
    uint8_t valid_byte = 0;
    for (size_t i = unroll_count; i < entry_count; ++i) {
      bool valid = vals[i] != null_val;
      valid_byte |= valid << (i & 7);
      null_count += !valid;
    }
    is_valid_data[unroll_count >> 3] = valid_byte;
  }

  if (!null_count) {
    is_valid.reset();
  }

  // TODO: support date/time + scaling
  // TODO: support booleans
  if (null_count) {
    out.reset(
        new arrow::NumericArray<ARROW_TYPE>(entry_count, values, is_valid, null_count));
  } else {
    out.reset(new arrow::NumericArray<ARROW_TYPE>(entry_count, values));
  }
}

#ifndef _MSC_VER
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
#endif

std::pair<key_t, std::shared_ptr<arrow::Buffer>> get_shm_buffer(size_t size) {
#ifdef _MSC_VER
  throw std::runtime_error("Arrow IPC not yet supported on Windows.");
  return std::make_pair(0, nullptr);
#else
  auto [key, ipc_ptr] = get_shm(size);
  std::shared_ptr<arrow::Buffer> buffer(
      new arrow::MutableBuffer(static_cast<uint8_t*>(ipc_ptr), size));
  return std::make_pair<key_t, std::shared_ptr<arrow::Buffer>>(std::move(key),
                                                               std::move(buffer));
#endif
}

void remap_string_values(const ArrowResultSetConverter::ColumnBuilder& column_builder,
                         const std::vector<uint8_t>& bitmap,
                         std::vector<int64_t>& vec1d) {
  /*
   remap negative values if ArrowStringRemapMode == ONLY_TRANSIENT_STRINGS_REMAPPED or
   everything if ALL_STRINGS_REMAPPED
  */

  auto all_strings_remapped_bitmap = [&column_builder, &vec1d, &bitmap]() {
    for (size_t i = 0; i < vec1d.size(); i++) {
      if (bitmap[i]) {
        vec1d[i] = column_builder.string_remapping.at(vec1d[i]);
      }
    }
  };

  auto all_strings_remapped = [&column_builder, &vec1d]() {
    for (size_t i = 0; i < vec1d.size(); i++) {
      vec1d[i] = column_builder.string_remapping.at(vec1d[i]);
    }
  };

  auto only_transient_strings_remapped = [&column_builder, &vec1d]() {
    for (size_t i = 0; i < vec1d.size(); i++) {
      if (vec1d[i] < 0) {
        vec1d[i] = column_builder.string_remapping.at(vec1d[i]);
      }
    }
  };

  auto only_transient_strings_remapped_bitmap = [&column_builder, &vec1d, &bitmap]() {
    for (size_t i = 0; i < vec1d.size(); i++) {
      if (bitmap[i] && vec1d[i] < 0) {
        vec1d[i] = column_builder.string_remapping.at(vec1d[i]);
      }
    }
  };

  switch (column_builder.string_remap_mode) {
    case ArrowStringRemapMode::ALL_STRINGS_REMAPPED:
      bitmap.empty() ? all_strings_remapped() : all_strings_remapped_bitmap();
      break;
    case ArrowStringRemapMode::ONLY_TRANSIENT_STRINGS_REMAPPED:
      bitmap.empty() ? only_transient_strings_remapped()
                     : only_transient_strings_remapped_bitmap();
      break;
    default:
      UNREACHABLE();
  }
}

}  // namespace

namespace arrow {

key_t get_and_copy_to_shm(const std::shared_ptr<Buffer>& data) {
#ifdef _MSC_VER
  throw std::runtime_error("Arrow IPC not yet supported on Windows.");
#else
  auto [key, ipc_ptr] = get_shm(data->size());
  // copy the arrow records buffer to shared memory
  // TODO(ptaylor): I'm sure it's possible to tell Arrow's RecordBatchStreamWriter to
  // write directly to the shared memory segment as a sink
  memcpy(ipc_ptr, data->data(), data->size());
  // detach from the shared memory segment
  shmdt(ipc_ptr);
  return key;
#endif
}

}  // namespace arrow

//! Serialize an Arrow result to IPC memory. Users are responsible for freeing all CPU IPC
//! buffers using deallocateArrowResultBuffer. GPU buffers will become owned by the caller
//! upon deserialization, and will be automatically freed when they go out of scope.
ArrowResult ArrowResultSetConverter::getArrowResult() const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<arrow::RecordBatch> record_batch = convertToArrow();

  struct BuildResultParams {
    int64_t schemaSize() const {
      return serialized_schema ? serialized_schema->size() : 0;
    };
    int64_t dictSize() const { return serialized_dict ? serialized_dict->size() : 0; };
    int64_t totalSize() const { return schemaSize() + records_size + dictSize(); }
    bool hasRecordBatch() const { return records_size > 0; }
    bool hasDict() const { return dictSize() > 0; }

    int64_t records_size{0};
    std::shared_ptr<arrow::Buffer> serialized_schema{nullptr};
    std::shared_ptr<arrow::Buffer> serialized_dict{nullptr};
  } result_params;

  if (device_type_ == ExecutorDeviceType::CPU ||
      transport_method_ == ArrowTransport::WIRE) {
    const auto getWireResult = [&]() -> ArrowResult {
      auto timer = DEBUG_TIMER("serialize batch to wire");
      const auto total_size = result_params.totalSize();
      std::vector<char> record_handle_data(total_size);
      auto serialized_records =
          arrow::MutableBuffer::Wrap(record_handle_data.data(), total_size);

      ARROW_ASSIGN_OR_THROW(auto writer, arrow::Buffer::GetWriter(serialized_records));

      ARROW_THROW_NOT_OK(writer->Write(
          reinterpret_cast<const uint8_t*>(result_params.serialized_schema->data()),
          result_params.schemaSize()));

      if (result_params.hasDict()) {
        ARROW_THROW_NOT_OK(writer->Write(
            reinterpret_cast<const uint8_t*>(result_params.serialized_dict->data()),
            result_params.dictSize()));
      }

      arrow::io::FixedSizeBufferWriter stream(SliceMutableBuffer(
          serialized_records, result_params.schemaSize() + result_params.dictSize()));

      if (result_params.hasRecordBatch()) {
        ARROW_THROW_NOT_OK(arrow::ipc::SerializeRecordBatch(
            *record_batch, arrow::ipc::IpcWriteOptions::Defaults(), &stream));
      }

      return {std::vector<char>(0),
              0,
              std::vector<char>(0),
              serialized_records->size(),
              std::string{""},
              std::move(record_handle_data)};
    };

    const auto getShmResult = [&]() -> ArrowResult {
      auto timer = DEBUG_TIMER("serialize batch to shared memory");
      std::shared_ptr<arrow::Buffer> serialized_records;
      std::vector<char> schema_handle_buffer;
      std::vector<char> record_handle_buffer(sizeof(key_t), 0);
      key_t records_shm_key = IPC_PRIVATE;
      const int64_t total_size = result_params.totalSize();

      std::tie(records_shm_key, serialized_records) = get_shm_buffer(total_size);

      memcpy(serialized_records->mutable_data(),
             result_params.serialized_schema->data(),
             (size_t)result_params.schemaSize());

      if (result_params.hasDict()) {
        memcpy(serialized_records->mutable_data() + result_params.schemaSize(),
               result_params.serialized_dict->data(),
               (size_t)result_params.dictSize());
      }

      arrow::io::FixedSizeBufferWriter stream(SliceMutableBuffer(
          serialized_records, result_params.schemaSize() + result_params.dictSize()));

      if (result_params.hasRecordBatch()) {
        ARROW_THROW_NOT_OK(arrow::ipc::SerializeRecordBatch(
            *record_batch, arrow::ipc::IpcWriteOptions::Defaults(), &stream));
      }

      memcpy(&record_handle_buffer[0],
             reinterpret_cast<const unsigned char*>(&records_shm_key),
             sizeof(key_t));

      return {schema_handle_buffer,
              0,
              record_handle_buffer,
              serialized_records->size(),
              std::string{""}};
    };

    arrow::ipc::DictionaryFieldMapper mapper(*record_batch->schema());
    auto options = arrow::ipc::IpcWriteOptions::Defaults();
    auto dict_stream = arrow::io::BufferOutputStream::Create(1024).ValueOrDie();

    // If our record batch is going to be empty, we omit it entirely,
    // only serializing the schema.
    if (!record_batch->num_rows()) {
      ARROW_ASSIGN_OR_THROW(result_params.serialized_schema,
                            arrow::ipc::SerializeSchema(*record_batch->schema(),
                                                        arrow::default_memory_pool()));

      switch (transport_method_) {
        case ArrowTransport::WIRE:
          return getWireResult();
        case ArrowTransport::SHARED_MEMORY:
          return getShmResult();
        default:
          UNREACHABLE();
      }
    }

    ARROW_ASSIGN_OR_THROW(auto dictionaries, CollectDictionaries(*record_batch, mapper));

    ARROW_LOG("CPU") << "found " << dictionaries.size() << " dictionaries";

    for (auto& pair : dictionaries) {
      arrow::ipc::IpcPayload payload;
      int64_t dictionary_id = pair.first;
      const auto& dictionary = pair.second;

      ARROW_THROW_NOT_OK(
          GetDictionaryPayload(dictionary_id, dictionary, options, &payload));
      int32_t metadata_length = 0;
      ARROW_THROW_NOT_OK(
          WriteIpcPayload(payload, options, dict_stream.get(), &metadata_length));
    }
    result_params.serialized_dict = dict_stream->Finish().ValueOrDie();

    ARROW_ASSIGN_OR_THROW(result_params.serialized_schema,
                          arrow::ipc::SerializeSchema(*record_batch->schema(),
                                                      arrow::default_memory_pool()));

    ARROW_THROW_NOT_OK(
        arrow::ipc::GetRecordBatchSize(*record_batch, &result_params.records_size));

    switch (transport_method_) {
      case ArrowTransport::WIRE:
        return getWireResult();
      case ArrowTransport::SHARED_MEMORY:
        return getShmResult();
      default:
        UNREACHABLE();
    }
  }
#ifdef HAVE_CUDA
  CHECK(device_type_ == ExecutorDeviceType::GPU);

  // Copy the schema to the schema handle
  auto out_stream_result = arrow::io::BufferOutputStream::Create(1024);
  ARROW_THROW_NOT_OK(out_stream_result.status());
  auto out_stream = std::move(out_stream_result).ValueOrDie();

  arrow::ipc::DictionaryFieldMapper mapper(*record_batch->schema());
  arrow::ipc::DictionaryMemo current_memo;
  arrow::ipc::DictionaryMemo serialized_memo;

  arrow::ipc::IpcPayload schema_payload;
  ARROW_THROW_NOT_OK(arrow::ipc::GetSchemaPayload(*record_batch->schema(),
                                                  arrow::ipc::IpcWriteOptions::Defaults(),
                                                  mapper,
                                                  &schema_payload));
  int32_t schema_payload_length = 0;
  ARROW_THROW_NOT_OK(arrow::ipc::WriteIpcPayload(schema_payload,
                                                 arrow::ipc::IpcWriteOptions::Defaults(),
                                                 out_stream.get(),
                                                 &schema_payload_length));
  ARROW_ASSIGN_OR_THROW(auto dictionaries, CollectDictionaries(*record_batch, mapper));
  ARROW_LOG("GPU") << "Dictionary "
                   << "found dicts: " << dictionaries.size();

  ARROW_THROW_NOT_OK(
      arrow::ipc::internal::CollectDictionaries(*record_batch, &current_memo));

  // now try a dictionary
  std::shared_ptr<arrow::Schema> dummy_schema;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dict_batches;

  for (const auto& pair : dictionaries) {
    arrow::ipc::IpcPayload payload;
    const auto& dict_id = pair.first;
    CHECK_GE(dict_id, 0);
    ARROW_LOG("GPU") << "Dictionary "
                     << "dict_id: " << dict_id;
    const auto& dict = pair.second;
    CHECK(dict);

    if (!dummy_schema) {
      auto dummy_field = std::make_shared<arrow::Field>("", dict->type());
      dummy_schema = std::make_shared<arrow::Schema>(
          std::vector<std::shared_ptr<arrow::Field>>{dummy_field});
    }
    dict_batches.emplace_back(
        arrow::RecordBatch::Make(dummy_schema, dict->length(), {dict}));
  }

  if (!dict_batches.empty()) {
    ARROW_THROW_NOT_OK(arrow::ipc::WriteRecordBatchStream(
        dict_batches, arrow::ipc::IpcWriteOptions::Defaults(), out_stream.get()));
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
  ARROW_ASSIGN_OR_THROW(manager, arrow::cuda::CudaDeviceManager::Instance());
  std::shared_ptr<arrow::cuda::CudaContext> context;
  ARROW_ASSIGN_OR_THROW(context, manager->GetContext(device_id_));

  std::shared_ptr<arrow::cuda::CudaBuffer> device_serialized;
  ARROW_ASSIGN_OR_THROW(device_serialized,
                        SerializeRecordBatch(*record_batch, context.get()));

  std::shared_ptr<arrow::cuda::CudaIpcMemHandle> cuda_handle;
  ARROW_ASSIGN_OR_THROW(cuda_handle, device_serialized->ExportForIpc());

  std::shared_ptr<arrow::Buffer> serialized_cuda_handle;
  ARROW_ASSIGN_OR_THROW(serialized_cuda_handle,
                        cuda_handle->Serialize(arrow::default_memory_pool()));

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
    arrow::ipc::DictionaryFieldMapper* mapper) const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<arrow::RecordBatch> arrow_copy = convertToArrow();
  std::shared_ptr<arrow::Buffer> serialized_records, serialized_schema;

  ARROW_ASSIGN_OR_THROW(
      serialized_schema,
      arrow::ipc::SerializeSchema(*arrow_copy->schema(), arrow::default_memory_pool()));

  if (arrow_copy->num_rows()) {
    auto timer = DEBUG_TIMER("serialize records");
    ARROW_THROW_NOT_OK(arrow_copy->Validate());
    ARROW_ASSIGN_OR_THROW(serialized_records,
                          arrow::ipc::SerializeRecordBatch(
                              *arrow_copy, arrow::ipc::IpcWriteOptions::Defaults()));
  } else {
    ARROW_ASSIGN_OR_THROW(serialized_records, arrow::AllocateBuffer(0));
  }
  return {serialized_schema, serialized_records};
}

std::shared_ptr<arrow::RecordBatch> ArrowResultSetConverter::convertToArrow() const {
  auto timer = DEBUG_TIMER(__func__);
  const auto col_count = results_->colCount();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  CHECK(col_names_.empty() || col_names_.size() == col_count);
  for (size_t i = 0; i < col_count; ++i) {
    const auto ti = results_->getColType(i);
    fields.push_back(makeField(col_names_.empty() ? "" : col_names_[i], ti));
  }
#if ARROW_CONVERTER_DEBUG
  VLOG(1) << "Arrow fields: ";
  for (const auto& f : fields) {
    VLOG(1) << "\t" << f->ToString(true);
  }
#endif
  return getArrowBatch(arrow::schema(fields));
}

std::shared_ptr<arrow::RecordBatch> ArrowResultSetConverter::getArrowBatch(
    const std::shared_ptr<arrow::Schema>& schema) const {
  std::vector<std::shared_ptr<arrow::Array>> result_columns;

  // First, check if the result set is empty.
  // If so, we return an arrow result set that only
  // contains the schema (no record batch will be serialized).
  if (results_->isEmpty()) {
    for (auto& field : schema->fields()) {
      result_columns.push_back(arrow::MakeArrayOfNull(field->type(), 0).ValueOrDie());
    }
    return ARROW_RECORDBATCH_MAKE(schema, 0, result_columns);
  }

  const size_t entry_count = top_n_ < 0
                                 ? results_->entryCount()
                                 : std::min(size_t(top_n_), results_->entryCount());

  const auto col_count = results_->colCount();
  size_t row_count = 0;

  result_columns.resize(col_count);
  std::vector<ColumnBuilder> builders(col_count);

  // Create array builders
  for (size_t i = 0; i < col_count; ++i) {
    initializeColumnBuilder(builders[i], results_->getColType(i), i, schema->field(i));
  }

  // TODO(miyu): speed up for columnar buffers
  auto fetch = [&](std::vector<std::shared_ptr<ValueArray>>& value_seg,
                   std::vector<std::shared_ptr<std::vector<bool>>>& null_bitmap_seg,
                   const std::vector<bool>& non_lazy_cols,
                   const size_t start_entry,
                   const size_t end_entry) -> size_t {
    CHECK_EQ(value_seg.size(), col_count);
    CHECK_EQ(null_bitmap_seg.size(), col_count);
    const auto local_entry_count = end_entry - start_entry;
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

        if (auto scalar_value = boost::get<ScalarTargetValue>(&row[j])) {
          // TODO(miyu): support more types other than scalar.
          CHECK(scalar_value);
          const auto& column = builders[j];
          switch (column.physical_type) {
            case kBOOLEAN:
              create_or_append_value<bool, int64_t>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<int64_t>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kTINYINT:
              create_or_append_value<int8_t, int64_t>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<int64_t>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kSMALLINT:
              create_or_append_value<int16_t, int64_t>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<int64_t>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kINT:
              create_or_append_value<int32_t, int64_t>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<int64_t>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kBIGINT:
              create_or_append_value<int64_t, int64_t>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<int64_t>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kDECIMAL:
              create_or_append_value<int64_t, int64_t>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<int64_t>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kFLOAT:
              create_or_append_value<float, float>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<float>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kDOUBLE:
              create_or_append_value<double, double>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<double>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kTIME:
              create_or_append_value<int32_t, int64_t>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<int64_t>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kDATE:
              device_type_ == ExecutorDeviceType::GPU
                  ? create_or_append_value<int64_t, int64_t>(
                        *scalar_value, value_seg[j], local_entry_count)
                  : create_or_append_value<int32_t, int64_t>(
                        *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<int64_t>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kTIMESTAMP:
              create_or_append_value<int64_t, int64_t>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<int64_t>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kTEXT:
              create_or_append_value<std::string, NullableString>(
                  *scalar_value, value_seg[j], local_entry_count);
              create_or_append_validity<NullableString>(
                  *scalar_value, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            default:
              // TODO(miyu): support more scalar types.
              throw std::runtime_error(column.col_type.get_type_name() +
                                       " is not supported in Arrow result sets.");
          }
        } else if (auto array = boost::get<ArrayTargetValue>(&row[j])) {
          // array := Boost::optional<std::vector<ScalarTargetValue>>
          const auto& column = builders[j];
          switch (column.col_type.get_subtype()) {
            case kBOOLEAN:
              create_or_append_value<int8_t, int64_t>(
                  *array, value_seg[j], local_entry_count);
              create_or_append_validity(
                  *array, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kTINYINT:
              create_or_append_value<int8_t, int64_t>(
                  *array, value_seg[j], local_entry_count);
              create_or_append_validity(
                  *array, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kSMALLINT:
              create_or_append_value<int16_t, int64_t>(
                  *array, value_seg[j], local_entry_count);
              create_or_append_validity(
                  *array, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kINT:
              create_or_append_value<int32_t, int64_t>(
                  *array, value_seg[j], local_entry_count);
              create_or_append_validity(
                  *array, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kBIGINT:
              create_or_append_value<int64_t, int64_t>(
                  *array, value_seg[j], local_entry_count);
              create_or_append_validity(
                  *array, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kFLOAT:
              create_or_append_value<float, float>(
                  *array, value_seg[j], local_entry_count);
              create_or_append_validity(
                  *array, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kDOUBLE:
              create_or_append_value<double, double>(
                  *array, value_seg[j], local_entry_count);
              create_or_append_validity(
                  *array, column.col_type, null_bitmap_seg[j], local_entry_count);
              break;
            case kTEXT:
              if (column.col_type.is_dict_encoded_type()) {
                create_or_append_value<int64_t, int64_t>(
                    *array, value_seg[j], local_entry_count);
                create_or_append_validity(
                    *array, column.col_type, null_bitmap_seg[j], local_entry_count);
                break;
              }
            default:
              throw std::runtime_error(column.col_type.get_type_name() +
                                       " is not supported in Arrow result sets.");
          }
        }
      }
    }
    return seg_row_count;
  };

  auto convert_columns = [&](std::vector<std::shared_ptr<arrow::Array>>& result,
                             const std::vector<bool>& non_lazy_cols,
                             const size_t start_col,
                             const size_t end_col) {
    for (size_t col = start_col; col < end_col; ++col) {
      if (!non_lazy_cols.empty() && !non_lazy_cols[col]) {
        continue;
      }

      const auto& column = builders[col];
      switch (column.physical_type) {
        case kTINYINT:
          convert_column<int8_t>(results_, col, entry_count, result[col]);
          break;
        case kSMALLINT:
          convert_column<int16_t>(results_, col, entry_count, result[col]);
          break;
        case kINT:
          convert_column<int32_t>(results_, col, entry_count, result[col]);
          break;
        case kBIGINT:
          convert_column<int64_t>(results_, col, entry_count, result[col]);
          break;
        case kFLOAT:
          convert_column<float>(results_, col, entry_count, result[col]);
          break;
        case kDOUBLE:
          convert_column<double>(results_, col, entry_count, result[col]);
          break;
        default:
          throw std::runtime_error(column.col_type.get_type_name() +
                                   " is not supported in Arrow column converter.");
      }
    }
  };

  std::vector<std::shared_ptr<ValueArray>> column_values(col_count, nullptr);
  std::vector<std::shared_ptr<std::vector<bool>>> null_bitmaps(col_count, nullptr);
  const bool multithreaded = entry_count > 10000 && !results_->isTruncated();
  // Don't believe we ever output directly from a table function, but this
  // might be possible with a future query plan optimization
  bool use_columnar_converter = results_->isDirectColumnarConversionPossible() &&
                                (results_->getQueryMemDesc().getQueryDescriptionType() ==
                                     QueryDescriptionType::Projection ||
                                 results_->getQueryMemDesc().getQueryDescriptionType() ==
                                     QueryDescriptionType::TableFunction) &&
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
      // Treat them as lazy.
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
      if (builders[i].field->type()->id() == arrow::Type::DICTIONARY) {
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
        auto timer = DEBUG_TIMER("append rows to arrow single thread");
        for (int i = 0; i < schema->num_fields(); ++i) {
          if (!non_lazy_cols.empty() && non_lazy_cols[i]) {
            continue;
          }

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

namespace {

std::shared_ptr<arrow::DataType> get_arrow_type(const SQLTypeInfo& sql_type,
                                                const ExecutorDeviceType device_type) {
  switch (get_physical_type(sql_type)) {
    case kBOOLEAN:
      return arrow::boolean();
    case kTINYINT:
      return arrow::int8();
    case kSMALLINT:
      return arrow::int16();
    case kINT:
      return arrow::int32();
    case kBIGINT:
      return arrow::int64();
    case kFLOAT:
      return arrow::float32();
    case kDOUBLE:
      return arrow::float64();
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      if (sql_type.is_dict_encoded_string()) {
        auto value_type = std::make_shared<arrow::StringType>();
        return arrow::dictionary(arrow::int32(), value_type, false);
      }
      return arrow::utf8();
    case kDECIMAL:
    case kNUMERIC:
      return arrow::decimal(sql_type.get_precision(), sql_type.get_scale());
    case kTIME:
      return time32(arrow::TimeUnit::SECOND);
    case kDATE: {
      // TODO(wamsi) : Remove date64() once date32() support is added in cuDF. date32()
      // Currently support for date32() is missing in cuDF.Hence, if client requests for
      // date on GPU, return date64() for the time being, till support is added.
      if (device_type == ExecutorDeviceType::GPU) {
        return arrow::date64();
      } else {
        return arrow::date32();
      }
    }
    case kTIMESTAMP:
      switch (sql_type.get_precision()) {
        case 0:
          return timestamp(arrow::TimeUnit::SECOND);
        case 3:
          return timestamp(arrow::TimeUnit::MILLI);
        case 6:
          return timestamp(arrow::TimeUnit::MICRO);
        case 9:
          return timestamp(arrow::TimeUnit::NANO);
        default:
          throw std::runtime_error(
              "Unsupported timestamp precision for Arrow result sets: " +
              std::to_string(sql_type.get_precision()));
      }
    case kARRAY:
      switch (sql_type.get_subtype()) {
        case kBOOLEAN:
          return arrow::list(arrow::boolean());
        case kTINYINT:
          return arrow::list(arrow::int8());
        case kSMALLINT:
          return arrow::list(arrow::int16());
        case kINT:
          return arrow::list(arrow::int32());
        case kBIGINT:
          return arrow::list(arrow::int64());
        case kFLOAT:
          return arrow::list(arrow::float32());
        case kDOUBLE:
          return arrow::list(arrow::float64());
        case kTEXT:
          if (sql_type.is_dict_encoded_type()) {
            auto value_type = std::make_shared<arrow::StringType>();
            return arrow::list(arrow::dictionary(arrow::int32(), value_type, false));
          }
        default:
          throw std::runtime_error("Unsupported array type for Arrow result sets: " +
                                   sql_type.get_type_name());
      }
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
#ifndef _MSC_VER
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
#endif
}

void ArrowResultSetConverter::initializeColumnBuilder(
    ColumnBuilder& column_builder,
    const SQLTypeInfo& col_type,
    const size_t results_col_slot_idx,
    const std::shared_ptr<arrow::Field>& field) const {
  column_builder.field = field;
  column_builder.col_type = col_type;
  column_builder.physical_type = col_type.is_dict_encoded_string()
                                     ? get_dict_index_type(col_type)
                                     : get_physical_type(col_type);

  auto value_type = field->type();
  if (col_type.is_dict_encoded_type()) {
    auto timer = DEBUG_TIMER("Translate string dictionary to Arrow dictionary");
    if (!col_type.is_array()) {
      column_builder.builder.reset(new arrow::StringDictionary32Builder());
    }
    // add values to the builder
    const auto& dict_key = col_type.getStringDictKey();

    // ResultSet::rowCount(), unlike ResultSet::entryCount(), will return
    // the actual number of rows in the result set, taking into account
    // things like any limit and offset set
    const size_t result_set_rows = results_->rowCount();
    // result_set_rows guaranteed > 0 by parent
    CHECK_GT(result_set_rows, 0UL);

    const auto sdp = results_->getStringDictionaryProxy(dict_key);
    const size_t dictionary_proxy_entries = sdp->entryCount();
    const double dictionary_to_result_size_ratio =
        static_cast<double>(dictionary_proxy_entries) / result_set_rows;

    // We are conservative with when we do a bulk dictionary fetch,
    // even though it is generally more efficient than dictionary unique value "plucking",
    // for the following reasons:
    // 1) The number of actual distinct dictionary values can be much lower than the
    // number of result rows, but without getting the expression range (and that would
    // only work in some cases), we don't know by how much
    // 2) Regardless of the effect of #1, the size of the dictionary generated via
    // the "pluck" method will always be at worst equal in size, and very likely
    // significantly smaller, than the dictionary created by the bulk dictionary
    // fetch method, and smaller Arrow dictionaries are always a win when it comes to
    // sending the Arrow results over the wire, and for lowering the processing load
    // for clients (which often is a web browser with a lot less compute and memory
    // resources than our server.)

    const bool do_dictionary_bulk_fetch =
        result_set_rows > min_result_size_for_bulk_dictionary_fetch_ &&
        dictionary_to_result_size_ratio <=
            max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch_;

    arrow::StringBuilder str_array_builder;

    if (do_dictionary_bulk_fetch) {
      VLOG(1) << "Arrow dictionary creation: bulk copying all dictionary "
              << " entries for column at offset " << results_col_slot_idx << ". "
              << "Column has " << dictionary_proxy_entries << " string entries"
              << " for a result set with " << result_set_rows << " rows.";
      column_builder.string_remap_mode =
          ArrowStringRemapMode::ONLY_TRANSIENT_STRINGS_REMAPPED;
      const auto str_list = results_->getStringDictionaryPayloadCopy(dict_key);
      ARROW_THROW_NOT_OK(str_array_builder.AppendValues(str_list));

      // When we fetch the bulk dictionary, we need to also fetch
      // the transient entries only contained in the proxy.
      // These values are always negative (starting at -2), and so need
      // to be remapped to point to the corresponding entries in the Arrow
      // dictionary (they are placed at the end after the materialized
      // string entries from StringDictionary)

      int32_t crt_transient_id = static_cast<int32_t>(str_list.size());
      auto const& transient_vecmap = sdp->getTransientVector();
      for (unsigned index = 0; index < transient_vecmap.size(); ++index) {
        ARROW_THROW_NOT_OK(str_array_builder.Append(*transient_vecmap[index]));
        auto const old_id = StringDictionaryProxy::transientIndexToId(index);
        CHECK(column_builder.string_remapping
                  .insert(std::make_pair(old_id, crt_transient_id++))
                  .second);
      }
    } else {
      // Pluck unique dictionary values from ResultSet column
      VLOG(1) << "Arrow dictionary creation: serializing unique result set dictionary "
              << " entries for column at offset " << results_col_slot_idx << ". "
              << "Column has " << dictionary_proxy_entries << " string entries"
              << " for a result set with " << result_set_rows << " rows.";
      column_builder.string_remap_mode = ArrowStringRemapMode::ALL_STRINGS_REMAPPED;

      // ResultSet::getUniqueStringsForDictEncodedTargetCol returns a pair of two vectors,
      // the first of int32_t values containing the unique string ids found for
      // results_col_slot_idx in the result set, the second containing the associated
      // unique strings. Note that the unique string for a unique string id are both
      // placed at the same offset in their respective vectors

      auto unique_ids_and_strings =
          results_->getUniqueStringsForDictEncodedTargetCol(results_col_slot_idx);
      const auto& unique_ids = unique_ids_and_strings.first;
      const auto& unique_strings = unique_ids_and_strings.second;
      ARROW_THROW_NOT_OK(str_array_builder.AppendValues(unique_strings));
      const int32_t num_unique_strings = unique_strings.size();
      CHECK_EQ(num_unique_strings, unique_ids.size());
      // We need to remap ALL string id values given the Arrow dictionary
      // will have "holes", i.e. it is a sparse representation of the underlying
      // StringDictionary
      for (int32_t unique_string_idx = 0; unique_string_idx < num_unique_strings;
           ++unique_string_idx) {
        CHECK(
            column_builder.string_remapping
                .insert(std::make_pair(unique_ids[unique_string_idx], unique_string_idx))
                .second);
      }
      // Note we don't need to get transients from proxy as they are already handled in
      // ResultSet::getUniqueStringsForDictEncodedTargetCol
    }

    std::shared_ptr<arrow::StringArray> string_array;
    ARROW_THROW_NOT_OK(str_array_builder.Finish(&string_array));

    if (col_type.is_array()) {
      column_builder.string_array = std::move(string_array);
      ARROW_THROW_NOT_OK(arrow::MakeBuilder(
          arrow::default_memory_pool(), value_type, &column_builder.builder));
    } else {
      auto dict_builder =
          dynamic_cast<arrow::StringDictionary32Builder*>(column_builder.builder.get());
      CHECK(dict_builder);

      ARROW_THROW_NOT_OK(dict_builder->InsertMemoValues(*string_array));
    }
  } else {
    ARROW_THROW_NOT_OK(arrow::MakeBuilder(
        arrow::default_memory_pool(), value_type, &column_builder.builder));
  }
}

std::shared_ptr<arrow::Array> ArrowResultSetConverter::finishColumnBuilder(
    ColumnBuilder& column_builder) const {
  std::shared_ptr<arrow::Array> values;
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
      return std::is_same<BUILDER_TYPE, arrow::Date32Builder>::value
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
void appendToColumnBuilder<arrow::Decimal128Builder, int64_t>(
    ArrowResultSetConverter::ColumnBuilder& column_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) {
  std::vector<int64_t> vals = boost::get<std::vector<int64_t>>(values);
  auto typed_builder =
      dynamic_cast<arrow::Decimal128Builder*>(column_builder.builder.get());
  CHECK(typed_builder);
  CHECK_EQ(is_valid->size(), vals.size());
  if (column_builder.field->nullable()) {
    CHECK(is_valid.get());
    for (size_t i = 0; i < vals.size(); i++) {
      const auto v = vals[i];
      const auto valid = (*is_valid)[i];
      if (valid) {
        ARROW_THROW_NOT_OK(typed_builder->Append(v));
      } else {
        ARROW_THROW_NOT_OK(typed_builder->AppendNull());
      }
    }
  } else {
    for (const auto& v : vals) {
      ARROW_THROW_NOT_OK(typed_builder->Append(v));
    }
  }
}

template <>
void appendToColumnBuilder<arrow::StringBuilder, std::string>(
    ArrowResultSetConverter::ColumnBuilder& column_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) {
  std::vector<std::string> vals = boost::get<std::vector<std::string>>(values);
  auto typed_builder = dynamic_cast<arrow::StringBuilder*>(column_builder.builder.get());
  CHECK(typed_builder);
  CHECK_EQ(is_valid->size(), vals.size());

  if (column_builder.field->nullable()) {
    CHECK(is_valid.get());

    // TODO: Generate this instead of the boolean bitmap
    std::vector<uint8_t> transformed_bitmap;
    transformed_bitmap.reserve(is_valid->size());
    std::for_each(
        is_valid->begin(), is_valid->end(), [&transformed_bitmap](const bool is_valid) {
          transformed_bitmap.push_back(is_valid ? 1 : 0);
        });
    ARROW_THROW_NOT_OK(typed_builder->AppendValues(vals, transformed_bitmap.data()));
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
  // remap negative values if ArrowStringRemapMode == ONLY_TRANSIENT_STRINGS_REMAPPED or
  // everything if ALL_STRINGS_REMAPPED
  CHECK(column_builder.string_remap_mode != ArrowStringRemapMode::INVALID);
  for (size_t i = 0; i < vals.size(); i++) {
    auto& val = vals[i];
    if ((column_builder.string_remap_mode == ArrowStringRemapMode::ALL_STRINGS_REMAPPED ||
         val < 0) &&
        (*is_valid)[i]) {
      vals[i] = column_builder.string_remapping.at(val);
    }
  }

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

template <typename BUILDER_TYPE, typename VALUE_TYPE>
void appendToListColumnBuilder(ArrowResultSetConverter::ColumnBuilder& column_builder,
                               const ValueArray& values,
                               const std::shared_ptr<std::vector<bool>>& is_valid) {
  Vec2<VALUE_TYPE> vals = boost::get<Vec2<VALUE_TYPE>>(values);
  auto list_builder = dynamic_cast<arrow::ListBuilder*>(column_builder.builder.get());
  CHECK(list_builder);

  auto value_builder = static_cast<BUILDER_TYPE*>(list_builder->value_builder());

  if (column_builder.field->nullable()) {
    for (size_t i = 0; i < vals.size(); i++) {
      if ((*is_valid)[i]) {
        const auto& val = vals[i];
        std::vector<uint8_t> bitmap(val.size());
        std::transform(val.begin(), val.end(), bitmap.begin(), [](VALUE_TYPE pvalue) {
          return static_cast<VALUE_TYPE>(pvalue) != null_type<VALUE_TYPE>::value;
        });
        ARROW_THROW_NOT_OK(list_builder->Append());
        if constexpr (std::is_same_v<BUILDER_TYPE, arrow::BooleanBuilder>) {
          std::vector<uint8_t> bval(val.size());
          std::copy(val.begin(), val.end(), bval.begin());
          ARROW_THROW_NOT_OK(
              value_builder->AppendValues(bval.data(), bval.size(), bitmap.data()));
        } else {
          ARROW_THROW_NOT_OK(
              value_builder->AppendValues(val.data(), val.size(), bitmap.data()));
        }
      } else {
        ARROW_THROW_NOT_OK(list_builder->AppendNull());
      }
    }
  } else {
    for (size_t i = 0; i < vals.size(); i++) {
      if ((*is_valid)[i]) {
        const auto& val = vals[i];
        ARROW_THROW_NOT_OK(list_builder->Append());
        if constexpr (std::is_same_v<BUILDER_TYPE, arrow::BooleanBuilder>) {
          std::vector<uint8_t> bval(val.size());
          std::copy(val.begin(), val.end(), bval.begin());
          ARROW_THROW_NOT_OK(value_builder->AppendValues(bval.data(), bval.size()));
        } else {
          ARROW_THROW_NOT_OK(value_builder->AppendValues(val.data(), val.size()));
        }
      } else {
        ARROW_THROW_NOT_OK(list_builder->AppendNull());
      }
    }
  }
}

template <>
void appendToListColumnBuilder<arrow::StringDictionaryBuilder, int64_t>(
    ArrowResultSetConverter::ColumnBuilder& column_builder,
    const ValueArray& values,
    const std::shared_ptr<std::vector<bool>>& is_valid) {
  Vec2<int64_t> vec2d = boost::get<Vec2<int64_t>>(values);

  auto* list_builder = dynamic_cast<arrow::ListBuilder*>(column_builder.builder.get());
  CHECK(list_builder);

  // todo: fix value_builder being a StringDictionaryBuilder and not
  // StringDictionary32Builder
  auto* value_builder =
      dynamic_cast<arrow::StringDictionaryBuilder*>(list_builder->value_builder());
  CHECK(value_builder);

  if (column_builder.field->nullable()) {
    for (size_t i = 0; i < vec2d.size(); i++) {
      if ((*is_valid)[i]) {
        auto& vec1d = vec2d[i];
        std::vector<uint8_t> bitmap(vec1d.size());
        std::transform(vec1d.begin(), vec1d.end(), bitmap.begin(), [](int64_t pvalue) {
          return pvalue != null_type<int32_t>::value;
        });
        ARROW_THROW_NOT_OK(list_builder->Append());
        ARROW_THROW_NOT_OK(value_builder->InsertMemoValues(*column_builder.string_array));
        remap_string_values(column_builder, bitmap, vec1d);
        ARROW_THROW_NOT_OK(value_builder->AppendIndices(
            vec1d.data(), static_cast<int64_t>(vec1d.size()), bitmap.data()));
      } else {
        ARROW_THROW_NOT_OK(list_builder->AppendNull());
      }
    }
  } else {
    for (size_t i = 0; i < vec2d.size(); i++) {
      if ((*is_valid)[i]) {
        auto& vec1d = vec2d[i];
        ARROW_THROW_NOT_OK(list_builder->Append());
        remap_string_values(column_builder, {}, vec1d);
        ARROW_THROW_NOT_OK(value_builder->AppendIndices(vec1d.data(), vec1d.size()));
      } else {
        ARROW_THROW_NOT_OK(list_builder->AppendNull());
      }
    }
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
    appendToColumnBuilder<arrow::StringDictionary32Builder, int32_t>(
        column_builder, values, is_valid);
    return;
  }
  switch (column_builder.physical_type) {
    case kBOOLEAN:
      appendToColumnBuilder<arrow::BooleanBuilder, bool>(
          column_builder, values, is_valid);
      break;
    case kTINYINT:
      appendToColumnBuilder<arrow::Int8Builder, int8_t>(column_builder, values, is_valid);
      break;
    case kSMALLINT:
      appendToColumnBuilder<arrow::Int16Builder, int16_t>(
          column_builder, values, is_valid);
      break;
    case kINT:
      appendToColumnBuilder<arrow::Int32Builder, int32_t>(
          column_builder, values, is_valid);
      break;
    case kBIGINT:
      appendToColumnBuilder<arrow::Int64Builder, int64_t>(
          column_builder, values, is_valid);
      break;
    case kDECIMAL:
      appendToColumnBuilder<arrow::Decimal128Builder, int64_t>(
          column_builder, values, is_valid);
      break;
    case kFLOAT:
      appendToColumnBuilder<arrow::FloatBuilder, float>(column_builder, values, is_valid);
      break;
    case kDOUBLE:
      appendToColumnBuilder<arrow::DoubleBuilder, double>(
          column_builder, values, is_valid);
      break;
    case kTIME:
      appendToColumnBuilder<arrow::Time32Builder, int32_t>(
          column_builder, values, is_valid);
      break;
    case kTIMESTAMP:
      appendToColumnBuilder<arrow::TimestampBuilder, int64_t>(
          column_builder, values, is_valid);
      break;
    case kDATE:
      device_type_ == ExecutorDeviceType::GPU
          ? appendToColumnBuilder<arrow::Date64Builder, int64_t>(
                column_builder, values, is_valid)
          : appendToColumnBuilder<arrow::Date32Builder, int32_t>(
                column_builder, values, is_valid);
      break;
    case kARRAY:
      if (column_builder.col_type.get_subtype() == kBOOLEAN) {
        appendToListColumnBuilder<arrow::BooleanBuilder, int8_t>(
            column_builder, values, is_valid);
        break;
      } else if (column_builder.col_type.get_subtype() == kTINYINT) {
        appendToListColumnBuilder<arrow::Int8Builder, int8_t>(
            column_builder, values, is_valid);
        break;
      } else if (column_builder.col_type.get_subtype() == kSMALLINT) {
        appendToListColumnBuilder<arrow::Int16Builder, int16_t>(
            column_builder, values, is_valid);
        break;
      } else if (column_builder.col_type.get_subtype() == kINT) {
        appendToListColumnBuilder<arrow::Int32Builder, int32_t>(
            column_builder, values, is_valid);
        break;
      } else if (column_builder.col_type.get_subtype() == kBIGINT) {
        appendToListColumnBuilder<arrow::Int64Builder, int64_t>(
            column_builder, values, is_valid);
        break;
      } else if (column_builder.col_type.get_subtype() == kFLOAT) {
        appendToListColumnBuilder<arrow::FloatBuilder, float>(
            column_builder, values, is_valid);
        break;
      } else if (column_builder.col_type.get_subtype() == kDOUBLE) {
        appendToListColumnBuilder<arrow::DoubleBuilder, double>(
            column_builder, values, is_valid);
        break;
      } else if (column_builder.col_type.is_dict_encoded_type()) {
        appendToListColumnBuilder<arrow::StringDictionaryBuilder, int64_t>(
            column_builder, values, is_valid);
        break;
      } else {
        throw std::runtime_error(column_builder.col_type.get_type_name() +
                                 " is not supported in Arrow result sets.");
      }
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      appendToColumnBuilder<arrow::StringBuilder, std::string>(
          column_builder, values, is_valid);
      break;
    default:
      // TODO(miyu): support more scalar types.
      throw std::runtime_error(column_builder.col_type.get_type_name() +
                               " is not supported in Arrow result sets.");
  }
}
