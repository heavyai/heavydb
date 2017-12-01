/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "Execute.h"
#include "ResultSet.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <string>

#include "arrow/api.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/api.h"

#include "ArrowUtil.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <future>

typedef boost::variant<std::vector<bool>,
                       std::vector<int8_t>,
                       std::vector<int16_t>,
                       std::vector<int32_t>,
                       std::vector<int64_t>,
                       std::vector<float>,
                       std::vector<double>,
                       std::vector<std::string>>
    ValueArray;

namespace {

bool is_dict_enc_str(const SQLTypeInfo& ti) {
  return ti.is_string() && ti.get_compression() == kENCODING_DICT;
}

SQLTypes get_dict_index_type(const SQLTypeInfo& ti) {
  CHECK(is_dict_enc_str(ti));
  switch (ti.get_size()) {
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

SQLTypeInfo get_dict_index_type_info(const SQLTypeInfo& ti) {
  CHECK(is_dict_enc_str(ti));
  switch (ti.get_size()) {
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

SQLTypes get_physical_type(const SQLTypeInfo& ti) {
  auto logical_type = ti.get_type();
  if (IS_INTEGER(logical_type)) {
    switch (ti.get_size()) {
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
  if (is_dict_enc_str(col_type)) {
    is_valid = inline_int_null_val(col_type) != static_cast<int32_t>(*pvalue);
  } else if (col_type.is_integer()) {
    is_valid = inline_int_null_val(col_type) != static_cast<int64_t>(*pvalue);
  } else {
    CHECK(col_type.is_fp());
    is_valid = inline_fp_null_val(col_type) != static_cast<double>(*pvalue);
  }
  if (!null_bitmap) {
    null_bitmap = std::make_shared<std::vector<bool>>();
    null_bitmap->reserve(max_size);
  }
  CHECK(null_bitmap);
  null_bitmap->push_back(is_valid);
}

}  // namespace

namespace arrow {

static TypePtr get_arrow_type(const SQLTypeInfo& mapd_type, const std::shared_ptr<Array>& dict_values) {
  switch (get_physical_type(mapd_type)) {
    case kBOOLEAN:
      return boolean();
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
    case kTEXT:
      if (is_dict_enc_str(mapd_type)) {
        CHECK(dict_values);
        const auto index_type = get_arrow_type(get_dict_index_type_info(mapd_type), nullptr);
        return dictionary(index_type, dict_values);
      }
    case kDECIMAL:
    case kNUMERIC:
    case kCHAR:
    case kVARCHAR:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kARRAY:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    default:
      throw std::runtime_error(mapd_type.get_type_name() + " is not supported in temporary table.");
  }
  return nullptr;
}

struct ColumnBuilder {
  std::shared_ptr<Field> field;
  std::unique_ptr<arrow::ArrayBuilder> builder;
  SQLTypeInfo col_type;
  SQLTypes physical_type;

  void init(const SQLTypeInfo& col_type, const std::shared_ptr<Field>& field) {
    this->field = field;

    this->col_type = col_type;
    this->physical_type = is_dict_enc_str(col_type) ? get_dict_index_type(col_type) : get_physical_type(col_type);

    auto value_type = field->type();
    if (value_type->id() == Type::DICTIONARY) {
      value_type = static_cast<const DictionaryType&>(*value_type).index_type();
    }
    ARROW_THROW_NOT_OK(arrow::MakeBuilder(default_memory_pool(), value_type, &this->builder));
  }

  void reserve(size_t row_count) { ARROW_THROW_NOT_OK(builder->Reserve(static_cast<int64_t>(row_count))); }

  template <typename BuilderType, typename C_TYPE>
  inline void append_to_builder(const ValueArray& values, const std::shared_ptr<std::vector<bool>>& is_valid) {
    const std::vector<C_TYPE>& vals = boost::get<std::vector<C_TYPE>>(values);

    auto typed_builder = static_cast<BuilderType*>(this->builder.get());
    if (this->field->nullable()) {
      CHECK(is_valid.get());
      ARROW_THROW_NOT_OK(typed_builder->Append(vals, *is_valid));
    } else {
      ARROW_THROW_NOT_OK(typed_builder->Append(vals));
    }
  }

  std::shared_ptr<Array> finish() {
    std::shared_ptr<Array> values;
    ARROW_THROW_NOT_OK(this->builder->Finish(&values));
    if (this->field->type()->id() == Type::DICTIONARY) {
      return std::make_shared<DictionaryArray>(this->field->type(), values);
    } else {
      return values;
    }
  }

  void append(const ValueArray& values, const std::shared_ptr<std::vector<bool>>& is_valid) {
    switch (this->physical_type) {
      case kBOOLEAN:
        append_to_builder<BooleanBuilder, bool>(values, is_valid);
        break;
      case kSMALLINT:
        append_to_builder<Int16Builder, int16_t>(values, is_valid);
        break;
      case kINT:
        append_to_builder<Int32Builder, int32_t>(values, is_valid);
        break;
      case kBIGINT:
        append_to_builder<Int64Builder, int64_t>(values, is_valid);
        break;
      case kFLOAT:
        append_to_builder<FloatBuilder, float>(values, is_valid);
        break;
      case kDOUBLE:
        append_to_builder<DoubleBuilder, double>(values, is_valid);
        break;
      default:
        // TODO(miyu): support more scalar types.
        CHECK(false);
    }
  }
};

std::shared_ptr<Field> make_field(const std::string name,
                                  const SQLTypeInfo& target_type,
                                  const std::shared_ptr<Array>& dictionary) {
  return field(name, get_arrow_type(target_type, dictionary), !target_type.get_notnull());
}

void print_serialized_schema(const uint8_t* data, const size_t length) {
  io::BufferReader reader(std::make_shared<arrow::Buffer>(data, length));
  std::shared_ptr<Schema> schema;
  ARROW_THROW_NOT_OK(ipc::ReadSchema(&reader, &schema));

  std::cout << "Arrow Schema: " << std::endl;
  ARROW_THROW_NOT_OK(PrettyPrint(*schema, {}, &std::cout));
}

void print_serialized_records(const uint8_t* data, const size_t length, const std::shared_ptr<Schema>& schema) {
  if (data == nullptr || !length) {
    std::cout << "No row found" << std::endl;
    return;
  }
  std::shared_ptr<RecordBatch> batch;

  io::BufferReader buffer_reader(std::make_shared<arrow::Buffer>(data, length));
  ARROW_THROW_NOT_OK(ipc::ReadRecordBatch(schema, &buffer_reader, &batch));
}

key_t get_and_copy_to_shm(const std::shared_ptr<Buffer>& data) {
  if (!data->size()) {
    return IPC_PRIVATE;
  }
  // Generate a new key for a shared memory segment. Keys to shared memory segments
  // are OS global, so we need to try a new key if we encounter a collision. It seems
  // incremental keygen would be deterministically worst-case. If we use a hash
  // (like djb2) + nonce, we could still get collisions if multiple clients specify
  // the same nonce, so using rand() in lieu of a better approach
  // TODO(ptaylor): Is this common? Are these assumptions true?
  auto key = static_cast<key_t>(rand());
  const auto shmsz = data->size();
  int shmid = -1;
  // IPC_CREAT - indicates we want to create a new segment for this key if it doesn't exist
  // IPC_EXCL - ensures failure if a segment already exists for this key
  while ((shmid = shmget(key, shmsz, IPC_CREAT | IPC_EXCL | 0666)) < 0) {
    // If shmget fails and errno is one of these four values, try a new key.
    // TODO(ptaylor): is checking for the last three values really necessary? Checking them by default to be safe.
    // EEXIST - a shared memory segment is already associated with this key
    // EACCES - a shared memory segment is already associated with this key, but we don't have permission to access it
    // EINVAL - a shared memory segment is already associated with this key, but the size is less than shmsz
    // ENOENT - IPC_CREAT was not set in shmflg and no shared memory segment associated with key was found
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

  // copy the arrow records buffer to shared memory
  // TODO(ptaylor): I'm sure it's possible to tell Arrow's RecordBatchStreamWriter to write
  // directly to the shared memory segment as a sink
  memcpy(ipc_ptr, data->data(), data->size());
  // detach from the shared memory segment
  shmdt(ipc_ptr);
  return key;
}

}  // namespace arrow

std::shared_ptr<const std::vector<std::string>> ResultSet::getDictionary(const int dict_id) const {
  const auto sdp = executor_ ? executor_->getStringDictionaryProxy(dict_id, row_set_mem_owner_, false)
                             : row_set_mem_owner_->getStringDictProxy(dict_id);
  return sdp->getDictionary()->copyStrings();
}

std::shared_ptr<arrow::RecordBatch> ResultSet::getArrowBatch(const std::shared_ptr<arrow::Schema>& schema) const {
  std::vector<std::shared_ptr<arrow::Array>> result_columns;

  const auto entry_count = entryCount();
  if (!entry_count) {
    return std::make_shared<arrow::RecordBatch>(schema, 0, result_columns);
  }
  const auto col_count = colCount();
  size_t row_count = 0;

  std::vector<arrow::ColumnBuilder> builders(col_count);

  // Create array builders
  for (size_t i = 0; i < col_count; ++i) {
    builders[i].init(getColType(i), schema->field(i));
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
      auto row = getRowAtNoTranslations(i);
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
            create_or_append_value<bool, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kSMALLINT:
            create_or_append_value<int16_t, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kINT:
            create_or_append_value<int32_t, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kBIGINT:
            create_or_append_value<int64_t, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kFLOAT:
            create_or_append_value<float, float>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<float>(*scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          case kDOUBLE:
            create_or_append_value<double, double>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<double>(*scalar_value, column.col_type, null_bitmap_seg[j], entry_count);
            break;
          default:
            // TODO(miyu): support more scalar types.
            CHECK(false);
        }
      }
    }
    return seg_row_count;
  };

  std::vector<std::shared_ptr<ValueArray>> column_values(col_count, nullptr);
  std::vector<std::shared_ptr<std::vector<bool>>> null_bitmaps(col_count, nullptr);
  const bool multithreaded = entry_count > 10000 && !isTruncated();
  if (multithreaded) {
    const size_t cpu_count = cpu_threads();
    std::vector<std::future<size_t>> child_threads;
    std::vector<std::vector<std::shared_ptr<ValueArray>>> column_value_segs(
        cpu_count, std::vector<std::shared_ptr<ValueArray>>(col_count, nullptr));
    std::vector<std::vector<std::shared_ptr<std::vector<bool>>>> null_bitmap_segs(
        cpu_count, std::vector<std::shared_ptr<std::vector<bool>>>(col_count, nullptr));
    const auto stride = (entry_count + cpu_count - 1) / cpu_count;
    for (size_t i = 0, start_entry = 0; start_entry < entry_count; ++i, start_entry += stride) {
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
      builders[i].reserve(row_count);
      for (size_t j = 0; j < cpu_count; ++j) {
        if (!column_value_segs[j][i]) {
          continue;
        }
        builders[i].append(*column_value_segs[j][i], null_bitmap_segs[j][i]);
      }
    }
  } else {
    row_count = fetch(column_values, null_bitmaps, size_t(0), entry_count);
    for (int i = 0; i < schema->num_fields(); ++i) {
      builders[i].reserve(row_count);
      builders[i].append(*column_values[i], null_bitmaps[i]);
    }
  }

  for (size_t i = 0; i < col_count; ++i) {
    result_columns.push_back(builders[i].finish());
  }
  return std::make_shared<arrow::RecordBatch>(schema, row_count, result_columns);
}

std::shared_ptr<arrow::RecordBatch> ResultSet::convertToArrow(const std::vector<std::string>& col_names,
                                                              arrow::ipc::DictionaryMemo& memo) const {
  const auto col_count = colCount();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  CHECK(col_names.empty() || col_names.size() == col_count);
  for (size_t i = 0; i < col_count; ++i) {
    const auto ti = getColType(i);
    std::shared_ptr<arrow::Array> dict;
    if (is_dict_enc_str(ti)) {
      const int dict_id = ti.get_comp_param();
      if (memo.HasDictionaryId(dict_id)) {
        ARROW_THROW_NOT_OK(memo.GetDictionary(dict_id, &dict));
      } else {
        auto str_list = getDictionary(dict_id);

        arrow::StringBuilder builder;
        for (const std::string& val : *str_list) {
          ARROW_THROW_NOT_OK(builder.Append(val));
        }
        ARROW_THROW_NOT_OK(builder.Finish(&dict));
        ARROW_THROW_NOT_OK(memo.AddDictionary(dict_id, dict));
      }
    }
    fields.push_back(arrow::make_field(col_names.empty() ? "" : col_names[i], ti, dict));
  }
  return getArrowBatch(arrow::schema(fields));
}

ResultSet::SerializedArrowOutput ResultSet::getSerializedArrowOutput(const std::vector<std::string>& col_names) const {
  arrow::ipc::DictionaryMemo dict_memo;
  std::shared_ptr<arrow::RecordBatch> arrow_copy = convertToArrow(col_names, dict_memo);
  std::shared_ptr<arrow::Buffer> serialized_records, serialized_schema;

  ARROW_THROW_NOT_OK(
      arrow::ipc::SerializeSchema(*arrow_copy->schema(), arrow::default_memory_pool(), &serialized_schema));

  ARROW_THROW_NOT_OK(arrow::ipc::SerializeRecordBatch(*arrow_copy, arrow::default_memory_pool(), &serialized_records));
  return {serialized_schema, serialized_records};
}

// WARN(ptaylor): users are responsible for detaching and removing shared memory segments, e.g.,
//   int shmid = shmget(...);
//   auto ipc_ptr = shmat(shmid, ...);
//   ...
//   shmdt(ipc_ptr);
//   shmctl(shmid, IPC_RMID, 0);
ArrowResult ResultSet::getArrowCopyOnCpu(const std::vector<std::string>& col_names) const {
  const auto serialized_arrow_output = getSerializedArrowOutput(col_names);
  const auto& serialized_schema = serialized_arrow_output.schema;
  const auto& serialized_records = serialized_arrow_output.records;

  const auto schema_key = arrow::get_and_copy_to_shm(serialized_schema);
  CHECK(schema_key != IPC_PRIVATE);
  std::vector<char> schema_handle_buffer(sizeof(key_t), 0);
  memcpy(&schema_handle_buffer[0], reinterpret_cast<const unsigned char*>(&schema_key), sizeof(key_t));

  const auto record_key = arrow::get_and_copy_to_shm(serialized_records);
  std::vector<char> record_handle_buffer(sizeof(key_t), 0);
  memcpy(&record_handle_buffer[0], reinterpret_cast<const unsigned char*>(&record_key), sizeof(key_t));

  return {schema_handle_buffer, serialized_schema->size(), record_handle_buffer, serialized_records->size(), nullptr};
}

// WARN(miyu): users are responsible to free all device copies, e.g.,
//   cudaIpcMemHandle_t mem_handle = ...
//   void* dev_ptr;
//   cudaIpcOpenMemHandle(&dev_ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess);
//   ...
//   cudaIpcCloseMemHandle(dev_ptr);
//   cudaFree(dev_ptr);
//
// TODO(miyu): verify if the server still needs to free its own copies after last uses
ArrowResult ResultSet::getArrowCopyOnGpu(Data_Namespace::DataMgr* data_mgr,
                                         const size_t device_id,
                                         const std::vector<std::string>& col_names) const {
  const auto serialized_arrow_output = getSerializedArrowOutput(col_names);
  const auto& serialized_schema = serialized_arrow_output.schema;

  const auto schema_key = arrow::get_and_copy_to_shm(serialized_schema);
  CHECK(schema_key != IPC_PRIVATE);
  std::vector<char> schema_handle_buffer(sizeof(key_t), 0);
  memcpy(&schema_handle_buffer[0], reinterpret_cast<const unsigned char*>(&schema_key), sizeof(key_t));

#ifdef HAVE_CUDA
  const auto& serialized_records = serialized_arrow_output.records;
  if (serialized_records->size()) {
    CHECK(data_mgr && data_mgr->cudaMgr_);
    auto dev_ptr =
        reinterpret_cast<CUdeviceptr>(data_mgr->cudaMgr_->allocateDeviceMem(serialized_records->size(), device_id));
    CUipcMemHandle record_handle;
    cuIpcGetMemHandle(&record_handle, dev_ptr);
    data_mgr->cudaMgr_->copyHostToDevice(reinterpret_cast<int8_t*>(dev_ptr),
                                         reinterpret_cast<const int8_t*>(serialized_records->data()),
                                         serialized_records->size(),
                                         device_id);
    std::vector<char> record_handle_buffer(sizeof(record_handle), 0);
    memcpy(&record_handle_buffer[0], reinterpret_cast<unsigned char*>(&record_handle), sizeof(CUipcMemHandle));

    return {schema_handle_buffer,
            serialized_schema->size(),
            record_handle_buffer,
            serialized_records->size(),
            reinterpret_cast<int8_t*>(dev_ptr)};
  }
#endif
  return {schema_handle_buffer, serialized_schema->size(), {}, 0, nullptr};
}

ArrowResult ResultSet::getArrowCopy(Data_Namespace::DataMgr* data_mgr,
                                    const ExecutorDeviceType device_type,
                                    const size_t device_id,
                                    const std::vector<std::string>& col_names) const {
  if (device_type == ExecutorDeviceType::CPU) {
    return getArrowCopyOnCpu(col_names);
  }

  CHECK(device_type == ExecutorDeviceType::GPU);
  return getArrowCopyOnGpu(data_mgr, device_id, col_names);
}

void deallocate_arrow_result(const ArrowResult& result,
                             const ExecutorDeviceType device_type,
                             const size_t device_id,
                             Data_Namespace::DataMgr* data_mgr) {
  // Remove shared memory on sysmem
  CHECK_EQ(sizeof(key_t), result.sm_handle.size());
  key_t schema_key;
  memcpy(&schema_key, &result.sm_handle[0], sizeof(key_t));
  auto shm_id = shmget(schema_key, result.sm_size, 0666);
  if (shm_id < 0) {
    throw std::runtime_error("failed to get an valid shm ID w/ given shm key of the schema");
  }
  if (-1 == shmctl(shm_id, IPC_RMID, 0)) {
    throw std::runtime_error("failed to deallocate Arrow schema on errorno(" + std::to_string(errno) + ")");
  }

  if (device_type == ExecutorDeviceType::CPU) {
    CHECK_EQ(sizeof(key_t), result.df_handle.size());
    key_t df_key;
    memcpy(&df_key, &result.df_handle[0], sizeof(key_t));
    auto shm_id = shmget(df_key, result.df_size, 0666);
    if (shm_id < 0) {
      throw std::runtime_error("failed to get an valid shm ID w/ given shm key of the data");
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

  data_mgr->cudaMgr_->freeDeviceMem(result.df_dev_ptr);
}
