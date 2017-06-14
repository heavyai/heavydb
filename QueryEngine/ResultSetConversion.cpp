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

#include "ResultSet.h"
#include "Execute.h"

#ifdef ENABLE_ARROW_CONVERTER
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <errno.h>
#include <string>

#define MAPD_ARROW_VERSION (ARROW_VERSION_MAJOR * 10000 + ARROW_VERSION_MINOR * 100 + ARROW_VERSION_PATCH)
#if MAPD_ARROW_VERSION != 300
#error MapD only supports Apache Arrow release 0.3.0
#endif

#include "arrow/builder.h"
#include "arrow/buffer.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/metadata.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"
#include "arrow/pretty_print.h"
#include "arrow/memory_pool.h"
#include "arrow/type.h"

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
                       std::vector<double>>
    ValueArray;

namespace {

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
  if (col_type.is_integer()) {
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

#define ASSERT_OK(expr)           \
  do {                            \
    Status s = (expr);            \
    if (!s.ok()) {                \
      LOG(ERROR) << s.ToString(); \
    }                             \
  } while (0)

#define RETURN_IF_NOT_OK(s) \
  do {                      \
    arrow::Status _s = (s); \
    if (!_s.ok()) {         \
      return nullptr;       \
    }                       \
  } while (0);

TypePtr get_arrow_type(const SQLTypeInfo& mapd_type) {
  switch (get_physical_type(mapd_type)) {
    case kBOOLEAN:
      return std::make_shared<BooleanType>();
    case kSMALLINT:
      return std::make_shared<Int16Type>();
    case kINT:
      return std::make_shared<Int32Type>();
    case kBIGINT:
      return std::make_shared<Int64Type>();
    case kFLOAT:
      return std::make_shared<FloatType>();
    case kDOUBLE:
      return std::make_shared<DoubleType>();
    case kDECIMAL:
    case kNUMERIC:
    case kCHAR:
    case kVARCHAR:
    case kTIME:
    case kTIMESTAMP:
    case kTEXT:
    case kDATE:
    case kARRAY:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    default:
      throw std::runtime_error(mapd_type.get_type_name() + " is not supported in temporary table.");
  }
  return nullptr;
}
std::shared_ptr<Field> make_field(const std::string name, const SQLTypeInfo& target_type) {
  return std::make_shared<Field>(name, get_arrow_type(target_type), !target_type.get_notnull());
}

// Cited from arrow/test-util.h
template <typename TYPE, typename C_TYPE>
std::shared_ptr<Array> array_from_vector(const std::shared_ptr<std::vector<bool>> is_valid,
                                         const std::vector<C_TYPE>& values) {
  std::shared_ptr<Array> out;
  MemoryPool* pool = default_memory_pool();
  typename TypeTraits<TYPE>::BuilderType builder(pool);
  if (is_valid) {
    for (size_t i = 0; i < values.size(); ++i) {
      if ((*is_valid)[i]) {
        ASSERT_OK(builder.Append(values[i]));
      } else {
        ASSERT_OK(builder.AppendNull());
      }
    }
  } else {
    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_OK(builder.Append(values[i]));
    }
  }
  ASSERT_OK(builder.Finish(&out));
  return out;
}

std::shared_ptr<Array> generate_column(const Field& field,
                                       const std::shared_ptr<std::vector<bool>> is_valid,
                                       const ValueArray& values) {
  switch (field.type()->id()) {
    case Type::BOOL: {
      auto vals_bool = boost::get<std::vector<bool>>(&values);
      CHECK(vals_bool);
      return array_from_vector<BooleanType, bool>(is_valid, *vals_bool);
    }
    case Type::INT8: {
      auto vals_i8 = boost::get<std::vector<int8_t>>(&values);
      CHECK(vals_i8);
      return array_from_vector<Int8Type, int8_t>(is_valid, *vals_i8);
    }
    case Type::INT16: {
      auto vals_i16 = boost::get<std::vector<int16_t>>(&values);
      CHECK(vals_i16);
      return array_from_vector<Int16Type, int16_t>(is_valid, *vals_i16);
    }
    case Type::INT32: {
      auto vals_i32 = boost::get<std::vector<int32_t>>(&values);
      CHECK(vals_i32);
      return array_from_vector<Int32Type, int32_t>(is_valid, *vals_i32);
    }
    case Type::INT64: {
      auto vals_i64 = boost::get<std::vector<int64_t>>(&values);
      CHECK(vals_i64);
      return array_from_vector<Int64Type, int64_t>(is_valid, *vals_i64);
    }
    case Type::FLOAT: {
      auto vals_float = boost::get<std::vector<float>>(&values);
      CHECK(vals_float);
      return array_from_vector<FloatType, float>(is_valid, *vals_float);
    }
    case Type::DOUBLE: {
      auto vals_double = boost::get<std::vector<double>>(&values);
      CHECK(vals_double);
      return array_from_vector<DoubleType, double>(is_valid, *vals_double);
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

std::vector<std::shared_ptr<Array>> generate_columns(
    const std::vector<std::shared_ptr<Field>>& fields,
    const std::vector<std::shared_ptr<std::vector<bool>>>& null_bitmaps,
    const std::vector<std::shared_ptr<ValueArray>>& column_values) {
  const auto col_count = fields.size();
  CHECK_GT(col_count, 0);
  std::vector<std::shared_ptr<arrow::Array>> columns(col_count, nullptr);
  auto generate = [&](const size_t i, std::shared_ptr<arrow::Array>& column) {
    column = generate_column(*fields[i], null_bitmaps[i], *column_values[i]);
  };
  if (col_count > 1) {
    std::vector<std::future<void>> child_threads;
    for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
      child_threads.push_back(std::async(std::launch::async, generate, col_idx, std::ref(columns[col_idx])));
    }
    for (auto& child : child_threads) {
      child.get();
    }
  } else {
    generate(0, columns[0]);
  }
  return columns;
}

std::shared_ptr<ValueArray> create_value_array(const Field& field, const size_t value_count) {
  switch (field.type()->id()) {
    case Type::BOOL: {
      auto array = std::make_shared<ValueArray>(std::vector<bool>());
      boost::get<std::vector<bool>>(*array).reserve(value_count);
      return array;
    }
    case Type::INT8: {
      auto array = std::make_shared<ValueArray>(std::vector<int8_t>());
      boost::get<std::vector<int8_t>>(*array).reserve(value_count);
      return array;
    }
    case Type::INT16: {
      auto array = std::make_shared<ValueArray>(std::vector<int16_t>());
      boost::get<std::vector<int16_t>>(*array).reserve(value_count);
      return array;
    }
    case Type::INT32: {
      auto array = std::make_shared<ValueArray>(std::vector<int32_t>());
      boost::get<std::vector<int32_t>>(*array).reserve(value_count);
      return array;
    }
    case Type::INT64: {
      auto array = std::make_shared<ValueArray>(std::vector<int64_t>());
      boost::get<std::vector<int64_t>>(*array).reserve(value_count);
      return array;
    }
    case Type::FLOAT: {
      auto array = std::make_shared<ValueArray>(std::vector<float>());
      boost::get<std::vector<float>>(*array).reserve(value_count);
      return array;
    }
    case Type::DOUBLE: {
      auto array = std::make_shared<ValueArray>(std::vector<double>());
      boost::get<std::vector<double>>(*array).reserve(value_count);
      return array;
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

void append_value_array(ValueArray& dst, const ValueArray& src, const Field& field) {
  switch (field.type()->id()) {
    case Type::BOOL: {
      auto dst_bool = boost::get<std::vector<bool>>(&dst);
      auto src_bool = boost::get<std::vector<bool>>(&src);
      CHECK(dst_bool && src_bool);
      dst_bool->insert(dst_bool->end(), src_bool->begin(), src_bool->end());
    } break;
    case Type::INT8: {
      auto dst_i8 = boost::get<std::vector<int8_t>>(&dst);
      auto src_i8 = boost::get<std::vector<int8_t>>(&src);
      CHECK(dst_i8 && src_i8);
      dst_i8->insert(dst_i8->end(), src_i8->begin(), src_i8->end());
    } break;
    case Type::INT16: {
      auto dst_i16 = boost::get<std::vector<int16_t>>(&dst);
      auto src_i16 = boost::get<std::vector<int16_t>>(&src);
      CHECK(dst_i16 && src_i16);
      dst_i16->insert(dst_i16->end(), src_i16->begin(), src_i16->end());
    } break;
    case Type::INT32: {
      auto dst_i32 = boost::get<std::vector<int32_t>>(&dst);
      auto src_i32 = boost::get<std::vector<int32_t>>(&src);
      CHECK(dst_i32 && src_i32);
      dst_i32->insert(dst_i32->end(), src_i32->begin(), src_i32->end());
    } break;
    case Type::INT64: {
      auto dst_i64 = boost::get<std::vector<int64_t>>(&dst);
      auto src_i64 = boost::get<std::vector<int64_t>>(&src);
      CHECK(dst_i64 && src_i64);
      dst_i64->insert(dst_i64->end(), src_i64->begin(), src_i64->end());
    } break;
    case Type::FLOAT: {
      auto dst_flt = boost::get<std::vector<float>>(&dst);
      auto src_flt = boost::get<std::vector<float>>(&src);
      CHECK(dst_flt && src_flt);
      dst_flt->insert(dst_flt->end(), src_flt->begin(), src_flt->end());
    } break;
    case Type::DOUBLE: {
      auto dst_dbl = boost::get<std::vector<double>>(&dst);
      auto src_dbl = boost::get<std::vector<double>>(&src);
      CHECK(dst_dbl && src_dbl);
      dst_dbl->insert(dst_dbl->end(), src_dbl->begin(), src_dbl->end());
    } break;
    default:
      CHECK(false);
      break;
  }
}

void print_serialization(const uint8_t* data, const size_t length) {
  const auto record_buffer = std::make_shared<arrow::Buffer>(data, length);
  auto buf_reader = std::make_shared<io::BufferReader>(record_buffer);
  std::shared_ptr<ipc::StreamReader> reader;
  ipc::StreamReader::Open(buf_reader, &reader);

  std::shared_ptr<RecordBatch> batch;
  reader->GetNextRecordBatch(&batch);
  std::cout << "Schema: " << batch->schema()->ToString() << std::endl;
  for (int i = 0; i < batch->num_columns(); ++i) {
    const auto& column = *batch->column(i);
    std::cout << "Column #" << i << ": ";
    PrettyPrint(column, 0, &std::cout);
  }
}

std::shared_ptr<PoolBuffer> serialize_arrow_records(const RecordBatch& rb) {
  int64_t rb_sz = 0;
  ASSERT_OK(ipc::GetRecordBatchSize(rb, &rb_sz));
  auto buffer = std::make_shared<PoolBuffer>(default_memory_pool());
  buffer->Reserve(rb_sz);
  io::BufferOutputStream sink(buffer);
  std::shared_ptr<ipc::StreamWriter> writer;
  RETURN_IF_NOT_OK(ipc::StreamWriter::Open(&sink, rb.schema(), &writer));
  RETURN_IF_NOT_OK(writer->WriteRecordBatch(rb));
  writer->Close();
  return buffer;
}

std::shared_ptr<Buffer> serialize_arrow_schema(const Schema& schema) {
  ipc::DictionaryMemo memo;
  std::shared_ptr<Buffer> buffer;
  ASSERT_OK(ipc::WriteSchemaMessage(schema, &memo, &buffer));
  return buffer;
}

#undef RETURN_IF_NOT_OK
#undef ASSERT_OK

}  // namespace arrow

std::pair<std::vector<std::shared_ptr<arrow::Array>>, size_t> ResultSet::getArrowColumns(
    const std::vector<std::shared_ptr<arrow::Field>>& fields) const {
  const auto entry_count = entryCount();
  if (!entry_count) {
    return {{}, 0};
  }
  const auto col_count = colCount();
  size_t row_count = 0;

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
      auto row = getRowAt(i);
      if (row.empty()) {
        continue;
      }
      ++seg_row_count;
      for (size_t j = 0; j < col_count; ++j) {
        const auto& col_type = getColType(j);
        auto scalar_value = boost::get<ScalarTargetValue>(&row[j]);
        // TODO(miyu): support more types other than scalar.
        CHECK(scalar_value);
        switch (get_physical_type(col_type)) {
          case kBOOLEAN:
            create_or_append_value<bool, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kSMALLINT:
            create_or_append_value<int16_t, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kINT:
            create_or_append_value<int32_t, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kBIGINT:
            create_or_append_value<int64_t, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kFLOAT:
            create_or_append_value<float, float>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<float>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kDOUBLE:
            create_or_append_value<double, double>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<double>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
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
    for (size_t i = 0; i < fields.size(); ++i) {
      column_values[i] = arrow::create_value_array(*fields[i], row_count);
      if (fields[i]->nullable()) {
        null_bitmaps[i] = std::make_shared<std::vector<bool>>();
        null_bitmaps[i]->reserve(row_count);
      }
      for (size_t j = 0; j < cpu_count; ++j) {
        if (!column_value_segs[j][i]) {
          continue;
        }
        arrow::append_value_array(*column_values[i], *column_value_segs[j][i], *fields[i]);
        if (fields[i]->nullable()) {
          CHECK(null_bitmap_segs[j][i]);
          null_bitmaps[i]->insert(
              null_bitmaps[i]->end(), null_bitmap_segs[j][i]->begin(), null_bitmap_segs[j][i]->end());
        }
      }
    }
  } else {
    row_count = fetch(column_values, null_bitmaps, size_t(0), entry_count);
  }

  return {generate_columns(fields, null_bitmaps, column_values), row_count};
}

arrow::RecordBatch ResultSet::convertToArrow(const std::vector<std::string>& col_names) const {
  const auto col_count = colCount();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  if (col_names.empty()) {
    for (size_t i = 0; i < col_count; ++i) {
      fields.push_back(arrow::make_field("", getColType(i)));
    }
  } else {
    CHECK_EQ(col_names.size(), col_count);
    for (size_t i = 0; i < col_count; ++i) {
      fields.push_back(arrow::make_field(col_names[i], getColType(i)));
    }
  }
  auto schema = std::make_shared<arrow::Schema>(fields);
  std::vector<std::shared_ptr<arrow::Array>> columns;
  size_t row_count = 0;
  if (col_count > 0) {
    std::tie(columns, row_count) = getArrowColumns(fields);
  }
  return arrow::RecordBatch(schema, row_count, columns);
}

// WARN(ptaylor): users are responsible for detaching and removing shared memory segments, e.g.,
//   int shmid = shmget(...);
//   char *ipc_ptr = (char* shmat(shmid, ...));
//   ...
//   shmdt(ipc_ptr);
//   shmctl(shmid, IPC_RMID, 0);
std::tuple<std::shared_ptr<arrow::Buffer>, std::vector<char>, int64_t> ResultSet::getArrowCopy(
    Data_Namespace::DataMgr* data_mgr,
    const std::vector<std::string>& col_names) const {
  auto arrow_copy = convertToArrow(col_names);
  auto serialized_schema = serialize_arrow_schema(*arrow_copy.schema());
  auto serialized_records = serialize_arrow_records(arrow_copy);
  // Generate a new key for a shared memory segment. Keys to shared memory segments
  // are OS global, so we need to try a new key if we encounter a collision. It seems
  // incremental keygen would be deterministically worst-case. If we use a hash
  // (like djb2) + nonce, we could still get collisions if multiple clients specify
  // the same nonce, so using rand() in lieu of a better approach
  // TODO(ptaylor): Is this common? Are these assumptions true?
  auto key = static_cast<key_t>(rand());
  const auto shmsz = serialized_records->size();
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
  // TODO(ptaylor): I'm sure it's possible to tell Arrow's StreamWriter to write
  // directly to the shared memory segment as a sink
  memcpy(ipc_ptr, serialized_records->data(), serialized_records->size());
  // detach from the shared memory segment
  shmdt(ipc_ptr);
  // cast the shared memory key to a string, and then to a char vec, so we can
  // keep the same method signature as the original `getArrowDeviceCopy` below.
  std::vector<char> handle_buffer(sizeof(key_t), 0);
  memcpy(&handle_buffer[0], reinterpret_cast<unsigned char*>(&key), sizeof(key_t));
  return std::tuple<std::shared_ptr<arrow::Buffer>, std::vector<char>, int64_t>{
      serialized_schema, handle_buffer, serialized_records->size()};
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
std::tuple<std::shared_ptr<arrow::Buffer>, std::vector<char>, int64_t> ResultSet::getArrowDeviceCopy(
    Data_Namespace::DataMgr* data_mgr,
    const size_t device_id,
    const std::vector<std::string>& col_names) const {
  auto arrow_copy = convertToArrow(col_names);
  auto serialized_schema = serialize_arrow_schema(*arrow_copy.schema());
#ifdef HAVE_CUDA
  auto serialized_records = serialize_arrow_records(arrow_copy);
  CHECK(data_mgr && data_mgr->cudaMgr_);
  auto dev_ptr =
      reinterpret_cast<CUdeviceptr>(data_mgr->cudaMgr_->allocateDeviceMem(serialized_records->size(), device_id));
  CUipcMemHandle ipc_mem_handle;
  cuIpcGetMemHandle(&ipc_mem_handle, dev_ptr);
  data_mgr->cudaMgr_->copyHostToDevice(reinterpret_cast<int8_t*>(dev_ptr),
                                       reinterpret_cast<const int8_t*>(serialized_records->data()),
                                       serialized_records->size(),
                                       device_id);
  std::vector<char> handle_buffer(sizeof(ipc_mem_handle), 0);
  memcpy(&handle_buffer[0], reinterpret_cast<unsigned char*>(&ipc_mem_handle), sizeof(ipc_mem_handle));
  return std::tuple<std::shared_ptr<arrow::Buffer>, std::vector<char>, int64_t>{
      serialized_schema, handle_buffer, serialized_records->size()};
#else
  return std::tuple<std::shared_ptr<arrow::Buffer>, std::vector<char>, int64_t>{serialized_schema, {}, 0};
#endif
}

#endif  // ENABLE_ARROW_CONVERTER
