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

#ifdef ENABLE_ARROW_CONVERTER
#include "ArrowResultSet.h"
#include "RelAlgExecutionDescriptor.h"

#include <arrow/array.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>

namespace {

SQLTypeInfo type_from_arrow_field(const arrow::Field& field) {
  switch (field.type()->id()) {
    case arrow::Type::INT16:
      return SQLTypeInfo(kSMALLINT, !field.nullable());
    case arrow::Type::INT32:
      return SQLTypeInfo(kINT, !field.nullable());
    case arrow::Type::INT64:
      return SQLTypeInfo(kBIGINT, !field.nullable());
    case arrow::Type::FLOAT:
      return SQLTypeInfo(kFLOAT, !field.nullable());
    case arrow::Type::DOUBLE:
      return SQLTypeInfo(kDOUBLE, !field.nullable());
    case arrow::Type::DICTIONARY:
      return SQLTypeInfo(kTEXT, !field.nullable(), kENCODING_DICT);
    default:
      CHECK(false);
  }
  CHECK(false);
  return SQLTypeInfo();
}

}  // namespace

ArrowResultSet::ArrowResultSet(const std::shared_ptr<arrow::Schema>& schema,
                               const std::shared_ptr<arrow::RecordBatch>& record_batch,
                               const std::vector<std::shared_ptr<arrow::PoolBuffer>>& pool_buffers)
    : record_batch_(record_batch), pool_buffers_(pool_buffers), crt_row_idx_(0) {
  CHECK_EQ(schema->num_fields(), record_batch->num_columns());
  for (int i = 0; i < schema->num_fields(); ++i) {
    column_metainfo_.emplace_back(schema->field(i)->name(), type_from_arrow_field(*schema->field(i)));
  }
}

std::vector<TargetValue> ArrowResultSet::getNextRow(const bool translate_strings, const bool decimal_to_double) const {
  if (crt_row_idx_ == rowCount()) {
    return {};
  }
  CHECK_LT(crt_row_idx_, rowCount());
  std::vector<TargetValue> row;
  for (int i = 0; i < record_batch_->num_columns(); ++i) {
    const auto& column = *record_batch_->column(i);
    const auto& column_typeinfo = getColType(i);
    switch (column_typeinfo.get_type()) {
      case kSMALLINT: {
        CHECK(dynamic_cast<const arrow::NumericArray<arrow::Int16Type>*>(&column));
        const auto& i16_column = static_cast<const arrow::NumericArray<arrow::Int16Type>&>(column);
        row.emplace_back(i16_column.IsNull(crt_row_idx_) ? inline_int_null_val(column_typeinfo)
                                                         : static_cast<int64_t>(i16_column.Value(crt_row_idx_)));
        break;
      }
      case kINT: {
        CHECK(dynamic_cast<const arrow::NumericArray<arrow::Int32Type>*>(&column));
        const auto& i32_column = static_cast<const arrow::NumericArray<arrow::Int32Type>&>(column);
        row.emplace_back(i32_column.IsNull(crt_row_idx_) ? inline_int_null_val(column_typeinfo)
                                                         : static_cast<int64_t>(i32_column.Value(crt_row_idx_)));
        break;
      }
      case kBIGINT: {
        CHECK(dynamic_cast<const arrow::NumericArray<arrow::Int64Type>*>(&column));
        const auto& i64_column = static_cast<const arrow::NumericArray<arrow::Int64Type>&>(column);
        row.emplace_back(i64_column.IsNull(crt_row_idx_) ? inline_int_null_val(column_typeinfo)
                                                         : static_cast<int64_t>(i64_column.Value(crt_row_idx_)));
        break;
      }
      case kFLOAT: {
        CHECK(dynamic_cast<const arrow::NumericArray<arrow::FloatType>*>(&column));
        const auto& float_column = static_cast<const arrow::NumericArray<arrow::FloatType>&>(column);
        row.emplace_back(float_column.IsNull(crt_row_idx_) ? inline_fp_null_value<float>()
                                                           : float_column.Value(crt_row_idx_));
        break;
      }
      case kDOUBLE: {
        CHECK(dynamic_cast<const arrow::NumericArray<arrow::DoubleType>*>(&column));
        const auto& double_column = static_cast<const arrow::NumericArray<arrow::DoubleType>&>(column);
        row.emplace_back(double_column.IsNull(crt_row_idx_) ? inline_fp_null_value<double>()
                                                            : double_column.Value(crt_row_idx_));
        break;
      }
      case kTEXT: {
        CHECK_EQ(kENCODING_DICT, column_typeinfo.get_compression());
        CHECK(dynamic_cast<const arrow::DictionaryArray*>(&column));
        const auto& dict_column = static_cast<const arrow::DictionaryArray&>(column);
        if (dict_column.IsNull(crt_row_idx_)) {
          row.emplace_back(NullableString(nullptr));
        } else {
          const auto& indices = static_cast<const arrow::NumericArray<arrow::Int32Type>&>(*dict_column.indices());
          const auto& dictionary = static_cast<const arrow::StringArray&>(*dict_column.dictionary());
          row.emplace_back(dictionary.GetString(indices.Value(crt_row_idx_)));
        }
        break;
      }
      default:
        CHECK(false);
    }
  }
  ++crt_row_idx_;
  return row;
}

size_t ArrowResultSet::colCount() const {
  return column_metainfo_.size();
}

SQLTypeInfo ArrowResultSet::getColType(const size_t col_idx) const {
  CHECK_LT(col_idx, column_metainfo_.size());
  return column_metainfo_[col_idx].get_type_info();
}

bool ArrowResultSet::definitelyHasNoRows() const {
  return !rowCount();
}

size_t ArrowResultSet::rowCount() const {
  return record_batch_->num_rows();
}

std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(const ExecutionResult& results) {
  const auto& targets_meta = results.getTargetsMeta();
  std::vector<std::string> col_names;
  for (const auto& target_meta : targets_meta) {
    col_names.push_back(target_meta.get_resname());
  }
  const auto serialized_arrow_output = results.getRows()->getSerializedArrowOutput(col_names);
  const auto schema_payload =
      std::make_shared<arrow::Buffer>(serialized_arrow_output.schema->data(), serialized_arrow_output.schema->size());
  auto schema_buffer = std::make_shared<arrow::io::BufferReader>(schema_payload);
  std::shared_ptr<arrow::ipc::RecordBatchStreamReader> schema_reader;
  arrow::ipc::RecordBatchStreamReader::Open(schema_buffer, &schema_reader);
  auto schema = schema_reader->schema();
  const auto records_payload =
      std::make_shared<arrow::Buffer>(serialized_arrow_output.records->data(), serialized_arrow_output.records->size());
  auto records_buffer_reader = std::make_shared<arrow::io::BufferReader>(records_payload);

  std::shared_ptr<arrow::ipc::Message> message;
  arrow::ipc::ReadMessage(records_buffer_reader.get(), &message);

  // The buffer offsets start at 0, so we must construct a
  // RandomAccessFile according to that frame of reference
  std::shared_ptr<arrow::Buffer> body_payload;
  records_buffer_reader->Read(message->body_length(), &body_payload);
  arrow::io::BufferReader body_reader(body_payload);

  std::shared_ptr<arrow::RecordBatch> batch;
  arrow::ipc::ReadRecordBatch(*message, schema, &body_reader, &batch);

  return boost::make_unique<ArrowResultSet>(
      schema,
      batch,
      std::vector<std::shared_ptr<arrow::PoolBuffer>>{serialized_arrow_output.schema, serialized_arrow_output.records});
}

#endif  // ENABLE_ARROW_CONVERTER
