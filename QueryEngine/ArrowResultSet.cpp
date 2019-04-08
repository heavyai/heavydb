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

#include "ArrowResultSet.h"
#include "ArrowUtil.h"
#include "RelAlgExecutionDescriptor.h"

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>

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

ArrowResultSet::ArrowResultSet(const std::shared_ptr<arrow::RecordBatch>& record_batch)
    : record_batch_(record_batch), crt_row_idx_(0) {
  auto schema = record_batch->schema();
  for (int i = 0; i < schema->num_fields(); ++i) {
    std::shared_ptr<arrow::Field> field = schema->field(i);
    SQLTypeInfo type_info = type_from_arrow_field(*schema->field(i));
    column_metainfo_.emplace_back(field->name(), type_info);
    columns_.emplace_back(record_batch->column(i));
  }
}

std::vector<TargetValue> ArrowResultSet::getRowAt(const size_t index) const {
  if (index >= rowCount()) {
    return {};
  }

  CHECK_LT(index, rowCount());
  std::vector<TargetValue> row;
  for (int i = 0; i < record_batch_->num_columns(); ++i) {
    const auto& column = *columns_[i];
    const auto& column_typeinfo = getColType(i);
    switch (column_typeinfo.get_type()) {
      case kSMALLINT: {
        CHECK_EQ(arrow::Type::INT16, column.type_id());
        const auto& i16_column = static_cast<const arrow::Int16Array&>(column);
        row.emplace_back(i16_column.IsNull(index)
                             ? inline_int_null_val(column_typeinfo)
                             : static_cast<int64_t>(i16_column.Value(index)));
        break;
      }
      case kINT: {
        CHECK_EQ(arrow::Type::INT32, column.type_id());
        const auto& i32_column = static_cast<const arrow::Int32Array&>(column);
        row.emplace_back(i32_column.IsNull(index)
                             ? inline_int_null_val(column_typeinfo)
                             : static_cast<int64_t>(i32_column.Value(index)));
        break;
      }
      case kBIGINT: {
        CHECK_EQ(arrow::Type::INT64, column.type_id());
        const auto& i64_column = static_cast<const arrow::Int64Array&>(column);
        row.emplace_back(i64_column.IsNull(index)
                             ? inline_int_null_val(column_typeinfo)
                             : static_cast<int64_t>(i64_column.Value(index)));
        break;
      }
      case kFLOAT: {
        CHECK_EQ(arrow::Type::FLOAT, column.type_id());
        const auto& float_column = static_cast<const arrow::FloatArray&>(column);
        row.emplace_back(float_column.IsNull(index) ? inline_fp_null_value<float>()
                                                    : float_column.Value(index));
        break;
      }
      case kDOUBLE: {
        CHECK_EQ(arrow::Type::DOUBLE, column.type_id());
        const auto& double_column = static_cast<const arrow::DoubleArray&>(column);
        row.emplace_back(double_column.IsNull(index) ? inline_fp_null_value<double>()
                                                     : double_column.Value(index));
        break;
      }
      case kTEXT: {
        CHECK_EQ(kENCODING_DICT, column_typeinfo.get_compression());
        CHECK_EQ(arrow::Type::DICTIONARY, column.type_id());
        const auto& dict_column = static_cast<const arrow::DictionaryArray&>(column);
        if (dict_column.IsNull(index)) {
          row.emplace_back(NullableString(nullptr));
        } else {
          const auto& indices =
              static_cast<const arrow::Int32Array&>(*dict_column.indices());
          const auto& dictionary =
              static_cast<const arrow::StringArray&>(*dict_column.dictionary());
          row.emplace_back(dictionary.GetString(indices.Value(index)));
        }
        break;
      }
      default:
        CHECK(false);
    }
  }
  return row;
}

std::vector<TargetValue> ArrowResultSet::getNextRow(const bool translate_strings,
                                                    const bool decimal_to_double) const {
  if (crt_row_idx_ == rowCount()) {
    return {};
  }
  CHECK_LT(crt_row_idx_, rowCount());
  auto row = getRowAt(crt_row_idx_);
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

std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(
    const ExecutionResult* results,
    const std::shared_ptr<ResultSet>& rows) {
  std::vector<std::string> col_names;

  if (results) {
    const auto& targets_meta = results->getTargetsMeta();
    for (auto& meta : targets_meta) {
      col_names.push_back(meta.get_resname());
    }
  } else {
    for (unsigned int i = 0; i < rows->colCount(); i++) {
      col_names.push_back("col_" + std::to_string(i));
    }
  }
  const auto serialized_arrow_output = rows->getSerializedArrowOutput(col_names, -1);

  arrow::io::BufferReader schema_reader(serialized_arrow_output.schema);

  std::shared_ptr<arrow::Schema> schema;
  ARROW_THROW_NOT_OK(arrow::ipc::ReadSchema(&schema_reader, &schema));

  std::shared_ptr<arrow::RecordBatch> batch;
  arrow::io::BufferReader records_reader(serialized_arrow_output.records);
  ARROW_THROW_NOT_OK(arrow::ipc::ReadRecordBatch(schema, &records_reader, &batch));

  CHECK_EQ(schema->num_fields(), batch->num_columns());

  // NOTE(wesm): About memory ownership

  // After calling ReadRecordBatch, the buffers inside arrow::RecordBatch now
  // share ownership of the memory in serialized_arrow_output.records (zero
  // copy). Not necessary to retain these buffers. Same is true of any
  // dictionaries contained in serialized_arrow_output.schema; the arrays
  // reference that memory (zero copy).

  return boost::make_unique<ArrowResultSet>(batch);
}

std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(
    const ExecutionResult& results) {
  return result_set_arrow_loopback(&results, results.getRows());
}
