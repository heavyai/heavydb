/*
 * Copyright 2020 OmniSci, Inc.
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

#include "LazyParquetChunkLoader.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/column_scanner.h>
#include <parquet/exception.h>
#include <parquet/platform.h>
#include <parquet/types.h>

#include "ParquetDecimalEncoder.h"
#include "ParquetFixedLengthEncoder.h"
#include "ParquetShared.h"
#include "ParquetTimeEncoder.h"
#include "ParquetTimestampEncoder.h"

namespace foreign_storage {

namespace {

template <typename V>
std::shared_ptr<ParquetInPlaceEncoder> create_parquet_decimal_encoder_with_omnisci_type(
    const ColumnDescriptor* column_descriptor,
    const parquet::ColumnDescriptor* parquet_column_descriptor,
    AbstractBuffer* buffer) {
  switch (parquet_column_descriptor->physical_type()) {
    case parquet::Type::INT32:
      return std::make_shared<ParquetDecimalEncoder<V, int32_t>>(
          buffer, column_descriptor, parquet_column_descriptor);
    case parquet::Type::INT64:
      return std::make_shared<ParquetDecimalEncoder<V, int64_t>>(
          buffer, column_descriptor, parquet_column_descriptor);
    case parquet::Type::FIXED_LEN_BYTE_ARRAY:
      return std::make_shared<ParquetDecimalEncoder<V, parquet::FixedLenByteArray>>(
          buffer, column_descriptor, parquet_column_descriptor);
    case parquet::Type::BYTE_ARRAY:
      return std::make_shared<ParquetDecimalEncoder<V, parquet::ByteArray>>(
          buffer, column_descriptor, parquet_column_descriptor);
    default:
      UNREACHABLE();
  }
  return {};
}

std::shared_ptr<ParquetInPlaceEncoder> create_parquet_decimal_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  if (parquet_column->logical_type()->is_decimal()) {
    if (omnisci_column->columnType.get_compression() == kENCODING_NONE) {
      return create_parquet_decimal_encoder_with_omnisci_type<int64_t>(
          omnisci_column, parquet_column, buffer);
    } else if (omnisci_column->columnType.get_compression() == kENCODING_FIXED) {
      switch (omnisci_column->columnType.get_comp_param()) {
        case 16:
          return create_parquet_decimal_encoder_with_omnisci_type<int16_t>(
              omnisci_column, parquet_column, buffer);
        case 32:
          return create_parquet_decimal_encoder_with_omnisci_type<int32_t>(
              omnisci_column, parquet_column, buffer);
        default:
          UNREACHABLE();
      }
    } else {
      UNREACHABLE();
    }
  }
  return {};
}

std::shared_ptr<ParquetInPlaceEncoder> create_parquet_integral_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  auto column_type = omnisci_column->columnType;
  if (auto int_logical_column = dynamic_cast<const parquet::IntLogicalType*>(
          parquet_column->logical_type().get())) {
    if (int_logical_column->is_signed()) {  // signed
      switch (column_type.get_size()) {
        case 8:
          CHECK(column_type.get_compression() == kENCODING_NONE);
          return std::make_shared<ParquetFixedLengthEncoder<int64_t, int64_t>>(
              buffer, omnisci_column, parquet_column);
        case 4:
          return std::make_shared<ParquetFixedLengthEncoder<int32_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        case 2:
          return std::make_shared<ParquetFixedLengthEncoder<int16_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        case 1:
          return std::make_shared<ParquetFixedLengthEncoder<int8_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        default:
          UNREACHABLE();
      }
    } else {  // unsigned, requires using a larger bit depth signed integer within omnisci
              // to prevent the possibility of loss of information
      switch (column_type.get_size()) {
        case 8:
          CHECK(column_type.get_compression() == kENCODING_NONE);
          return std::make_shared<
              ParquetUnsignedFixedLengthEncoder<int64_t, int32_t, uint32_t>>(
              buffer, omnisci_column, parquet_column);
        case 4:
          return std::make_shared<
              ParquetUnsignedFixedLengthEncoder<int32_t, int32_t, uint16_t>>(
              buffer, omnisci_column, parquet_column);
        case 2:
          return std::make_shared<
              ParquetUnsignedFixedLengthEncoder<int16_t, int32_t, uint8_t>>(
              buffer, omnisci_column, parquet_column);
        default:
          UNREACHABLE();
      }
    }
  }
  return {};
}

std::shared_ptr<ParquetInPlaceEncoder> create_parquet_none_type_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  auto column_type = omnisci_column->columnType;
  if (parquet_column->logical_type()
          ->is_none()) {  // boolean, int32, int64, float & double
    if (column_type.get_compression() == kENCODING_NONE) {
      switch (column_type.get_type()) {
        case kBIGINT:
          return std::make_shared<ParquetFixedLengthEncoder<int64_t, int64_t>>(
              buffer, omnisci_column, parquet_column);
        case kINT:
          return std::make_shared<ParquetFixedLengthEncoder<int32_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        case kBOOLEAN:
          return std::make_shared<ParquetFixedLengthEncoder<int8_t, bool>>(
              buffer, omnisci_column, parquet_column);
        case kFLOAT:
          return std::make_shared<ParquetFixedLengthEncoder<float, float>>(
              buffer, omnisci_column, parquet_column);
        case kDOUBLE:
          return std::make_shared<ParquetFixedLengthEncoder<double, double>>(
              buffer, omnisci_column, parquet_column);
        default:
          UNREACHABLE();
      }
    } else {
      UNREACHABLE();
    }
  }
  return {};
}

std::shared_ptr<ParquetInPlaceEncoder> create_parquet_timestamp_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  auto column_type = omnisci_column->columnType;
  if (parquet_column->logical_type()->is_timestamp()) {
    auto precision = column_type.get_precision();
    if (precision == 0) {
      return std::make_shared<ParquetTimestampEncoder<int64_t, int64_t>>(
          buffer, omnisci_column, parquet_column);
    } else {
      return std::make_shared<ParquetFixedLengthEncoder<int64_t, int64_t>>(
          buffer, omnisci_column, parquet_column);
    }
  }
  return {};
}

std::shared_ptr<ParquetInPlaceEncoder> create_parquet_time_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  auto column_type = omnisci_column->columnType;
  if (auto time_logical_column = dynamic_cast<const parquet::TimeLogicalType*>(
          parquet_column->logical_type().get())) {
    if (column_type.get_compression() == kENCODING_NONE) {
      if (time_logical_column->time_unit() == parquet::LogicalType::TimeUnit::MILLIS) {
        return std::make_shared<ParquetTimeEncoder<int64_t, int32_t>>(
            buffer, omnisci_column, parquet_column);
      } else {
        return std::make_shared<ParquetTimeEncoder<int64_t, int64_t>>(
            buffer, omnisci_column, parquet_column);
      }
    } else if (column_type.get_compression() == kENCODING_FIXED) {
      return std::make_shared<ParquetTimeEncoder<int32_t, int32_t>>(
          buffer, omnisci_column, parquet_column);
    } else {
      UNREACHABLE();
    }
  }
  return {};
}

std::shared_ptr<ParquetInPlaceEncoder> create_parquet_date_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  auto column_type = omnisci_column->columnType;
  if (parquet_column->logical_type()->is_date()) {
    if (column_type.get_compression() == kENCODING_DATE_IN_DAYS) {
      return std::make_shared<ParquetFixedLengthEncoder<int32_t, int32_t>>(
          buffer, omnisci_column, parquet_column);
    } else {
      UNREACHABLE();
    }
  }
  return {};
}

std::shared_ptr<ParquetInPlaceEncoder> create_parquet_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  if (auto encoder =
          create_parquet_decimal_encoder(omnisci_column, parquet_column, buffer)) {
    return encoder;
  }
  if (auto encoder =
          create_parquet_integral_encoder(omnisci_column, parquet_column, buffer)) {
    return encoder;
  }
  if (auto encoder =
          create_parquet_none_type_encoder(omnisci_column, parquet_column, buffer)) {
    return encoder;
  }
  if (auto encoder =
          create_parquet_timestamp_encoder(omnisci_column, parquet_column, buffer)) {
    return encoder;
  }
  if (auto encoder =
          create_parquet_time_encoder(omnisci_column, parquet_column, buffer)) {
    return encoder;
  }
  if (auto encoder =
          create_parquet_date_encoder(omnisci_column, parquet_column, buffer)) {
    return encoder;
  }
  UNREACHABLE();
  return {};
}

void append_row_groups(const Interval<RowGroupType>& row_group_interval,
                       const int logical_column_index,
                       const ColumnDescriptor* column_descriptor,
                       const parquet::ColumnDescriptor* parquet_column_descriptor,
                       AbstractBuffer* buffer,
                       parquet::ParquetFileReader* reader,
                       int16_t* def_levels,
                       int16_t* rep_levels) {
  auto encoder =
      create_parquet_encoder(column_descriptor, parquet_column_descriptor, buffer);
  CHECK(encoder.get());
  for (int row_group_index = row_group_interval.start;
       row_group_index <= row_group_interval.end;
       ++row_group_index) {
    int64_t values_read = 0;
    auto group_reader = reader->RowGroup(row_group_index);
    std::shared_ptr<parquet::ColumnReader> col_reader =
        group_reader->Column(logical_column_index);

    if (col_reader->descr()->max_repetition_level() > 0) {
      throw std::runtime_error("Nested schema detected in column '" +
                               parquet_column_descriptor->name() +
                               "'. Only flat schemas are supported.");
    }

    while (col_reader->HasNext()) {
      size_t buffer_size = buffer->size();
      CHECK(buffer->reservedSize() - buffer_size >=
            LazyParquetChunkLoader::batch_reader_num_elements);
      uint8_t* values = reinterpret_cast<uint8_t*>(buffer->getMemoryPtr() + buffer_size);
      int64_t levels_read =
          parquet::ScanAllValues(LazyParquetChunkLoader::batch_reader_num_elements,
                                 def_levels,
                                 rep_levels,
                                 values,
                                 &values_read,
                                 col_reader.get());
      encoder->appendData(def_levels, values_read, levels_read);
    }
  }
}

bool validate_decimal_mapping(const ColumnDescriptor* omnisci_column,
                              const parquet::ColumnDescriptor* parquet_column) {
  if (auto decimal_logical_column = dynamic_cast<const parquet::DecimalLogicalType*>(
          parquet_column->logical_type().get())) {
    return omnisci_column->columnType.get_precision() ==
               decimal_logical_column->precision() &&
           omnisci_column->columnType.get_scale() == decimal_logical_column->scale() &&
           omnisci_column->columnType.is_decimal() &&
           (omnisci_column->columnType.get_compression() == kENCODING_NONE ||
            omnisci_column->columnType.get_compression() == kENCODING_FIXED);
  }
  return false;
}

bool validate_integral_mapping(const ColumnDescriptor* omnisci_column,
                               const parquet::ColumnDescriptor* parquet_column) {
  if (auto int_logical_column = dynamic_cast<const parquet::IntLogicalType*>(
          parquet_column->logical_type().get())) {
    if (int_logical_column->is_signed()) {  // signed
      if (omnisci_column->columnType.get_compression() == kENCODING_NONE) {
        switch (int_logical_column->bit_width()) {
          case 64:
            return omnisci_column->columnType.get_type() == kBIGINT;
          case 32:
            return omnisci_column->columnType.get_type() == kINT;
          case 16:
            return omnisci_column->columnType.get_type() == kSMALLINT;
          case 8:
            return omnisci_column->columnType.get_type() == kTINYINT;
          default:
            UNREACHABLE();
        }
      } else if (omnisci_column->columnType.get_compression() == kENCODING_FIXED) {
        return omnisci_column->columnType.get_comp_param() ==
               int_logical_column->bit_width();
      }
    } else {  // unsigned
      if (omnisci_column->columnType.get_compression() == kENCODING_NONE) {
        switch (int_logical_column->bit_width()) {
          case 64:
            return false;
          case 32:
            return omnisci_column->columnType.get_type() == kBIGINT;
          case 16:
            return omnisci_column->columnType.get_type() == kINT;
          case 8:
            return omnisci_column->columnType.get_type() == kSMALLINT;
          default:
            UNREACHABLE();
        }
      } else if (omnisci_column->columnType.get_compression() == kENCODING_FIXED) {
        switch (int_logical_column->bit_width()) {
          case 64:
          case 32:
            return false;
          case 16:
            return omnisci_column->columnType.get_comp_param() == 32;
          case 8:
            return omnisci_column->columnType.get_comp_param() == 16;
          default:
            UNREACHABLE();
        }
      }
    }
  }
  return false;
}

bool validate_none_type_mapping(const ColumnDescriptor* omnisci_column,
                                const parquet::ColumnDescriptor* parquet_column) {
  return parquet_column->logical_type()->is_none() &&
         omnisci_column->columnType.get_compression() == kENCODING_NONE &&
         ((parquet_column->physical_type() == parquet::Type::BOOLEAN &&
           omnisci_column->columnType.get_type() == kBOOLEAN) ||
          (parquet_column->physical_type() == parquet::Type::INT32 &&
           omnisci_column->columnType.get_type() == kINT) ||
          (parquet_column->physical_type() == parquet::Type::INT64 &&
           omnisci_column->columnType.get_type() == kBIGINT) ||
          (parquet_column->physical_type() == parquet::Type::FLOAT &&
           omnisci_column->columnType.get_type() == kFLOAT) ||
          (parquet_column->physical_type() == parquet::Type::DOUBLE &&
           omnisci_column->columnType.get_type() == kDOUBLE));
}

bool validate_timestamp_mapping(const ColumnDescriptor* omnisci_column,
                                const parquet::ColumnDescriptor* parquet_column) {
  if (!(omnisci_column->columnType.get_type() == kTIMESTAMP &&
        omnisci_column->columnType.get_compression() == kENCODING_NONE)) {
    return false;
  }
  if (auto timestamp_logical_column = dynamic_cast<const parquet::TimestampLogicalType*>(
          parquet_column->logical_type().get())) {
    if (!timestamp_logical_column->is_adjusted_to_utc()) {
      return false;
    }
    return omnisci_column->columnType.get_dimension() == 0 ||
           ((omnisci_column->columnType.get_dimension() == 9 &&
             timestamp_logical_column->time_unit() ==
                 parquet::LogicalType::TimeUnit::NANOS) ||
            (omnisci_column->columnType.get_dimension() == 6 &&
             timestamp_logical_column->time_unit() ==
                 parquet::LogicalType::TimeUnit::MICROS) ||
            (omnisci_column->columnType.get_dimension() == 3 &&
             timestamp_logical_column->time_unit() ==
                 parquet::LogicalType::TimeUnit::MILLIS));
  }
  return false;
}

bool validate_time_mapping(const ColumnDescriptor* omnisci_column,
                           const parquet::ColumnDescriptor* parquet_column) {
  if (!(omnisci_column->columnType.get_type() == kTIME &&
        (omnisci_column->columnType.get_compression() == kENCODING_NONE ||
         (omnisci_column->columnType.get_compression() == kENCODING_FIXED &&
          omnisci_column->columnType.get_comp_param() == 32)))) {
    return false;
  }
  if (auto time_logical_column = dynamic_cast<const parquet::TimeLogicalType*>(
          parquet_column->logical_type().get())) {
    if (!time_logical_column->is_adjusted_to_utc()) {
      return false;
    }
    return omnisci_column->columnType.get_compression() == kENCODING_NONE ||
           time_logical_column->time_unit() == parquet::LogicalType::TimeUnit::MILLIS;
  }
  return false;
}

bool validate_date_mapping(const ColumnDescriptor* omnisci_column,
                           const parquet::ColumnDescriptor* parquet_column) {
  if (!(omnisci_column->columnType.get_type() == kDATE &&
        (omnisci_column->columnType.get_compression() == kENCODING_DATE_IN_DAYS &&
         omnisci_column->columnType.get_comp_param() ==
             0)  // DATE ENCODING DAYS (32) specifies comp_param of 0
        )) {
    return false;
  }
  return parquet_column->logical_type()->is_date();
}

}  // namespace

bool LazyParquetChunkLoader::isColumnMappingSupported(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column) {
  if (validate_decimal_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  if (validate_integral_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  if (validate_none_type_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  if (validate_timestamp_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  if (validate_time_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  if (validate_date_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  return false;
}

LazyParquetChunkLoader::LazyParquetChunkLoader(const std::string& file_name)
    : file_name_(file_name)
    , def_levels_(batch_reader_num_elements)
    , rep_levels_(batch_reader_num_elements) {}

void LazyParquetChunkLoader::loadChunk(const Interval<RowGroupType>& row_group_interval,
                                       const int parquet_column_index,
                                       Chunk_NS::Chunk& chunk) {
  std::unique_ptr<parquet::arrow::FileReader> reader;
  open_parquet_table(file_name_, reader);

  int num_row_groups, num_columns;
  std::tie(num_row_groups, num_columns) = get_parquet_table_size(reader);
  CHECK(row_group_interval.start >= 0 && row_group_interval.end < num_row_groups);
  CHECK(parquet_column_index >= 0 && parquet_column_index < num_columns);

  auto column_descriptor = chunk.getColumnDesc();
  auto buffer = chunk.getBuffer();
  CHECK(buffer);

  auto parquet_column_descriptor = get_column_descriptor(reader, parquet_column_index);

  append_row_groups(row_group_interval,
                    parquet_column_index,
                    column_descriptor,
                    parquet_column_descriptor,
                    buffer,
                    reader->parquet_reader(),
                    def_levels_.data(),
                    rep_levels_.data());
}

}  // namespace foreign_storage
