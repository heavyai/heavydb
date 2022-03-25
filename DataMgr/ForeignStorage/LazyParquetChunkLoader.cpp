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
#include <parquet/statistics.h>
#include <parquet/types.h>

#include "ForeignStorageException.h"
#include "FsiChunkUtils.h"
#include "ParquetArrayDetectEncoder.h"
#include "ParquetArrayImportEncoder.h"
#include "ParquetDateInDaysFromTimestampEncoder.h"
#include "ParquetDateInSecondsEncoder.h"
#include "ParquetDecimalEncoder.h"
#include "ParquetDetectStringEncoder.h"
#include "ParquetFixedLengthArrayEncoder.h"
#include "ParquetFixedLengthEncoder.h"
#include "ParquetGeospatialEncoder.h"
#include "ParquetGeospatialImportEncoder.h"
#include "ParquetStringEncoder.h"
#include "ParquetStringImportEncoder.h"
#include "ParquetStringNoneEncoder.h"
#include "ParquetTimeEncoder.h"
#include "ParquetTimestampEncoder.h"
#include "ParquetVariableLengthArrayEncoder.h"
#include "Shared/misc.h"
#include "StringDictionary/StringDictionary.h"
#include "TypedParquetDetectBuffer.h"

namespace foreign_storage {

namespace {

bool within_range(int64_t lower_bound, int64_t upper_bound, int64_t value) {
  return value >= lower_bound && value <= upper_bound;
}

bool is_valid_parquet_string(const parquet::ColumnDescriptor* parquet_column) {
  return (parquet_column->logical_type()->is_none() &&
          parquet_column->physical_type() == parquet::Type::BYTE_ARRAY) ||
         parquet_column->logical_type()->is_string();
}

/**
 * @brief Detect a valid list parquet column.
 *
 * @param parquet_column - the parquet column descriptor of the column to
 * detect
 *
 * @return true if it is a valid parquet list column
 *
 * Note: the notion of a valid parquet list column is adapted from the parquet
 * schema specification for logical type definitions:
 *
 *    <list-repetition> group <name> (LIST) {
 *      repeated group list {
 *        <element-repetition> <element-type> element;
 *      }
 *    }
 *
 *  Testing has shown that there are small deviations from this specification in
 *  at least one library-- pyarrow-- where the innermost schema node is named
 *  "item" as opposed to "element".
 *
 *  The following is also true of the schema definition.
 *
 *    * The outer-most level must be a group annotated with LIST that contains a
 *      single field named list. The repetition of this level must be either
 *      optional or required and determines whether the list is nullable.
 *
 *    * The middle level, named list, must be a repeated group with a single field
 *      named element.
 *
 *    * The element field encodes the list's element type and
 *      repetition. Element repetition must be required or optional.
 *
 *  FSI further restricts lists to be defined only at the top level, meaning
 *  directly below the root schema node.
 */
bool is_valid_parquet_list_column(const parquet::ColumnDescriptor* parquet_column) {
  const parquet::schema::Node* node = parquet_column->schema_node().get();
  if ((node->name() != "element" && node->name() != "item") ||
      !(node->is_required() ||
        node->is_optional())) {  // ensure first innermost node is named "element"
                                 // which is required by the parquet specification;
                                 // however testing shows that pyarrow generates this
                                 // column with the name of "item"
                                 // this field must be either required or optional
    return false;
  }
  node = node->parent();
  if (!node) {  // required nested structure
    return false;
  }
  if (node->name() != "list" || !node->is_repeated() ||
      !node->is_group()) {  // ensure second innermost node is named "list" which is
                            // a repeated group; this is
                            // required by the parquet specification
    return false;
  }
  node = node->parent();
  if (!node) {  // required nested structure
    return false;
  }
  if (!node->logical_type()->is_list() ||
      !(node->is_optional() ||
        node->is_required())) {  // ensure third outermost node has logical type LIST
                                 // which is either optional or required; this is required
                                 // by the parquet specification
    return false;
  }
  node =
      node->parent();  // this must now be the root node of schema which is required by
                       // FSI (lists can not be embedded into a deeper nested structure)
  if (!node) {         // required nested structure
    return false;
  }
  node = node->parent();
  if (node) {  // implies the previous node was not the root node
    return false;
  }
  return true;
}

template <typename V, typename NullType>
std::shared_ptr<ParquetEncoder> create_parquet_decimal_encoder_with_omnisci_type(
    const ColumnDescriptor* column_descriptor,
    const parquet::ColumnDescriptor* parquet_column_descriptor,
    AbstractBuffer* buffer) {
  switch (parquet_column_descriptor->physical_type()) {
    case parquet::Type::INT32:
      return std::make_shared<ParquetDecimalEncoder<V, int32_t, NullType>>(
          buffer, column_descriptor, parquet_column_descriptor);
    case parquet::Type::INT64:
      return std::make_shared<ParquetDecimalEncoder<V, int64_t, NullType>>(
          buffer, column_descriptor, parquet_column_descriptor);
    case parquet::Type::FIXED_LEN_BYTE_ARRAY:
      return std::make_shared<
          ParquetDecimalEncoder<V, parquet::FixedLenByteArray, NullType>>(
          buffer, column_descriptor, parquet_column_descriptor);
    case parquet::Type::BYTE_ARRAY:
      return std::make_shared<ParquetDecimalEncoder<V, parquet::ByteArray, NullType>>(
          buffer, column_descriptor, parquet_column_descriptor);
    default:
      UNREACHABLE();
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_decimal_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan_or_for_import) {
  if (parquet_column->logical_type()->is_decimal()) {
    if (omnisci_column->columnType.get_compression() == kENCODING_NONE) {
      return create_parquet_decimal_encoder_with_omnisci_type<int64_t, int64_t>(
          omnisci_column, parquet_column, buffer);
    }
    CHECK(omnisci_column->columnType.get_compression() == kENCODING_FIXED);
    if (is_metadata_scan_or_for_import) {
      switch (omnisci_column->columnType.get_comp_param()) {
        case 16:
          return create_parquet_decimal_encoder_with_omnisci_type<int64_t, int16_t>(
              omnisci_column, parquet_column, buffer);
        case 32:
          return create_parquet_decimal_encoder_with_omnisci_type<int64_t, int32_t>(
              omnisci_column, parquet_column, buffer);
        default:
          UNREACHABLE();
      }
    } else {
      switch (omnisci_column->columnType.get_comp_param()) {
        case 16:
          return create_parquet_decimal_encoder_with_omnisci_type<int16_t, int16_t>(
              omnisci_column, parquet_column, buffer);
        case 32:
          return create_parquet_decimal_encoder_with_omnisci_type<int32_t, int32_t>(
              omnisci_column, parquet_column, buffer);
        default:
          UNREACHABLE();
      }
    }
  }
  return {};
}

/**
 * @brief Create a signed or unsigned integral parquet encoder using types.
 *
 * @param buffer - buffer used within the encoder
 * @param omnisci_data_type_byte_size - size in number of bytes of OmniSci type
 * @param parquet_data_type_byte_size - size in number of bytes of Parquet physical type
 * @param is_signed - flag indicating if Parquet column is signed
 *
 * @return a std::shared_ptr to an integral encoder
 *
 * See the documentation for ParquetFixedLengthEncoder and
 * ParquetUnsignedFixedLengthEncoder for a description of the semantics of the
 * templated types `V`, `T`, `U`, and `NullType`.
 */
template <typename V, typename T, typename U, typename NullType>
std::shared_ptr<ParquetEncoder>
create_parquet_signed_or_unsigned_integral_encoder_with_types(
    AbstractBuffer* buffer,
    const size_t omnisci_data_type_byte_size,
    const size_t parquet_data_type_byte_size,
    const bool is_signed) {
  CHECK(sizeof(NullType) == omnisci_data_type_byte_size);
  if (is_signed) {
    return std::make_shared<ParquetFixedLengthEncoder<V, T, NullType>>(
        buffer, omnisci_data_type_byte_size, parquet_data_type_byte_size);
  } else {
    return std::make_shared<ParquetUnsignedFixedLengthEncoder<V, T, U, NullType>>(
        buffer, omnisci_data_type_byte_size, parquet_data_type_byte_size);
  }
}

/**
 * @brief Create a integral parquet encoder using types.
 *
 * @param buffer - buffer used within the encoder
 * @param omnisci_data_type_byte_size - size in number of bytes of OmniSci type
 * @param parquet_data_type_byte_size - size in number of bytes of Parquet physical type
 * @param bit_width - bit width specified for the Parquet column
 * @param is_signed - flag indicating if Parquet column is signed
 *
 * @return a std::shared_ptr to an integral encoder
 *
 * See the documentation for ParquetFixedLengthEncoder and
 * ParquetUnsignedFixedLengthEncoder for a description of the semantics of the
 * templated type `V` and `NullType`.
 *
 * Note, this function determines the appropriate bit depth integral encoder to
 * create, while `create_parquet_signed_or_unsigned_integral_encoder_with_types`
 * determines whether to create a signed or unsigned integral encoder.
 */
template <typename V, typename NullType>
std::shared_ptr<ParquetEncoder> create_parquet_integral_encoder_with_omnisci_type(
    AbstractBuffer* buffer,
    const size_t omnisci_data_type_byte_size,
    const size_t parquet_data_type_byte_size,
    const int bit_width,
    const bool is_signed) {
  switch (bit_width) {
    case 8:
      return create_parquet_signed_or_unsigned_integral_encoder_with_types<V,
                                                                           int32_t,
                                                                           uint8_t,
                                                                           NullType>(
          buffer, omnisci_data_type_byte_size, parquet_data_type_byte_size, is_signed);
    case 16:
      return create_parquet_signed_or_unsigned_integral_encoder_with_types<V,
                                                                           int32_t,
                                                                           uint16_t,
                                                                           NullType>(
          buffer, omnisci_data_type_byte_size, parquet_data_type_byte_size, is_signed);
    case 32:
      return create_parquet_signed_or_unsigned_integral_encoder_with_types<V,
                                                                           int32_t,
                                                                           uint32_t,
                                                                           NullType>(
          buffer, omnisci_data_type_byte_size, parquet_data_type_byte_size, is_signed);
    case 64:
      return create_parquet_signed_or_unsigned_integral_encoder_with_types<V,
                                                                           int64_t,
                                                                           uint64_t,
                                                                           NullType>(
          buffer, omnisci_data_type_byte_size, parquet_data_type_byte_size, is_signed);
    default:
      UNREACHABLE();
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_integral_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan_or_for_import) {
  auto column_type = omnisci_column->columnType;
  auto physical_type = parquet_column->physical_type();

  int bit_width = -1;
  int is_signed = false;
  // handle the integral case with no Parquet annotation
  if (parquet_column->logical_type()->is_none() && column_type.is_integer()) {
    if (physical_type == parquet::Type::INT32) {
      bit_width = 32;
    } else if (physical_type == parquet::Type::INT64) {
      bit_width = 64;
    } else {
      UNREACHABLE();
    }
    is_signed = true;
  }
  // handle the integral case with Parquet annotation
  if (auto int_logical_column = dynamic_cast<const parquet::IntLogicalType*>(
          parquet_column->logical_type().get())) {
    bit_width = int_logical_column->bit_width();
    is_signed = int_logical_column->is_signed();
  }

  if (bit_width == -1) {  // no valid logical type (with or without annotation) found
    return {};
  }

  const size_t omnisci_data_type_byte_size = column_type.get_size();
  const size_t parquet_data_type_byte_size = parquet::GetTypeByteSize(physical_type);

  switch (omnisci_data_type_byte_size) {
    case 8:
      CHECK(column_type.get_compression() == kENCODING_NONE);
      return create_parquet_integral_encoder_with_omnisci_type<int64_t, int64_t>(
          buffer,
          omnisci_data_type_byte_size,
          parquet_data_type_byte_size,
          bit_width,
          is_signed);
    case 4:
      if (is_metadata_scan_or_for_import && column_type.get_type() == kBIGINT) {
        return create_parquet_integral_encoder_with_omnisci_type<int64_t, int32_t>(
            buffer,
            omnisci_data_type_byte_size,
            parquet_data_type_byte_size,
            bit_width,
            is_signed);
      }
      return create_parquet_integral_encoder_with_omnisci_type<int32_t, int32_t>(
          buffer,
          omnisci_data_type_byte_size,
          parquet_data_type_byte_size,
          bit_width,
          is_signed);
    case 2:
      if (is_metadata_scan_or_for_import) {
        switch (column_type.get_type()) {
          case kBIGINT:
            return create_parquet_integral_encoder_with_omnisci_type<int64_t, int16_t>(
                buffer,
                omnisci_data_type_byte_size,
                parquet_data_type_byte_size,
                bit_width,
                is_signed);
          case kINT:
            return create_parquet_integral_encoder_with_omnisci_type<int32_t, int16_t>(
                buffer,
                omnisci_data_type_byte_size,
                parquet_data_type_byte_size,
                bit_width,
                is_signed);
          case kSMALLINT:
            break;
          default:
            UNREACHABLE();
        }
      }
      return create_parquet_integral_encoder_with_omnisci_type<int16_t, int16_t>(
          buffer,
          omnisci_data_type_byte_size,
          parquet_data_type_byte_size,
          bit_width,
          is_signed);
    case 1:
      if (is_metadata_scan_or_for_import) {
        switch (column_type.get_type()) {
          case kBIGINT:
            return create_parquet_integral_encoder_with_omnisci_type<int64_t, int8_t>(
                buffer,
                omnisci_data_type_byte_size,
                parquet_data_type_byte_size,
                bit_width,
                is_signed);
          case kINT:
            return create_parquet_integral_encoder_with_omnisci_type<int32_t, int8_t>(
                buffer,
                omnisci_data_type_byte_size,
                parquet_data_type_byte_size,
                bit_width,
                is_signed);
          case kSMALLINT:
            return create_parquet_integral_encoder_with_omnisci_type<int16_t, int8_t>(
                buffer,
                omnisci_data_type_byte_size,
                parquet_data_type_byte_size,
                bit_width,
                is_signed);
          case kTINYINT:
            break;
          default:
            UNREACHABLE();
        }
      }
      return create_parquet_integral_encoder_with_omnisci_type<int8_t, int8_t>(
          buffer,
          omnisci_data_type_byte_size,
          parquet_data_type_byte_size,
          bit_width,
          is_signed);
    default:
      UNREACHABLE();
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_floating_point_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  auto column_type = omnisci_column->columnType;
  if (!column_type.is_fp()) {
    return {};
  }
  CHECK_EQ(column_type.get_compression(), kENCODING_NONE);
  switch (column_type.get_type()) {
    case kFLOAT:
      switch (parquet_column->physical_type()) {
        case parquet::Type::FLOAT:
          return std::make_shared<ParquetFixedLengthEncoder<float, float>>(
              buffer, omnisci_column, parquet_column);
        case parquet::Type::DOUBLE:
          return std::make_shared<ParquetFixedLengthEncoder<float, double>>(
              buffer, omnisci_column, parquet_column);
        default:
          UNREACHABLE();
      }
    case kDOUBLE:
      CHECK(parquet_column->physical_type() == parquet::Type::DOUBLE);
      return std::make_shared<ParquetFixedLengthEncoder<double, double>>(
          buffer, omnisci_column, parquet_column);
    default:
      UNREACHABLE();
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_none_type_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  auto column_type = omnisci_column->columnType;
  if (parquet_column->logical_type()->is_none() &&
      !omnisci_column->columnType.is_string()) {  // boolean
    if (column_type.get_compression() == kENCODING_NONE) {
      switch (column_type.get_type()) {
        case kBOOLEAN:
          return std::make_shared<ParquetFixedLengthEncoder<int8_t, bool>>(
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

template <typename V, typename T, typename NullType>
std::shared_ptr<ParquetEncoder> create_parquet_timestamp_encoder_with_types(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  if (auto timestamp_logical_type = dynamic_cast<const parquet::TimestampLogicalType*>(
          parquet_column->logical_type().get())) {
    switch (timestamp_logical_type->time_unit()) {
      case parquet::LogicalType::TimeUnit::MILLIS:
        return std::make_shared<ParquetTimestampEncoder<V, T, 1000L, NullType>>(
            buffer, omnisci_column, parquet_column);
      case parquet::LogicalType::TimeUnit::MICROS:
        return std::make_shared<ParquetTimestampEncoder<V, T, 1000L * 1000L, NullType>>(
            buffer, omnisci_column, parquet_column);
      case parquet::LogicalType::TimeUnit::NANOS:
        return std::make_shared<
            ParquetTimestampEncoder<V, T, 1000L * 1000L * 1000L, NullType>>(
            buffer, omnisci_column, parquet_column);
      default:
        UNREACHABLE();
    }
  } else {
    UNREACHABLE();
  }
  return {};
}

template <typename V, typename T, typename NullType>
std::shared_ptr<ParquetEncoder> create_parquet_date_from_timestamp_encoder_with_types(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan_or_for_import) {
  if (auto timestamp_logical_type = dynamic_cast<const parquet::TimestampLogicalType*>(
          parquet_column->logical_type().get())) {
    switch (timestamp_logical_type->time_unit()) {
      case parquet::LogicalType::TimeUnit::MILLIS:
        if (is_metadata_scan_or_for_import) {
          return std::make_shared<
              ParquetDateInSecondsFromTimestampEncoder<V, T, 1000L, NullType>>(
              buffer, omnisci_column, parquet_column);
        }
        return std::make_shared<
            ParquetDateInDaysFromTimestampEncoder<V, T, 1000L, NullType>>(
            buffer, omnisci_column, parquet_column);
      case parquet::LogicalType::TimeUnit::MICROS:
        if (is_metadata_scan_or_for_import) {
          return std::make_shared<
              ParquetDateInSecondsFromTimestampEncoder<V, T, 1000L * 1000L, NullType>>(
              buffer, omnisci_column, parquet_column);
        }
        return std::make_shared<
            ParquetDateInDaysFromTimestampEncoder<V, T, 1000L * 1000L, NullType>>(
            buffer, omnisci_column, parquet_column);
      case parquet::LogicalType::TimeUnit::NANOS:
        if (is_metadata_scan_or_for_import) {
          return std::make_shared<
              ParquetDateInSecondsFromTimestampEncoder<V,
                                                       T,
                                                       1000L * 1000L * 1000L,
                                                       NullType>>(
              buffer, omnisci_column, parquet_column);
        }
        return std::make_shared<
            ParquetDateInDaysFromTimestampEncoder<V, T, 1000L * 1000L * 1000L, NullType>>(
            buffer, omnisci_column, parquet_column);
      default:
        UNREACHABLE();
    }
  } else {
    UNREACHABLE();
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_timestamp_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan_or_for_import) {
  auto column_type = omnisci_column->columnType;
  auto precision = column_type.get_precision();
  if (parquet_column->logical_type()->is_timestamp()) {
    if (column_type.get_compression() == kENCODING_NONE) {
      if (precision == 0) {
        return create_parquet_timestamp_encoder_with_types<int64_t, int64_t, int64_t>(
            omnisci_column, parquet_column, buffer);
      } else {
        return std::make_shared<ParquetFixedLengthEncoder<int64_t, int64_t, int64_t>>(
            buffer, omnisci_column, parquet_column);
      }
    } else if (column_type.get_compression() == kENCODING_FIXED) {
      CHECK(column_type.get_comp_param() == 32);
      if (is_metadata_scan_or_for_import) {
        return create_parquet_timestamp_encoder_with_types<int64_t, int64_t, int32_t>(
            omnisci_column, parquet_column, buffer);
      } else {
        return create_parquet_timestamp_encoder_with_types<int32_t, int64_t, int32_t>(
            omnisci_column, parquet_column, buffer);
      }
    }
  } else if (parquet_column->logical_type()->is_none() && column_type.is_timestamp()) {
    if (parquet_column->physical_type() == parquet::Type::INT32) {
      CHECK(column_type.get_compression() == kENCODING_FIXED &&
            column_type.get_comp_param() == 32);
      if (is_metadata_scan_or_for_import) {
        return std::make_shared<ParquetFixedLengthEncoder<int64_t, int32_t, int32_t>>(
            buffer, omnisci_column, parquet_column);
      } else {
        return std::make_shared<ParquetFixedLengthEncoder<int32_t, int32_t, int32_t>>(
            buffer, omnisci_column, parquet_column);
      }
    } else if (parquet_column->physical_type() == parquet::Type::INT64) {
      if (column_type.get_compression() == kENCODING_NONE) {
        return std::make_shared<ParquetFixedLengthEncoder<int64_t, int64_t, int64_t>>(
            buffer, omnisci_column, parquet_column);
      } else if (column_type.get_compression() == kENCODING_FIXED) {
        CHECK(column_type.get_comp_param() == 32);
        if (is_metadata_scan_or_for_import) {
          return std::make_shared<ParquetFixedLengthEncoder<int64_t, int64_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        } else {
          return std::make_shared<ParquetFixedLengthEncoder<int32_t, int64_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        }
      }
    } else {
      UNREACHABLE();
    }
  }
  return {};
}

template <typename V, typename T, typename NullType>
std::shared_ptr<ParquetEncoder> create_parquet_time_encoder_with_types(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  if (auto time_logical_type = dynamic_cast<const parquet::TimeLogicalType*>(
          parquet_column->logical_type().get())) {
    switch (time_logical_type->time_unit()) {
      case parquet::LogicalType::TimeUnit::MILLIS:
        return std::make_shared<ParquetTimeEncoder<V, T, 1000L, NullType>>(
            buffer, omnisci_column, parquet_column);
      case parquet::LogicalType::TimeUnit::MICROS:
        return std::make_shared<ParquetTimeEncoder<V, T, 1000L * 1000L, NullType>>(
            buffer, omnisci_column, parquet_column);
      case parquet::LogicalType::TimeUnit::NANOS:
        return std::make_shared<
            ParquetTimeEncoder<V, T, 1000L * 1000L * 1000L, NullType>>(
            buffer, omnisci_column, parquet_column);
      default:
        UNREACHABLE();
    }
  } else {
    UNREACHABLE();
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_time_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan_or_for_import) {
  auto column_type = omnisci_column->columnType;
  if (auto time_logical_column = dynamic_cast<const parquet::TimeLogicalType*>(
          parquet_column->logical_type().get())) {
    if (column_type.get_compression() == kENCODING_NONE) {
      if (time_logical_column->time_unit() == parquet::LogicalType::TimeUnit::MILLIS) {
        return create_parquet_time_encoder_with_types<int64_t, int32_t, int64_t>(
            omnisci_column, parquet_column, buffer);
      } else {
        return create_parquet_time_encoder_with_types<int64_t, int64_t, int64_t>(
            omnisci_column, parquet_column, buffer);
      }
    } else if (column_type.get_compression() == kENCODING_FIXED) {
      if (is_metadata_scan_or_for_import) {
        if (time_logical_column->time_unit() == parquet::LogicalType::TimeUnit::MILLIS) {
          CHECK(parquet_column->physical_type() == parquet::Type::INT32);
          return create_parquet_time_encoder_with_types<int64_t, int32_t, int32_t>(
              omnisci_column, parquet_column, buffer);
        } else {
          CHECK(time_logical_column->time_unit() ==
                    parquet::LogicalType::TimeUnit::MICROS ||
                time_logical_column->time_unit() ==
                    parquet::LogicalType::TimeUnit::NANOS);
          CHECK(parquet_column->physical_type() == parquet::Type::INT64);
          return create_parquet_time_encoder_with_types<int64_t, int64_t, int32_t>(
              omnisci_column, parquet_column, buffer);
        }
      } else {
        if (time_logical_column->time_unit() == parquet::LogicalType::TimeUnit::MILLIS) {
          CHECK(parquet_column->physical_type() == parquet::Type::INT32);
          return create_parquet_time_encoder_with_types<int32_t, int32_t, int32_t>(
              omnisci_column, parquet_column, buffer);
        } else {
          CHECK(time_logical_column->time_unit() ==
                    parquet::LogicalType::TimeUnit::MICROS ||
                time_logical_column->time_unit() ==
                    parquet::LogicalType::TimeUnit::NANOS);
          CHECK(parquet_column->physical_type() == parquet::Type::INT64);
          return create_parquet_time_encoder_with_types<int32_t, int64_t, int32_t>(
              omnisci_column, parquet_column, buffer);
        }
      }
    } else {
      UNREACHABLE();
    }
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_date_from_timestamp_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan_or_for_import) {
  auto column_type = omnisci_column->columnType;
  if (parquet_column->logical_type()->is_timestamp() && column_type.is_date()) {
    CHECK(column_type.get_compression() == kENCODING_DATE_IN_DAYS);
    if (is_metadata_scan_or_for_import) {
      if (column_type.get_comp_param() ==
          0) {  // DATE ENCODING FIXED (32) uses comp param 0
        return create_parquet_date_from_timestamp_encoder_with_types<int64_t,
                                                                     int64_t,
                                                                     int32_t>(
            omnisci_column, parquet_column, buffer, true);
      } else if (column_type.get_comp_param() == 16) {
        return create_parquet_date_from_timestamp_encoder_with_types<int64_t,
                                                                     int64_t,
                                                                     int16_t>(
            omnisci_column, parquet_column, buffer, true);
      } else {
        UNREACHABLE();
      }
    } else {
      if (column_type.get_comp_param() ==
          0) {  // DATE ENCODING FIXED (32) uses comp param 0
        return create_parquet_date_from_timestamp_encoder_with_types<int32_t,
                                                                     int64_t,
                                                                     int32_t>(
            omnisci_column, parquet_column, buffer, false);
      } else if (column_type.get_comp_param() == 16) {
        return create_parquet_date_from_timestamp_encoder_with_types<int16_t,
                                                                     int64_t,
                                                                     int16_t>(
            omnisci_column, parquet_column, buffer, false);
      } else {
        UNREACHABLE();
      }
    }
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_date_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan_or_for_import) {
  auto column_type = omnisci_column->columnType;
  if (parquet_column->logical_type()->is_date() && column_type.is_date()) {
    if (column_type.get_compression() == kENCODING_DATE_IN_DAYS) {
      if (is_metadata_scan_or_for_import) {
        if (column_type.get_comp_param() ==
            0) {  // DATE ENCODING FIXED (32) uses comp param 0
          return std::make_shared<ParquetDateInSecondsEncoder</*NullType=*/int32_t>>(
              buffer);
        } else if (column_type.get_comp_param() == 16) {
          return std::make_shared<ParquetDateInSecondsEncoder</*NullType=*/int16_t>>(
              buffer);
        } else {
          UNREACHABLE();
        }
      } else {
        if (column_type.get_comp_param() ==
            0) {  // DATE ENCODING FIXED (32) uses comp param 0
          return std::make_shared<ParquetFixedLengthEncoder<int32_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        } else if (column_type.get_comp_param() == 16) {
          return std::make_shared<ParquetFixedLengthEncoder<int16_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        } else {
          UNREACHABLE();
        }
      }
    } else if (column_type.get_compression() == kENCODING_NONE) {  // for array types
      return std::make_shared<ParquetDateInSecondsEncoder</*NullType=*/int64_t>>(
          buffer, omnisci_column, parquet_column);
    } else {
      UNREACHABLE();
    }
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_string_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    const Chunk_NS::Chunk& chunk,
    StringDictionary* string_dictionary,
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
    bool is_for_import,
    const bool is_for_detect) {
  auto column_type = omnisci_column->columnType;
  if (!is_valid_parquet_string(parquet_column) ||
      !omnisci_column->columnType.is_string()) {
    return {};
  }
  if (column_type.get_compression() == kENCODING_NONE) {
    if (is_for_import) {
      return std::make_shared<ParquetStringImportEncoder>(chunk.getBuffer());
    } else {
      return std::make_shared<ParquetStringNoneEncoder>(chunk.getBuffer(),
                                                        chunk.getIndexBuf());
    }
  } else if (column_type.get_compression() == kENCODING_DICT) {
    if (!is_for_detect) {  // non-detect use case
      chunk_metadata.emplace_back(std::make_unique<ChunkMetadata>());
      std::unique_ptr<ChunkMetadata>& logical_chunk_metadata = chunk_metadata.back();
      logical_chunk_metadata->sqlType = omnisci_column->columnType;
      switch (column_type.get_size()) {
        case 1:
          return std::make_shared<ParquetStringEncoder<uint8_t>>(
              chunk.getBuffer(),
              string_dictionary,
              is_for_import ? nullptr : logical_chunk_metadata.get());
        case 2:
          return std::make_shared<ParquetStringEncoder<uint16_t>>(
              chunk.getBuffer(),
              string_dictionary,
              is_for_import ? nullptr : logical_chunk_metadata.get());
        case 4:
          return std::make_shared<ParquetStringEncoder<int32_t>>(
              chunk.getBuffer(),
              string_dictionary,
              is_for_import ? nullptr : logical_chunk_metadata.get());
        default:
          UNREACHABLE();
      }
    } else {  // detect use-case
      return std::make_shared<ParquetDetectStringEncoder>(chunk.getBuffer());
    }
  } else {
    UNREACHABLE();
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_geospatial_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    std::list<Chunk_NS::Chunk>& chunks,
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
    const RenderGroupAnalyzerMap* render_group_analyzer_map,
    const bool is_metadata_scan,
    const bool is_for_import) {
  auto column_type = omnisci_column->columnType;
  if (!is_valid_parquet_string(parquet_column) || !column_type.is_geometry()) {
    return {};
  }
  if (is_for_import) {
    return std::make_shared<ParquetGeospatialImportEncoder>(chunks);  // no RGAMap
  }
  if (is_metadata_scan) {
    return std::make_shared<ParquetGeospatialEncoder>(render_group_analyzer_map);
  }
  for (auto chunks_iter = chunks.begin(); chunks_iter != chunks.end(); ++chunks_iter) {
    chunk_metadata.emplace_back(std::make_unique<ChunkMetadata>());
    auto& chunk_metadata_ptr = chunk_metadata.back();
    chunk_metadata_ptr->sqlType = chunks_iter->getColumnDesc()->columnType;
  }
  return std::make_shared<ParquetGeospatialEncoder>(
      parquet_column, chunks, chunk_metadata, render_group_analyzer_map);
}

// forward declare `create_parquet_array_encoder`: `create_parquet_encoder` and
// `create_parquet_array_encoder` each make use of each other, so
// one of the two functions must have a forward declaration
std::shared_ptr<ParquetEncoder> create_parquet_array_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    std::list<Chunk_NS::Chunk>& chunks,
    StringDictionary* string_dictionary,
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
    const bool is_metadata_scan,
    const bool is_for_import,
    const bool is_for_detect);

/**
 * @brief Create a Parquet specific encoder for a Parquet to OmniSci mapping.
 *
 * @param omnisci_column - the descriptor of OmniSci column
 * @param parquet_column - the descriptor of Parquet column
 * @param chunks - list of chunks to populate (the case of more than one chunk
 * happens only if a logical column expands to multiple physical columns)
 * @param string_dictionary - string dictionary used in encoding for string dictionary
 * encoded columns
 * @param chunk_metadata - similar to the list of chunks, a list of chunk metadata that is
 * populated
 * @param is_metadata_scan - a flag indicating if the encoders created should be for a
 * metadata scan
 * @param is_for_import - a flag indicating if the encoders created should be
 * for import
 *
 *
 * @return  An appropriate Parquet encoder for the use case defined by the Parquet to
 * OmniSci mapping.
 *
 * Notes:
 *
 * - In the case of a metadata scan, the type of the encoder created may
 *   significantly change (for example in bit width.) This is because it is
 *   common for OmniSci to store metadata in a different format altogether
 *   than the data itself (see for example FixedLengthEncoder.)
 * - This function and the function `isColumnMappingSupported` work in
 *   conjunction with each other. For example, once a mapping is known to be
 *   allowed (since `isColumnMappingSupported` returned true) this
 *   function does not have to check many corner cases exhaustively as it would
 *   be redundant with what was checked in `isColumnMappingSupported`.
 */
std::shared_ptr<ParquetEncoder> create_parquet_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    std::list<Chunk_NS::Chunk>& chunks,
    StringDictionary* string_dictionary,
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
    const RenderGroupAnalyzerMap* render_group_analyzer_map,
    const bool is_metadata_scan = false,
    const bool is_for_import = false,
    const bool is_for_detect = false) {
  CHECK(!(is_metadata_scan && is_for_import));
  auto buffer = chunks.empty() ? nullptr : chunks.begin()->getBuffer();
  if (auto encoder = create_parquet_geospatial_encoder(omnisci_column,
                                                       parquet_column,
                                                       chunks,
                                                       chunk_metadata,
                                                       render_group_analyzer_map,
                                                       is_metadata_scan,
                                                       is_for_import)) {
    return encoder;
  }
  if (auto encoder = create_parquet_array_encoder(omnisci_column,
                                                  parquet_column,
                                                  chunks,
                                                  string_dictionary,
                                                  chunk_metadata,
                                                  is_metadata_scan,
                                                  is_for_import,
                                                  is_for_detect)) {
    return encoder;
  }
  if (auto encoder = create_parquet_decimal_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan || is_for_import)) {
    return encoder;
  }
  if (auto encoder = create_parquet_integral_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan || is_for_import)) {
    return encoder;
  }
  if (auto encoder =
          create_parquet_floating_point_encoder(omnisci_column, parquet_column, buffer)) {
    return encoder;
  }
  if (auto encoder = create_parquet_timestamp_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan || is_for_import)) {
    return encoder;
  }
  if (auto encoder =
          create_parquet_none_type_encoder(omnisci_column, parquet_column, buffer)) {
    return encoder;
  }
  if (auto encoder = create_parquet_time_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan || is_for_import)) {
    return encoder;
  }
  if (auto encoder = create_parquet_date_from_timestamp_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan || is_for_import)) {
    return encoder;
  }
  if (auto encoder = create_parquet_date_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan || is_for_import)) {
    return encoder;
  }
  if (auto encoder = create_parquet_string_encoder(
          omnisci_column,
          parquet_column,
          chunks.empty() ? Chunk_NS::Chunk{} : *chunks.begin(),
          string_dictionary,
          chunk_metadata,
          is_for_import,
          is_for_detect)) {
    return encoder;
  }
  UNREACHABLE();
  return {};
}

/**
 * Intended to be used for the import case.
 */
std::shared_ptr<ParquetEncoder> create_parquet_encoder_for_import(
    std::list<Chunk_NS::Chunk>& chunks,
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    StringDictionary* string_dictionary,
    const RenderGroupAnalyzerMap* render_group_analyzer_map) {
  std::list<std::unique_ptr<ChunkMetadata>> chunk_metadata;
  return create_parquet_encoder(omnisci_column,
                                parquet_column,
                                chunks,
                                string_dictionary,
                                chunk_metadata,
                                render_group_analyzer_map,
                                false,
                                true);
}

/**
 * Intended to be used only with metadata scan. Creates an incomplete encoder
 * capable of updating metadata.
 */
std::shared_ptr<ParquetEncoder> create_parquet_encoder_for_metadata_scan(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    const RenderGroupAnalyzerMap* render_group_analyzer_map) {
  std::list<Chunk_NS::Chunk> chunks;
  std::list<std::unique_ptr<ChunkMetadata>> chunk_metadata;
  return create_parquet_encoder(omnisci_column,
                                parquet_column,
                                chunks,
                                nullptr,
                                chunk_metadata,
                                render_group_analyzer_map,
                                true);
}

std::shared_ptr<ParquetEncoder> create_parquet_array_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    std::list<Chunk_NS::Chunk>& chunks,
    StringDictionary* string_dictionary,
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
    const bool is_metadata_scan,
    const bool is_for_import,
    const bool is_for_detect) {
  bool is_valid_parquet_list = is_valid_parquet_list_column(parquet_column);
  if (!is_valid_parquet_list || !omnisci_column->columnType.is_array()) {
    return {};
  }
  std::unique_ptr<ColumnDescriptor> omnisci_column_sub_type_column =
      get_sub_type_column_descriptor(omnisci_column);
  auto encoder = create_parquet_encoder(omnisci_column_sub_type_column.get(),
                                        parquet_column,
                                        chunks,
                                        string_dictionary,
                                        chunk_metadata,
                                        nullptr,
                                        is_metadata_scan,
                                        is_for_import,
                                        is_for_detect);
  CHECK(encoder.get());
  auto scalar_encoder = std::dynamic_pointer_cast<ParquetScalarEncoder>(encoder);
  CHECK(scalar_encoder);
  if (!is_for_import) {
    if (!is_for_detect) {
      if (omnisci_column->columnType.is_fixlen_array()) {
        encoder = std::make_shared<ParquetFixedLengthArrayEncoder>(
            is_metadata_scan ? nullptr : chunks.begin()->getBuffer(),
            scalar_encoder,
            omnisci_column);
      } else {
        encoder = std::make_shared<ParquetVariableLengthArrayEncoder>(
            is_metadata_scan ? nullptr : chunks.begin()->getBuffer(),
            is_metadata_scan ? nullptr : chunks.begin()->getIndexBuf(),
            scalar_encoder,
            omnisci_column);
      }
    } else {  // is_for_detect
      encoder = std::make_shared<ParquetArrayDetectEncoder>(
          chunks.begin()->getBuffer(), scalar_encoder, omnisci_column);
    }
  } else {  // is_for_import
    encoder = std::make_shared<ParquetArrayImportEncoder>(
        chunks.begin()->getBuffer(), scalar_encoder, omnisci_column);
  }
  return encoder;
}

void validate_definition_levels(
    const parquet::ParquetFileReader* reader,
    const int row_group_index,
    const int column_index,
    const int16_t* def_levels,
    const int64_t num_levels,
    const parquet::ColumnDescriptor* parquet_column_descriptor) {
  bool is_valid_parquet_list = is_valid_parquet_list_column(parquet_column_descriptor);
  if (!is_valid_parquet_list) {
    return;
  }
  std::unique_ptr<parquet::RowGroupMetaData> group_metadata =
      reader->metadata()->RowGroup(row_group_index);
  auto column_metadata = group_metadata->ColumnChunk(column_index);
  auto stats = validate_and_get_column_metadata_statistics(column_metadata.get());
  if (!stats->HasMinMax()) {
    auto find_it = std::find_if(def_levels,
                                def_levels + num_levels,
                                [](const int16_t def_level) { return def_level == 3; });
    if (find_it != def_levels + num_levels) {
      throw std::runtime_error(
          "No minimum and maximum statistic set in list column but non-null & non-empty "
          "array/value detected.");
    }
  }
}

void validate_max_repetition_and_definition_level(
    const ColumnDescriptor* omnisci_column_descriptor,
    const parquet::ColumnDescriptor* parquet_column_descriptor) {
  bool is_valid_parquet_list = is_valid_parquet_list_column(parquet_column_descriptor);
  if (is_valid_parquet_list && !omnisci_column_descriptor->columnType.is_array()) {
    throw std::runtime_error(
        "Unsupported mapping detected. Column '" + parquet_column_descriptor->name() +
        "' detected to be a parquet list but HeavyDB mapped column '" +
        omnisci_column_descriptor->columnName + "' is not an array.");
  }
  if (is_valid_parquet_list) {
    if (parquet_column_descriptor->max_repetition_level() != 1 ||
        parquet_column_descriptor->max_definition_level() != 3) {
      throw std::runtime_error(
          "Incorrect schema max repetition level detected in column '" +
          parquet_column_descriptor->name() +
          "'. Expected a max repetition level of 1 and max definition level of 3 for "
          "list column but column has a max "
          "repetition level of " +
          std::to_string(parquet_column_descriptor->max_repetition_level()) +
          " and a max definition level of " +
          std::to_string(parquet_column_descriptor->max_definition_level()) + ".");
    }
  } else {
    if (parquet_column_descriptor->max_repetition_level() != 0 ||
        parquet_column_descriptor->max_definition_level() != 1) {
      throw std::runtime_error(
          "Incorrect schema max repetition level detected in column '" +
          parquet_column_descriptor->name() +
          "'. Expected a max repetition level of 0 and max definition level of 1 for "
          "flat column but column has a max "
          "repetition level of " +
          std::to_string(parquet_column_descriptor->max_repetition_level()) +
          " and a max definition level of " +
          std::to_string(parquet_column_descriptor->max_definition_level()) + ".");
    }
  }
}

void resize_values_buffer(const ColumnDescriptor* omnisci_column,
                          const parquet::ColumnDescriptor* parquet_column,
                          std::vector<int8_t>& values) {
  auto max_type_byte_size =
      std::max(omnisci_column->columnType.get_size(),
               parquet::GetTypeByteSize(parquet_column->physical_type()));
  size_t values_size =
      LazyParquetChunkLoader::batch_reader_num_elements * max_type_byte_size;
  values.resize(values_size);
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

SQLTypeInfo suggest_decimal_mapping(const parquet::ColumnDescriptor* parquet_column) {
  if (auto decimal_logical_column = dynamic_cast<const parquet::DecimalLogicalType*>(
          parquet_column->logical_type().get())) {
    auto parquet_precision = decimal_logical_column->precision();
    auto parquet_scale = decimal_logical_column->scale();
    if (parquet_precision > 18) {
      throw ForeignStorageException(
          "Parquet column \"" + parquet_column->ToString() +
          "\" has decimal precision of " + std::to_string(parquet_precision) +
          " which is too high to import, maximum precision supported is 18.");
    }
    SQLTypeInfo type;
    type.set_type(kDECIMAL);
    type.set_compression(kENCODING_NONE);
    type.set_precision(parquet_precision);
    type.set_scale(parquet_scale);
    type.set_fixed_size();
    return type;
  }
  UNREACHABLE()
      << " a Parquet column's decimal logical type failed to be read appropriately";
  return {};
}

bool validate_floating_point_mapping(const ColumnDescriptor* omnisci_column,
                                     const parquet::ColumnDescriptor* parquet_column) {
  if (!omnisci_column->columnType.is_fp()) {
    return false;
  }
  // check if mapping is a valid coerced or non-coerced floating point mapping
  // with no annotation (floating point columns have no annotation in the
  // Parquet specification)
  if (omnisci_column->columnType.get_compression() == kENCODING_NONE) {
    return (parquet_column->physical_type() == parquet::Type::DOUBLE) ||
           (parquet_column->physical_type() == parquet::Type::FLOAT &&
            omnisci_column->columnType.get_type() == kFLOAT);
  }
  return false;
}

SQLTypeInfo suggest_floating_point_mapping(
    const parquet::ColumnDescriptor* parquet_column) {
  SQLTypeInfo type;
  if (parquet_column->physical_type() == parquet::Type::FLOAT) {
    type.set_type(kFLOAT);
  } else if (parquet_column->physical_type() == parquet::Type::DOUBLE) {
    type.set_type(kDOUBLE);
  } else {
    UNREACHABLE();
  }
  type.set_compression(kENCODING_NONE);
  type.set_fixed_size();
  return type;
}

bool validate_integral_mapping(const ColumnDescriptor* omnisci_column,
                               const parquet::ColumnDescriptor* parquet_column) {
  if (!omnisci_column->columnType.is_integer()) {
    return false;
  }
  if (auto int_logical_column = dynamic_cast<const parquet::IntLogicalType*>(
          parquet_column->logical_type().get())) {
    CHECK(omnisci_column->columnType.get_compression() == kENCODING_NONE ||
          omnisci_column->columnType.get_compression() == kENCODING_FIXED);
    const int bits_per_byte = 8;
    // unsigned types are permitted to map to a wider integral type in order to avoid
    // precision loss
    const int bit_widening_factor = int_logical_column->is_signed() ? 1 : 2;
    return omnisci_column->columnType.get_size() * bits_per_byte <=
           int_logical_column->bit_width() * bit_widening_factor;
  }
  // check if mapping is a valid coerced or non-coerced integral mapping with no
  // annotation
  if ((omnisci_column->columnType.get_compression() == kENCODING_NONE ||
       omnisci_column->columnType.get_compression() == kENCODING_FIXED)) {
    return (parquet_column->physical_type() == parquet::Type::INT64) ||
           (parquet_column->physical_type() == parquet::Type::INT32 &&
            omnisci_column->columnType.get_size() <= 4);
  }
  return false;
}

SQLTypeInfo suggest_integral_mapping(const parquet::ColumnDescriptor* parquet_column) {
  SQLTypeInfo type;
  type.set_compression(kENCODING_NONE);
  if (auto int_logical_column = dynamic_cast<const parquet::IntLogicalType*>(
          parquet_column->logical_type().get())) {
    auto bit_width = int_logical_column->bit_width();
    if (!int_logical_column->is_signed()) {
      if (within_range(33, 64, bit_width)) {
        throw ForeignStorageException(
            "Unsigned integer column \"" + parquet_column->name() +
            "\" in Parquet file with 64 bit-width has no supported type for ingestion "
            "that will not result in data loss");
      } else if (within_range(17, 32, bit_width)) {
        type.set_type(kBIGINT);
      } else if (within_range(9, 16, bit_width)) {
        type.set_type(kINT);
      } else if (within_range(0, 8, bit_width)) {
        type.set_type(kSMALLINT);
      }
    } else {
      if (within_range(33, 64, bit_width)) {
        type.set_type(kBIGINT);
      } else if (within_range(17, 32, bit_width)) {
        type.set_type(kINT);
      } else if (within_range(9, 16, bit_width)) {
        type.set_type(kSMALLINT);
      } else if (within_range(0, 8, bit_width)) {
        type.set_type(kTINYINT);
      }
    }
    type.set_fixed_size();
    return type;
  }

  CHECK(parquet_column->logical_type()->is_none());
  if (parquet_column->physical_type() == parquet::Type::INT32) {
    type.set_type(kINT);
  } else {
    CHECK(parquet_column->physical_type() == parquet::Type::INT64);
    type.set_type(kBIGINT);
  }
  type.set_fixed_size();
  return type;
}

bool is_nanosecond_precision(const ColumnDescriptor* omnisci_column) {
  return omnisci_column->columnType.get_dimension() == 9;
}

bool is_nanosecond_precision(
    const parquet::TimestampLogicalType* timestamp_logical_column) {
  return timestamp_logical_column->time_unit() == parquet::LogicalType::TimeUnit::NANOS;
}

bool is_microsecond_precision(const ColumnDescriptor* omnisci_column) {
  return omnisci_column->columnType.get_dimension() == 6;
}

bool is_microsecond_precision(
    const parquet::TimestampLogicalType* timestamp_logical_column) {
  return timestamp_logical_column->time_unit() == parquet::LogicalType::TimeUnit::MICROS;
}

bool is_millisecond_precision(const ColumnDescriptor* omnisci_column) {
  return omnisci_column->columnType.get_dimension() == 3;
}

bool is_millisecond_precision(
    const parquet::TimestampLogicalType* timestamp_logical_column) {
  return timestamp_logical_column->time_unit() == parquet::LogicalType::TimeUnit::MILLIS;
}

bool validate_none_type_mapping(const ColumnDescriptor* omnisci_column,
                                const parquet::ColumnDescriptor* parquet_column) {
  bool is_none_encoded_mapping =
      omnisci_column->columnType.get_compression() == kENCODING_NONE &&
      (parquet_column->physical_type() == parquet::Type::BOOLEAN &&
       omnisci_column->columnType.get_type() == kBOOLEAN);
  return parquet_column->logical_type()->is_none() && is_none_encoded_mapping;
}

SQLTypeInfo suggest_boolean_type_mapping(
    const parquet::ColumnDescriptor* parquet_column) {
  SQLTypeInfo type;
  type.set_compression(kENCODING_NONE);
  type.set_type(kBOOLEAN);
  type.set_fixed_size();
  return type;
}

bool validate_timestamp_mapping(const ColumnDescriptor* omnisci_column,
                                const parquet::ColumnDescriptor* parquet_column) {
  if (!(omnisci_column->columnType.get_type() == kTIMESTAMP &&
        ((omnisci_column->columnType.get_compression() == kENCODING_NONE) ||
         (omnisci_column->columnType.get_compression() == kENCODING_FIXED &&
          omnisci_column->columnType.get_comp_param() == 32)))) {
    return false;
  }
  // check the annotated case
  if (auto timestamp_logical_column = dynamic_cast<const parquet::TimestampLogicalType*>(
          parquet_column->logical_type().get())) {
    if (omnisci_column->columnType.get_compression() == kENCODING_NONE) {
      return omnisci_column->columnType.get_dimension() == 0 ||
             ((is_nanosecond_precision(omnisci_column) &&
               is_nanosecond_precision(timestamp_logical_column)) ||
              (is_microsecond_precision(omnisci_column) &&
               is_microsecond_precision(timestamp_logical_column)) ||
              (is_millisecond_precision(omnisci_column) &&
               is_millisecond_precision(timestamp_logical_column)));
    }
    if (omnisci_column->columnType.get_compression() == kENCODING_FIXED) {
      return omnisci_column->columnType.get_dimension() == 0;
    }
  }
  // check the unannotated case
  if (parquet_column->logical_type()->is_none() &&
      ((parquet_column->physical_type() == parquet::Type::INT32 &&
        omnisci_column->columnType.get_compression() == kENCODING_FIXED &&
        omnisci_column->columnType.get_comp_param() == 32) ||
       parquet_column->physical_type() == parquet::Type::INT64)) {
    return true;
  }
  return false;
}

SQLTypeInfo suggest_timestamp_mapping(const parquet::ColumnDescriptor* parquet_column) {
  if (auto timestamp_logical_column = dynamic_cast<const parquet::TimestampLogicalType*>(
          parquet_column->logical_type().get())) {
    SQLTypeInfo type;
    type.set_type(kTIMESTAMP);
    type.set_compression(kENCODING_NONE);
    if (is_nanosecond_precision(timestamp_logical_column)) {
      type.set_precision(9);
    } else if (is_microsecond_precision(timestamp_logical_column)) {
      type.set_precision(6);
    } else if (is_millisecond_precision(timestamp_logical_column)) {
      type.set_precision(3);
    }
    type.set_fixed_size();
    return type;
  }
  UNREACHABLE();
  return {};
}

bool validate_time_mapping(const ColumnDescriptor* omnisci_column,
                           const parquet::ColumnDescriptor* parquet_column) {
  if (!(omnisci_column->columnType.get_type() == kTIME &&
        (omnisci_column->columnType.get_compression() == kENCODING_NONE ||
         (omnisci_column->columnType.get_compression() == kENCODING_FIXED &&
          omnisci_column->columnType.get_comp_param() == 32)))) {
    return false;
  }
  if (parquet_column->logical_type()->is_time()) {
    return true;
  }
  return false;
}

SQLTypeInfo suggest_time_mapping(const parquet::ColumnDescriptor* parquet_column) {
  CHECK(parquet_column->logical_type()->is_time());
  SQLTypeInfo type;
  type.set_type(kTIME);
  type.set_compression(kENCODING_NONE);
  type.set_fixed_size();
  return type;
}

bool validate_date_mapping(const ColumnDescriptor* omnisci_column,
                           const parquet::ColumnDescriptor* parquet_column) {
  if (!(omnisci_column->columnType.get_type() == kDATE &&
        ((omnisci_column->columnType.get_compression() == kENCODING_DATE_IN_DAYS &&
          (omnisci_column->columnType.get_comp_param() ==
               0  // DATE ENCODING DAYS (32) specifies comp_param of 0
           || omnisci_column->columnType.get_comp_param() == 16)) ||
         omnisci_column->columnType.get_compression() ==
             kENCODING_NONE  // for array types
         ))) {
    return false;
  }
  return parquet_column->logical_type()->is_date() ||
         parquet_column->logical_type()
             ->is_timestamp();  // to support TIMESTAMP -> DATE coercion
}

SQLTypeInfo suggest_date_mapping(const parquet::ColumnDescriptor* parquet_column) {
  CHECK(parquet_column->logical_type()->is_date());
  SQLTypeInfo type;
  type.set_type(kDATE);
  type.set_compression(kENCODING_NONE);
  type.set_fixed_size();
  return type;
}

bool validate_string_mapping(const ColumnDescriptor* omnisci_column,
                             const parquet::ColumnDescriptor* parquet_column) {
  return is_valid_parquet_string(parquet_column) &&
         omnisci_column->columnType.is_string() &&
         (omnisci_column->columnType.get_compression() == kENCODING_NONE ||
          omnisci_column->columnType.get_compression() == kENCODING_DICT);
}

SQLTypeInfo suggest_string_mapping(const parquet::ColumnDescriptor* parquet_column) {
  CHECK(is_valid_parquet_string(parquet_column));
  SQLTypeInfo type;
  type.set_type(kTEXT);
  type.set_compression(kENCODING_DICT);
  type.set_comp_param(32);
  type.set_fixed_size();
  return type;
}

bool validate_array_mapping(const ColumnDescriptor* omnisci_column,
                            const parquet::ColumnDescriptor* parquet_column) {
  if (is_valid_parquet_list_column(parquet_column) &&
      omnisci_column->columnType.is_array()) {
    auto omnisci_column_sub_type_column = get_sub_type_column_descriptor(omnisci_column);
    return LazyParquetChunkLoader::isColumnMappingSupported(
        omnisci_column_sub_type_column.get(), parquet_column);
  }
  return false;
}

bool validate_geospatial_mapping(const ColumnDescriptor* omnisci_column,
                                 const parquet::ColumnDescriptor* parquet_column) {
  return is_valid_parquet_string(parquet_column) &&
         omnisci_column->columnType.is_geometry();
}

void validate_equal_schema(const parquet::arrow::FileReader* reference_file_reader,
                           const parquet::arrow::FileReader* new_file_reader,
                           const std::string& reference_file_path,
                           const std::string& new_file_path) {
  const auto reference_num_columns =
      reference_file_reader->parquet_reader()->metadata()->num_columns();
  const auto new_num_columns =
      new_file_reader->parquet_reader()->metadata()->num_columns();
  if (reference_num_columns != new_num_columns) {
    throw std::runtime_error{"Parquet file \"" + new_file_path +
                             "\" has a different schema. Please ensure that all Parquet "
                             "files use the same schema. Reference Parquet file: \"" +
                             reference_file_path + "\" has " +
                             std::to_string(reference_num_columns) +
                             " columns. New Parquet file \"" + new_file_path + "\" has " +
                             std::to_string(new_num_columns) + " columns."};
  }

  for (int i = 0; i < reference_num_columns; i++) {
    validate_equal_column_descriptor(get_column_descriptor(reference_file_reader, i),
                                     get_column_descriptor(new_file_reader, i),
                                     reference_file_path,
                                     new_file_path);
  }
}

void validate_allowed_mapping(const parquet::ColumnDescriptor* parquet_column,
                              const ColumnDescriptor* omnisci_column) {
  parquet::Type::type physical_type = parquet_column->physical_type();
  auto logical_type = parquet_column->logical_type();
  bool allowed_type =
      LazyParquetChunkLoader::isColumnMappingSupported(omnisci_column, parquet_column);
  if (!allowed_type) {
    if (logical_type->is_timestamp()) {
      auto timestamp_type =
          dynamic_cast<const parquet::TimestampLogicalType*>(logical_type.get());
      CHECK(timestamp_type);

      if (!timestamp_type->is_adjusted_to_utc()) {
        LOG(WARNING) << "Non-UTC timezone specified in Parquet file for column \""
                     << omnisci_column->columnName
                     << "\". Only UTC timezone is currently supported.";
      }
    }
    std::string parquet_type;
    if (parquet_column->logical_type()->is_none()) {
      parquet_type = parquet::TypeToString(physical_type);
    } else {
      parquet_type = logical_type->ToString();
    }
    std::string omnisci_type = omnisci_column->columnType.get_type_name();
    throw std::runtime_error{"Conversion from Parquet type \"" + parquet_type +
                             "\" to HeavyDB type \"" + omnisci_type +
                             "\" is not allowed. Please use an appropriate column type."};
  }
}

SQLTypeInfo suggest_column_scalar_type(const parquet::ColumnDescriptor* parquet_column) {
  // decimal case
  if (parquet_column->logical_type()->is_decimal()) {
    return suggest_decimal_mapping(parquet_column);
  }
  // float case
  if (parquet_column->logical_type()->is_none() &&
      (parquet_column->physical_type() == parquet::Type::FLOAT ||
       parquet_column->physical_type() == parquet::Type::DOUBLE)) {
    return suggest_floating_point_mapping(parquet_column);
  }
  // integral case
  if ((parquet_column->logical_type()->is_none() &&
       (parquet_column->physical_type() == parquet::Type::INT32 ||
        parquet_column->physical_type() == parquet::Type::INT64)) ||
      parquet_column->logical_type()->is_int()) {
    return suggest_integral_mapping(parquet_column);
  }
  // boolean case
  if (parquet_column->logical_type()->is_none() &&
      parquet_column->physical_type() == parquet::Type::BOOLEAN) {
    return suggest_boolean_type_mapping(parquet_column);
  }
  // timestamp case
  if (parquet_column->logical_type()->is_timestamp()) {
    return suggest_timestamp_mapping(parquet_column);
  }
  // time case
  if (parquet_column->logical_type()->is_time()) {
    return suggest_time_mapping(parquet_column);
  }
  // date case
  if (parquet_column->logical_type()->is_date()) {
    return suggest_date_mapping(parquet_column);
  }
  // string case
  if (is_valid_parquet_string(parquet_column)) {
    return suggest_string_mapping(parquet_column);
  }

  throw ForeignStorageException("Unsupported data type detected for column: " +
                                parquet_column->ToString());
}

void validate_number_of_columns(
    const std::shared_ptr<parquet::FileMetaData>& file_metadata,
    const std::string& file_path,
    const ForeignTableSchema& schema) {
  if (schema.numLogicalColumns() != file_metadata->num_columns()) {
    throw_number_of_columns_mismatch_error(
        schema.numLogicalColumns(), file_metadata->num_columns(), file_path);
  }
}

void throw_missing_metadata_error(const int row_group_index,
                                  const int column_index,
                                  const std::string& file_path) {
  throw std::runtime_error{
      "Statistics metadata is required for all row groups. Metadata is missing for "
      "row group index: " +
      std::to_string(row_group_index) +
      ", column index: " + std::to_string(column_index) + ", file path: " + file_path};
}

void throw_row_group_larger_than_fragment_size_error(const int row_group_index,
                                                     const int64_t max_row_group_size,
                                                     const int fragment_size,
                                                     const std::string& file_path) {
  auto metadata_scan_exception = MetadataScanInfeasibleFragmentSizeException{
      "Parquet file has a row group size that is larger than the fragment size. "
      "Please set the table fragment size to a number that is larger than the "
      "row group size. Row group index: " +
      std::to_string(row_group_index) +
      ", row group size: " + std::to_string(max_row_group_size) +
      ", fragment size: " + std::to_string(fragment_size) + ", file path: " + file_path};
  metadata_scan_exception.min_feasible_fragment_size_ = max_row_group_size;
  throw metadata_scan_exception;
}

void validate_column_mapping_and_row_group_metadata(
    const std::shared_ptr<parquet::FileMetaData>& file_metadata,
    const std::string& file_path,
    const ForeignTableSchema& schema) {
  auto column_it = schema.getLogicalColumns().begin();
  for (int i = 0; i < file_metadata->num_columns(); ++i, ++column_it) {
    const parquet::ColumnDescriptor* descr = file_metadata->schema()->Column(i);
    try {
      validate_allowed_mapping(descr, *column_it);
    } catch (std::runtime_error& e) {
      std::stringstream error_message;
      error_message << e.what() << " Parquet column: " << descr->name()
                    << ", HeavyDB column: " << (*column_it)->columnName
                    << ", Parquet file: " << file_path << ".";
      throw std::runtime_error(error_message.str());
    }

    auto fragment_size = schema.getForeignTable()->maxFragRows;
    int64_t max_row_group_size = 0;
    int max_row_group_index = 0;
    for (int r = 0; r < file_metadata->num_row_groups(); ++r) {
      auto group_metadata = file_metadata->RowGroup(r);
      auto num_rows = group_metadata->num_rows();
      if (num_rows == 0) {
        continue;
      } else if (num_rows > max_row_group_size) {
        max_row_group_size = num_rows;
        max_row_group_index = r;
      }

      auto column_chunk = group_metadata->ColumnChunk(i);
      bool contains_metadata = column_chunk->is_stats_set();
      if (contains_metadata) {
        auto stats = column_chunk->statistics();
        bool is_all_nulls = stats->null_count() == column_chunk->num_values();
        bool is_list = is_valid_parquet_list_column(file_metadata->schema()->Column(i));
        // Given a list, it is possible it has no min or max if it is comprised
        // only of empty lists & nulls. This can not be detected by comparing
        // the null count; therefore we afford list types the benefit of the
        // doubt in this situation.
        if (!(stats->HasMinMax() || is_all_nulls || is_list)) {
          contains_metadata = false;
        }
      }

      if (!contains_metadata) {
        throw_missing_metadata_error(r, i, file_path);
      }
    }

    if (max_row_group_size > fragment_size) {
      throw_row_group_larger_than_fragment_size_error(
          max_row_group_index, max_row_group_size, fragment_size, file_path);
    }
  }
}

void validate_parquet_metadata(
    const std::shared_ptr<parquet::FileMetaData>& file_metadata,
    const std::string& file_path,
    const ForeignTableSchema& schema) {
  validate_number_of_columns(file_metadata, file_path, schema);
  validate_column_mapping_and_row_group_metadata(file_metadata, file_path, schema);
}

std::list<RowGroupMetadata> metadata_scan_rowgroup_interval(
    const std::map<int, std::shared_ptr<ParquetEncoder>>& encoder_map,
    const RowGroupInterval& row_group_interval,
    const ReaderPtr& reader,
    const ForeignTableSchema& schema) {
  std::list<RowGroupMetadata> row_group_metadata;
  auto column_interval =
      Interval<ColumnType>{schema.getLogicalAndPhysicalColumns().front()->columnId,
                           schema.getLogicalAndPhysicalColumns().back()->columnId};

  auto file_metadata = reader->parquet_reader()->metadata();
  for (int row_group = row_group_interval.start_index;
       row_group <= row_group_interval.end_index;
       ++row_group) {
    auto& row_group_metadata_item = row_group_metadata.emplace_back();
    row_group_metadata_item.row_group_index = row_group;
    row_group_metadata_item.file_path = row_group_interval.file_path;

    std::unique_ptr<parquet::RowGroupMetaData> group_metadata =
        file_metadata->RowGroup(row_group);

    for (int column_id = column_interval.start; column_id <= column_interval.end;
         column_id++) {
      const auto column_descriptor = schema.getColumnDescriptor(column_id);
      auto parquet_column_index = schema.getParquetColumnIndex(column_id);
      auto encoder_map_iter =
          encoder_map.find(schema.getLogicalColumn(column_id)->columnId);
      CHECK(encoder_map_iter != encoder_map.end());
      try {
        auto metadata = encoder_map_iter->second->getRowGroupMetadata(
            group_metadata.get(), parquet_column_index, column_descriptor->columnType);
        row_group_metadata_item.column_chunk_metadata.emplace_back(metadata);
      } catch (const std::exception& e) {
        std::stringstream error_message;
        error_message << e.what() << " in row group " << row_group << " of Parquet file '"
                      << row_group_interval.file_path << "'.";
        throw std::runtime_error(error_message.str());
      }
    }
  }
  return row_group_metadata;
}

std::map<int, std::shared_ptr<ParquetEncoder>> populate_encoder_map_for_import(
    const std::map<int, Chunk_NS::Chunk> chunks,
    const ForeignTableSchema& schema,
    const ReaderPtr& reader,
    const std::map<int, StringDictionary*> column_dictionaries,
    const int64_t num_rows,
    const RenderGroupAnalyzerMap* render_group_analyzer_map) {
  std::map<int, std::shared_ptr<ParquetEncoder>> encoder_map;
  auto file_metadata = reader->parquet_reader()->metadata();
  for (auto& [column_id, chunk] : chunks) {
    const auto column_descriptor = schema.getColumnDescriptor(column_id);
    if (column_descriptor->isGeoPhyCol) {  // skip physical columns
      continue;
    }
    auto parquet_column_descriptor =
        file_metadata->schema()->Column(schema.getParquetColumnIndex(column_id));
    auto find_it = column_dictionaries.find(column_id);
    StringDictionary* dictionary =
        (find_it == column_dictionaries.end() ? nullptr : find_it->second);
    std::list<Chunk_NS::Chunk> chunks_for_import;
    chunks_for_import.push_back(chunk);
    if (column_descriptor->columnType.is_geometry()) {
      for (int i = 0; i < column_descriptor->columnType.get_physical_cols(); ++i) {
        chunks_for_import.push_back(chunks.at(column_id + i + 1));
      }
    }
    encoder_map[column_id] = create_parquet_encoder_for_import(chunks_for_import,
                                                               column_descriptor,
                                                               parquet_column_descriptor,
                                                               dictionary,
                                                               render_group_analyzer_map);

    // reserve space in buffer when num-elements known ahead of time for types
    // of known size (for example dictionary encoded strings)
    auto encoder = shared::get_from_map(encoder_map, column_id);
    if (auto inplace_encoder = dynamic_cast<ParquetInPlaceEncoder*>(encoder.get())) {
      inplace_encoder->reserve(num_rows);
    }
  }
  return encoder_map;
}

std::map<int, std::shared_ptr<ParquetEncoder>> populate_encoder_map_for_metadata_scan(
    const Interval<ColumnType>& column_interval,
    const ForeignTableSchema& schema,
    const ReaderPtr& reader,
    const RenderGroupAnalyzerMap* render_group_analyzer_map,
    const bool do_metadata_stats_validation) {
  std::map<int, std::shared_ptr<ParquetEncoder>> encoder_map;
  auto file_metadata = reader->parquet_reader()->metadata();
  for (int column_id = column_interval.start; column_id <= column_interval.end;
       column_id++) {
    const auto column_descriptor = schema.getColumnDescriptor(column_id);
    auto parquet_column_descriptor =
        file_metadata->schema()->Column(schema.getParquetColumnIndex(column_id));
    encoder_map[column_id] = create_parquet_encoder_for_metadata_scan(
        column_descriptor, parquet_column_descriptor, render_group_analyzer_map);
    if (!do_metadata_stats_validation) {
      shared::get_from_map(encoder_map, column_id)->disableMetadataStatsValidation();
    }
    column_id += column_descriptor->columnType.get_physical_cols();
  }
  return encoder_map;
}
}  // namespace

std::list<std::unique_ptr<ChunkMetadata>> LazyParquetChunkLoader::appendRowGroups(
    const std::vector<RowGroupInterval>& row_group_intervals,
    const int parquet_column_index,
    const ColumnDescriptor* column_descriptor,
    std::list<Chunk_NS::Chunk>& chunks,
    StringDictionary* string_dictionary,
    RejectedRowIndices* rejected_row_indices,
    const bool is_for_detect,
    const std::optional<int64_t> max_levels_read) {
  auto timer = DEBUG_TIMER(__func__);
  std::list<std::unique_ptr<ChunkMetadata>> chunk_metadata;
  // `def_levels` and `rep_levels` below are used to store the read definition
  // and repetition levels of the Dremel encoding implemented by the Parquet
  // format
  std::vector<int16_t> def_levels(LazyParquetChunkLoader::batch_reader_num_elements);
  std::vector<int16_t> rep_levels(LazyParquetChunkLoader::batch_reader_num_elements);
  std::vector<int8_t> values;

  CHECK(!row_group_intervals.empty());
  const auto& first_file_path = row_group_intervals.front().file_path;

  auto first_file_reader = file_reader_cache_->getOrInsert(first_file_path, file_system_);
  auto first_parquet_column_descriptor =
      get_column_descriptor(first_file_reader, parquet_column_index);
  resize_values_buffer(column_descriptor, first_parquet_column_descriptor, values);
  auto encoder = create_parquet_encoder(column_descriptor,
                                        first_parquet_column_descriptor,
                                        chunks,
                                        string_dictionary,
                                        chunk_metadata,
                                        render_group_analyzer_map_,
                                        false,
                                        false,
                                        is_for_detect);
  CHECK(encoder.get());

  if (rejected_row_indices) {  // error tracking is enabled
    encoder->initializeErrorTracking(column_descriptor->columnType);
  }

  bool early_exit = false;
  int64_t total_levels_read = 0;
  for (const auto& row_group_interval : row_group_intervals) {
    const auto& file_path = row_group_interval.file_path;
    auto file_reader = file_reader_cache_->getOrInsert(file_path, file_system_);

    auto [num_row_groups, num_columns] = get_parquet_table_size(file_reader);
    CHECK(row_group_interval.start_index >= 0 &&
          row_group_interval.end_index < num_row_groups);
    CHECK(parquet_column_index >= 0 && parquet_column_index < num_columns);

    parquet::ParquetFileReader* parquet_reader = file_reader->parquet_reader();
    auto parquet_column_descriptor =
        get_column_descriptor(file_reader, parquet_column_index);
    validate_equal_column_descriptor(first_parquet_column_descriptor,
                                     parquet_column_descriptor,
                                     first_file_path,
                                     file_path);

    validate_max_repetition_and_definition_level(column_descriptor,
                                                 parquet_column_descriptor);
    int64_t values_read = 0;
    for (int row_group_index = row_group_interval.start_index;
         row_group_index <= row_group_interval.end_index;
         ++row_group_index) {
      auto group_reader = parquet_reader->RowGroup(row_group_index);
      std::shared_ptr<parquet::ColumnReader> col_reader =
          group_reader->Column(parquet_column_index);

      try {
        while (col_reader->HasNext()) {
          int64_t levels_read =
              parquet::ScanAllValues(LazyParquetChunkLoader::batch_reader_num_elements,
                                     def_levels.data(),
                                     rep_levels.data(),
                                     reinterpret_cast<uint8_t*>(values.data()),
                                     &values_read,
                                     col_reader.get());

          validate_definition_levels(parquet_reader,
                                     row_group_index,
                                     parquet_column_index,
                                     def_levels.data(),
                                     levels_read,
                                     parquet_column_descriptor);

          if (rejected_row_indices) {  // error tracking is enabled
            encoder->appendDataTrackErrors(def_levels.data(),
                                           rep_levels.data(),
                                           values_read,
                                           levels_read,
                                           values.data());
          } else {  // no error tracking enabled
            encoder->appendData(def_levels.data(),
                                rep_levels.data(),
                                values_read,
                                levels_read,
                                values.data());
          }

          if (max_levels_read.has_value()) {
            total_levels_read += levels_read;
            if (total_levels_read >= max_levels_read.value()) {
              early_exit = true;
              break;
            }
          }
        }
        if (auto array_encoder = dynamic_cast<ParquetArrayEncoder*>(encoder.get())) {
          array_encoder->finalizeRowGroup();
        }
      } catch (const std::exception& error) {
        throw ForeignStorageException(
            std::string(error.what()) + " Row group: " + std::to_string(row_group_index) +
            ", Parquet column: '" + col_reader->descr()->path()->ToDotString() +
            "', Parquet file: '" + file_path + "'");
      }
      if (max_levels_read.has_value() && early_exit) {
        break;
      }
    }
    if (max_levels_read.has_value() && early_exit) {
      break;
    }
  }

  if (rejected_row_indices) {  // error tracking is enabled
    *rejected_row_indices = encoder->getRejectedRowIndices();
  }
  return chunk_metadata;
}

SQLTypeInfo LazyParquetChunkLoader::suggestColumnMapping(
    const parquet::ColumnDescriptor* parquet_column) {
  auto type = suggest_column_scalar_type(parquet_column);

  // array case
  if (is_valid_parquet_list_column(parquet_column)) {
    return type.get_array_type();
  }

  return type;
}

bool LazyParquetChunkLoader::isColumnMappingSupported(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column) {
  if (validate_geospatial_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  if (validate_array_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  if (validate_decimal_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  if (validate_floating_point_mapping(omnisci_column, parquet_column)) {
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
  if (validate_string_mapping(omnisci_column, parquet_column)) {
    return true;
  }
  return false;
}

LazyParquetChunkLoader::LazyParquetChunkLoader(
    std::shared_ptr<arrow::fs::FileSystem> file_system,
    FileReaderMap* file_map,
    const RenderGroupAnalyzerMap* render_group_analyzer_map)
    : file_system_(file_system)
    , file_reader_cache_(file_map)
    , render_group_analyzer_map_{render_group_analyzer_map} {}

std::list<std::unique_ptr<ChunkMetadata>> LazyParquetChunkLoader::loadChunk(
    const std::vector<RowGroupInterval>& row_group_intervals,
    const int parquet_column_index,
    std::list<Chunk_NS::Chunk>& chunks,
    StringDictionary* string_dictionary,
    RejectedRowIndices* rejected_row_indices) {
  CHECK(!chunks.empty());
  auto const& chunk = *chunks.begin();
  auto column_descriptor = chunk.getColumnDesc();
  auto buffer = chunk.getBuffer();
  CHECK(buffer);

  try {
    auto metadata = appendRowGroups(row_group_intervals,
                                    parquet_column_index,
                                    column_descriptor,
                                    chunks,
                                    string_dictionary,
                                    rejected_row_indices);
    return metadata;
  } catch (const std::exception& error) {
    throw ForeignStorageException(error.what());
  }

  return {};
}

struct ParquetBatchData {
  ParquetBatchData()
      : def_levels(LazyParquetChunkLoader::batch_reader_num_elements)
      , rep_levels(LazyParquetChunkLoader::batch_reader_num_elements) {}
  std::vector<int16_t> def_levels;
  std::vector<int16_t> rep_levels;
  std::vector<int8_t> values;
  int64_t values_read;
  int64_t levels_read;
};

class ParquetRowGroupReader {
 public:
  ParquetRowGroupReader(std::shared_ptr<parquet::ColumnReader> col_reader,
                        const ColumnDescriptor* column_descriptor,
                        const parquet::ColumnDescriptor* parquet_column_descriptor,
                        ParquetEncoder* encoder,
                        InvalidRowGroupIndices& invalid_indices,
                        const int row_group_index,
                        const int parquet_column_index,
                        const parquet::ParquetFileReader* parquet_reader)
      : col_reader_(col_reader)
      , column_descriptor_(column_descriptor)
      , parquet_column_descriptor_(parquet_column_descriptor)
      , encoder_(encoder)
      , invalid_indices_(invalid_indices)
      , row_group_index_(row_group_index)
      , parquet_column_index_(parquet_column_index)
      , parquet_reader_(parquet_reader) {
    import_encoder = dynamic_cast<ParquetImportEncoder*>(encoder);
    CHECK(import_encoder);
  }

  void readAndValidateRowGroup() {
    while (col_reader_->HasNext()) {
      ParquetBatchData batch_data;
      resize_values_buffer(
          column_descriptor_, parquet_column_descriptor_, batch_data.values);
      batch_data.levels_read =
          parquet::ScanAllValues(LazyParquetChunkLoader::batch_reader_num_elements,
                                 batch_data.def_levels.data(),
                                 batch_data.rep_levels.data(),
                                 reinterpret_cast<uint8_t*>(batch_data.values.data()),
                                 &batch_data.values_read,
                                 col_reader_.get());
      SQLTypeInfo column_type = column_descriptor_->columnType.is_array()
                                    ? column_descriptor_->columnType.get_subtype()
                                    : column_descriptor_->columnType;
      validate_definition_levels(parquet_reader_,
                                 row_group_index_,
                                 parquet_column_index_,
                                 batch_data.def_levels.data(),
                                 batch_data.levels_read,
                                 parquet_column_descriptor_);
      import_encoder->validateAndAppendData(batch_data.def_levels.data(),
                                            batch_data.rep_levels.data(),
                                            batch_data.values_read,
                                            batch_data.levels_read,
                                            batch_data.values.data(),
                                            column_type,
                                            invalid_indices_);
    }
    if (auto array_encoder = dynamic_cast<ParquetArrayEncoder*>(encoder_)) {
      array_encoder->finalizeRowGroup();
    }
  }

  void eraseInvalidRowGroupData(const InvalidRowGroupIndices& invalid_indices) {
    import_encoder->eraseInvalidIndicesInBuffer(invalid_indices);
  }

 private:
  std::shared_ptr<parquet::ColumnReader> col_reader_;
  const ColumnDescriptor* column_descriptor_;
  const parquet::ColumnDescriptor* parquet_column_descriptor_;
  ParquetEncoder* encoder_;
  ParquetImportEncoder* import_encoder;
  InvalidRowGroupIndices& invalid_indices_;
  const int row_group_index_;
  const int parquet_column_index_;
  const parquet::ParquetFileReader* parquet_reader_;
};

std::pair<size_t, size_t> LazyParquetChunkLoader::loadRowGroups(
    const RowGroupInterval& row_group_interval,
    const std::map<int, Chunk_NS::Chunk>& chunks,
    const ForeignTableSchema& schema,
    const std::map<int, StringDictionary*>& column_dictionaries,
    const int num_threads) {
  auto timer = DEBUG_TIMER(__func__);

  const auto& file_path = row_group_interval.file_path;

  // do not use caching with file-readers, open a new one for every request
  auto file_reader_owner = open_parquet_table(file_path, file_system_);
  auto file_reader = file_reader_owner.get();
  auto file_metadata = file_reader->parquet_reader()->metadata();

  validate_number_of_columns(file_metadata, file_path, schema);

  // check for fixed length encoded columns and indicate to the user
  // they should not be used
  for (const auto column_descriptor : schema.getLogicalColumns()) {
    auto parquet_column_index = schema.getParquetColumnIndex(column_descriptor->columnId);
    auto parquet_column = file_metadata->schema()->Column(parquet_column_index);
    try {
      validate_allowed_mapping(parquet_column, column_descriptor);
    } catch (std::runtime_error& e) {
      std::stringstream error_message;
      error_message << e.what() << " Parquet column: " << parquet_column->name()
                    << ", HeavyDB column: " << column_descriptor->columnName
                    << ", Parquet file: " << file_path << ".";
      throw std::runtime_error(error_message.str());
    }
  }

  CHECK(row_group_interval.start_index == row_group_interval.end_index);
  auto row_group_index = row_group_interval.start_index;
  std::map<int, ParquetRowGroupReader> row_group_reader_map;

  parquet::ParquetFileReader* parquet_reader = file_reader->parquet_reader();
  auto group_reader = parquet_reader->RowGroup(row_group_index);

  std::vector<InvalidRowGroupIndices> invalid_indices_per_thread(num_threads);

  auto encoder_map = populate_encoder_map_for_import(chunks,
                                                     schema,
                                                     file_reader,
                                                     column_dictionaries,
                                                     group_reader->metadata()->num_rows(),
                                                     render_group_analyzer_map_);

  std::vector<std::set<int>> partitions(num_threads);
  std::map<int, int> column_id_to_thread;
  for (auto& [column_id, encoder] : encoder_map) {
    auto thread_id = column_id % num_threads;
    column_id_to_thread[column_id] = thread_id;
    partitions[thread_id].insert(column_id);
  }

  for (auto& [column_id, encoder] : encoder_map) {
    const auto& column_descriptor = schema.getColumnDescriptor(column_id);
    const auto parquet_column_index = schema.getParquetColumnIndex(column_id);
    auto parquet_column_descriptor =
        file_metadata->schema()->Column(parquet_column_index);

    // validate
    auto [num_row_groups, num_columns] = get_parquet_table_size(file_reader);
    CHECK(row_group_interval.start_index >= 0 &&
          row_group_interval.end_index < num_row_groups);
    CHECK(parquet_column_index >= 0 && parquet_column_index < num_columns);
    validate_max_repetition_and_definition_level(column_descriptor,
                                                 parquet_column_descriptor);

    std::shared_ptr<parquet::ColumnReader> col_reader =
        group_reader->Column(parquet_column_index);

    row_group_reader_map.insert(
        {column_id,
         ParquetRowGroupReader(col_reader,
                               column_descriptor,
                               parquet_column_descriptor,
                               shared::get_from_map(encoder_map, column_id).get(),
                               invalid_indices_per_thread[shared::get_from_map(
                                   column_id_to_thread, column_id)],
                               row_group_index,
                               parquet_column_index,
                               parquet_reader)});
  }

  std::vector<std::future<void>> futures;
  for (int ithread = 0; ithread < num_threads; ++ithread) {
    auto column_ids_for_thread = partitions[ithread];
    futures.emplace_back(
        std::async(std::launch::async, [&row_group_reader_map, column_ids_for_thread] {
          for (const auto column_id : column_ids_for_thread) {
            shared::get_from_map(row_group_reader_map, column_id)
                .readAndValidateRowGroup();  // reads and validate entire row group per
                                             // column
          }
        }));
  }

  for (auto& future : futures) {
    future.wait();
  }

  for (auto& future : futures) {
    future.get();
  }

  // merge/reduce invalid indices
  InvalidRowGroupIndices invalid_indices;
  for (auto& thread_invalid_indices : invalid_indices_per_thread) {
    invalid_indices.merge(thread_invalid_indices);
  }

  for (auto& [_, reader] : row_group_reader_map) {
    reader.eraseInvalidRowGroupData(
        invalid_indices);  // removes invalid encoded data in buffers
  }

  // update the element count for each encoder
  for (const auto column_descriptor : schema.getLogicalColumns()) {
    auto column_id = column_descriptor->columnId;
    auto db_encoder = shared::get_from_map(chunks, column_id).getBuffer()->getEncoder();
    CHECK(static_cast<size_t>(group_reader->metadata()->num_rows()) >=
          invalid_indices.size());
    size_t updated_num_elems = db_encoder->getNumElems() +
                               group_reader->metadata()->num_rows() -
                               invalid_indices.size();
    db_encoder->setNumElems(updated_num_elems);
    if (column_descriptor->columnType.is_geometry()) {
      for (int i = 0; i < column_descriptor->columnType.get_physical_cols(); ++i) {
        auto db_encoder =
            shared::get_from_map(chunks, column_id + i + 1).getBuffer()->getEncoder();
        db_encoder->setNumElems(updated_num_elems);
      }
    }
  }

  return {group_reader->metadata()->num_rows() - invalid_indices.size(),
          invalid_indices.size()};
}

struct PreviewBuffers {
  std::list<std::unique_ptr<TypedParquetDetectBuffer>> detect_buffers;
  std::list<Chunk_NS::Chunk> column_chunks;
};

DataPreview LazyParquetChunkLoader::previewFiles(const std::vector<std::string>& files,
                                                 const size_t max_num_rows) {
  CHECK(!files.empty());

  auto first_file = *files.begin();
  auto first_file_reader = file_reader_cache_->getOrInsert(*files.begin(), file_system_);

  for (auto current_file_it = ++files.begin(); current_file_it != files.end();
       ++current_file_it) {
    auto file_reader = file_reader_cache_->getOrInsert(*current_file_it, file_system_);
    validate_equal_schema(first_file_reader, file_reader, first_file, *current_file_it);
  }

  auto first_file_metadata = first_file_reader->parquet_reader()->metadata();
  auto num_columns = first_file_metadata->num_columns();

  DataPreview data_preview;
  data_preview.num_rejected_rows = 0;

  auto current_file_it = files.begin();
  while (data_preview.sample_rows.size() < max_num_rows &&
         current_file_it != files.end()) {
    size_t total_num_rows = data_preview.sample_rows.size();
    size_t max_num_rows_to_append = max_num_rows - data_preview.sample_rows.size();

    // gather enough rows in row groups to produce required samples
    std::vector<RowGroupInterval> row_group_intervals;
    for (; current_file_it != files.end(); ++current_file_it) {
      const auto& file_path = *current_file_it;
      auto file_reader = file_reader_cache_->getOrInsert(file_path, file_system_);
      auto file_metadata = file_reader->parquet_reader()->metadata();
      auto num_row_groups = file_metadata->num_row_groups();
      int end_row_group = 0;
      for (int i = 0; i < num_row_groups && total_num_rows < max_num_rows; ++i) {
        const size_t next_num_rows = file_metadata->RowGroup(i)->num_rows();
        total_num_rows += next_num_rows;
        end_row_group = i;
      }
      row_group_intervals.push_back(RowGroupInterval{file_path, 0, end_row_group});
    }

    auto rejected_row_indices = std::make_unique<RejectedRowIndices>();
    PreviewBuffers preview_buffers;
    for (int i = 0; i < num_columns; ++i) {
      auto col = first_file_metadata->schema()->Column(i);
      ColumnDescriptor cd;
      auto sql_type = LazyParquetChunkLoader::suggestColumnMapping(col);
      cd.columnType = sql_type;
      cd.columnName =
          sql_type.is_array() ? col->path()->ToDotVector()[0] + "_array" : col->name();
      cd.isSystemCol = false;
      cd.isVirtualCol = false;
      cd.tableId = -1;
      cd.columnId = i + 1;
      data_preview.column_names.emplace_back(cd.columnName);
      data_preview.column_types.emplace_back(sql_type);
      preview_buffers.detect_buffers.push_back(
          std::make_unique<TypedParquetDetectBuffer>());
      auto& detect_buffer = preview_buffers.detect_buffers.back();
      auto chunk = preview_buffers.column_chunks.emplace_back(&cd);
      chunk.setBuffer(detect_buffer.get());
      auto chunk_list = std::list<Chunk_NS::Chunk>{chunk};
      appendRowGroups(row_group_intervals,
                      i,
                      &cd,
                      chunk_list,
                      nullptr,
                      rejected_row_indices.get(),
                      true,
                      max_num_rows_to_append);
    }

    size_t num_rows = 0;
    auto buffers_it = preview_buffers.detect_buffers.begin();
    for (int i = 0; i < num_columns; ++i, ++buffers_it) {
      CHECK(buffers_it != preview_buffers.detect_buffers.end());
      auto& strings = buffers_it->get()->getStrings();
      if (i == 0) {
        num_rows = strings.size();
      } else {
        CHECK_EQ(num_rows, strings.size());
      }
    }

    size_t num_rejected_rows = rejected_row_indices->size();
    data_preview.num_rejected_rows += num_rejected_rows;
    CHECK_GE(num_rows, num_rejected_rows);
    auto row_count = num_rows - num_rejected_rows;

    auto offset_row = data_preview.sample_rows.size();
    data_preview.sample_rows.resize(std::min(offset_row + row_count, max_num_rows));

    for (size_t irow = 0, rows_appended = 0;
         irow < num_rows && offset_row + rows_appended < max_num_rows;
         ++irow) {
      if (rejected_row_indices->find(irow) != rejected_row_indices->end()) {
        continue;
      }
      auto& row_data = data_preview.sample_rows[offset_row + rows_appended];
      row_data.resize(num_columns);
      auto buffers_it = preview_buffers.detect_buffers.begin();
      for (int i = 0; i < num_columns; ++i, ++buffers_it) {
        CHECK(buffers_it != preview_buffers.detect_buffers.end());
        auto& strings = buffers_it->get()->getStrings();
        row_data[i] = strings[irow];
      }
      ++rows_appended;
    }
  }

  // attempt to detect geo columns
  for (int i = 0; i < num_columns; ++i) {
    auto tentative_geo_type =
        foreign_storage::detect_geo_type(data_preview.sample_rows, i);
    if (tentative_geo_type.has_value()) {
      data_preview.column_types[i].set_type(tentative_geo_type.value());
      data_preview.column_types[i].set_compression(kENCODING_NONE);
    }
  }

  return data_preview;
}

std::list<RowGroupMetadata> LazyParquetChunkLoader::metadataScan(
    const std::vector<std::string>& file_paths,
    const ForeignTableSchema& schema,
    const bool do_metadata_stats_validation) {
  auto timer = DEBUG_TIMER(__func__);
  auto column_interval =
      Interval<ColumnType>{schema.getLogicalAndPhysicalColumns().front()->columnId,
                           schema.getLogicalAndPhysicalColumns().back()->columnId};
  CHECK(!file_paths.empty());

  // The encoder map needs to be populated before we can start scanning rowgroups, so we
  // peel the first file_path out of the async loop below to perform population.
  const auto& first_path = *file_paths.begin();
  auto first_reader = file_reader_cache_->insert(first_path, file_system_);
  validate_parquet_metadata(
      first_reader->parquet_reader()->metadata(), first_path, schema);
  auto encoder_map = populate_encoder_map_for_metadata_scan(column_interval,
                                                            schema,
                                                            first_reader,
                                                            render_group_analyzer_map_,
                                                            do_metadata_stats_validation);
  const auto num_row_groups = get_parquet_table_size(first_reader).first;
  auto row_group_metadata = metadata_scan_rowgroup_interval(
      encoder_map, {first_path, 0, num_row_groups - 1}, first_reader, schema);

  // We want each (filepath->FileReader) pair in the cache to be initialized before we
  // multithread so that we are not adding keys in a concurrent environment, so we add
  // cache entries for each path and initialize to an empty unique_ptr if the file has not
  // yet been opened.
  // Since we have already performed the first iteration, we skip it in the thread groups
  // so as not to process it twice.
  std::vector<std::string> cache_subset;
  for (auto path_it = ++(file_paths.begin()); path_it != file_paths.end(); ++path_it) {
    file_reader_cache_->initializeIfEmpty(*path_it);
    cache_subset.emplace_back(*path_it);
  }

  // Iterate asyncronously over any paths beyond the first.
  auto paths_per_thread = partition_for_threads(cache_subset, g_max_import_threads);
  std::vector<std::future<std::list<RowGroupMetadata>>> futures;
  for (const auto& path_group : paths_per_thread) {
    futures.emplace_back(std::async(
        std::launch::async,
        [&](const auto& paths, const auto& file_reader_cache) {
          std::list<RowGroupMetadata> reduced_metadata;
          for (const auto& path : paths.get()) {
            auto reader = file_reader_cache.get().getOrInsert(path, file_system_);
            validate_equal_schema(first_reader, reader, first_path, path);
            validate_parquet_metadata(reader->parquet_reader()->metadata(), path, schema);
            const auto num_row_groups = get_parquet_table_size(reader).first;
            const auto interval = RowGroupInterval{path, 0, num_row_groups - 1};
            reduced_metadata.splice(
                reduced_metadata.end(),
                metadata_scan_rowgroup_interval(encoder_map, interval, reader, schema));
          }
          return reduced_metadata;
        },
        std::ref(path_group),
        std::ref(*file_reader_cache_)));
  }

  // Reduce all the row_group results.
  for (auto& future : futures) {
    row_group_metadata.splice(row_group_metadata.end(), future.get());
  }
  return row_group_metadata;
}

}  // namespace foreign_storage
