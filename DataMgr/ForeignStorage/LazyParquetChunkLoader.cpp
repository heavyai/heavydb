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

#include "ParquetDateInSecondsEncoder.h"
#include "ParquetDecimalEncoder.h"
#include "ParquetFixedLengthArrayEncoder.h"
#include "ParquetFixedLengthEncoder.h"
#include "ParquetGeospatialEncoder.h"
#include "ParquetStringEncoder.h"
#include "ParquetStringNoneEncoder.h"
#include "ParquetTimeEncoder.h"
#include "ParquetTimestampEncoder.h"
#include "ParquetVariableLengthArrayEncoder.h"

namespace foreign_storage {

namespace {

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

template <typename V>
std::shared_ptr<ParquetEncoder> create_parquet_decimal_encoder_with_omnisci_type(
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

std::shared_ptr<ParquetEncoder> create_parquet_decimal_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan) {
  if (parquet_column->logical_type()->is_decimal()) {
    if (omnisci_column->columnType.get_compression() == kENCODING_NONE ||
        is_metadata_scan) {
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

std::shared_ptr<ParquetEncoder> create_parquet_integral_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan) {
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
          if (is_metadata_scan && column_type.get_type() == kBIGINT) {
            return std::make_shared<ParquetFixedLengthEncoder<int64_t, int32_t>>(
                buffer, omnisci_column, parquet_column);
          }
          return std::make_shared<ParquetFixedLengthEncoder<int32_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        case 2:
          if (is_metadata_scan) {
            switch (column_type.get_type()) {
              case kBIGINT:
                return std::make_shared<ParquetFixedLengthEncoder<int64_t, int32_t>>(
                    buffer, omnisci_column, parquet_column);
              case kINT:
                return std::make_shared<ParquetFixedLengthEncoder<int32_t, int32_t>>(
                    buffer, omnisci_column, parquet_column);
              case kSMALLINT:
                break;
              default:
                UNREACHABLE();
            }
          }
          return std::make_shared<ParquetFixedLengthEncoder<int16_t, int32_t>>(
              buffer, omnisci_column, parquet_column);
        case 1:
          if (is_metadata_scan) {
            switch (column_type.get_type()) {
              case kBIGINT:
                return std::make_shared<ParquetFixedLengthEncoder<int64_t, int32_t>>(
                    buffer, omnisci_column, parquet_column);
              case kINT:
                return std::make_shared<ParquetFixedLengthEncoder<int32_t, int32_t>>(
                    buffer, omnisci_column, parquet_column);
              case kSMALLINT:
                return std::make_shared<ParquetFixedLengthEncoder<int16_t, int32_t>>(
                    buffer, omnisci_column, parquet_column);
              case kTINYINT:
                break;
              default:
                UNREACHABLE();
            }
          }
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
          if (is_metadata_scan && column_type.get_type() == kBIGINT) {
            return std::make_shared<
                ParquetUnsignedFixedLengthEncoder<int64_t, int32_t, uint16_t>>(
                buffer, omnisci_column, parquet_column);
          }
          return std::make_shared<
              ParquetUnsignedFixedLengthEncoder<int32_t, int32_t, uint16_t>>(
              buffer, omnisci_column, parquet_column);
        case 2:
          if (is_metadata_scan) {
            switch (column_type.get_type()) {
              case kBIGINT:
                return std::make_shared<
                    ParquetUnsignedFixedLengthEncoder<int64_t, int32_t, uint8_t>>(
                    buffer, omnisci_column, parquet_column);
              case kINT:
                return std::make_shared<
                    ParquetUnsignedFixedLengthEncoder<int32_t, int32_t, uint8_t>>(
                    buffer, omnisci_column, parquet_column);
              case kSMALLINT:
                break;
              default:
                UNREACHABLE();
            }
          }
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

std::shared_ptr<ParquetEncoder> create_parquet_none_type_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  auto column_type = omnisci_column->columnType;
  if (parquet_column->logical_type()->is_none() &&
      !omnisci_column->columnType.is_string()) {  // boolean, int32, int64, float & double
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

std::shared_ptr<ParquetEncoder> create_parquet_timestamp_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer) {
  auto column_type = omnisci_column->columnType;
  if (parquet_column->logical_type()->is_timestamp()) {
    auto precision = column_type.get_precision();
    CHECK(column_type.get_compression() == kENCODING_NONE);
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

std::shared_ptr<ParquetEncoder> create_parquet_time_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan) {
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
      if (is_metadata_scan) {
        return std::make_shared<ParquetTimeEncoder<int64_t, int32_t>>(
            buffer, omnisci_column, parquet_column);
      } else {
        return std::make_shared<ParquetTimeEncoder<int32_t, int32_t>>(
            buffer, omnisci_column, parquet_column);
      }
    } else {
      UNREACHABLE();
    }
  }
  return {};
}

std::shared_ptr<ParquetEncoder> create_parquet_date_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    AbstractBuffer* buffer,
    const bool is_metadata_scan) {
  auto column_type = omnisci_column->columnType;
  if (parquet_column->logical_type()->is_date()) {
    if (column_type.get_compression() == kENCODING_DATE_IN_DAYS) {
      if (is_metadata_scan) {
        return std::make_shared<ParquetDateInSecondsEncoder>(
            buffer, omnisci_column, parquet_column);
      } else {
        return std::make_shared<ParquetFixedLengthEncoder<int32_t, int32_t>>(
            buffer, omnisci_column, parquet_column);
      }
    } else if (column_type.get_compression() == kENCODING_NONE) {  // for array types
      return std::make_shared<ParquetDateInSecondsEncoder>(
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
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata) {
  auto column_type = omnisci_column->columnType;
  if (!is_valid_parquet_string(parquet_column) ||
      !omnisci_column->columnType.is_string()) {
    return {};
  }
  if (column_type.get_compression() == kENCODING_NONE) {
    return std::make_shared<ParquetStringNoneEncoder>(chunk.getBuffer(),
                                                      chunk.getIndexBuf());
  } else if (column_type.get_compression() == kENCODING_DICT) {
    chunk_metadata.emplace_back(std::make_unique<ChunkMetadata>());
    auto& logical_chunk_metadata = chunk_metadata.back();
    logical_chunk_metadata->sqlType = omnisci_column->columnType;
    switch (column_type.get_size()) {
      case 1:
        return std::make_shared<ParquetStringEncoder<uint8_t>>(
            chunk.getBuffer(), string_dictionary, logical_chunk_metadata);
      case 2:
        return std::make_shared<ParquetStringEncoder<uint16_t>>(
            chunk.getBuffer(), string_dictionary, logical_chunk_metadata);
      case 4:
        return std::make_shared<ParquetStringEncoder<int32_t>>(
            chunk.getBuffer(), string_dictionary, logical_chunk_metadata);
      default:
        UNREACHABLE();
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
    const bool is_metadata_scan) {
  auto column_type = omnisci_column->columnType;
  if (!is_valid_parquet_string(parquet_column) || !column_type.is_geometry()) {
    return {};
  }
  if (is_metadata_scan) {
    return std::make_shared<ParquetGeospatialEncoder>();
  }
  for (auto chunks_iter = chunks.begin(); chunks_iter != chunks.end(); ++chunks_iter) {
    chunk_metadata.emplace_back(std::make_unique<ChunkMetadata>());
    auto& chunk_metadata_ptr = chunk_metadata.back();
    chunk_metadata_ptr->sqlType = chunks_iter->getColumnDesc()->columnType;
  }
  return std::make_shared<ParquetGeospatialEncoder>(
      parquet_column, chunks, chunk_metadata);
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
    const bool is_metadata_scan);

std::shared_ptr<ParquetEncoder> create_parquet_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    std::list<Chunk_NS::Chunk>& chunks,
    StringDictionary* string_dictionary,
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
    const bool is_metadata_scan = false) {
  auto buffer = chunks.empty() ? nullptr : chunks.begin()->getBuffer();
  if (auto encoder = create_parquet_geospatial_encoder(
          omnisci_column, parquet_column, chunks, chunk_metadata, is_metadata_scan)) {
    return encoder;
  }
  if (auto encoder = create_parquet_array_encoder(omnisci_column,
                                                  parquet_column,
                                                  chunks,
                                                  string_dictionary,
                                                  chunk_metadata,
                                                  is_metadata_scan)) {
    return encoder;
  }
  if (auto encoder = create_parquet_decimal_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan)) {
    return encoder;
  }
  if (auto encoder = create_parquet_integral_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan)) {
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
  if (auto encoder = create_parquet_time_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan)) {
    return encoder;
  }
  if (auto encoder = create_parquet_date_encoder(
          omnisci_column, parquet_column, buffer, is_metadata_scan)) {
    return encoder;
  }
  if (auto encoder = create_parquet_string_encoder(
          omnisci_column,
          parquet_column,
          chunks.empty() ? Chunk_NS::Chunk{} : *chunks.begin(),
          string_dictionary,
          chunk_metadata)) {
    return encoder;
  }
  UNREACHABLE();
  return {};
}

/**
 * Intended to be used only with metadata scan. Creates an incomplete encoder
 * capable of updating metadata.
 */
std::shared_ptr<ParquetEncoder> create_parquet_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column) {
  std::list<Chunk_NS::Chunk> chunks;
  std::list<std::unique_ptr<ChunkMetadata>> chunk_metadata;
  return create_parquet_encoder(
      omnisci_column, parquet_column, chunks, nullptr, chunk_metadata, true);
}

std::shared_ptr<ParquetEncoder> create_parquet_array_encoder(
    const ColumnDescriptor* omnisci_column,
    const parquet::ColumnDescriptor* parquet_column,
    std::list<Chunk_NS::Chunk>& chunks,
    StringDictionary* string_dictionary,
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
    const bool is_metadata_scan) {
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
                                        is_metadata_scan);
  CHECK(encoder.get());
  auto scalar_encoder = std::dynamic_pointer_cast<ParquetScalarEncoder>(encoder);
  CHECK(scalar_encoder);
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
  return encoder;
}

void validate_max_repetition_and_definition_level(
    const ColumnDescriptor* omnisci_column_descriptor,
    const parquet::ColumnDescriptor* parquet_column_descriptor) {
  bool is_valid_parquet_list = is_valid_parquet_list_column(parquet_column_descriptor);
  if (is_valid_parquet_list && !omnisci_column_descriptor->columnType.is_array()) {
    throw std::runtime_error(
        "Unsupported mapping detected. Column '" + parquet_column_descriptor->name() +
        "' detected to be a parquet list but OmniSci mapped column '" +
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

std::list<std::unique_ptr<ChunkMetadata>> append_row_groups(
    const std::vector<RowGroupInterval>& row_group_intervals,
    const int parquet_column_index,
    const ColumnDescriptor* column_descriptor,
    std::list<Chunk_NS::Chunk>& chunks,
    StringDictionary* string_dictionary,
    std::shared_ptr<arrow::fs::FileSystem> file_system) {
  std::list<std::unique_ptr<ChunkMetadata>> chunk_metadata;
  // `def_levels` and `rep_levels` below are used to store the read definition
  // and repetition levels of the Dremel encoding implemented by the Parquet
  // format
  std::vector<int16_t> def_levels(LazyParquetChunkLoader::batch_reader_num_elements);
  std::vector<int16_t> rep_levels(LazyParquetChunkLoader::batch_reader_num_elements);
  std::vector<int8_t> values;

  CHECK(!row_group_intervals.empty());
  std::unique_ptr<parquet::arrow::FileReader> first_file_reader;
  const auto& first_file_path = row_group_intervals.front().file_path;
  open_parquet_table(first_file_path, first_file_reader, file_system);
  auto first_parquet_column_descriptor =
      get_column_descriptor(first_file_reader.get(), parquet_column_index);
  resize_values_buffer(column_descriptor, first_parquet_column_descriptor, values);
  auto encoder = create_parquet_encoder(column_descriptor,
                                        first_parquet_column_descriptor,
                                        chunks,
                                        string_dictionary,
                                        chunk_metadata);
  CHECK(encoder.get());

  for (const auto& row_group_interval : row_group_intervals) {
    std::unique_ptr<parquet::arrow::FileReader> file_reader;
    open_parquet_table(row_group_interval.file_path, file_reader, file_system);

    int num_row_groups, num_columns;
    std::tie(num_row_groups, num_columns) = get_parquet_table_size(file_reader);
    CHECK(row_group_interval.start_index >= 0 &&
          row_group_interval.end_index < num_row_groups);
    CHECK(parquet_column_index >= 0 && parquet_column_index < num_columns);

    parquet::ParquetFileReader* parquet_reader = file_reader->parquet_reader();
    auto parquet_column_descriptor =
        get_column_descriptor(file_reader.get(), parquet_column_index);
    validate_equal_column_descriptor(first_parquet_column_descriptor,
                                     parquet_column_descriptor,
                                     first_file_path,
                                     row_group_interval.file_path);

    validate_max_repetition_and_definition_level(column_descriptor,
                                                 parquet_column_descriptor);
    int64_t values_read = 0;
    for (int row_group_index = row_group_interval.start_index;
         row_group_index <= row_group_interval.end_index;
         ++row_group_index) {
      auto group_reader = parquet_reader->RowGroup(row_group_index);
      std::shared_ptr<parquet::ColumnReader> col_reader =
          group_reader->Column(parquet_column_index);

      while (col_reader->HasNext()) {
        int64_t levels_read =
            parquet::ScanAllValues(LazyParquetChunkLoader::batch_reader_num_elements,
                                   def_levels.data(),
                                   rep_levels.data(),
                                   reinterpret_cast<uint8_t*>(values.data()),
                                   &values_read,
                                   col_reader.get());
        encoder->appendData(def_levels.data(),
                            rep_levels.data(),
                            values_read,
                            levels_read,
                            !col_reader->HasNext(),
                            values.data());
      }
    }
  }
  return chunk_metadata;
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
        ((omnisci_column->columnType.get_compression() == kENCODING_DATE_IN_DAYS &&
          omnisci_column->columnType.get_comp_param() ==
              0)  // DATE ENCODING DAYS (32) specifies comp_param of 0
         || omnisci_column->columnType.get_compression() ==
                kENCODING_NONE  // for array types
         ))) {
    return false;
  }
  return parquet_column->logical_type()->is_date();
}

bool validate_string_mapping(const ColumnDescriptor* omnisci_column,
                             const parquet::ColumnDescriptor* parquet_column) {
  return is_valid_parquet_string(parquet_column) &&
         omnisci_column->columnType.is_string() &&
         (omnisci_column->columnType.get_compression() == kENCODING_NONE ||
          omnisci_column->columnType.get_compression() == kENCODING_DICT);
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
        throw std::runtime_error{
            "Non-UTC timezone specified in Parquet file for column \"" +
            omnisci_column->columnName + "\". Only UTC timezone is currently supported."};
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
                             "\" to OmniSci type \"" + omnisci_type +
                             "\" is not allowed. Please use an appropriate column type."};
  }
}

void throw_mismatched_column_number_error(
    const std::shared_ptr<parquet::FileMetaData>& file_metadata,
    const std::string& file_path,
    const ForeignTableSchema& schema) {
  std::stringstream error_msg;
  error_msg << "Mismatched number of logical columns in table '"
            << schema.getForeignTable()->tableName << "' (" << schema.numLogicalColumns()
            << " columns) with parquet file '" << file_path << "' ("
            << file_metadata->num_columns() << " columns.)";
  throw std::runtime_error(error_msg.str());
}

void validate_number_of_columns(
    const std::shared_ptr<parquet::FileMetaData>& file_metadata,
    const std::string& file_path,
    const ForeignTableSchema& schema) {
  if (schema.numLogicalColumns() != file_metadata->num_columns()) {
    throw_mismatched_column_number_error(file_metadata, file_path, schema);
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
  throw std::runtime_error{
      "Parquet file has a row group size that is larger than the fragment size. "
      "Please set the table fragment size to a number that is larger than the "
      "row group size. Row group index: " +
      std::to_string(row_group_index) +
      ", row group size: " + std::to_string(max_row_group_size) +
      ", fragment size: " + std::to_string(fragment_size) + ", file path: " + file_path};
}

void validate_column_mapping_and_row_group_metadata(
    const std::shared_ptr<parquet::FileMetaData>& file_metadata,
    const std::string& file_path,
    const ForeignTableSchema& schema) {
  auto column_it = schema.getLogicalColumns().begin();
  for (int i = 0; i < file_metadata->num_columns(); ++i, ++column_it) {
    const parquet::ColumnDescriptor* descr = file_metadata->schema()->Column(i);
    validate_allowed_mapping(descr, *column_it);

    auto fragment_size = schema.getForeignTable()->maxFragRows;
    int64_t max_row_group_size = 0;
    int max_row_group_index = 0;
    for (int r = 0; r < file_metadata->num_row_groups(); ++r) {
      auto group_metadata = file_metadata->RowGroup(r);
      auto num_rows = group_metadata->num_rows();
      if (num_rows > max_row_group_size) {
        max_row_group_size = num_rows;
        max_row_group_index = r;
      }

      auto column_chunk = group_metadata->ColumnChunk(i);
      bool contains_metadata = column_chunk->is_stats_set();
      if (contains_metadata) {
        auto stats = column_chunk->statistics();
        bool is_all_nulls = stats->null_count() == column_chunk->num_values();
        if (!stats->HasMinMax() && !is_all_nulls) {
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

void metadata_scan_rowgroup_interval(
    const std::map<int, std::shared_ptr<ParquetEncoder>>& encoder_map,
    const RowGroupInterval& row_group_interval,
    const std::unique_ptr<parquet::arrow::FileReader>& reader,
    const ForeignTableSchema& schema,
    std::list<RowGroupMetadata>& row_group_metadata) {
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
      auto metadata = encoder_map_iter->second->getRowGroupMetadata(
          group_metadata.get(), parquet_column_index, column_descriptor->columnType);
      row_group_metadata_item.column_chunk_metadata.emplace_back(metadata);
    }
  }
}

void populate_encoder_map(std::map<int, std::shared_ptr<ParquetEncoder>>& encoder_map,
                          const Interval<ColumnType>& column_interval,
                          const ForeignTableSchema& schema,
                          const std::unique_ptr<parquet::arrow::FileReader>& reader) {
  auto file_metadata = reader->parquet_reader()->metadata();
  for (int column_id = column_interval.start; column_id <= column_interval.end;
       column_id++) {
    const auto column_descriptor = schema.getColumnDescriptor(column_id);
    auto parquet_column_descriptor =
        file_metadata->schema()->Column(schema.getParquetColumnIndex(column_id));
    encoder_map[column_id] =
        create_parquet_encoder(column_descriptor, parquet_column_descriptor);
    column_id += column_descriptor->columnType.get_physical_cols();
  }
}

}  // namespace

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
    std::shared_ptr<arrow::fs::FileSystem> file_system)
    : file_system_(file_system) {}

std::list<std::unique_ptr<ChunkMetadata>> LazyParquetChunkLoader::loadChunk(
    const std::vector<RowGroupInterval>& row_group_intervals,
    const int parquet_column_index,
    std::list<Chunk_NS::Chunk>& chunks,
    StringDictionary* string_dictionary) {
  CHECK(!chunks.empty());
  auto const& chunk = *chunks.begin();
  auto column_descriptor = chunk.getColumnDesc();
  auto buffer = chunk.getBuffer();
  CHECK(buffer);

  auto metadata = append_row_groups(row_group_intervals,
                                    parquet_column_index,
                                    column_descriptor,
                                    chunks,
                                    string_dictionary,
                                    file_system_);
  return metadata;
}

std::list<RowGroupMetadata> LazyParquetChunkLoader::metadataScan(
    const std::set<std::string>& file_paths,
    const ForeignTableSchema& schema) {
  std::list<RowGroupMetadata> row_group_metadata;
  auto column_interval =
      Interval<ColumnType>{schema.getLogicalAndPhysicalColumns().front()->columnId,
                           schema.getLogicalAndPhysicalColumns().back()->columnId};
  CHECK(!file_paths.empty());
  std::unique_ptr<parquet::arrow::FileReader> first_file_reader;
  const auto& first_file_path = *file_paths.begin();
  open_parquet_table(first_file_path, first_file_reader, file_system_);
  std::map<int, std::shared_ptr<ParquetEncoder>> encoder_map;
  for (const auto& file_path : file_paths) {
    std::unique_ptr<parquet::arrow::FileReader> reader;
    open_parquet_table(file_path, reader, file_system_);
    validate_equal_schema(
        first_file_reader.get(), reader.get(), first_file_path, file_path);
    int num_row_groups = get_parquet_table_size(reader).first;
    auto row_group_interval = RowGroupInterval{file_path, 0, num_row_groups - 1};
    validate_parquet_metadata(reader->parquet_reader()->metadata(), file_path, schema);
    if (file_path == first_file_path) {
      populate_encoder_map(encoder_map, column_interval, schema, first_file_reader);
    }
    metadata_scan_rowgroup_interval(
        encoder_map, row_group_interval, reader, schema, row_group_metadata);
  }
  return row_group_metadata;
}

}  // namespace foreign_storage
