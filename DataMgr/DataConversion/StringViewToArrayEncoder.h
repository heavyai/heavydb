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

#pragma once

#include "BaseConvertEncoder.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "ImportExport/Importer.h"  // composeNullArray

#include "DataMgr/ArrayNoneEncoder.h"
#include "DataMgr/FixedLengthArrayNoneEncoder.h"

namespace data_conversion {

template <typename ScalarEncoderType>
class StringViewToArrayEncoder
    : public TypedBaseConvertEncoder<typename ScalarEncoderType::DataType,
                                     typename ScalarEncoderType::MetadataType> {
 public:
  using DstDataType = typename ScalarEncoderType::DataType;

  StringViewToArrayEncoder(const Chunk_NS::Chunk& scalar_temp_chunk,
                           const Chunk_NS::Chunk& dst_chunk,
                           const bool error_tracking_enabled)
      : TypedBaseConvertEncoder<typename ScalarEncoderType::DataType,
                                typename ScalarEncoderType::MetadataType>(
            error_tracking_enabled)
      , dst_chunk_(dst_chunk)
      , scalar_temp_chunk_(scalar_temp_chunk)
      , scalar_encoder_(scalar_temp_chunk, error_tracking_enabled) {
    initialize();
  }

  void encodeAndAppendData(const int8_t* data, const size_t num_elements) override {
    auto typed_data = reinterpret_cast<const std::string_view*>(data);

    const import_export::CopyParams default_copy_params;

    std::optional<std::vector<bool>> error_occurred = std::nullopt;

    if (BaseConvertEncoder::error_tracking_enabled_) {
      error_occurred = std::vector<bool>{};
      error_occurred->reserve(num_elements);
    }

    clearLocalState(num_elements);

    encodeScalarData(num_elements, typed_data, default_copy_params, error_occurred);

    auto current_data = reinterpret_cast<int8_t*>(
        scalar_encoder_.getDstChunk().getBuffer()->getMemoryPtr());
    size_t current_scalar_offset = 0;

    appendArrayDatums(num_elements, error_occurred, current_data, current_scalar_offset);
  }

  void appendArrayDatums(const size_t num_elements,
                         std::optional<std::vector<bool>>& error_occurred,
                         int8_t* current_data,
                         size_t current_scalar_offset) {
    for (size_t i = 0; i < num_elements; ++i) {
      auto array_size = array_sizes_[i];
      if (BaseConvertEncoder::error_tracking_enabled_) {
        BaseConvertEncoder::delete_buffer_->push_back(false);
      }
      if (is_null_[i]) {
        if (dst_type_info_.get_notnull()) {
          if (BaseConvertEncoder::error_tracking_enabled_) {
            BaseConvertEncoder::delete_buffer_->back() = true;
          } else {
            throw std::runtime_error("NULL value not allowed in NOT NULL column");
          }
        }
        array_datums_.push_back(
            import_export::ImporterUtils::composeNullArray(dst_type_info_));
      } else {
        if (BaseConvertEncoder::error_tracking_enabled_) {
          for (size_t j = current_scalar_offset; j < current_scalar_offset + array_size;
               ++j) {
            if ((*scalar_encoder_.getDeleteBuffer())[j]) {
              (*error_occurred)[i] = true;
              break;
            }
          }
          current_scalar_offset += array_size;
        }
        if (dst_type_info_.is_fixlen_array() &&
            array_size * sizeof(DstDataType) !=
                static_cast<size_t>(dst_type_info_.get_size())) {
          if (BaseConvertEncoder::error_tracking_enabled_) {
            array_datums_.push_back(
                import_export::ImporterUtils::composeNullArray(dst_type_info_));
            current_data += sizeof(DstDataType) * array_size;
            BaseConvertEncoder::delete_buffer_->back() = true;
            continue;
          } else {
            throw std::runtime_error(
                "Incorrect number of elements (" + std::to_string(array_size) +
                ") in array for fixed length array of size " +
                std::to_string(dst_type_info_.get_size() / sizeof(DstDataType)));
          }
        } else {
          if (BaseConvertEncoder::error_tracking_enabled_) {
            if ((*error_occurred)[i]) {
              array_datums_.push_back(
                  import_export::ImporterUtils::composeNullArray(dst_type_info_));
              BaseConvertEncoder::delete_buffer_->back() = true;
              continue;
            }
          }
          array_datums_.emplace_back(
              sizeof(DstDataType) * array_size, current_data, false, DoNothingDeleter{});
          current_data += sizeof(DstDataType) * array_size;
        }
      }
    }

    if (dst_type_info_.is_varlen_array()) {
      auto encoder = dynamic_cast<ArrayNoneEncoder*>(buffer_->getEncoder());
      CHECK(encoder);
      encoder->appendData(&array_datums_, 0, num_elements, false);
    } else if (dst_type_info_.is_fixlen_array()) {
      auto encoder = dynamic_cast<FixedLengthArrayNoneEncoder*>(buffer_->getEncoder());
      CHECK(encoder);
      encoder->appendData(&array_datums_, 0, num_elements, false);
    } else {
      UNREACHABLE();
    }
  }

  void clearLocalState(const size_t num_elements) {
    scalar_encoder_.clear();
    array_datums_.clear();
    array_sizes_.clear();
    is_null_.clear();
    array_sizes_.reserve(num_elements);
    array_datums_.reserve(num_elements);
    is_null_.reserve(num_elements);
  }

  void encodeScalarData(const size_t num_elements,
                        const std::string_view* typed_data,
                        const import_export::CopyParams& default_copy_params,
                        std::optional<std::vector<bool>>& error_occurred) {
    for (size_t i = 0; i < num_elements; ++i) {
      if (BaseConvertEncoder::error_tracking_enabled_) {
        error_occurred->push_back(false);
      }
      if (typed_data[i].empty()) {
        is_null_.push_back(true);
        array_sizes_.push_back(0);
        BaseConvertEncoder::has_nulls_ = true;
        continue;
      }
      is_null_.push_back(false);
      array_.clear();
      array_views_.clear();
      try {
        std::string s{typed_data[i]};
        import_export::delimited_parser::parse_string_array(
            s, default_copy_params, array_, false);
        array_sizes_.push_back(array_.size());
        for (const auto& s : array_) {
          array_views_.emplace_back(s.data(), s.length());
        }
      } catch (std::exception& except) {
        if (BaseConvertEncoder::error_tracking_enabled_) {
          error_occurred->back() = true;
          array_sizes_.push_back(0);
          continue;
        } else {
          throw except;
        }
      }
      scalar_encoder_.encodeAndAppendData(reinterpret_cast<int8_t*>(array_views_.data()),
                                          array_views_.size());
    }

    scalar_encoder_.finalize(num_elements);
  }

 private:
  void initialize() {
    auto type_info = dst_chunk_.getColumnDesc()->columnType;
    dst_type_info_ = type_info;
    buffer_ = dst_chunk_.getBuffer();
  }

  SQLTypeInfo dst_type_info_;
  const Chunk_NS::Chunk& dst_chunk_;
  const Chunk_NS::Chunk& scalar_temp_chunk_;
  AbstractBuffer* buffer_;

  ScalarEncoderType scalar_encoder_;

  std::vector<std::string> array_;
  std::vector<std::string_view> array_views_;
  std::vector<ArrayDatum> array_datums_;
  std::vector<size_t> array_sizes_;
  std::vector<bool> is_null_;
};

}  // namespace data_conversion
