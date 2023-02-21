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
#include "StringViewToArrayEncoder.h"

namespace data_conversion {

template <typename DataType, typename MetadataType = DataType>
class StringViewToScalarEncoder : public TypedBaseConvertEncoder<DataType, MetadataType> {
 public:
  StringViewToScalarEncoder(const Chunk_NS::Chunk& dst_chunk,
                            const bool error_tracking_enabled)
      : TypedBaseConvertEncoder<DataType, MetadataType>(error_tracking_enabled)
      , dst_chunk_(dst_chunk)
      , is_date_in_days_(dst_chunk_.getColumnDesc()->columnType.is_date_in_days())
      , date_days_overflow_validator_(std::nullopt)
      , decimal_overflow_validator_(std::nullopt) {
    initialize();
  }

  void encodeAndAppendData(const int8_t* data, const size_t num_elements) override {
    auto typed_data = reinterpret_cast<const std::string_view*>(data);
    for (size_t i = 0; i < num_elements; ++i) {
      auto converted_value = convertAndUpdateMetadata(typed_data[i]);
      buffer_->append(reinterpret_cast<int8_t*>(&converted_value), sizeof(DataType));
    }
  }

  void clear() override {
    buffer_->resetToEmpty();
    BaseConvertEncoder::clear();
  }

  const Chunk_NS::Chunk& getDstChunk() const { return dst_chunk_; }

  std::optional<std::vector<bool>>& getDeleteBuffer() {
    return BaseConvertEncoder::delete_buffer_;
  }

 private:
  void initialize() {
    auto type_info = dst_chunk_.getColumnDesc()->columnType;
    dst_type_info_ = type_info;
    buffer_ = dst_chunk_.getBuffer();
    if (is_date_in_days_) {
      date_days_overflow_validator_ = DateDaysOverflowValidator{dst_type_info_};
    }
    if (dst_type_info_.is_decimal()) {
      decimal_overflow_validator_ = DecimalOverflowValidator(dst_type_info_);
    }
  }

  DataType convertAndUpdateMetadata(const std::string_view& typed_value) {
    if (BaseConvertEncoder::isNull(typed_value)) {
      if (!BaseConvertEncoder::error_tracking_enabled_) {
        if (dst_type_info_.get_notnull()) {
          throw std::runtime_error("NULL value not allowed in NOT NULL column");
        }
        BaseConvertEncoder::has_nulls_ = true;
      } else {
        if (dst_type_info_.get_notnull()) {
          BaseConvertEncoder::delete_buffer_->push_back(true);
        } else {
          BaseConvertEncoder::delete_buffer_->push_back(false);
        }
      }
      return TypedBaseConvertEncoder<DataType, MetadataType>::getNull();
    }

    DataType converted_value{};
    try {
      converted_value = convert(typed_value);
      TypedBaseConvertEncoder<DataType, MetadataType>::updateMetadataStats(
          converted_value, is_date_in_days_);
      if (BaseConvertEncoder::error_tracking_enabled_) {
        BaseConvertEncoder::delete_buffer_->push_back(false);
      }
    } catch (std::exception& except) {
      if (BaseConvertEncoder::error_tracking_enabled_) {
        converted_value = TypedBaseConvertEncoder<DataType, MetadataType>::getNull();
        BaseConvertEncoder::delete_buffer_->push_back(true);
      } else {
        throw;
      }
    }
    return converted_value;
  }

  DataType convert(const std::string_view& typed_value) {
    if constexpr (std::is_same<DataType, std::string_view>::value) {
      if (dst_type_info_.is_none_encoded_string()) {
        return typed_value;
      }
    } else {
      CHECK(!dst_type_info_.is_none_encoded_string());

      auto& type_info = dst_type_info_;

      // TODO: remove this CHECK if it shows up in profiling
      CHECK(type_info.is_integer() || type_info.is_boolean() || type_info.is_fp() ||
            type_info.is_decimal() || type_info.is_time_or_date());

      // TODO: the call to `StringToDatum` and the switch statement below can be
      // merged into one switch calling the appropriate parsing subroutine and may
      // improve performance. Profile if improvement is observed.
      Datum d = StringToDatum(typed_value, const_cast<SQLTypeInfo&>(type_info));
      DataType result{};
      switch (type_info.get_type()) {
        case kBOOLEAN:
          result = d.boolval;
          break;
        case kBIGINT:
        case kTIME:
        case kTIMESTAMP:
          result = d.bigintval;
          break;
        case kNUMERIC:
        case kDECIMAL:
          if (type_info.get_compression() == kENCODING_FIXED) {
            decimal_overflow_validator_->validate(d.bigintval);
          }
          result = d.bigintval;
          break;
        case kDATE:
          if (is_date_in_days_) {
            date_days_overflow_validator_->validate(d.bigintval);
            result = DateConverters::get_epoch_days_from_seconds(d.bigintval);
          } else {
            result = d.bigintval;
          }
          break;
        case kINT:
          result = d.intval;
          break;
        case kSMALLINT:
          result = d.smallintval;
          break;
        case kTINYINT:
          result = d.tinyintval;
          break;
        case kFLOAT:
          result = d.floatval;
          break;
        case kDOUBLE:
          result = d.doubleval;
          break;
        default:
          UNREACHABLE();
      }

      return result;
    }
  }

  SQLTypeInfo dst_type_info_;
  const Chunk_NS::Chunk& dst_chunk_;
  AbstractBuffer* buffer_;

  const bool is_date_in_days_;  // stored as a const bool to aid compiler in optimizing
                                // code paths related to this flag

  std::optional<DateDaysOverflowValidator> date_days_overflow_validator_;
  std::optional<DecimalOverflowValidator> decimal_overflow_validator_;
};

}  // namespace data_conversion
