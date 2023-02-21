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

template <typename IdType>
class StringViewToStringDictEncoder : public TypedBaseConvertEncoder<IdType> {
 public:
  StringViewToStringDictEncoder(const Chunk_NS::Chunk& dst_chunk,
                                const bool error_tracking_enabled)
      : TypedBaseConvertEncoder<IdType>(error_tracking_enabled), dst_chunk_(dst_chunk) {
    initialize();
  }

  void encodeAndAppendData(const int8_t* data, const size_t num_elements) override {
    auto typed_data = reinterpret_cast<const std::string_view*>(data);

    CHECK(!BaseConvertEncoder::error_tracking_enabled_)
        << " unimplemented case for this encoder";

    std::vector<std::string_view> input_buffer(typed_data, typed_data + num_elements);
    CHECK(string_dict_);
    dict_encoding_output_buffer_.resize(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      if (input_buffer[i].size() > StringDictionary::MAX_STRLEN) {
        if (!BaseConvertEncoder::error_tracking_enabled_) {
          throw std::runtime_error("String length of " +
                                   std::to_string(input_buffer[i].size()) +
                                   " exceeds allowed maximum string length of " +
                                   std::to_string(StringDictionary::MAX_STRLEN));
        } else {
          BaseConvertEncoder::delete_buffer_->push_back(true);
          input_buffer[i] = {};  // set to NULL/empty string to process
        }
      } else if (BaseConvertEncoder::isNull(input_buffer[i])) {
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
      } else {
        if (BaseConvertEncoder::error_tracking_enabled_) {
          BaseConvertEncoder::delete_buffer_->push_back(false);
        }
      }
    }
    string_dict_->getOrAddBulk<IdType, std::string_view>(
        input_buffer, dict_encoding_output_buffer_.data());
    for (size_t i = 0; i < num_elements; ++i) {
      if (!BaseConvertEncoder::isNull(input_buffer[i])) {
        TypedBaseConvertEncoder<IdType>::updateMetadataStats(
            dict_encoding_output_buffer_[i]);
      }
    }

    buffer_->append(reinterpret_cast<int8_t*>(dict_encoding_output_buffer_.data()),
                    num_elements * sizeof(IdType));
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
    auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(
        type_info.getStringDictKey().db_id);
    string_dict_ = catalog->getMetadataForDict(type_info.getStringDictKey().dict_id, true)
                       ->stringDict.get();
    dst_type_info_ = type_info;
    buffer_ = dst_chunk_.getBuffer();
  }

  SQLTypeInfo dst_type_info_;
  const Chunk_NS::Chunk& dst_chunk_;
  StringDictionary* string_dict_;

  std::vector<IdType> dict_encoding_output_buffer_;

  AbstractBuffer* buffer_;
};

}  // namespace data_conversion
