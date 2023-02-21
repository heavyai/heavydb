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

namespace data_conversion {

class StringViewToStringNoneEncoder : public BaseConvertEncoder {
 public:
  StringViewToStringNoneEncoder(const Chunk_NS::Chunk& dst_chunk,
                                const bool error_tracking_enabled)
      : BaseConvertEncoder(error_tracking_enabled), dst_chunk_(dst_chunk) {
    initialize();
  }

  void encodeAndAppendData(const int8_t* data, const size_t num_elements) override {
    auto typed_data = reinterpret_cast<const std::string_view*>(data);
    if (BaseConvertEncoder::error_tracking_enabled_) {
      // scan for any strings that may be an error
      bool tracking_individual_strings_required = false;
      size_t first_error_index = 0;
      for (size_t i = 0; i < num_elements; ++i) {
        if (typed_data[i].size() > StringDictionary::MAX_STRLEN) {
          tracking_individual_strings_required = true;
          first_error_index = i;
          break;
        }
      }

      if (!tracking_individual_strings_required) {
        for (size_t i = 0; i < num_elements; ++i) {
          delete_buffer_->push_back(false);
        }
        StringNoneEncoder* encoder = getEncoder();
        auto metadata = encoder->appendData(typed_data, 0, num_elements, false);
        has_nulls_ |= metadata->chunkStats.has_nulls;
      } else {
        std::vector<std::string_view> tracked_strings(num_elements);
        for (size_t i = first_error_index; i < num_elements; ++i) {
          if (typed_data[i].size() > StringDictionary::MAX_STRLEN) {
            tracked_strings[i] = {};
            delete_buffer_->push_back(true);
          } else {
            tracked_strings[i] = typed_data[i];
            delete_buffer_->push_back(false);
          }
          auto metadata =
              getEncoder()->appendData(tracked_strings.data(), 0, num_elements, false);
          has_nulls_ |= metadata->chunkStats.has_nulls;
        }
      }
    } else {
      for (size_t i = 0; i < num_elements; ++i) {
        if (typed_data[i].size() > StringDictionary::MAX_STRLEN) {
          throw std::runtime_error("String length of " +
                                   std::to_string(typed_data[i].size()) +
                                   " exceeds allowed maximum string length of " +
                                   std::to_string(StringDictionary::MAX_STRLEN));
        }
      }
      auto metadata = dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder())
                          ->appendData(typed_data, 0, num_elements, false);
      has_nulls_ |= metadata->chunkStats.has_nulls;
    }
  }
  StringNoneEncoder* getEncoder() const {
    auto encoder = dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder());
    CHECK(encoder);
    return encoder;
  }

 private:
  void initialize() {
    auto type_info = dst_chunk_.getColumnDesc()->columnType;
    dst_type_info_ = type_info;
    buffer_ = dst_chunk_.getBuffer();
  }

  SQLTypeInfo dst_type_info_;
  const Chunk_NS::Chunk& dst_chunk_;
  AbstractBuffer* buffer_;
};

}  // namespace data_conversion
