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

#include "Catalog/Catalog.h"
#include "Catalog/ColumnDescriptor.h"
#include "Catalog/SysCatalog.h"
#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/ForeignStorage/SharedMetadataValidator.h"
#include "DataMgr/StringNoneEncoder.h"
#include "Shared/types.h"
#include "StringDictionary/StringDictionary.h"

#include <stdexcept>

namespace data_conversion {

class BaseConvertEncoder {
 public:
  virtual ~BaseConvertEncoder(){};

  BaseConvertEncoder(const bool error_tracking_enabled)
      : delete_buffer_(std::nullopt)
      , error_tracking_enabled_(error_tracking_enabled)
      , has_nulls_(false)
      , num_elements_(0) {}

  void initializeDeleteBuffer(const size_t size_hint) {
    if (!delete_buffer_.has_value()) {
      delete_buffer_ = std::vector<bool>{};
    }
    delete_buffer_->clear();
    delete_buffer_->reserve(size_hint);
  }

  virtual void finalize(const size_t rows_appended) { num_elements_ = rows_appended; }

  virtual void encodeAndAppendData(const int8_t* data, const size_t num_elements) = 0;

  virtual std::shared_ptr<ChunkMetadata> getMetadata(const Chunk_NS::Chunk& chunk) const {
    auto chunk_metadata = std::make_shared<ChunkMetadata>();
    auto dst_type_info = chunk.getColumnDesc()->columnType;
    chunk_metadata->sqlType = dst_type_info;
    chunk_metadata->chunkStats.has_nulls = has_nulls_;
    chunk_metadata->numElements = num_elements_;
    chunk_metadata->numBytes = chunk.getBuffer()->size();
    return chunk_metadata;
  }

 protected:
  virtual void clear() {
    num_elements_ = 0;
    if (error_tracking_enabled_) {
      delete_buffer_->clear();
    }
  }

  template <typename DataType>
  bool isNull(const DataType& typed_value) {
    if constexpr (std::is_arithmetic<DataType>::value) {
      auto null = foreign_storage::get_null_value<DataType>();
      if (typed_value == null) {
        return true;
      }
    } else if constexpr (std::is_same<DataType, std::string>::value ||
                         std::is_same<DataType, std::string_view>::value) {
      if (typed_value.empty()) {
        return true;
      }
    }
    return false;
  }

  std::optional<std::vector<bool>> delete_buffer_;
  const bool error_tracking_enabled_;

  bool has_nulls_;
  size_t num_elements_;
};

template <typename DataType_, typename MetadataType_ = DataType_>
class TypedBaseConvertEncoder : public BaseConvertEncoder {
 public:
  using DataType = DataType_;
  using MetadataType = MetadataType_;

  TypedBaseConvertEncoder(const bool error_tracking_enabled)
      : BaseConvertEncoder(error_tracking_enabled) {
    min_ = std::numeric_limits<MetadataType>::max();
    max_ = std::numeric_limits<MetadataType>::lowest();
  }

  std::shared_ptr<ChunkMetadata> getMetadata(
      const Chunk_NS::Chunk& chunk) const override {
    auto metadata = BaseConvertEncoder::getMetadata(chunk);
    metadata->fillChunkStats(min_, max_, has_nulls_);
    return metadata;
  }

 protected:
  DataType getNull() const {
    if constexpr (std::is_arithmetic<DataType>::value) {
      auto null = foreign_storage::get_null_value<DataType>();
      return null;
    } else if constexpr (std::is_same<DataType, std::string>::value ||
                         std::is_same<DataType, std::string_view>::value) {
      return std::string{};  // empty_string
    } else {
      return nullptr;
    }
  }

  void updateMetadataStats(const DataType& typed_value,
                           const bool is_date_in_days = false) {
    if (is_date_in_days) {
      const MetadataType to_compare =
          DateConverters::get_epoch_seconds_from_days(typed_value);
      min_ = std::min<MetadataType>(min_, to_compare);
      max_ = std::max<MetadataType>(max_, to_compare);
    } else {
      min_ = std::min<MetadataType>(min_, typed_value);
      max_ = std::max<MetadataType>(max_, typed_value);
    }
  }

  MetadataType min_, max_;
};

}  // namespace data_conversion
