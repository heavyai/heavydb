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
#include "BaseConvertEncoder.h"

namespace data_conversion {

namespace {

template <typename DataType>
bool is_null(const DataType& typed_value) {
  if constexpr (std::is_arithmetic<DataType>::value) {
    auto null = foreign_storage::get_null_value<DataType>();
    if (typed_value == null) {
      return true;
    } else {
      return false;
    }
  }

  UNREACHABLE();

  return false;
}

template <typename T>
std::vector<std::string_view> get_materialized_string_views(
    const size_t num_elements,
    const StringDictionary* string_dict,
    const T* ids) {
  std::vector<std::string_view> materialized_string_views(num_elements);
  std::transform(ids,
                 ids + num_elements,
                 materialized_string_views.begin(),
                 [&string_dict](const T& id) -> std::string_view {
                   if (is_null(id)) {
                     return std::string_view(nullptr, 0);
                   }
                   return string_dict->getStringView(id);
                 });
  return materialized_string_views;
}

std::vector<std::string_view> get_materialized_string_views(
    const size_t num_elements,
    const StringDictionary* string_dict,
    const int8_t* ids,
    const SQLTypeInfo& type_info) {
  switch (type_info.get_size()) {
    case 1:
      return get_materialized_string_views(
          num_elements, string_dict, reinterpret_cast<const uint8_t*>(ids));
      break;
    case 2:
      return get_materialized_string_views(
          num_elements, string_dict, reinterpret_cast<const uint16_t*>(ids));
      break;
    case 4:
      return get_materialized_string_views(
          num_elements, string_dict, reinterpret_cast<const int32_t*>(ids));
      break;
    default:
      UNREACHABLE();
  }

  return {};
}
}  // namespace

class BaseSource {
 public:
  virtual ~BaseSource() = default;

  virtual std::pair<const int8_t*, size_t> getSourceData() = 0;
};

class StringViewSource : public BaseSource {
 public:
  StringViewSource(const Chunk_NS::Chunk& input) : input_(input) {}

  std::pair<const int8_t*, size_t> getSourceData() override {
    auto buffer = input_.getBuffer();
    auto src_type_info = input_.getColumnDesc()->columnType;
    auto dict_key = src_type_info.getStringDictKey();
    auto num_elements = buffer->getEncoder()->getNumElems();

    if (src_type_info.is_dict_encoded_string()) {
      auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(dict_key.db_id);

      auto src_string_dictionary =
          catalog->getMetadataForDict(dict_key.dict_id, true)->stringDict.get();
      CHECK(src_string_dictionary);
      src_string_views_ = get_materialized_string_views(
          num_elements, src_string_dictionary, buffer->getMemoryPtr(), src_type_info);
    } else if (src_type_info.is_none_encoded_string()) {
      auto index_buffer = input_.getIndexBuf();
      src_string_views_.resize(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        src_string_views_[i] = StringNoneEncoder::getStringAtIndex(
            index_buffer->getMemoryPtr(), buffer->getMemoryPtr(), i);
      }
    } else {
      UNREACHABLE() << "unknown string type";
    }

    return {reinterpret_cast<int8_t*>(src_string_views_.data()), num_elements};
  }

 private:
  std::vector<std::string_view> src_string_views_;

  const Chunk_NS::Chunk& input_;
};

}  // namespace data_conversion
