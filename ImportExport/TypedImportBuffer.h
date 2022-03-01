/*
 * Copyright 2022 OmniSci, Inc.
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

#include "Shared/checked_alloc.h"

class TColumn;
class TDatum;
class CopyParams;

namespace arrow {

class Array;

}  // namespace arrow

namespace import_export {

using ArraySliceRange = std::pair<size_t, size_t>;

class Importer;

struct BadRowsTracker {
  std::mutex mutex;
  std::set<int64_t> rows;
  std::atomic<int> nerrors;
  std::string file_name;
  int row_group;
  Importer* importer;
};

class TypedImportBuffer : boost::noncopyable {
 public:
  TypedImportBuffer(const ColumnDescriptor* col_desc, StringDictionary* string_dict)
      : column_desc_(col_desc), string_dict_(string_dict) {
    switch (col_desc->columnType.get_type()) {
      case kBOOLEAN:
        bool_buffer_ = new std::vector<int8_t>();
        break;
      case kTINYINT:
        tinyint_buffer_ = new std::vector<int8_t>();
        break;
      case kSMALLINT:
        smallint_buffer_ = new std::vector<int16_t>();
        break;
      case kINT:
        int_buffer_ = new std::vector<int32_t>();
        break;
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        bigint_buffer_ = new std::vector<int64_t>();
        break;
      case kFLOAT:
        float_buffer_ = new std::vector<float>();
        break;
      case kDOUBLE:
        double_buffer_ = new std::vector<double>();
        break;
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
        string_buffer_ = new std::vector<std::string>();
        if (col_desc->columnType.get_compression() == kENCODING_DICT) {
          switch (col_desc->columnType.get_size()) {
            case 1:
              string_dict_i8_buffer_ = new std::vector<uint8_t>();
              break;
            case 2:
              string_dict_i16_buffer_ = new std::vector<uint16_t>();
              break;
            case 4:
              string_dict_i32_buffer_ = new std::vector<int32_t>();
              break;
            default:
              CHECK(false);
          }
        }
        break;
      case kDATE:
      case kTIME:
      case kTIMESTAMP:
        bigint_buffer_ = new std::vector<int64_t>();
        break;
      case kARRAY:
        if (IS_STRING(col_desc->columnType.get_subtype())) {
          CHECK(col_desc->columnType.get_compression() == kENCODING_DICT);
          string_array_buffer_ = new std::vector<std::vector<std::string>>();
          string_array_dict_buffer_ = new std::vector<ArrayDatum>();
        } else {
          array_buffer_ = new std::vector<ArrayDatum>();
        }
        break;
      default:
        CHECK(false);
    }
  }

  ~TypedImportBuffer() {
    switch (column_desc_->columnType.get_type()) {
      case kBOOLEAN:
        delete bool_buffer_;
        break;
      case kTINYINT:
        delete tinyint_buffer_;
        break;
      case kSMALLINT:
        delete smallint_buffer_;
        break;
      case kINT:
        delete int_buffer_;
        break;
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        delete bigint_buffer_;
        break;
      case kFLOAT:
        delete float_buffer_;
        break;
      case kDOUBLE:
        delete double_buffer_;
        break;
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
        delete string_buffer_;
        if (column_desc_->columnType.get_compression() == kENCODING_DICT) {
          switch (column_desc_->columnType.get_size()) {
            case 1:
              delete string_dict_i8_buffer_;
              break;
            case 2:
              delete string_dict_i16_buffer_;
              break;
            case 4:
              delete string_dict_i32_buffer_;
              break;
          }
        }
        break;
      case kDATE:
      case kTIME:
      case kTIMESTAMP:
        delete bigint_buffer_;
        break;
      case kARRAY:
        if (IS_STRING(column_desc_->columnType.get_subtype())) {
          delete string_array_buffer_;
          delete string_array_dict_buffer_;
        } else {
          delete array_buffer_;
        }
        break;
      default:
        CHECK(false);
    }
  }

  void addBoolean(const int8_t v) { bool_buffer_->push_back(v); }

  void addTinyint(const int8_t v) { tinyint_buffer_->push_back(v); }

  void addSmallint(const int16_t v) { smallint_buffer_->push_back(v); }

  void addInt(const int32_t v) { int_buffer_->push_back(v); }

  void addBigint(const int64_t v) { bigint_buffer_->push_back(v); }

  void addFloat(const float v) { float_buffer_->push_back(v); }

  void addDouble(const double v) { double_buffer_->push_back(v); }

  void addString(const std::string_view v) { string_buffer_->emplace_back(v); }

  void addArray(const ArrayDatum& v) { array_buffer_->push_back(v); }

  std::vector<std::string>& addStringArray() {
    string_array_buffer_->emplace_back();
    return string_array_buffer_->back();
  }

  void addStringArray(const std::vector<std::string>& arr) {
    string_array_buffer_->push_back(arr);
  }

  void addDictEncodedString(const std::vector<std::string>& string_vec);

  void addDictEncodedStringArray(
      const std::vector<std::vector<std::string>>& string_array_vec) {
    CHECK(string_dict_);

    // first check data is ok
    for (auto& p : string_array_vec) {
      for (const auto& str : p) {
        if (str.size() > StringDictionary::MAX_STRLEN) {
          throw std::runtime_error("String too long for dictionary encoding.");
        }
      }
    }

    std::vector<std::vector<int32_t>> ids_array(0);
    string_dict_->getOrAddBulkArray(string_array_vec, ids_array);

    for (auto& p : ids_array) {
      size_t len = p.size() * sizeof(int32_t);
      auto a = static_cast<int32_t*>(checked_malloc(len));
      memcpy(a, &p[0], len);
      // TODO: distinguish between empty and NULL
      string_array_dict_buffer_->push_back(
          ArrayDatum(len, reinterpret_cast<int8_t*>(a), len == 0));
    }
  }

  const SQLTypeInfo& getTypeInfo() const { return column_desc_->columnType; }

  const ColumnDescriptor* getColumnDesc() const { return column_desc_; }

  StringDictionary* getStringDictionary() const { return string_dict_; }

  int8_t* getAsBytes() const {
    switch (column_desc_->columnType.get_type()) {
      case kBOOLEAN:
        return reinterpret_cast<int8_t*>(bool_buffer_->data());
      case kTINYINT:
        return reinterpret_cast<int8_t*>(tinyint_buffer_->data());
      case kSMALLINT:
        return reinterpret_cast<int8_t*>(smallint_buffer_->data());
      case kINT:
        return reinterpret_cast<int8_t*>(int_buffer_->data());
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        return reinterpret_cast<int8_t*>(bigint_buffer_->data());
      case kFLOAT:
        return reinterpret_cast<int8_t*>(float_buffer_->data());
      case kDOUBLE:
        return reinterpret_cast<int8_t*>(double_buffer_->data());
      case kDATE:
      case kTIME:
      case kTIMESTAMP:
        return reinterpret_cast<int8_t*>(bigint_buffer_->data());
      default:
        abort();
    }
  }

  size_t getElementSize() const {
    switch (column_desc_->columnType.get_type()) {
      case kBOOLEAN:
        return sizeof((*bool_buffer_)[0]);
      case kTINYINT:
        return sizeof((*tinyint_buffer_)[0]);
      case kSMALLINT:
        return sizeof((*smallint_buffer_)[0]);
      case kINT:
        return sizeof((*int_buffer_)[0]);
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        return sizeof((*bigint_buffer_)[0]);
      case kFLOAT:
        return sizeof((*float_buffer_)[0]);
      case kDOUBLE:
        return sizeof((*double_buffer_)[0]);
      case kDATE:
      case kTIME:
      case kTIMESTAMP:
        return sizeof((*bigint_buffer_)[0]);
      default:
        abort();
    }
  }

  std::vector<std::string>* getStringBuffer() const { return string_buffer_; }

  std::vector<ArrayDatum>* getArrayBuffer() const { return array_buffer_; }

  std::vector<std::vector<std::string>>* getStringArrayBuffer() const {
    return string_array_buffer_;
  }

  std::vector<ArrayDatum>* getStringArrayDictBuffer() const {
    return string_array_dict_buffer_;
  }

  int8_t* getStringDictBuffer() const {
    switch (column_desc_->columnType.get_size()) {
      case 1:
        return reinterpret_cast<int8_t*>(string_dict_i8_buffer_->data());
      case 2:
        return reinterpret_cast<int8_t*>(string_dict_i16_buffer_->data());
      case 4:
        return reinterpret_cast<int8_t*>(string_dict_i32_buffer_->data());
      default:
        abort();
    }
  }

  bool stringDictCheckpoint() {
    if (string_dict_ == nullptr) {
      return true;
    }
    return string_dict_->checkpoint();
  }

  void clear() {
    switch (column_desc_->columnType.get_type()) {
      case kBOOLEAN: {
        bool_buffer_->clear();
        break;
      }
      case kTINYINT: {
        tinyint_buffer_->clear();
        break;
      }
      case kSMALLINT: {
        smallint_buffer_->clear();
        break;
      }
      case kINT: {
        int_buffer_->clear();
        break;
      }
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL: {
        bigint_buffer_->clear();
        break;
      }
      case kFLOAT: {
        float_buffer_->clear();
        break;
      }
      case kDOUBLE: {
        double_buffer_->clear();
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR: {
        string_buffer_->clear();
        if (column_desc_->columnType.get_compression() == kENCODING_DICT) {
          switch (column_desc_->columnType.get_size()) {
            case 1:
              string_dict_i8_buffer_->clear();
              break;
            case 2:
              string_dict_i16_buffer_->clear();
              break;
            case 4:
              string_dict_i32_buffer_->clear();
              break;
            default:
              CHECK(false);
          }
        }
        break;
      }
      case kDATE:
      case kTIME:
      case kTIMESTAMP:
        bigint_buffer_->clear();
        break;
      case kARRAY: {
        if (IS_STRING(column_desc_->columnType.get_subtype())) {
          string_array_buffer_->clear();
          string_array_dict_buffer_->clear();
        } else {
          array_buffer_->clear();
        }
        break;
      }
      default:
        CHECK(false);
    }
  }

  size_t add_values(const ColumnDescriptor* cd, const TColumn& data);

  size_t add_arrow_values(const ColumnDescriptor* cd,
                          const arrow::Array& data,
                          const bool exact_type_match,
                          const ArraySliceRange& slice_range,
                          BadRowsTracker* bad_rows_tracker);

  void add_value(const ColumnDescriptor* cd,
                 const std::string_view val,
                 const bool is_null,
                 const CopyParams& copy_params);

  void add_value(const ColumnDescriptor* cd, const TDatum& val, const bool is_null);

  void addDefaultValues(const ColumnDescriptor* cd, size_t num_rows);

  void pop_value();

  template <typename DATA_TYPE>
  size_t convert_arrow_val_to_import_buffer(const ColumnDescriptor* cd,
                                            const arrow::Array& array,
                                            std::vector<DATA_TYPE>& buffer,
                                            const ArraySliceRange& slice_range,
                                            BadRowsTracker* const bad_rows_tracker);
  template <typename DATA_TYPE>
  auto del_values(std::vector<DATA_TYPE>& buffer, BadRowsTracker* const bad_rows_tracker);
  auto del_values(const SQLTypes type, BadRowsTracker* const bad_rows_tracker);

  static std::vector<DataBlockPtr> get_data_block_pointers(
      const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers);

  std::vector<std::unique_ptr<TypedImportBuffer>>* import_buffers;
  size_t col_idx;

 private:
  union {
    std::vector<int8_t>* bool_buffer_;
    std::vector<int8_t>* tinyint_buffer_;
    std::vector<int16_t>* smallint_buffer_;
    std::vector<int32_t>* int_buffer_;
    std::vector<int64_t>* bigint_buffer_;
    std::vector<float>* float_buffer_;
    std::vector<double>* double_buffer_;
    std::vector<std::string>* string_buffer_;
    std::vector<ArrayDatum>* array_buffer_;
    std::vector<std::vector<std::string>>* string_array_buffer_;
  };
  union {
    std::vector<uint8_t>* string_dict_i8_buffer_;
    std::vector<uint16_t>* string_dict_i16_buffer_;
    std::vector<int32_t>* string_dict_i32_buffer_;
    std::vector<ArrayDatum>* string_array_dict_buffer_;
  };
  const ColumnDescriptor* column_desc_;
  StringDictionary* string_dict_;
};

}  // namespace import_export
