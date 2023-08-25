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

/*
 * @file Importer.h
 * @brief Importer class for table import from file
 *
 */

#ifndef _IMPORTER_H_
#define _IMPORTER_H_

#include <atomic>
#include <boost/filesystem.hpp>
#include <boost/noncopyable.hpp>
#include <boost/tokenizer.hpp>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "AbstractImporter.h"
#include "Catalog/Catalog.h"
#include "Catalog/TableDescriptor.h"
#include "DataMgr/Chunk/Chunk.h"
#if defined(ENABLE_IMPORT_PARQUET)
#include "DataMgr/ForeignStorage/DataPreview.h"
#endif
#include "Fragmenter/Fragmenter.h"
#include "Geospatial/GDAL.h"
#include "ImportExport/CopyParams.h"
#include "Logger/Logger.h"
#include "Shared/ThreadController.h"
#include "Shared/checked_alloc.h"
#include "Shared/fixautotools.h"
// Some builds of boost::geometry require iostream, but don't explicitly include it.
// Placing in own section to ensure it's included after iostream.
#include <boost/geometry/index/rtree.hpp>

class TDatum;
class TColumn;

namespace arrow {

class Array;

}  // namespace arrow

namespace import_export {

class Importer;

using ArraySliceRange = std::pair<size_t, size_t>;

struct BadRowsTracker {
  std::mutex mutex;
  std::set<int64_t> rows;
  std::atomic<int> nerrors;
  std::string file_name;
  int row_group;
  Importer* importer;
};

class ImporterUtils {
 public:
  static ArrayDatum composeNullArray(const SQLTypeInfo& ti);
  static ArrayDatum composeNullPointCoords(const SQLTypeInfo& coords_ti,
                                           const SQLTypeInfo& geo_ti);
};

class TypedImportBuffer : boost::noncopyable {
 public:
  using OptionalStringVector = std::optional<std::vector<std::string>>;
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
          string_array_buffer_ = new std::vector<OptionalStringVector>();
          string_array_dict_buffer_ = new std::vector<ArrayDatum>();
        } else {
          array_buffer_ = new std::vector<ArrayDatum>();
        }
        break;
      case kPOINT:
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        geo_string_buffer_ = new std::vector<std::string>();
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
      case kPOINT:
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        delete geo_string_buffer_;
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

  void addDictStringWithTruncation(std::string_view v) {
    if (v.size() > StringDictionary::MAX_STRLEN) {
      v = v.substr(0, StringDictionary::MAX_STRLEN);
    }
    string_buffer_->emplace_back(v);
  }

  void addGeoString(const std::string_view v) { geo_string_buffer_->emplace_back(v); }

  void addArray(const ArrayDatum& v) { array_buffer_->push_back(v); }

  OptionalStringVector& addStringArray() {
    string_array_buffer_->emplace_back(std::vector<std::string>{});
    return string_array_buffer_->back();
  }

  void addStringArray(const OptionalStringVector& arr) {
    string_array_buffer_->push_back(arr);
  }

  void addDictEncodedString(const std::vector<std::string>& string_vec);

  void addDictEncodedStringArray(
      const std::vector<OptionalStringVector>& string_array_vec) {
    CHECK(string_dict_);

    // first check data is ok
    for (auto& p : string_array_vec) {
      if (!p) {
        continue;
      }
      for (const auto& str : *p) {
        if (str.size() > StringDictionary::MAX_STRLEN) {
          throw std::runtime_error("String too long for dictionary encoding.");
        }
      }
    }

    // to avoid copying, create a string view of each string in the
    // `string_array_vec` where the array holding the string is *not null*
    std::vector<std::vector<std::string_view>> string_view_array_vec;
    for (auto& p : string_array_vec) {
      if (!p) {
        continue;
      }
      auto& array = string_view_array_vec.emplace_back();
      for (const auto& str : *p) {
        array.emplace_back(str);
      }
    }

    std::vector<std::vector<int32_t>> ids_array(0);
    string_dict_->getOrAddBulkArray(string_view_array_vec, ids_array);

    size_t i, j;
    for (i = 0, j = 0; i < string_array_vec.size(); ++i) {
      if (!string_array_vec[i]) {  // null array
        string_array_dict_buffer_->push_back(
            ImporterUtils::composeNullArray(column_desc_->columnType));
      } else {  // non-null array
        auto& p = ids_array[j++];
        size_t len = p.size() * sizeof(int32_t);
        auto a = static_cast<int32_t*>(checked_malloc(len));
        memcpy(a, &p[0], len);
        string_array_dict_buffer_->push_back(
            ArrayDatum(len, reinterpret_cast<int8_t*>(a), false));
      }
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

  std::vector<std::string>* getGeoStringBuffer() const { return geo_string_buffer_; }

  std::vector<ArrayDatum>* getArrayBuffer() const { return array_buffer_; }

  std::vector<OptionalStringVector>* getStringArrayBuffer() const {
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
      case kPOINT:
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        geo_string_buffer_->clear();
        break;
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
                 const CopyParams& copy_params,
                 const bool check_not_null = true);

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
    std::vector<std::string>* geo_string_buffer_;
    std::vector<ArrayDatum>* array_buffer_;
    std::vector<OptionalStringVector>* string_array_buffer_;
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

class Loader {
  using LoadCallbackType =
      std::function<bool(const std::vector<std::unique_ptr<TypedImportBuffer>>&,
                         std::vector<DataBlockPtr>&,
                         size_t)>;

 public:
  // ParquetDataWrapper
  Loader(Catalog_Namespace::Catalog& c,
         const TableDescriptor* t,
         LoadCallbackType load_callback = nullptr)
      : catalog_(c)
      , table_desc_(t)
      , column_descs_(c.getAllColumnMetadataForTable(t->tableId, false, false, true))
      , load_callback_(load_callback) {
    init();
  }

  virtual ~Loader() {}

  Catalog_Namespace::Catalog& getCatalog() const { return catalog_; }
  const TableDescriptor* getTableDesc() const { return table_desc_; }
  const std::list<const ColumnDescriptor*>& get_column_descs() const {
    return column_descs_;
  }

  StringDictionary* getStringDict(const ColumnDescriptor* cd) const {
    if ((cd->columnType.get_type() != kARRAY ||
         !IS_STRING(cd->columnType.get_subtype())) &&
        (!cd->columnType.is_string() ||
         cd->columnType.get_compression() != kENCODING_DICT)) {
      return nullptr;
    }
    return dict_map_.at(cd->columnId);
  }

  virtual bool load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                    const size_t row_count,
                    const Catalog_Namespace::SessionInfo* session_info);
  virtual bool loadNoCheckpoint(
      const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
      const size_t row_count,
      const Catalog_Namespace::SessionInfo* session_info);
  virtual void checkpoint();
  virtual std::vector<Catalog_Namespace::TableEpochInfo> getTableEpochs() const;
  virtual void setTableEpochs(
      const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs);

  void setAddingColumns(const bool adding_columns) { adding_columns_ = adding_columns; }
  bool isAddingColumns() const { return adding_columns_; }
  void dropColumns(const std::vector<int>& columns);
  std::string getErrorMessage() { return error_msg_; };

 protected:
  void init();

  virtual bool loadImpl(
      const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
      size_t row_count,
      bool checkpoint,
      const Catalog_Namespace::SessionInfo* session_info);

  using OneShardBuffers = std::vector<std::unique_ptr<TypedImportBuffer>>;
  void distributeToShards(std::vector<OneShardBuffers>& all_shard_import_buffers,
                          std::vector<size_t>& all_shard_row_counts,
                          const OneShardBuffers& import_buffers,
                          const size_t row_count,
                          const size_t shard_count,
                          const Catalog_Namespace::SessionInfo* session_info);

  Catalog_Namespace::Catalog& catalog_;
  const TableDescriptor* table_desc_;
  std::list<const ColumnDescriptor*> column_descs_;
  LoadCallbackType load_callback_;
  Fragmenter_Namespace::InsertData insert_data_;
  std::map<int, StringDictionary*> dict_map_;

 private:
  bool loadToShard(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                   size_t row_count,
                   const TableDescriptor* shard_table,
                   bool checkpoint,
                   const Catalog_Namespace::SessionInfo* session_info);
  void distributeToShardsNewColumns(
      std::vector<OneShardBuffers>& all_shard_import_buffers,
      std::vector<size_t>& all_shard_row_counts,
      const OneShardBuffers& import_buffers,
      const size_t row_count,
      const size_t shard_count,
      const Catalog_Namespace::SessionInfo* session_info);
  void distributeToShardsExistingColumns(
      std::vector<OneShardBuffers>& all_shard_import_buffers,
      std::vector<size_t>& all_shard_row_counts,
      const OneShardBuffers& import_buffers,
      const size_t row_count,
      const size_t shard_count,
      const Catalog_Namespace::SessionInfo* session_info);
  void fillShardRow(const size_t row_index,
                    OneShardBuffers& shard_output_buffers,
                    const OneShardBuffers& import_buffers);

  bool adding_columns_ = false;
  std::mutex loader_mutex_;
  std::string error_msg_;
};

struct ImportStatus {
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  size_t rows_completed;
  size_t rows_estimated;
  size_t rows_rejected;
  std::chrono::duration<size_t, std::milli> elapsed;
  bool load_failed = false;
  std::string load_msg;
  int thread_id;  // to recall thread_id after thread exit
  ImportStatus()
      : start(std::chrono::steady_clock::now())
      , rows_completed(0)
      , rows_estimated(0)
      , rows_rejected(0)
      , elapsed(0)
      , thread_id(0) {}

  ImportStatus& operator+=(const ImportStatus& is) {
    rows_completed += is.rows_completed;
    rows_rejected += is.rows_rejected;
    if (is.load_failed) {
      load_failed = true;
      load_msg = is.load_msg;
    }

    return *this;
  }
};

class DataStreamSink {
 public:
  DataStreamSink() {}
  DataStreamSink(const CopyParams& copy_params, const std::string file_path)
      : copy_params(copy_params), file_path(file_path) {}
  virtual ~DataStreamSink() {}
  virtual ImportStatus importDelimited(
      const std::string& file_path,
      const bool decompressed,
      const Catalog_Namespace::SessionInfo* session_info) = 0;
#ifdef ENABLE_IMPORT_PARQUET
  virtual void import_parquet(std::vector<std::string>& file_paths,
                              const Catalog_Namespace::SessionInfo* session_info);
  virtual void import_local_parquet(
      const std::string& file_path,
      const Catalog_Namespace::SessionInfo* session_info) = 0;
#endif
  const CopyParams& get_copy_params() const {
    return copy_params;
  }
  void import_compressed(std::vector<std::string>& file_paths,
                         const Catalog_Namespace::SessionInfo* session_info);

 protected:
  ImportStatus archivePlumber(const Catalog_Namespace::SessionInfo* session_info);

  CopyParams copy_params;
  const std::string file_path;
  FILE* p_file = nullptr;
  ImportStatus import_status_;
  heavyai::shared_mutex import_mutex_;
  size_t total_file_size{0};
  std::vector<size_t> file_offsets;
  std::mutex file_offsets_mutex;
};

class Detector : public DataStreamSink {
 public:
  Detector(const boost::filesystem::path& fp, CopyParams& cp);

#ifdef ENABLE_IMPORT_PARQUET
  void import_local_parquet(const std::string& file_path,
                            const Catalog_Namespace::SessionInfo* session_info) override;
#endif
  static SQLTypes detect_sqltype(const std::string& str);
  std::vector<std::string> get_headers();
  std::vector<std::vector<std::string>> raw_rows;
  std::vector<std::vector<std::string>> get_sample_rows(size_t n);
  bool has_headers = false;

  std::vector<SQLTypeInfo> getBestColumnTypes() const;

 private:
  void init();
  void read_file();
  void detect_row_delimiter();
  void split_raw_data();
  std::vector<SQLTypes> detect_column_types(const std::vector<std::string>& row);
  static bool more_restrictive_sqltype(const SQLTypes a, const SQLTypes b);
  void find_best_sqltypes();
  std::vector<SQLTypes> find_best_sqltypes(
      const std::vector<std::vector<std::string>>& raw_rows,
      const CopyParams& copy_params);
  std::vector<SQLTypes> find_best_sqltypes(
      const std::vector<std::vector<std::string>>::const_iterator& row_begin,
      const std::vector<std::vector<std::string>>::const_iterator& row_end,
      const CopyParams& copy_params);

  std::vector<EncodingType> find_best_encodings(
      const std::vector<std::vector<std::string>>::const_iterator& row_begin,
      const std::vector<std::vector<std::string>>::const_iterator& row_end,
      const std::vector<SQLTypes>& best_types);

  bool detect_headers(const std::vector<SQLTypes>& first_types,
                      const std::vector<SQLTypes>& rest_types);
  void find_best_sqltypes_and_headers();
  ImportStatus importDelimited(
      const std::string& file_path,
      const bool decompressed,
      const Catalog_Namespace::SessionInfo* session_info) override;
  std::string raw_data;
  boost::filesystem::path file_path;
  std::chrono::duration<double> timeout{1};
  std::string line1;
#if defined(ENABLE_IMPORT_PARQUET)
  std::optional<foreign_storage::DataPreview> data_preview_;
#endif
  std::vector<SQLTypes> best_sqltypes;
  std::vector<EncodingType> best_encodings;
};

class Importer : public DataStreamSink, public AbstractImporter {
 public:
  Importer(Catalog_Namespace::Catalog& c,
           const TableDescriptor* t,
           const std::string& f,
           const CopyParams& p);
  Importer(Loader* providedLoader, const std::string& f, const CopyParams& p);
  ~Importer() override;
  ImportStatus import(const Catalog_Namespace::SessionInfo* session_info) override;
  ImportStatus importDelimited(
      const std::string& file_path,
      const bool decompressed,
      const Catalog_Namespace::SessionInfo* session_info) override;
  ImportStatus importGDAL(const std::map<std::string, std::string>& colname_to_src,
                          const Catalog_Namespace::SessionInfo* session_info,
                          const bool is_raster);
  const CopyParams& get_copy_params() const { return copy_params; }
  const std::list<const ColumnDescriptor*>& get_column_descs() const {
    return loader->get_column_descs();
  }
  void load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
            size_t row_count,
            const Catalog_Namespace::SessionInfo* session_info);
  std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>>& get_import_buffers_vec() {
    return import_buffers_vec;
  }
  std::vector<std::unique_ptr<TypedImportBuffer>>& get_import_buffers(int i) {
    return import_buffers_vec[i];
  }
  const bool* get_is_array() const { return is_array_a.get(); }
#ifdef ENABLE_IMPORT_PARQUET
  void import_local_parquet(const std::string& file_path,
                            const Catalog_Namespace::SessionInfo* session_info) override;
#endif
  static ImportStatus get_import_status(const std::string& id);
  static void set_import_status(const std::string& id, const ImportStatus is);
  static const std::list<ColumnDescriptor> gdalToColumnDescriptors(
      const std::string& fileName,
      const bool is_raster,
      const std::string& geoColumnName,
      const CopyParams& copy_params);
  static void readMetadataSampleGDAL(
      const std::string& fileName,
      const std::string& geoColumnName,
      std::map<std::string, std::vector<std::string>>& metadata,
      int rowLimit,
      const CopyParams& copy_params);
  static bool gdalFileExists(const std::string& path, const CopyParams& copy_params);
  static bool gdalFileOrDirectoryExists(const std::string& path,
                                        const CopyParams& copy_params);
  static std::vector<std::string> gdalGetAllFilesInArchive(
      const std::string& archive_path,
      const CopyParams& copy_params);
  enum class GeoFileLayerContents { EMPTY, GEO, NON_GEO, UNSUPPORTED_GEO };
  struct GeoFileLayerInfo {
    GeoFileLayerInfo(const std::string& name_, GeoFileLayerContents contents_)
        : name(name_), contents(contents_) {}
    std::string name;
    GeoFileLayerContents contents;
  };
  static std::vector<GeoFileLayerInfo> gdalGetLayersInGeoFile(
      const std::string& file_name,
      const CopyParams& copy_params);
  Catalog_Namespace::Catalog& getCatalog() {
    return loader->getCatalog();
  }
  static void set_geo_physical_import_buffer(
      const Catalog_Namespace::Catalog& catalog,
      const ColumnDescriptor* cd,
      std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
      size_t& col_idx,
      std::vector<double>& coords,
      std::vector<double>& bounds,
      std::vector<int>& ring_sizes,
      std::vector<int>& poly_rings,
      const bool force_null = false);
  static void set_geo_physical_import_buffer_columnar(
      const Catalog_Namespace::Catalog& catalog,
      const ColumnDescriptor* cd,
      std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
      size_t& col_idx,
      std::vector<std::vector<double>>& coords_column,
      std::vector<std::vector<double>>& bounds_column,
      std::vector<std::vector<int>>& ring_sizes_column,
      std::vector<std::vector<int>>& poly_rings_column);
  void checkpoint(const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs);
  auto getLoader() const {
    return loader.get();
  }

 private:
  static bool gdalStatInternal(const std::string& path,
                               const CopyParams& copy_params,
                               bool also_dir);
  static Geospatial::GDAL::DataSourceUqPtr openGDALDataSource(
      const std::string& fileName,
      const CopyParams& copy_params);

  ImportStatus importGDALGeo(const std::map<std::string, std::string>& colname_to_src,
                             const Catalog_Namespace::SessionInfo* session_info);
  ImportStatus importGDALRaster(const Catalog_Namespace::SessionInfo* session_info);

  static const std::list<ColumnDescriptor> gdalToColumnDescriptorsGeo(
      const std::string& fileName,
      const std::string& geoColumnName,
      const CopyParams& copy_params);
  static const std::list<ColumnDescriptor> gdalToColumnDescriptorsRaster(
      const std::string& fileName,
      const std::string& geoColumnName,
      const CopyParams& copy_params);

  std::string import_id;
  size_t file_size;
  size_t max_threads;
  char* buffer[2];
  std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>> import_buffers_vec;
  std::unique_ptr<Loader> loader;
  std::unique_ptr<bool[]> is_array_a;
  static std::mutex init_gdal_mutex;
};

std::vector<std::unique_ptr<TypedImportBuffer>> setup_column_loaders(
    const TableDescriptor* td,
    Loader* loader);

std::vector<std::unique_ptr<TypedImportBuffer>> fill_missing_columns(
    const Catalog_Namespace::Catalog* cat,
    Fragmenter_Namespace::InsertData& insert_data);

std::unique_ptr<AbstractImporter> create_importer(
    Catalog_Namespace::Catalog& catalog,
    const TableDescriptor* td,
    const std::string& copy_from_source,
    const import_export::CopyParams& copy_params);

}  // namespace import_export

#endif  // _IMPORTER_H_
