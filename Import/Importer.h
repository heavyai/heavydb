/*
 * @file Importer.h
 * @author Wei Hong < wei@mapd.com>
 * @brief Importer class for table import from file
 */
#ifndef _IMPORTER_H_
#define _IMPORTER_H_

#include <string>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <boost/noncopyable.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <glog/logging.h>
#include "../Catalog/TableDescriptor.h"
#include "../Catalog/Catalog.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Shared/checked_alloc.h"
#include "../StringDictionary/StringDictionary.h"

class TDatum;

namespace Importer_NS {

struct CopyParams {
  char delimiter;
  std::string null_str;
  bool has_header;
  bool quoted;  // does the input have any quoted fields, default to false
  char quote;
  char escape;
  char line_delim;
  char array_delim;
  char array_begin;
  char array_end;
  int threads;

  CopyParams()
      : delimiter(','),
        null_str("\\N"),
        has_header(true),
        quoted(true),
        quote('"'),
        escape('"'),
        line_delim('\n'),
        array_delim(','),
        array_begin('{'),
        array_end('}'),
        threads(0) {}
};

class TypedImportBuffer : boost::noncopyable {
 public:
  TypedImportBuffer(const ColumnDescriptor* col_desc, StringDictionary* string_dict)
      : column_desc_(col_desc), string_dict_(string_dict) {
    switch (col_desc->columnType.get_type()) {
      case kBOOLEAN:
        bool_buffer_ = new std::vector<int8_t>();
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
          string_dict_buffer_ = new std::vector<int32_t>();
        }
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
        time_buffer_ = new std::vector<time_t>();
        break;
      case kARRAY:
        if (IS_STRING(col_desc->columnType.get_subtype())) {
          CHECK(col_desc->columnType.get_compression() == kENCODING_DICT);
          string_array_buffer_ = new std::vector<std::vector<std::string>>();
          string_array_dict_buffer_ = new std::vector<ArrayDatum>();
        } else
          array_buffer_ = new std::vector<ArrayDatum>();
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
          delete string_dict_buffer_;
        }
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
        delete time_buffer_;
        break;
      case kARRAY:
        if (IS_STRING(column_desc_->columnType.get_subtype())) {
          delete string_array_buffer_;
          delete string_array_dict_buffer_;
        } else
          delete array_buffer_;
        break;
      default:
        CHECK(false);
    }
  }

  void addBoolean(const int8_t v) { bool_buffer_->push_back(v); }

  void addSmallint(const int16_t v) { smallint_buffer_->push_back(v); }

  void addInt(const int32_t v) { int_buffer_->push_back(v); }

  void addBigint(const int64_t v) { bigint_buffer_->push_back(v); }

  void addFloat(const float v) { float_buffer_->push_back(v); }

  void addDouble(const double v) { double_buffer_->push_back(v); }

  void addString(const std::string& v) { string_buffer_->push_back(v); }

  void addArray(const ArrayDatum& v) { array_buffer_->push_back(v); }

  std::vector<std::string>& addStringArray() {
    string_array_buffer_->push_back(std::vector<std::string>());
    return string_array_buffer_->back();
  }

  void addTime(const time_t v) { time_buffer_->push_back(v); }

  void addDictEncodedString(const std::vector<std::string>& string_vec) {
    CHECK(string_dict_);
    for (const auto& str: string_vec) {
      if (str.size() > StringDictionary::MAX_STRLEN) {
        throw std::runtime_error("String too long for dictionary encoding.");
      }
    }
    string_dict_buffer_->resize(string_vec.size());
    string_dict_->getOrAddBulk(string_vec, string_dict_buffer_->data());
  }

  void addDictEncodedStringArray(const std::vector<std::vector<std::string>>& string_array_vec) {
    CHECK(string_dict_);
    for (auto& p : string_array_vec) {
      size_t len = p.size() * sizeof(int32_t);
      auto a = static_cast<int32_t*>(checked_malloc(len));
      for (const auto& str: p) {
        if (str.size() > StringDictionary::MAX_STRLEN) {
          throw std::runtime_error("String too long for dictionary encoding.");
        }
      }
      string_dict_->getOrAddBulk(p, a);
      string_array_dict_buffer_->push_back(ArrayDatum(len, reinterpret_cast<int8_t*>(a), len == 0));
    }
  }

  const SQLTypeInfo& getTypeInfo() const { return column_desc_->columnType; }

  const ColumnDescriptor* getColumnDesc() const { return column_desc_; }

  int8_t* getAsBytes() const {
    switch (column_desc_->columnType.get_type()) {
      case kBOOLEAN:
        return reinterpret_cast<int8_t*>(&((*bool_buffer_)[0]));
      case kSMALLINT:
        return reinterpret_cast<int8_t*>(&((*smallint_buffer_)[0]));
      case kINT:
        return reinterpret_cast<int8_t*>(&((*int_buffer_)[0]));
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        return reinterpret_cast<int8_t*>(&((*bigint_buffer_)[0]));
      case kFLOAT:
        return reinterpret_cast<int8_t*>(&((*float_buffer_)[0]));
      case kDOUBLE:
        return reinterpret_cast<int8_t*>(&((*double_buffer_)[0]));
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
        return reinterpret_cast<int8_t*>(&((*time_buffer_)[0]));
      default:
        CHECK(false);
    }
  }

  std::vector<std::string>* getStringBuffer() const { return string_buffer_; }

  std::vector<ArrayDatum>* getArrayBuffer() const { return array_buffer_; }

  std::vector<std::vector<std::string>>* getStringArrayBuffer() const { return string_array_buffer_; }

  std::vector<ArrayDatum>* getStringArrayDictBuffer() const { return string_array_dict_buffer_; }

  int8_t* getStringDictBuffer() const { return reinterpret_cast<int8_t*>(&((*string_dict_buffer_)[0])); }

  bool stringDictCheckpoint() {
    if (string_dict_ == nullptr)
      return true;
    return string_dict_->checkpoint();
  }

  void clear() {
    switch (column_desc_->columnType.get_type()) {
      case kBOOLEAN: {
        bool_buffer_->clear();
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
          string_dict_buffer_->clear();
        }
        break;
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        time_buffer_->clear();
        break;
      }
      case kARRAY: {
        if (IS_STRING(column_desc_->columnType.get_subtype())) {
          string_array_buffer_->clear();
          string_array_dict_buffer_->clear();
        } else
          array_buffer_->clear();
        break;
      }
      default:
        CHECK(false);
    }
  }

  void add_value(const ColumnDescriptor* cd, const std::string& val, const bool is_null, const CopyParams& copy_params);
  void add_value(const ColumnDescriptor* cd, const TDatum& val, const bool is_null);

 private:
  union {
    std::vector<int8_t>* bool_buffer_;
    std::vector<int16_t>* smallint_buffer_;
    std::vector<int32_t>* int_buffer_;
    std::vector<int64_t>* bigint_buffer_;
    std::vector<float>* float_buffer_;
    std::vector<double>* double_buffer_;
    std::vector<time_t>* time_buffer_;
    std::vector<std::string>* string_buffer_;
    std::vector<ArrayDatum>* array_buffer_;
    std::vector<std::vector<std::string>>* string_array_buffer_;
  };
  union {
    std::vector<int32_t>* string_dict_buffer_;
    std::vector<ArrayDatum>* string_array_dict_buffer_;
  };
  const ColumnDescriptor* column_desc_;
  StringDictionary* string_dict_;
};

class Loader {
 public:
  Loader(const Catalog_Namespace::Catalog& c, const TableDescriptor* t)
      : catalog(c), table_desc(t), column_descs(c.getAllColumnMetadataForTable(t->tableId, false, false)) {
    init();
  };
  const Catalog_Namespace::Catalog& get_catalog() const { return catalog; }
  const TableDescriptor* get_table_desc() const { return table_desc; }
  const std::list<const ColumnDescriptor*>& get_column_descs() const { return column_descs; }
  const Fragmenter_Namespace::InsertData& get_insert_data() const { return insert_data; }
  StringDictionary* get_string_dict(const ColumnDescriptor* cd) const {
    if ((cd->columnType.get_type() != kARRAY || !IS_STRING(cd->columnType.get_subtype())) &&
        (!cd->columnType.is_string() || cd->columnType.get_compression() != kENCODING_DICT))
      return nullptr;
    return dict_map.at(cd->columnId);
  }
  bool load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers, size_t row_count);
  void checkpoint() { catalog.get_dataMgr().checkpoint(); }

 private:
  const Catalog_Namespace::Catalog& catalog;
  const TableDescriptor* table_desc;
  std::list<const ColumnDescriptor*> column_descs;
  Fragmenter_Namespace::InsertData insert_data;
  std::map<int, StringDictionary*> dict_map;
  void init();
};

class Detector {
 public:
  Detector(const boost::filesystem::path& fp, CopyParams& cp) : file_path(fp), copy_params(cp) {
    read_file();
    init();
  };
  SQLTypes detect_sqltype(const std::string& str);
  std::vector<std::string> get_headers();
  const CopyParams& get_copy_params() const { return copy_params; }
  std::vector<std::vector<std::string>> raw_rows;
  std::vector<std::vector<std::string>> get_sample_rows(size_t n);
  std::vector<SQLTypes> best_sqltypes;
  std::vector<EncodingType> best_encodings;
  bool has_headers = false;

 private:
  void init();
  void read_file();
  void detect_row_delimiter();
  void split_raw_data();
  std::vector<SQLTypes> detect_column_types(const std::vector<std::string>& row);
  static bool more_restrictive_sqltype(const SQLTypes a, const SQLTypes b);
  void find_best_sqltypes();
  std::vector<SQLTypes> find_best_sqltypes(const std::vector<std::vector<std::string>>& raw_rows,
                                           const CopyParams& copy_params);
  std::vector<SQLTypes> find_best_sqltypes(const std::vector<std::vector<std::string>>::const_iterator& row_begin,
                                           const std::vector<std::vector<std::string>>::const_iterator& row_end,
                                           const CopyParams& copy_params);

  std::vector<EncodingType> find_best_encodings(const std::vector<std::vector<std::string>>::const_iterator& row_begin,
                                                const std::vector<std::vector<std::string>>::const_iterator& row_end,
                                                const std::vector<SQLTypes>& best_types);

  void detect_headers();
  bool detect_headers(const std::vector<std::vector<std::string>>& raw_rows);
  bool detect_headers(const std::vector<SQLTypes>& first_types, const std::vector<SQLTypes>& rest_types);
  void find_best_sqltypes_and_headers();
  std::vector<std::string> raw_data;
  boost::filesystem::path file_path;
  std::chrono::duration<double> timeout{1};
  CopyParams copy_params;
};

struct ImportStatus {
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  size_t rows_completed;
  size_t rows_estimated;
  std::chrono::duration<size_t, std::milli> elapsed;
  ImportStatus() : start(std::chrono::steady_clock::now()), rows_completed(0), rows_estimated(0), elapsed(0) {}
};

class Importer {
 public:
  Importer(const Catalog_Namespace::Catalog& c, const TableDescriptor* t, const std::string& f, const CopyParams& p);
  ~Importer();
  void import();
  const CopyParams& get_copy_params() const { return copy_params; }
  const std::list<const ColumnDescriptor*>& get_column_descs() const { return loader.get_column_descs(); }
  void load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers, size_t row_count) {
    if (!loader.load(import_buffers, row_count))
      load_failed = true;
  }
  std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>>& get_import_buffers_vec() { return import_buffers_vec; }
  std::vector<std::unique_ptr<TypedImportBuffer>>& get_import_buffers(int i) { return import_buffers_vec[i]; }
  const bool* get_is_array() const { return is_array_a.get(); }
  static ImportStatus get_import_status(const std::string& id);
  static void set_import_status(const std::string& id, const ImportStatus is);

 private:
  const std::string& file_path;
  std::string import_id;
  const CopyParams& copy_params;
  size_t file_size;
  int max_threads;
  FILE* p_file;
  char* buffer[2];
  int which_buf;
  std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>> import_buffers_vec;
  Loader loader;
  bool load_failed;
  std::unique_ptr<bool[]> is_array_a;
  ImportStatus import_status;
};
};
#endif  // _IMPORTER_H_
