/*
 * Copyright 2017 MapD Technologies, Inc.
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
#include <poly2tri/poly2tri.h>
#include "../Shared/fixautotools.h"
#include <ogrsf_frmts.h>
#include <gdal.h>
#include "../Shared/fixautotools.h"
#include "../Shared/ShapeDrawData.h"
#include "../Catalog/TableDescriptor.h"
#include "../Catalog/Catalog.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Shared/checked_alloc.h"

class TDatum;
class TColumn;

namespace arrow {

class Array;

}  // namespace arrow

namespace Importer_NS {

enum class TableType { DELIMITED, POLYGON };

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
  size_t max_reject;  // maximum number of records that can be rejected before copy is failed
  TableType table_type;
  bool plain_text = false;
  // s3/parquet related params
  bool is_parquet;
  std::string s3_access_key;  // per-query credentials to override the
  std::string s3_secret_key;  // settings in ~/.aws/credentials or environment
  std::string s3_region;
  // kafka related params
  size_t retry_count;
  size_t retry_wait;
  size_t batch_size;

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
        threads(0),
        max_reject(100000),
        table_type(TableType::DELIMITED),
        is_parquet(false),
        retry_count(100),
        retry_wait(5),
        batch_size(1000) {}

  CopyParams(char d, const std::string& n, char l, size_t b, size_t retries, size_t wait)
      : delimiter(d),
        null_str(n),
        has_header(true),
        quoted(true),
        quote('"'),
        escape('"'),
        line_delim(l),
        array_delim(','),
        array_begin('{'),
        array_end('}'),
        threads(0),
        max_reject(100000),
        table_type(TableType::DELIMITED),
        retry_count(retries),
        retry_wait(wait),
        batch_size(b) {}
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

  void addStringArray(const std::vector<std::string>& arr) { string_array_buffer_->push_back(arr); }

  void addTime(const time_t v) { time_buffer_->push_back(v); }

  void addDictEncodedString(const std::vector<std::string>& string_vec) {
    CHECK(string_dict_);
    for (const auto& str : string_vec) {
      if (str.size() > StringDictionary::MAX_STRLEN) {
        throw std::runtime_error("String too long for dictionary encoding.");
      }
    }
    switch (column_desc_->columnType.get_size()) {
      case 1:
        string_dict_i8_buffer_->resize(string_vec.size());
        string_dict_->getOrAddBulk(string_vec, string_dict_i8_buffer_->data());
        break;
      case 2:
        string_dict_i16_buffer_->resize(string_vec.size());
        string_dict_->getOrAddBulk(string_vec, string_dict_i16_buffer_->data());
        break;
      case 4:
        string_dict_i32_buffer_->resize(string_vec.size());
        string_dict_->getOrAddBulk(string_vec, string_dict_i32_buffer_->data());
        break;
      default:
        CHECK(false);
    }
  }

  void addDictEncodedStringArray(const std::vector<std::vector<std::string>>& string_array_vec) {
    CHECK(string_dict_);
    for (auto& p : string_array_vec) {
      size_t len = p.size() * sizeof(int32_t);
      auto a = static_cast<int32_t*>(checked_malloc(len));
      for (const auto& str : p) {
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

  StringDictionary* getStringDictionary() const { return string_dict_; }

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
        abort();
    }
  }

  size_t getElementSize() const {
    switch (column_desc_->columnType.get_type()) {
      case kBOOLEAN:
        return sizeof((*bool_buffer_)[0]);
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
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
        return sizeof((*time_buffer_)[0]);
      default:
        abort();
    }
  }

  std::vector<std::string>* getStringBuffer() const { return string_buffer_; }

  std::vector<ArrayDatum>* getArrayBuffer() const { return array_buffer_; }

  std::vector<std::vector<std::string>>* getStringArrayBuffer() const { return string_array_buffer_; }

  std::vector<ArrayDatum>* getStringArrayDictBuffer() const { return string_array_dict_buffer_; }

  int8_t* getStringDictBuffer() const {
    switch (column_desc_->columnType.get_size()) {
      case 1:
        return reinterpret_cast<int8_t*>(&((*string_dict_i8_buffer_)[0]));
      case 2:
        return reinterpret_cast<int8_t*>(&((*string_dict_i16_buffer_)[0]));
      case 4:
        return reinterpret_cast<int8_t*>(&((*string_dict_i32_buffer_)[0]));
      default:
        abort();
    }
  }

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

  size_t add_values(const ColumnDescriptor* cd, const TColumn& data);

  size_t add_arrow_values(const ColumnDescriptor* cd, const arrow::Array& data);

  void add_value(const ColumnDescriptor* cd, const std::string& val, const bool is_null, const CopyParams& copy_params);
  void add_value(const ColumnDescriptor* cd, const TDatum& val, const bool is_null);
  void pop_value();

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
    std::vector<uint8_t>* string_dict_i8_buffer_;
    std::vector<uint16_t>* string_dict_i16_buffer_;
    std::vector<int32_t>* string_dict_i32_buffer_;
    std::vector<ArrayDatum>* string_array_dict_buffer_;
  };
  const ColumnDescriptor* column_desc_;
  StringDictionary* string_dict_;
};

class Loader {
 public:
  Loader(Catalog_Namespace::Catalog& c, const TableDescriptor* t)
      : catalog(c), table_desc(t), column_descs(c.getAllColumnMetadataForTable(t->tableId, false, false)) {
    init();
  };
  Catalog_Namespace::Catalog& get_catalog() { return catalog; }
  const TableDescriptor* get_table_desc() const { return table_desc; }
  const std::list<const ColumnDescriptor*>& get_column_descs() const { return column_descs; }
  const Fragmenter_Namespace::InsertData& get_insert_data() const { return insert_data; }
  StringDictionary* get_string_dict(const ColumnDescriptor* cd) const {
    if ((cd->columnType.get_type() != kARRAY || !IS_STRING(cd->columnType.get_subtype())) &&
        (!cd->columnType.is_string() || cd->columnType.get_compression() != kENCODING_DICT))
      return nullptr;
    return dict_map.at(cd->columnId);
  }
  virtual bool load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers, size_t row_count);
  virtual bool loadNoCheckpoint(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                                size_t row_count);
  virtual bool loadImpl(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                        size_t row_count,
                        bool checkpoint);
  virtual void checkpoint();
  virtual int32_t getTableEpoch();
  virtual void setTableEpoch(const int32_t new_epoch);

 protected:
  Catalog_Namespace::Catalog& catalog;
  const TableDescriptor* table_desc;
  std::list<const ColumnDescriptor*> column_descs;
  Fragmenter_Namespace::InsertData insert_data;
  std::map<int, StringDictionary*> dict_map;
  void init();
  typedef std::vector<std::unique_ptr<TypedImportBuffer>> OneShardBuffers;
  void distributeToShards(std::vector<OneShardBuffers>& all_shard_import_buffers,
                          std::vector<size_t>& all_shard_row_counts,
                          const OneShardBuffers& import_buffers,
                          const size_t row_count,
                          const size_t shard_count);

 private:
  bool loadToShard(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                   size_t row_count,
                   const TableDescriptor* shard_table,
                   bool checkpoint);
};

struct ImportStatus {
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  size_t rows_completed;
  size_t rows_estimated;
  size_t rows_rejected;
  std::chrono::duration<size_t, std::milli> elapsed;
  bool load_truncated;
  int thread_id;  // to recall thread_id after thread exit
  ImportStatus()
      : start(std::chrono::steady_clock::now()),
        rows_completed(0),
        rows_estimated(0),
        rows_rejected(0),
        elapsed(0),
        load_truncated(0),
        thread_id(0) {}

  ImportStatus& operator+=(const ImportStatus& is) {
    rows_completed += is.rows_completed;
    rows_rejected += is.rows_rejected;

    return *this;
  }
};

class DataStreamSink {
 public:
  DataStreamSink() {}
  DataStreamSink(const CopyParams& copy_params, const std::string file_path)
      : copy_params(copy_params), file_path(file_path) {}
  virtual ~DataStreamSink() {}
  virtual ImportStatus importDelimited(const std::string& file_path, const bool decompressed) = 0;
  const CopyParams& get_copy_params() const { return copy_params; }
  void import_local_parquet(const std::string& file_path);
  void import_parquet(std::vector<std::string>& file_paths);
  void import_compressed(std::vector<std::string>& file_paths);

 protected:
  ImportStatus archivePlumber();

  CopyParams copy_params;
  const std::string file_path;
  FILE* p_file = nullptr;
  ImportStatus import_status;
  bool load_failed = false;
};

class Detector : public DataStreamSink {
 public:
  Detector(const boost::filesystem::path& fp, CopyParams& cp) : DataStreamSink(cp, fp.string()), file_path(fp) {
    read_file();
    init();
  };
  static SQLTypes detect_sqltype(const std::string& str);
  std::vector<std::string> get_headers();
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
  ImportStatus importDelimited(const std::string& file_path, const bool decompressed);
  std::string raw_data;
  boost::filesystem::path file_path;
  std::chrono::duration<double> timeout{1};
  std::string line1;
};

struct PolyData2d {
  std::vector<double> coords;
  std::vector<unsigned int> triangulation_indices;
  std::vector<Rendering::GL::Resources::IndirectDrawVertexData> lineDrawInfo;
  std::vector<Rendering::GL::Resources::IndirectDrawIndexData> polyDrawInfo;

  PolyData2d(unsigned int startVert = 0, unsigned int startIdx = 0)
      : _ended(true), _startVert(startVert), _startIdx(startIdx), _startLineIdx(0) {}
  ~PolyData2d() {}

  size_t numVerts() const {
    size_t s = coords.size();
    CHECK(s % 2 == 0);
    return s / 2;
  }
  size_t numTris() const {
    size_t s = triangulation_indices.size();
    CHECK_EQ(s % 3, 0);
    return s / 3;
  }
  size_t numLineLoops() const { return lineDrawInfo.size(); }
  size_t numIndices() const { return triangulation_indices.size(); }

  unsigned int startVert() const { return _startVert; }
  unsigned int startIdx() const { return _startIdx; }

  void beginPoly() {
    CHECK(_ended);
    _ended = false;

    if (!polyDrawInfo.size()) {
      polyDrawInfo.emplace_back(0, _startIdx, _startVert);
    }
  }

  void addTriangle(unsigned int idx0, unsigned int idx1, unsigned int idx2) {
    triangulation_indices.push_back(_startLineIdx + idx0);
    triangulation_indices.push_back(_startLineIdx + idx1);
    triangulation_indices.push_back(_startLineIdx + idx2);

    polyDrawInfo.back().count += 3;
  }

  void endPoly() {
    CHECK(!_ended);
    _ended = true;
  }

  void popPoly() {
    CHECK(_ended);
    CHECK(polyDrawInfo.size());
    if (triangulation_indices.empty()) {
      return;
    }
    CHECK_EQ(triangulation_indices.size() % 3, 0);
    auto itr = triangulation_indices.end() - 1;
    for (; itr >= triangulation_indices.begin(); itr -= 3) {
      if (*itr < _startLineIdx) {
        break;
      }
    }
    itr++;
    auto count = triangulation_indices.end() - itr;
    triangulation_indices.erase(itr, triangulation_indices.end());
    polyDrawInfo.back().count -= count;
  }

  void beginLine() {
    CHECK(_ended);
    _ended = false;

    if (!lineDrawInfo.size()) {
      // first line creates this
      lineDrawInfo.emplace_back(0, _startVert + numVerts());
    } else {
      // second and subsequent line, first
      // add an empty coord as a separator
      coords.push_back(-FLT_MAX);
      coords.push_back(-FLT_MAX);
      lineDrawInfo.back().count++;
    }

    _startLineIdx = numVerts();
  }

  void addLinePoint(const p2t::Point* vertPtr) {
    _addPoint(vertPtr->x, vertPtr->y);
    lineDrawInfo.back().count++;
  }

  void endLine(const bool add_extra_verts = true) {
    if (add_extra_verts) {
      // repeat the first 3 vertices to fully create the "loop"
      // since it will be drawn using the GL_LINE_STRIP_ADJACENCY
      // primitive type
      int numPointsThisLine = numVerts() - _startLineIdx;
      for (int i = 0; i < 3; ++i) {
        int idx = (_startLineIdx + (i % numPointsThisLine)) * 2;
        coords.push_back(coords[idx]);
        coords.push_back(coords[idx + 1]);
      }
      lineDrawInfo.back().count += 3;
    }

    _ended = true;
  }

  void popLine() {
    CHECK(_ended);
    CHECK(lineDrawInfo.size());
    auto count = lineDrawInfo.back().count;
    CHECK_LE(count * 2, coords.size());
    auto itr = coords.end();
    itr -= count * 2;
    coords.erase(itr, coords.end());
    lineDrawInfo.pop_back();
  }

 private:
  bool _ended;
  unsigned int _startVert;
  unsigned int _startIdx;
  unsigned int _startLineIdx;

  void _addPoint(double x, double y) {
    coords.push_back(x);
    coords.push_back(y);
  }
};

class ImporterUtils {
 public:
  static void parseStringArray(const std::string& s,
                               const CopyParams& copy_params,
                               std::vector<std::string>& string_vec) {
    if (s == copy_params.null_str || s.size() < 1) {
      return;
    }
    if (s[0] != copy_params.array_begin || s[s.size() - 1] != copy_params.array_end) {
      throw std::runtime_error("Malformed Array :" + s);
    }
    size_t last = 1;
    for (size_t i = s.find(copy_params.array_delim, 1); i != std::string::npos;
         i = s.find(copy_params.array_delim, last)) {
      if (i > last) {  // if not empty string - disallow empty strings for now
        if (s.substr(last, i - last).length() > StringDictionary::MAX_STRLEN)
          throw std::runtime_error("Array String too long : " + std::to_string(s.substr(last, i - last).length()) +
                                   " max is " + std::to_string(StringDictionary::MAX_STRLEN));

        string_vec.push_back(s.substr(last, i - last));
      }
      last = i + 1;
    }
    if (s.size() - 1 > last) {  // if not empty string - disallow empty strings for now
      if (s.substr(last, s.size() - 1 - last).length() > StringDictionary::MAX_STRLEN)
        throw std::runtime_error("Array String too long : " +
                                 std::to_string(s.substr(last, s.size() - 1 - last).length()) + " max is " +
                                 std::to_string(StringDictionary::MAX_STRLEN));

      string_vec.push_back(s.substr(last, s.size() - 1 - last));
    }
  }
};

class Importer : public DataStreamSink {
 public:
  Importer(Catalog_Namespace::Catalog& c, const TableDescriptor* t, const std::string& f, const CopyParams& p);
  Importer(Loader* providedLoader, const std::string& f, const CopyParams& p);
  ~Importer();
  ImportStatus import();
  ImportStatus importDelimited(const std::string& file_path, const bool decompressed);
  ImportStatus importShapefile();
  ImportStatus importGDAL(std::map<std::string, std::string> colname_to_src);
  const CopyParams& get_copy_params() const { return copy_params; }
  const std::list<const ColumnDescriptor*>& get_column_descs() const { return loader->get_column_descs(); }
  void load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers, size_t row_count);
  std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>>& get_import_buffers_vec() { return import_buffers_vec; }
  std::vector<std::unique_ptr<TypedImportBuffer>>& get_import_buffers(int i) { return import_buffers_vec[i]; }
  const bool* get_is_array() const { return is_array_a.get(); }
  static ImportStatus get_import_status(const std::string& id);
  static void set_import_status(const std::string& id, const ImportStatus is);
  static const std::list<ColumnDescriptor> gdalToColumnDescriptors(const std::string& fileName);
  static void readMetadataSampleGDAL(const std::string& fileName,
                                     std::map<std::string, std::vector<std::string>>& metadata,
                                     int rowLimit);

 private:
  void readVerticesFromGDAL(const std::string& fileName,
                            std::vector<PolyData2d>& polys,
                            std::pair<std::map<std::string, size_t>, std::vector<std::vector<std::string>>>& metadata);
  void readVerticesFromGDALGeometryZ(const std::string& fileName,
                                     OGRPolygon* poPolygon,
                                     PolyData2d& poly,
                                     const bool hasZ,
                                     const ssize_t featureIdx,
                                     const ssize_t multipolyIdx);
  void initGDAL();
  std::string import_id;
  size_t file_size;
  int max_threads;
  char* buffer[2];
  std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>> import_buffers_vec;
  std::unique_ptr<Loader> loader;
  std::unique_ptr<bool[]> is_array_a;
};
};
#endif  // _IMPORTER_H_
