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
#include "../StringDictionary/StringDictionary.h"

class TDatum;

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
        table_type(TableType::DELIMITED) {}
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
  virtual bool load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers, size_t row_count);
  virtual bool loadNoCheckpoint(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                                size_t row_count);
  virtual bool loadImpl(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                        size_t row_count,
                        bool checkpoint);

 protected:
  const Catalog_Namespace::Catalog& catalog;
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

class Detector {
 public:
  Detector(const boost::filesystem::path& fp, CopyParams& cp) : file_path(fp), copy_params(cp) {
    read_file();
    init();
  };
  static SQLTypes detect_sqltype(const std::string& str);
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
  std::string raw_data;
  boost::filesystem::path file_path;
  std::chrono::duration<double> timeout{1};
  CopyParams copy_params;
};

struct ImportStatus {
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  size_t rows_completed;
  size_t rows_estimated;
  size_t rows_rejected;
  std::chrono::duration<size_t, std::milli> elapsed;
  bool load_truncated;
  ImportStatus()
      : start(std::chrono::steady_clock::now()),
        rows_completed(0),
        rows_estimated(0),
        rows_rejected(0),
        elapsed(0),
        load_truncated(0) {}

  ImportStatus& operator+=(const ImportStatus& is) {
    rows_completed += is.rows_completed;
    rows_rejected += is.rows_rejected;

    return *this;
  }
};

struct PolyData2d {
  std::vector<double> coords;
  std::vector<unsigned int> triangulation_indices;
  std::vector<Rendering::GL::Resources::IndirectDrawVertexData> lineDrawInfo;
  std::vector<Rendering::GL::Resources::IndirectDrawIndexData> polyDrawInfo;

  PolyData2d(unsigned int startVert = 0, unsigned int startIdx = 0)
      : _ended(true), _startVert(startVert), _startIdx(startIdx), _startTriIdx(0) {}
  ~PolyData2d() {}

  size_t numVerts() const { return coords.size() / 2; }
  size_t numLineLoops() const { return lineDrawInfo.size(); }
  size_t numTris() const {
    CHECK(triangulation_indices.size() % 3 == 0);
    return triangulation_indices.size() / 3;
  }
  size_t numIndices() const { return triangulation_indices.size(); }

  unsigned int startVert() const { return _startVert; }
  unsigned int startIdx() const { return _startIdx; }

  void beginPoly() {
    assert(_ended);
    _ended = false;
    _startTriIdx = numVerts() - lineDrawInfo.back().count;

    if (!polyDrawInfo.size()) {
      // polyDrawInfo.emplace_back(0, _startIdx + triangulation_indices.size(), lineDrawInfo.back().firstIndex);
      polyDrawInfo.emplace_back(0, _startIdx, _startVert);
    }
  }

  void endPoly() {
    assert(!_ended);
    _ended = true;
  }

  void beginLine() {
    assert(_ended);
    _ended = false;

    lineDrawInfo.emplace_back(0, _startVert + numVerts());
  }

  void addLinePoint(const std::shared_ptr<p2t::Point>& vertPtr) {
    _addPoint(vertPtr->x, vertPtr->y);
    lineDrawInfo.back().count++;
  }

  bool endLine() {
    bool rtn = false;
    auto& lineDrawItem = lineDrawInfo.back();
    size_t idx0 = (lineDrawItem.firstIndex - _startVert) * 2;
    size_t idx1 = idx0 + (lineDrawItem.count - 1) * 2;
    if (coords[idx0] == coords[idx1] && coords[idx0 + 1] == coords[idx1 + 1]) {
      coords.pop_back();
      coords.pop_back();
      lineDrawItem.count--;
      rtn = true;
    }

    // repeat the first 3 vertices to fully create the "loop"
    // since it will be drawn using the GL_LINE_STRIP_ADJACENCY
    // primitive type
    int num = lineDrawItem.count;
    for (int i = 0; i < 3; ++i) {
      int idx = (idx0 + ((i % num) * 2));
      coords.push_back(coords[idx]);
      coords.push_back(coords[idx + 1]);
    }
    lineDrawItem.count += 3;

    // add an empty coord as a separator
    // coords.push_back(-10000000.0);
    // coords.push_back(-10000000.0);

    _ended = true;
    return rtn;
  }

  void addTriangle(unsigned int idx0, unsigned int idx1, unsigned int idx2) {
    // triangulation_indices.push_back(idx0);
    // triangulation_indices.push_back(idx1);
    // triangulation_indices.push_back(idx2);

    triangulation_indices.push_back(_startTriIdx + idx0);
    triangulation_indices.push_back(_startTriIdx + idx1);
    triangulation_indices.push_back(_startTriIdx + idx2);

    polyDrawInfo.back().count += 3;
  }

 private:
  bool _ended;
  unsigned int _startVert;
  unsigned int _startIdx;
  unsigned int _startTriIdx;

  void _addPoint(double x, double y) {
    coords.push_back(x);
    coords.push_back(y);
  }
};

class Importer {
 public:
  Importer(const Catalog_Namespace::Catalog& c, const TableDescriptor* t, const std::string& f, const CopyParams& p);
  Importer(Loader* providedLoader, const std::string& f, const CopyParams& p);
  ~Importer();
  ImportStatus import();
  ImportStatus importDelimited();
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
  void readVerticesFromGDALGeometryZ(const std::string& fileName, OGRPolygon* poPolygon, PolyData2d& poly, bool hasZ);
  void initGDAL();
  const std::string& file_path;
  std::string import_id;
  const CopyParams& copy_params;
  size_t file_size;
  int max_threads;
  FILE* p_file;
  char* buffer[2];
  int which_buf;
  std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>> import_buffers_vec;
  std::unique_ptr<Loader> loader;
  bool load_failed;
  std::unique_ptr<bool[]> is_array_a;
  ImportStatus import_status;
};
};
#endif  // _IMPORTER_H_
