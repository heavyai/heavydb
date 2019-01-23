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
 * @file Importer.cpp
 * @author Wei Hong <wei@mapd.com>
 * @brief Functions for Importer class
 */

#include <gdal.h>
#include <glog/logging.h>
#include <ogrsf_frmts.h>
#include <unistd.h>
#include <boost/algorithm/string.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/filesystem.hpp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <list>
#include <mutex>
#include <stack>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../QueryEngine/SqlTypesLayout.h"
#include "../QueryEngine/TypePunning.h"
#include "../Shared/geo_compression.h"
#include "../Shared/geo_types.h"
#include "../Shared/geosupport.h"
#include "../Shared/import_helpers.h"
#include "../Shared/mapd_glob.h"
#include "../Shared/mapdpath.h"
#include "../Shared/measure.h"
#include "../Shared/scope.h"
#include "../Shared/shard_key.h"
#include "../Shared/unreachable.h"

#include <iostream>
#include <vector>
#include "DataMgr/LockMgr.h"
#include "Importer.h"
#include "QueryRunner/QueryRunner.h"
#include "Utils/ChunkAccessorTable.h"
#include "gen-cpp/MapD.h"

#include <arrow/api.h>

#include "../Archive/PosixFileArchive.h"

#include "../Archive/S3Archive.h"

using std::ostream;

namespace {

struct OGRDataSourceDeleter {
  void operator()(OGRDataSource* datasource) {
    if (datasource) {
      GDALClose(datasource);
    }
  }
};
using OGRDataSourceUqPtr = std::unique_ptr<OGRDataSource, OGRDataSourceDeleter>;

struct OGRFeatureDeleter {
  void operator()(OGRFeature* feature) {
    if (feature) {
      OGRFeature::DestroyFeature(feature);
    }
  }
};
using OGRFeatureUqPtr = std::unique_ptr<OGRFeature, OGRFeatureDeleter>;

struct OGRSpatialReferenceDeleter {
  void operator()(OGRSpatialReference* ref) {
    if (ref) {
      OGRSpatialReference::DestroySpatialReference(ref);
    }
  }
};
using OGRSpatialReferenceUqPtr =
    std::unique_ptr<OGRSpatialReference, OGRSpatialReferenceDeleter>;

}  // namespace

namespace Importer_NS {

using FieldNameToIndexMapType = std::map<std::string, size_t>;
using ColumnNameToSourceNameMapType = std::map<std::string, std::string>;
using ColumnIdToRenderGroupAnalyzerMapType =
    std::map<int, std::shared_ptr<RenderGroupAnalyzer>>;
using FeaturePtrVector = std::vector<OGRFeatureUqPtr>;

#define DEBUG_TIMING false
#define DEBUG_RENDER_GROUP_ANALYZER 0
#define DEBUG_AWS_AUTHENTICATION 0

#define DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT 0

static constexpr bool PROMOTE_POLYGON_TO_MULTIPOLYGON = true;

static mapd_shared_mutex status_mutex;
static std::map<std::string, ImportStatus> import_status_map;

Importer::Importer(Catalog_Namespace::Catalog& c,
                   const TableDescriptor* t,
                   const std::string& f,
                   const CopyParams& p)
    : Importer(new Loader(c, t), f, p) {}

Importer::Importer(Loader* providedLoader, const std::string& f, const CopyParams& p)
    : DataStreamSink(p, f), loader(providedLoader) {
  import_id = boost::filesystem::path(file_path).filename().string();
  file_size = 0;
  max_threads = 0;
  p_file = nullptr;
  buffer[0] = nullptr;
  buffer[1] = nullptr;
  // we may be overallocating a little more memory here due to dropping phy cols.
  // it shouldn't be an issue because iteration of it is not supposed to go OOB.
  auto is_array = std::unique_ptr<bool[]>(new bool[loader->get_column_descs().size()]);
  int i = 0;
  bool has_array = false;
  // TODO: replace this ugly way of skipping phy cols once if isPhyGeo is defined
  int skip_physical_cols = 0;
  for (auto& p : loader->get_column_descs()) {
    // phy geo columns can't be in input file
    if (skip_physical_cols-- > 0) {
      continue;
    }
    // neither are rowid or $deleted$
    // note: columns can be added after rowid/$deleted$
    if (p->isVirtualCol || p->isDeletedCol) {
      continue;
    }
    skip_physical_cols = p->columnType.get_physical_cols();
    if (p->columnType.get_type() == kARRAY) {
      is_array.get()[i] = true;
      has_array = true;
    } else {
      is_array.get()[i] = false;
    }
    ++i;
  }
  if (has_array) {
    is_array_a = std::unique_ptr<bool[]>(is_array.release());
  } else {
    is_array_a = std::unique_ptr<bool[]>(nullptr);
  }
}

Importer::~Importer() {
  if (p_file != nullptr) {
    fclose(p_file);
  }
  if (buffer[0] != nullptr) {
    free(buffer[0]);
  }
  if (buffer[1] != nullptr) {
    free(buffer[1]);
  }
}

ImportStatus Importer::get_import_status(const std::string& import_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(status_mutex);
  return import_status_map.at(import_id);
}

void Importer::set_import_status(const std::string& import_id, ImportStatus is) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(status_mutex);
  is.end = std::chrono::steady_clock::now();
  is.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(is.end - is.start);
  import_status_map[import_id] = is;
}

static const std::string trim_space(const char* field, const size_t len) {
  size_t i = 0;
  size_t j = len;
  while (i < j && (field[i] == ' ' || field[i] == '\r')) {
    i++;
  }
  while (i < j && (field[j - 1] == ' ' || field[j - 1] == '\r')) {
    j--;
  }
  return std::string(field + i, j - i);
}

static const bool is_eol(const char& p, const std::string& line_delims) {
  for (auto i : line_delims) {
    if (p == i) {
      return true;
    }
  }
  return false;
}

static const char* get_row(const char* buf,
                           const char* buf_end,
                           const char* entire_buf_end,
                           const CopyParams& copy_params,
                           bool is_begin,
                           const bool* is_array,
                           std::vector<std::string>& row,
                           bool& try_single_thread) {
  const char* field = buf;
  const char* p;
  bool in_quote = false;
  bool in_array = false;
  bool has_escape = false;
  bool strip_quotes = false;
  try_single_thread = false;
  std::string line_endings({copy_params.line_delim, '\r', '\n'});
  for (p = buf; p < entire_buf_end; p++) {
    if (*p == copy_params.escape && p < entire_buf_end - 1 &&
        *(p + 1) == copy_params.quote) {
      p++;
      has_escape = true;
    } else if (copy_params.quoted && *p == copy_params.quote) {
      in_quote = !in_quote;
      if (in_quote) {
        strip_quotes = true;
      }
    } else if (!in_quote && is_array != nullptr && *p == copy_params.array_begin &&
               is_array[row.size()]) {
      in_array = true;
    } else if (!in_quote && is_array != nullptr && *p == copy_params.array_end &&
               is_array[row.size()]) {
      in_array = false;
    } else if (*p == copy_params.delimiter || is_eol(*p, line_endings)) {
      if (!in_quote && !in_array) {
        if (!has_escape && !strip_quotes) {
          std::string s = trim_space(field, p - field);
          row.push_back(s);
        } else {
          auto field_buf = std::unique_ptr<char[]>(new char[p - field + 1]);
          int j = 0, i = 0;
          for (; i < p - field; i++, j++) {
            if (has_escape && field[i] == copy_params.escape &&
                field[i + 1] == copy_params.quote) {
              field_buf[j] = copy_params.quote;
              i++;
            } else {
              field_buf[j] = field[i];
            }
          }
          std::string s = trim_space(field_buf.get(), j);
          if (copy_params.quoted && s.size() > 0 && s.front() == copy_params.quote) {
            s.erase(0, 1);
          }
          if (copy_params.quoted && s.size() > 0 && s.back() == copy_params.quote) {
            s.pop_back();
          }
          row.push_back(s);
        }
        field = p + 1;
        has_escape = false;
        strip_quotes = false;
      }
      if (is_eol(*p, line_endings) &&
          ((!in_quote && !in_array) || copy_params.threads != 1)) {
        while (p + 1 < buf_end && is_eol(*(p + 1), line_endings)) {
          p++;
        }
        break;
      }
    }
  }
  /*
  @TODO(wei) do error handling
  */
  if (in_quote) {
    LOG(ERROR) << "Unmatched quote.";
    try_single_thread = true;
  }
  if (in_array) {
    LOG(ERROR) << "Unmatched array.";
    try_single_thread = true;
  }
  return p;
}

int8_t* appendDatum(int8_t* buf, Datum d, const SQLTypeInfo& ti) {
  switch (ti.get_type()) {
    case kBOOLEAN:
      *(bool*)buf = d.boolval;
      return buf + sizeof(bool);
    case kNUMERIC:
    case kDECIMAL:
    case kBIGINT:
      *(int64_t*)buf = d.bigintval;
      return buf + sizeof(int64_t);
    case kINT:
      *(int32_t*)buf = d.intval;
      return buf + sizeof(int32_t);
    case kSMALLINT:
      *(int16_t*)buf = d.smallintval;
      return buf + sizeof(int16_t);
    case kTINYINT:
      *(int8_t*)buf = d.tinyintval;
      return buf + sizeof(int8_t);
    case kFLOAT:
      *(float*)buf = d.floatval;
      return buf + sizeof(float);
    case kDOUBLE:
      *(double*)buf = d.doubleval;
      return buf + sizeof(double);
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      *(time_t*)buf = d.timeval;
      return buf + sizeof(time_t);
    default:
      return NULL;
  }
  return NULL;
}

Datum NullDatum(SQLTypeInfo& ti) {
  Datum d;
  const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  switch (type) {
    case kBOOLEAN:
      d.boolval = inline_fixed_encoding_null_val(ti);
      break;
    case kBIGINT:
      d.bigintval = inline_fixed_encoding_null_val(ti);
      break;
    case kINT:
      d.intval = inline_fixed_encoding_null_val(ti);
      break;
    case kSMALLINT:
      d.smallintval = inline_fixed_encoding_null_val(ti);
      break;
    case kTINYINT:
      d.tinyintval = inline_fixed_encoding_null_val(ti);
      break;
    case kFLOAT:
      d.floatval = NULL_FLOAT;
      break;
    case kDOUBLE:
      d.doubleval = NULL_DOUBLE;
      break;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      d.timeval = inline_fixed_encoding_null_val(ti);
      break;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      throw std::runtime_error("Internal error: geometry type in NullDatum.");
    default:
      throw std::runtime_error("Internal error: invalid type in NullDatum.");
  }
  return d;
}

ArrayDatum StringToArray(const std::string& s,
                         const SQLTypeInfo& ti,
                         const CopyParams& copy_params) {
  SQLTypeInfo elem_ti = ti.get_elem_type();
  if (s[0] != copy_params.array_begin || s[s.size() - 1] != copy_params.array_end) {
    LOG(WARNING) << "Malformed array: " << s;
    return ArrayDatum(0, NULL, true);
  }
  std::vector<std::string> elem_strs;
  size_t last = 1;
  for (size_t i = s.find(copy_params.array_delim, 1); i != std::string::npos;
       i = s.find(copy_params.array_delim, last)) {
    elem_strs.push_back(s.substr(last, i - last));
    last = i + 1;
  }
  if (last + 1 <= s.size()) {
    elem_strs.push_back(s.substr(last, s.size() - 1 - last));
  }
  if (!elem_ti.is_string()) {
    size_t len = elem_strs.size() * elem_ti.get_size();
    int8_t* buf = (int8_t*)checked_malloc(len);
    int8_t* p = buf;
    for (auto& es : elem_strs) {
      auto e = trim_space(es.c_str(), es.length());
      bool is_null = (e == copy_params.null_str);
      if (!elem_ti.is_string() && e == "") {
        is_null = true;
      }
      if (elem_ti.is_number() || elem_ti.is_time()) {
        if (!isdigit(e[0]) && e[0] != '-') {
          is_null = true;
        }
      }
      Datum d = is_null ? NullDatum(elem_ti) : StringToDatum(e, elem_ti);
      p = appendDatum(p, d, elem_ti);
    }
    return ArrayDatum(len, buf, len == 0);
  }
  // must not be called for array of strings
  CHECK(false);
  return ArrayDatum(0, NULL, true);
}

void addBinaryStringArray(const TDatum& datum, std::vector<std::string>& string_vec) {
  const auto& arr = datum.val.arr_val;
  for (const auto& elem_datum : arr) {
    string_vec.push_back(elem_datum.val.str_val);
  }
}

Datum TDatumToDatum(const TDatum& datum, SQLTypeInfo& ti) {
  Datum d;
  const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  switch (type) {
    case kBOOLEAN:
      d.boolval = datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
      break;
    case kBIGINT:
      d.bigintval =
          datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
      break;
    case kINT:
      d.intval = datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
      break;
    case kSMALLINT:
      d.smallintval =
          datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
      break;
    case kTINYINT:
      d.tinyintval =
          datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
      break;
    case kFLOAT:
      d.floatval = datum.is_null ? NULL_FLOAT : datum.val.real_val;
      break;
    case kDOUBLE:
      d.doubleval = datum.is_null ? NULL_DOUBLE : datum.val.real_val;
      break;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      d.timeval = datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
      break;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      throw std::runtime_error("Internal error: geometry type in TDatumToDatum.");
    default:
      throw std::runtime_error("Internal error: invalid type in TDatumToDatum.");
  }
  return d;
}

ArrayDatum TDatumToArrayDatum(const TDatum& datum, const SQLTypeInfo& ti) {
  SQLTypeInfo elem_ti = ti.get_elem_type();
  CHECK(!elem_ti.is_string());
  size_t len = datum.val.arr_val.size() * elem_ti.get_size();
  int8_t* buf = (int8_t*)checked_malloc(len);
  int8_t* p = buf;
  for (auto& e : datum.val.arr_val) {
    p = appendDatum(p, TDatumToDatum(e, elem_ti), elem_ti);
  }
  return ArrayDatum(len, buf, len == 0);
}

static size_t find_beginning(const char* buffer,
                             size_t begin,
                             size_t end,
                             const CopyParams& copy_params) {
  // @TODO(wei) line_delim is in quotes note supported
  if (begin == 0 || (begin > 0 && buffer[begin - 1] == copy_params.line_delim)) {
    return 0;
  }
  size_t i;
  const char* buf = buffer + begin;
  for (i = 0; i < end - begin; i++) {
    if (buf[i] == copy_params.line_delim) {
      return i + 1;
    }
  }
  return i;
}

void TypedImportBuffer::add_value(const ColumnDescriptor* cd,
                                  const std::string& val,
                                  const bool is_null,
                                  const CopyParams& copy_params,
                                  const int64_t replicate_count) {
  set_replicate_count(replicate_count);
  const auto type = cd->columnType.get_type();
  switch (type) {
    case kBOOLEAN: {
      if (is_null) {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addBoolean(inline_fixed_encoding_null_val(cd->columnType));
      } else {
        SQLTypeInfo ti = cd->columnType;
        Datum d = StringToDatum(val, ti);
        addBoolean((int8_t)d.boolval);
      }
      break;
    }
    case kTINYINT: {
      if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
        SQLTypeInfo ti = cd->columnType;
        Datum d = StringToDatum(val, ti);
        addTinyint(d.tinyintval);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addTinyint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kSMALLINT: {
      if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
        SQLTypeInfo ti = cd->columnType;
        Datum d = StringToDatum(val, ti);
        addSmallint(d.smallintval);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addSmallint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kINT: {
      if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
        SQLTypeInfo ti = cd->columnType;
        Datum d = StringToDatum(val, ti);
        addInt(d.intval);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addInt(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kBIGINT: {
      if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
        SQLTypeInfo ti = cd->columnType;
        Datum d = StringToDatum(val, ti);
        addBigint(d.bigintval);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addBigint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kDECIMAL:
    case kNUMERIC: {
      if (!is_null) {
        SQLTypeInfo ti(kNUMERIC, 0, 0, false);
        Datum d = StringToDatum(val, ti);
        const auto converted_decimal_value =
            convert_decimal_value_to_scale(d.bigintval, ti, cd->columnType);
        addBigint(converted_decimal_value);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addBigint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kFLOAT:
      if (!is_null && (val[0] == '.' || isdigit(val[0]) || val[0] == '-')) {
        addFloat((float)std::atof(val.c_str()));
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addFloat(NULL_FLOAT);
      }
      break;
    case kDOUBLE:
      if (!is_null && (val[0] == '.' || isdigit(val[0]) || val[0] == '-')) {
        addDouble(std::atof(val.c_str()));
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addDouble(NULL_DOUBLE);
      }
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      // @TODO(wei) for now, use empty string for nulls
      if (is_null) {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addString(std::string());
      } else {
        if (val.length() > StringDictionary::MAX_STRLEN) {
          throw std::runtime_error("String too long for column " + cd->columnName +
                                   " was " + std::to_string(val.length()) + " max is " +
                                   std::to_string(StringDictionary::MAX_STRLEN));
        }
        addString(val);
      }
      break;
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
        SQLTypeInfo ti = cd->columnType;
        Datum d = StringToDatum(val, ti);
        addTime(d.timeval);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addTime(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kARRAY:
      if (is_null && cd->columnType.get_notnull()) {
        throw std::runtime_error("NULL for column " + cd->columnName);
      }
      if (IS_STRING(cd->columnType.get_subtype())) {
        std::vector<std::string>& string_vec = addStringArray();
        ImporterUtils::parseStringArray(val, copy_params, string_vec);
      } else {
        if (!is_null) {
          SQLTypeInfo ti = cd->columnType;
          ArrayDatum d = StringToArray(val, ti, copy_params);
          if (ti.get_size() > 0 && static_cast<size_t>(ti.get_size()) != d.length) {
            throw std::runtime_error("Fixed length array for column " + cd->columnName +
                                     " has incorrect length: " + val);
          }
          addArray(d);
        } else {
          addArray(ArrayDatum(0, NULL, true));
        }
      }
      break;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      addGeoString(val);
      break;
    default:
      CHECK(false) << "TypedImportBuffer::add_value() does not support type " << type;
  }
}

void TypedImportBuffer::pop_value() {
  const auto type = column_desc_->columnType.is_decimal()
                        ? decimal_to_int_type(column_desc_->columnType)
                        : column_desc_->columnType.get_type();
  switch (type) {
    case kBOOLEAN:
      bool_buffer_->pop_back();
      break;
    case kTINYINT:
      tinyint_buffer_->pop_back();
      break;
    case kSMALLINT:
      smallint_buffer_->pop_back();
      break;
    case kINT:
      int_buffer_->pop_back();
      break;
    case kBIGINT:
      bigint_buffer_->pop_back();
      break;
    case kFLOAT:
      float_buffer_->pop_back();
      break;
    case kDOUBLE:
      double_buffer_->pop_back();
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      string_buffer_->pop_back();
      break;
    case kDATE:
    case kTIME:
    case kTIMESTAMP:
      time_buffer_->pop_back();
      break;
    case kARRAY:
      if (IS_STRING(column_desc_->columnType.get_subtype())) {
        string_array_buffer_->pop_back();
      } else {
        array_buffer_->pop_back();
      }
      break;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      geo_string_buffer_->pop_back();
      break;
    default:
      CHECK(false) << "TypedImportBuffer::pop_value() does not support type " << type;
  }
}

namespace {

using namespace arrow;

#define ARROW_THROW_IF(cond, message)  \
  if ((cond)) {                        \
    LOG(ERROR) << message;             \
    throw std::runtime_error(message); \
  }

template <typename ArrayType, typename T>
inline void append_arrow_primitive(const Array& values,
                                   const T null_sentinel,
                                   std::vector<T>* buffer) {
  const auto& typed_values = static_cast<const ArrayType&>(values);
  buffer->reserve(typed_values.length());
  const T* raw_values = typed_values.raw_values();
  if (typed_values.null_count() > 0) {
    for (int64_t i = 0; i < typed_values.length(); i++) {
      if (typed_values.IsNull(i)) {
        buffer->push_back(null_sentinel);
      } else {
        buffer->push_back(raw_values[i]);
      }
    }
  } else {
    for (int64_t i = 0; i < typed_values.length(); i++) {
      buffer->push_back(raw_values[i]);
    }
  }
}

void append_arrow_boolean(const ColumnDescriptor* cd,
                          const Array& values,
                          std::vector<int8_t>* buffer) {
  ARROW_THROW_IF(values.type_id() != Type::BOOL, "Expected boolean col");
  const int8_t null_sentinel = inline_fixed_encoding_null_val(cd->columnType);
  const auto& typed_values = static_cast<const BooleanArray&>(values);
  buffer->reserve(typed_values.length());
  for (int64_t i = 0; i < typed_values.length(); i++) {
    if (typed_values.IsNull(i)) {
      buffer->push_back(null_sentinel);
    } else {
      buffer->push_back(static_cast<int8_t>(typed_values.Value(i)));
    }
  }
}

template <typename ArrowType, typename T>
void append_arrow_integer(const ColumnDescriptor* cd,
                          const Array& values,
                          std::vector<T>* buffer) {
  using ArrayType = typename TypeTraits<ArrowType>::ArrayType;
  const T null_sentinel = inline_fixed_encoding_null_val(cd->columnType);
  append_arrow_primitive<ArrayType, T>(values, null_sentinel, buffer);
}

void append_arrow_float(const ColumnDescriptor* cd,
                        const Array& values,
                        std::vector<float>* buffer) {
  ARROW_THROW_IF(values.type_id() != Type::FLOAT, "Expected float col");
  append_arrow_primitive<FloatArray, float>(values, NULL_FLOAT, buffer);
}

void append_arrow_double(const ColumnDescriptor* cd,
                         const Array& values,
                         std::vector<double>* buffer) {
  ARROW_THROW_IF(values.type_id() != Type::DOUBLE, "Expected double col");
  append_arrow_primitive<DoubleArray, double>(values, NULL_DOUBLE, buffer);
}

constexpr int64_t kMillisecondsInSecond = 1000L;
constexpr int64_t kMicrosecondsInSecond = 1000L * 1000L;
constexpr int64_t kNanosecondsinSecond = 1000L * 1000L * 1000L;
constexpr int32_t kSecondsInDay = 86400;

void append_arrow_time(const ColumnDescriptor* cd,
                       const Array& values,
                       std::vector<time_t>* buffer) {
  const time_t null_sentinel = inline_fixed_encoding_null_val(cd->columnType);
  if (values.type_id() == Type::TIME32) {
    const auto& typed_values = static_cast<const Time32Array&>(values);
    const auto& type = static_cast<const Time32Type&>(*values.type());

    buffer->reserve(typed_values.length());
    const int32_t* raw_values = typed_values.raw_values();
    const TimeUnit::type unit = type.unit();

    for (int64_t i = 0; i < typed_values.length(); i++) {
      if (typed_values.IsNull(i)) {
        buffer->push_back(null_sentinel);
      } else {
        switch (unit) {
          case TimeUnit::SECOND:
            buffer->push_back(static_cast<time_t>(raw_values[i]));
            break;
          case TimeUnit::MILLI:
            buffer->push_back(static_cast<time_t>(raw_values[i] / kMillisecondsInSecond));
            break;
          default:
            // unreachable code
            CHECK(false);
            break;
        }
      }
    }
  } else if (values.type_id() == Type::TIME64) {
    const auto& typed_values = static_cast<const Time64Array&>(values);
    const auto& type = static_cast<const Time64Type&>(*values.type());

    buffer->reserve(typed_values.length());
    const int64_t* raw_values = typed_values.raw_values();
    const TimeUnit::type unit = type.unit();

    for (int64_t i = 0; i < typed_values.length(); i++) {
      if (typed_values.IsNull(i)) {
        buffer->push_back(null_sentinel);
      } else {
        switch (unit) {
          case TimeUnit::MICRO:
            buffer->push_back(static_cast<time_t>(raw_values[i] / kMicrosecondsInSecond));
            break;
          case TimeUnit::NANO:
            buffer->push_back(static_cast<time_t>(raw_values[i] / kNanosecondsinSecond));
            break;
          default:
            // unreachable code
            CHECK(false);
            break;
        }
      }
    }
  } else {
    ARROW_THROW_IF(true, "Column was not time32 or time64");
  }
}

void append_arrow_timestamp(const ColumnDescriptor* cd,
                            const Array& values,
                            std::vector<time_t>* buffer) {
  ARROW_THROW_IF(values.type_id() != Type::TIMESTAMP, "Expected timestamp col");

  const time_t null_sentinel = inline_fixed_encoding_null_val(cd->columnType);
  const auto& typed_values = static_cast<const TimestampArray&>(values);
  const auto& type = static_cast<const TimestampType&>(*values.type());

  buffer->reserve(typed_values.length());
  const int64_t* raw_values = typed_values.raw_values();
  const TimeUnit::type unit = type.unit();

  for (int64_t i = 0; i < typed_values.length(); i++) {
    if (typed_values.IsNull(i)) {
      buffer->push_back(null_sentinel);
    } else {
      switch (unit) {
        case TimeUnit::SECOND:
          buffer->push_back(static_cast<time_t>(raw_values[i]));
          break;
        case TimeUnit::MILLI:
          buffer->push_back(static_cast<time_t>(raw_values[i] / kMillisecondsInSecond));
          break;
        case TimeUnit::MICRO:
          buffer->push_back(static_cast<time_t>(raw_values[i] / kMicrosecondsInSecond));
          break;
        case TimeUnit::NANO:
          buffer->push_back(static_cast<time_t>(raw_values[i] / kNanosecondsinSecond));
          break;
        default:
          break;
      }
    }
  }
}

void append_arrow_date(const ColumnDescriptor* cd,
                       const Array& values,
                       std::vector<time_t>* buffer) {
  const time_t null_sentinel = inline_fixed_encoding_null_val(cd->columnType);
  if (values.type_id() == Type::DATE32) {
    const auto& typed_values = static_cast<const Date32Array&>(values);

    buffer->reserve(typed_values.length());
    const int32_t* raw_values = typed_values.raw_values();

    for (int64_t i = 0; i < typed_values.length(); i++) {
      if (typed_values.IsNull(i)) {
        buffer->push_back(null_sentinel);
      } else {
        buffer->push_back(static_cast<time_t>(raw_values[i] * kSecondsInDay));
      }
    }
  } else if (values.type_id() == Type::DATE64) {
    const auto& typed_values = static_cast<const Date64Array&>(values);

    buffer->reserve(typed_values.length());
    const int64_t* raw_values = typed_values.raw_values();

    // Convert from milliseconds since UNIX epoch
    for (int64_t i = 0; i < typed_values.length(); i++) {
      if (typed_values.IsNull(i)) {
        buffer->push_back(null_sentinel);
      } else {
        buffer->push_back(static_cast<time_t>(raw_values[i] / 1000));
      }
    }
  } else {
    ARROW_THROW_IF(true, "Column was not date32 or date64");
  }
}

void append_arrow_date_i32(const ColumnDescriptor* cd,
                           const Array& values,
                           std::vector<time_t>* buffer) {
  const time_t null_sentinel = inline_fixed_encoding_null_val(cd->columnType);
  if (values.type_id() == Type::DATE32) {
    const auto& typed_values = static_cast<const Date32Array&>(values);

    buffer->reserve(typed_values.length());
    const int32_t* raw_values = typed_values.raw_values();

    for (int64_t i = 0; i < typed_values.length(); i++) {
      if (typed_values.IsNull(i)) {
        buffer->push_back(null_sentinel);
      } else {
        buffer->push_back(static_cast<time_t>(raw_values[i]));
      }
    }
  } else if (values.type_id() == Type::DATE64) {
    const auto& typed_values = static_cast<const Date64Array&>(values);

    buffer->reserve(typed_values.length());
    const int64_t* raw_values = typed_values.raw_values();

    // Convert from milliseconds since UNIX epoch into days since UNIX epoch
    for (int64_t i = 0; i < typed_values.length(); i++) {
      if (typed_values.IsNull(i)) {
        buffer->push_back(null_sentinel);
      } else {
        buffer->push_back(static_cast<int32_t>((raw_values[i] / 1000) / kSecondsInDay));
      }
    }
  } else {
    ARROW_THROW_IF(true, "Column was not date32 or date64");
  }
}

void append_arrow_binary(const ColumnDescriptor* cd,
                         const Array& values,
                         std::vector<std::string>* buffer) {
  ARROW_THROW_IF(values.type_id() != Type::BINARY && values.type_id() != Type::STRING,
                 "Expected binary col");

  const auto& typed_values = static_cast<const BinaryArray&>(values);
  buffer->reserve(typed_values.length());

  const char* bytes;
  int32_t bytes_length = 0;
  for (int64_t i = 0; i < typed_values.length(); i++) {
    if (typed_values.IsNull(i)) {
      // TODO(wesm): How are nulls handled for strings?
      buffer->push_back(std::string());
    } else {
      bytes = reinterpret_cast<const char*>(typed_values.GetValue(i, &bytes_length));
      buffer->push_back(std::string(bytes, bytes_length));
    }
  }
}

}  // namespace

size_t TypedImportBuffer::add_arrow_values(const ColumnDescriptor* cd,
                                           const arrow::Array& col) {
  const auto type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType)
                                                : cd->columnType.get_type();
  if (cd->columnType.get_notnull()) {
    // We can't have any null values for this column; to have them is an error
    if (col.null_count() > 0) {
      throw std::runtime_error("NULL not allowed for column " + cd->columnName);
    }
  }

  switch (type) {
    case kBOOLEAN:
      append_arrow_boolean(cd, col, bool_buffer_);
      break;
    case kTINYINT:
      ARROW_THROW_IF(col.type_id() != arrow::Type::INT8, "Expected int8 type");
      append_arrow_integer<arrow::Int8Type, int8_t>(cd, col, tinyint_buffer_);
      break;
    case kSMALLINT:
      ARROW_THROW_IF(col.type_id() != arrow::Type::INT16, "Expected int16 type");
      append_arrow_integer<arrow::Int16Type, int16_t>(cd, col, smallint_buffer_);
      break;
    case kINT:
      ARROW_THROW_IF(col.type_id() != arrow::Type::INT32, "Expected int32 type");
      append_arrow_integer<arrow::Int32Type, int32_t>(cd, col, int_buffer_);
      break;
    case kBIGINT:
      ARROW_THROW_IF(col.type_id() != arrow::Type::INT64, "Expected int64 type");
      append_arrow_integer<arrow::Int64Type, int64_t>(cd, col, bigint_buffer_);
      break;
    case kFLOAT:
      append_arrow_float(cd, col, float_buffer_);
      break;
    case kDOUBLE:
      append_arrow_double(cd, col, double_buffer_);
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      append_arrow_binary(cd, col, string_buffer_);
      break;
    case kTIME:
      append_arrow_time(cd, col, time_buffer_);
      break;
    case kTIMESTAMP:
      append_arrow_timestamp(cd, col, time_buffer_);
      break;
    case kDATE:
      if (cd->columnType.get_compression() == kENCODING_DATE_IN_DAYS) {
        append_arrow_date_i32(cd, col, time_buffer_);
        break;
      }
      append_arrow_date(cd, col, time_buffer_);
      break;
    case kARRAY:
      throw std::runtime_error("Arrow array appends not yet supported");
    default:
      throw std::runtime_error("Invalid Type");
  }
  return col.length();
}

size_t TypedImportBuffer::add_values(const ColumnDescriptor* cd, const TColumn& col) {
  size_t dataSize = 0;
  const auto type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType)
                                                : cd->columnType.get_type();
  if (cd->columnType.get_notnull()) {
    // We can't have any null values for this column; to have them is an error
    if (std::any_of(col.nulls.begin(), col.nulls.end(), [](int i) { return i != 0; })) {
      throw std::runtime_error("NULL for column " + cd->columnName);
    }
  }

  switch (type) {
    case kBOOLEAN: {
      dataSize = col.data.int_col.size();
      bool_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          bool_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        } else {
          bool_buffer_->push_back((int8_t)col.data.int_col[i]);
        }
      }
      break;
    }
    case kTINYINT: {
      dataSize = col.data.int_col.size();
      tinyint_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          tinyint_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        } else {
          tinyint_buffer_->push_back((int8_t)col.data.int_col[i]);
        }
      }
      break;
    }
    case kSMALLINT: {
      dataSize = col.data.int_col.size();
      smallint_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          smallint_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        } else {
          smallint_buffer_->push_back((int16_t)col.data.int_col[i]);
        }
      }
      break;
    }
    case kINT: {
      dataSize = col.data.int_col.size();
      int_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          int_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        } else {
          int_buffer_->push_back((int32_t)col.data.int_col[i]);
        }
      }
      break;
    }
    case kBIGINT: {
      dataSize = col.data.int_col.size();
      bigint_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          bigint_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        } else {
          bigint_buffer_->push_back((int64_t)col.data.int_col[i]);
        }
      }
      break;
    }
    case kFLOAT: {
      dataSize = col.data.real_col.size();
      float_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          float_buffer_->push_back(NULL_FLOAT);
        } else {
          float_buffer_->push_back((float)col.data.real_col[i]);
        }
      }
      break;
    }
    case kDOUBLE: {
      dataSize = col.data.real_col.size();
      double_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          double_buffer_->push_back(NULL_DOUBLE);
        } else {
          double_buffer_->push_back((double)col.data.real_col[i]);
        }
      }
      break;
    }
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      // TODO: for now, use empty string for nulls
      dataSize = col.data.str_col.size();
      string_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          string_buffer_->push_back(std::string());
        } else {
          string_buffer_->push_back(col.data.str_col[i]);
        }
      }
      break;
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      dataSize = col.data.int_col.size();
      time_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          time_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        } else {
          time_buffer_->push_back((time_t)col.data.int_col[i]);
        }
      }
      break;
    }
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON: {
      dataSize = col.data.str_col.size();
      geo_string_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          // TODO: add support for NULL geo
          geo_string_buffer_->push_back(std::string());
        } else {
          geo_string_buffer_->push_back(col.data.str_col[i]);
        }
      }
      break;
    }
    case kARRAY: {
      // TODO: add support for nulls inside array
      dataSize = col.data.arr_col.size();
      if (IS_STRING(cd->columnType.get_subtype())) {
        for (size_t i = 0; i < dataSize; i++) {
          std::vector<std::string>& string_vec = addStringArray();
          if (!col.nulls[i]) {
            size_t stringArrSize = col.data.arr_col[i].data.str_col.size();
            for (size_t str_idx = 0; str_idx != stringArrSize; ++str_idx) {
              string_vec.push_back(col.data.arr_col[i].data.str_col[str_idx]);
            }
          }
        }
      } else {
        auto elem_ti = cd->columnType.get_subtype();
        switch (elem_ti) {
          case kBOOLEAN: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(ArrayDatum(0, NULL, true));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int8_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(bool*)p = static_cast<bool>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(bool);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          case kTINYINT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(ArrayDatum(0, NULL, true));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int8_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(int8_t*)p = static_cast<int8_t>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(int8_t);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          case kSMALLINT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(ArrayDatum(0, NULL, true));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int16_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(int16_t*)p =
                      static_cast<int16_t>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(int16_t);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          case kINT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(ArrayDatum(0, NULL, true));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int32_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(int32_t*)p =
                      static_cast<int32_t>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(int32_t);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          case kBIGINT:
          case kNUMERIC:
          case kDECIMAL: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(ArrayDatum(0, NULL, true));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int64_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(int64_t*)p =
                      static_cast<int64_t>(col.data.arr_col[j].data.int_col[j]);
                  p += sizeof(int64_t);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          case kFLOAT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(ArrayDatum(0, NULL, true));
              } else {
                size_t len = col.data.arr_col[i].data.real_col.size();
                size_t byteSize = len * sizeof(float);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(float*)p = static_cast<float>(col.data.arr_col[i].data.real_col[j]);
                  p += sizeof(float);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          case kDOUBLE: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(ArrayDatum(0, NULL, true));
              } else {
                size_t len = col.data.arr_col[i].data.real_col.size();
                size_t byteSize = len * sizeof(double);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(double*)p = static_cast<double>(col.data.arr_col[i].data.real_col[j]);
                  p += sizeof(double);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          case kTIME:
          case kTIMESTAMP:
          case kDATE: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(ArrayDatum(0, NULL, true));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteWidth = sizeof(time_t);
                size_t byteSize = len * byteWidth;
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(time_t*)p = static_cast<time_t>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(time_t);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          default:
            throw std::runtime_error("Invalid Array Type");
        }
      }
      break;
    }
    default:
      throw std::runtime_error("Invalid Type");
  }
  return dataSize;
}

void TypedImportBuffer::add_value(const ColumnDescriptor* cd,
                                  const TDatum& datum,
                                  const bool is_null,
                                  const int64_t replicate_count) {
  set_replicate_count(replicate_count);
  const auto type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType)
                                                : cd->columnType.get_type();
  switch (type) {
    case kBOOLEAN: {
      if (is_null) {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addBoolean(inline_fixed_encoding_null_val(cd->columnType));
      } else {
        addBoolean((int8_t)datum.val.int_val);
      }
      break;
    }
    case kTINYINT:
      if (!is_null) {
        addTinyint((int8_t)datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addTinyint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kSMALLINT:
      if (!is_null) {
        addSmallint((int16_t)datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addSmallint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kINT:
      if (!is_null) {
        addInt((int32_t)datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addInt(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kBIGINT:
      if (!is_null) {
        addBigint(datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addBigint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kFLOAT:
      if (!is_null) {
        addFloat((float)datum.val.real_val);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addFloat(NULL_FLOAT);
      }
      break;
    case kDOUBLE:
      if (!is_null) {
        addDouble(datum.val.real_val);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addDouble(NULL_DOUBLE);
      }
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      // @TODO(wei) for now, use empty string for nulls
      if (is_null) {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addString(std::string());
      } else {
        addString(datum.val.str_val);
      }
      break;
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      if (!is_null) {
        addTime(datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addTime(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kARRAY:
      if (is_null && cd->columnType.get_notnull()) {
        throw std::runtime_error("NULL for column " + cd->columnName);
      }
      if (IS_STRING(cd->columnType.get_subtype())) {
        std::vector<std::string>& string_vec = addStringArray();
        addBinaryStringArray(datum, string_vec);
      } else {
        if (!is_null) {
          addArray(TDatumToArrayDatum(datum, cd->columnType));
        } else {
          addArray(ArrayDatum(0, NULL, true));
        }
      }
      break;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      if (is_null) {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addGeoString(std::string());
      } else {
        addGeoString(datum.val.str_val);
      }
      break;
    default:
      CHECK(false) << "TypedImportBuffer::add_value() does not support type " << type;
  }
}

template <typename T>
ostream& operator<<(ostream& out, const std::vector<T>& v) {
  out << "[";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last) {
      out << ", ";
    }
  }
  out << "]";
  return out;
}

bool importGeoFromLonLat(double lon, double lat, std::vector<double>& coords) {
  if (std::isinf(lat) || std::isnan(lat) || std::isinf(lon) || std::isnan(lon)) {
    return false;
  }
  auto point = new OGRPoint(lon, lat);
  // NOTE(adb): Use OGRSpatialReferenceUqPtr to ensure proper deletion
  // auto poSR0 = new OGRSpatialReference();
  // poSR0->importFromEPSG(4326);
  // point->assignSpatialReference(poSR0);

  // auto poSR = new OGRSpatialReference();
  // poSR->importFromEPSG(3857);
  // point->transformTo(poSR);

  coords.push_back(point->getX());
  coords.push_back(point->getY());
  return true;
}

uint64_t compress_coord(double coord, const SQLTypeInfo& ti, bool x) {
  if (ti.get_compression() == kENCODING_GEOINT && ti.get_comp_param() == 32) {
    return x ? Geo_namespace::compress_longitude_coord_geoint32(coord)
             : Geo_namespace::compress_lattitude_coord_geoint32(coord);
  }
  return *reinterpret_cast<uint64_t*>(may_alias_ptr(&coord));
}

std::vector<uint8_t> compress_coords(std::vector<double>& coords, const SQLTypeInfo& ti) {
  std::vector<uint8_t> compressed_coords;
  bool x = true;
  for (auto coord : coords) {
    auto coord_data_ptr = reinterpret_cast<uint64_t*>(&coord);
    uint64_t coord_data = *coord_data_ptr;
    size_t coord_data_size = sizeof(double);

    if (ti.get_output_srid() == 4326) {
      if (x) {
        if (coord < -180.0 || coord > 180.0) {
          throw std::runtime_error("WGS84 longitude " + std::to_string(coord) +
                                   " is out of bounds");
        }
      } else {
        if (coord < -90.0 || coord > 90.0) {
          throw std::runtime_error("WGS84 latitude " + std::to_string(coord) +
                                   " is out of bounds");
        }
      }
      if (ti.get_compression() == kENCODING_GEOINT && ti.get_comp_param() == 32) {
        coord_data = compress_coord(coord, ti, x);
        coord_data_size = ti.get_comp_param() / 8;
      }
      x = !x;
    }

    for (size_t i = 0; i < coord_data_size; i++) {
      compressed_coords.push_back(coord_data & 0xFF);
      coord_data >>= 8;
    }
  }
  return compressed_coords;
}

void Importer::set_geo_physical_import_buffer(
    const Catalog_Namespace::Catalog& catalog,
    const ColumnDescriptor* cd,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
    size_t& col_idx,
    std::vector<double>& coords,
    std::vector<double>& bounds,
    std::vector<int>& ring_sizes,
    std::vector<int>& poly_rings,
    int render_group,
    const int64_t replicate_count) {
  const auto col_ti = cd->columnType;
  const auto col_type = col_ti.get_type();
  auto columnId = cd->columnId;
  auto cd_coords = catalog.getMetadataForColumn(cd->tableId, ++columnId);
  std::vector<TDatum> td_coords_data;
  std::vector<uint8_t> compressed_coords = compress_coords(coords, col_ti);
  for (auto cc : compressed_coords) {
    TDatum td_byte;
    td_byte.val.int_val = cc;
    td_coords_data.push_back(td_byte);
  }
  TDatum tdd_coords;
  tdd_coords.val.arr_val = td_coords_data;
  tdd_coords.is_null = false;
  import_buffers[col_idx++]->add_value(cd_coords, tdd_coords, false, replicate_count);

  if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    // Create ring_sizes array value and add it to the physical column
    auto cd_ring_sizes = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    std::vector<TDatum> td_ring_sizes;
    for (auto ring_size : ring_sizes) {
      TDatum td_ring_size;
      td_ring_size.val.int_val = ring_size;
      td_ring_sizes.push_back(td_ring_size);
    }
    TDatum tdd_ring_sizes;
    tdd_ring_sizes.val.arr_val = td_ring_sizes;
    tdd_ring_sizes.is_null = false;
    import_buffers[col_idx++]->add_value(
        cd_ring_sizes, tdd_ring_sizes, false, replicate_count);
  }

  if (col_type == kMULTIPOLYGON) {
    // Create poly_rings array value and add it to the physical column
    auto cd_poly_rings = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    std::vector<TDatum> td_poly_rings;
    for (auto num_rings : poly_rings) {
      TDatum td_num_rings;
      td_num_rings.val.int_val = num_rings;
      td_poly_rings.push_back(td_num_rings);
    }
    TDatum tdd_poly_rings;
    tdd_poly_rings.val.arr_val = td_poly_rings;
    tdd_poly_rings.is_null = false;
    import_buffers[col_idx++]->add_value(
        cd_poly_rings, tdd_poly_rings, false, replicate_count);
  }

  if (col_type == kLINESTRING || col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    auto cd_bounds = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    std::vector<TDatum> td_bounds_data;
    for (auto b : bounds) {
      TDatum td_double;
      td_double.val.real_val = b;
      td_bounds_data.push_back(td_double);
    }
    TDatum tdd_bounds;
    tdd_bounds.val.arr_val = td_bounds_data;
    tdd_bounds.is_null = false;
    import_buffers[col_idx++]->add_value(cd_bounds, tdd_bounds, false, replicate_count);
  }

  if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    // Create render_group value and add it to the physical column
    auto cd_render_group = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    TDatum td_render_group;
    td_render_group.val.int_val = render_group;
    td_render_group.is_null = false;
    import_buffers[col_idx++]->add_value(
        cd_render_group, td_render_group, false, replicate_count);
  }
}

void Importer::set_geo_physical_import_buffer_columnar(
    const Catalog_Namespace::Catalog& catalog,
    const ColumnDescriptor* cd,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
    size_t& col_idx,
    std::vector<std::vector<double>>& coords_column,
    std::vector<std::vector<double>>& bounds_column,
    std::vector<std::vector<int>>& ring_sizes_column,
    std::vector<std::vector<int>>& poly_rings_column,
    int render_group,
    const int64_t replicate_count) {
  const auto col_ti = cd->columnType;
  const auto col_type = col_ti.get_type();
  auto columnId = cd->columnId;

  auto coords_row_count = coords_column.size();
  auto cd_coords = catalog.getMetadataForColumn(cd->tableId, ++columnId);
  for (auto coords : coords_column) {
    std::vector<TDatum> td_coords_data;
    std::vector<uint8_t> compressed_coords = compress_coords(coords, col_ti);
    for (auto cc : compressed_coords) {
      TDatum td_byte;
      td_byte.val.int_val = cc;
      td_coords_data.push_back(td_byte);
    }
    TDatum tdd_coords;
    tdd_coords.val.arr_val = td_coords_data;
    tdd_coords.is_null = false;
    import_buffers[col_idx]->add_value(cd_coords, tdd_coords, false, replicate_count);
  }
  col_idx++;

  if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    if (ring_sizes_column.size() != coords_row_count) {
      CHECK(false) << "Geometry import columnar: ring sizes column size mismatch";
    }
    // Create ring_sizes array value and add it to the physical column
    for (auto ring_sizes : ring_sizes_column) {
      auto cd_ring_sizes = catalog.getMetadataForColumn(cd->tableId, ++columnId);
      std::vector<TDatum> td_ring_sizes;
      for (auto ring_size : ring_sizes) {
        TDatum td_ring_size;
        td_ring_size.val.int_val = ring_size;
        td_ring_sizes.push_back(td_ring_size);
      }
      TDatum tdd_ring_sizes;
      tdd_ring_sizes.val.arr_val = td_ring_sizes;
      tdd_ring_sizes.is_null = false;
      import_buffers[col_idx]->add_value(
          cd_ring_sizes, tdd_ring_sizes, false, replicate_count);
    }
    col_idx++;
  }

  if (col_type == kMULTIPOLYGON) {
    if (poly_rings_column.size() != coords_row_count) {
      CHECK(false) << "Geometry import columnar: poly rings column size mismatch";
    }
    // Create poly_rings array value and add it to the physical column
    for (auto poly_rings : poly_rings_column) {
      auto cd_poly_rings = catalog.getMetadataForColumn(cd->tableId, ++columnId);
      std::vector<TDatum> td_poly_rings;
      for (auto num_rings : poly_rings) {
        TDatum td_num_rings;
        td_num_rings.val.int_val = num_rings;
        td_poly_rings.push_back(td_num_rings);
      }
      TDatum tdd_poly_rings;
      tdd_poly_rings.val.arr_val = td_poly_rings;
      tdd_poly_rings.is_null = false;
      import_buffers[col_idx]->add_value(
          cd_poly_rings, tdd_poly_rings, false, replicate_count);
    }
    col_idx++;
  }

  if (col_type == kLINESTRING || col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    if (bounds_column.size() != coords_row_count) {
      CHECK(false) << "Geometry import columnar: bounds column size mismatch";
    }
    for (auto bounds : bounds_column) {
      auto cd_bounds = catalog.getMetadataForColumn(cd->tableId, ++columnId);
      std::vector<TDatum> td_bounds_data;
      for (auto b : bounds) {
        TDatum td_double;
        td_double.val.real_val = b;
        td_bounds_data.push_back(td_double);
      }
      TDatum tdd_bounds;
      tdd_bounds.val.arr_val = td_bounds_data;
      tdd_bounds.is_null = false;
      import_buffers[col_idx]->add_value(cd_bounds, tdd_bounds, false, replicate_count);
    }
    col_idx++;
  }

  if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    // Create render_group value and add it to the physical column
    auto cd_render_group = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    TDatum td_render_group;
    td_render_group.val.int_val = render_group;
    td_render_group.is_null = false;
    for (decltype(coords_row_count) i = 0; i < coords_row_count; i++) {
      import_buffers[col_idx]->add_value(
          cd_render_group, td_render_group, false, replicate_count);
    }
    col_idx++;
  }
}

static ImportStatus import_thread_delimited(
    int thread_id,
    Importer* importer,
    std::shared_ptr<const char> sbuffer,
    size_t begin_pos,
    size_t end_pos,
    size_t total_size,
    const ColumnIdToRenderGroupAnalyzerMapType& columnIdToRenderGroupAnalyzerMap,
    size_t first_row_index_this_buffer) {
  ImportStatus import_status;
  int64_t total_get_row_time_us = 0;
  int64_t total_str_to_val_time_us = 0;
  auto buffer = sbuffer.get();
  auto load_ms = measure<>::execution([]() {});
  auto ms = measure<>::execution([&]() {
    const CopyParams& copy_params = importer->get_copy_params();
    const std::list<const ColumnDescriptor*>& col_descs = importer->get_column_descs();
    size_t begin = find_beginning(buffer, begin_pos, end_pos, copy_params);
    const char* thread_buf = buffer + begin_pos + begin;
    const char* thread_buf_end = buffer + end_pos;
    const char* buf_end = buffer + total_size;
    bool try_single_thread = false;
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers =
        importer->get_import_buffers(thread_id);
    auto us = measure<std::chrono::microseconds>::execution([&]() {});
    for (const auto& p : import_buffers) {
      p->clear();
    }
    std::vector<std::string> row;
    size_t row_index_plus_one = 0;
    for (const char* p = thread_buf; p < thread_buf_end; p++) {
      row.clear();
      if (DEBUG_TIMING) {
        us = measure<std::chrono::microseconds>::execution([&]() {
          p = get_row(p,
                      thread_buf_end,
                      buf_end,
                      copy_params,
                      p == thread_buf,
                      importer->get_is_array(),
                      row,
                      try_single_thread);
        });
        total_get_row_time_us += us;
      } else {
        p = get_row(p,
                    thread_buf_end,
                    buf_end,
                    copy_params,
                    p == thread_buf,
                    importer->get_is_array(),
                    row,
                    try_single_thread);
      }
      row_index_plus_one++;
      int phys_cols = 0;
      int point_cols = 0;
      for (const auto cd : col_descs) {
        const auto& col_ti = cd->columnType;
        phys_cols += col_ti.get_physical_cols();
        if (cd->columnType.get_type() == kPOINT) {
          point_cols++;
        }
      }
      auto num_cols = col_descs.size() - phys_cols;
      // Each POINT could consume two separate coords instead of a single WKT
      if (row.size() < num_cols || (num_cols + point_cols) < row.size()) {
        import_status.rows_rejected++;
        LOG(ERROR) << "Incorrect Row (expected " << num_cols << " columns, has "
                   << row.size() << "): " << row;
        if (import_status.rows_rejected > copy_params.max_reject) {
          break;
        }
        continue;
      }
      us = measure<std::chrono::microseconds>::execution([&]() {
        size_t import_idx = 0;
        size_t col_idx = 0;
        try {
          for (auto cd_it = col_descs.begin(); cd_it != col_descs.end(); cd_it++) {
            auto cd = *cd_it;
            const auto& col_ti = cd->columnType;
            if (col_ti.get_physical_cols() == 0) {
              // not geo

              // store the string (possibly null)
              bool is_null = (row[import_idx] == copy_params.null_str);
              if (!cd->columnType.is_string() && row[import_idx].empty()) {
                is_null = true;
              }
              import_buffers[col_idx]->add_value(
                  cd, row[import_idx], is_null, copy_params);

              // next
              ++import_idx;
              ++col_idx;
            } else {
              // geo

              // store null string in the base column
              import_buffers[col_idx]->add_value(
                  cd, copy_params.null_str, true, copy_params);

              // WKT from string we're not storing
              std::string wkt{row[import_idx]};

              // next
              ++import_idx;
              ++col_idx;

              SQLTypes col_type = col_ti.get_type();
              CHECK(IS_GEO(col_type));

              std::vector<double> coords;
              std::vector<double> bounds;
              std::vector<int> ring_sizes;
              std::vector<int> poly_rings;
              int render_group = 0;

              if (col_type == kPOINT && wkt.size() > 0 &&
                  (wkt[0] == '.' || isdigit(wkt[0]) || wkt[0] == '-')) {
                // Invalid WKT, looks more like a scalar.
                // Try custom POINT import: from two separate scalars rather than WKT
                // string
                double lon = std::atof(wkt.c_str());
                double lat = NAN;
                std::string lat_str{row[import_idx]};
                ++import_idx;
                if (lat_str.size() > 0 &&
                    (lat_str[0] == '.' || isdigit(lat_str[0]) || lat_str[0] == '-')) {
                  lat = std::atof(lat_str.c_str());
                }
                // Swap coordinates if this table uses a reverse order: lat/lon
                if (!copy_params.lonlat) {
                  std::swap(lat, lon);
                }
                // TODO: should check if POINT column should have been declared with SRID
                // WGS 84, EPSG 4326 ? if (col_ti.get_dimension() != 4326) {
                //  throw std::runtime_error("POINT column " + cd->columnName + " is not
                //  WGS84, cannot insert lon/lat");
                // }
                if (!importGeoFromLonLat(lon, lat, coords)) {
                  throw std::runtime_error(
                      "Cannot read lon/lat to insert into POINT column " +
                      cd->columnName);
                }
              } else {
                // import it
                SQLTypeInfo import_ti;
                if (!Geo_namespace::GeoTypesFactory::getGeoColumns(
                        wkt,
                        import_ti,
                        coords,
                        bounds,
                        ring_sizes,
                        poly_rings,
                        PROMOTE_POLYGON_TO_MULTIPOLYGON)) {
                  std::string msg =
                      "Failed to extract valid geometry from row " +
                      std::to_string(first_row_index_this_buffer + row_index_plus_one) +
                      " for column " + cd->columnName;
                  throw std::runtime_error(msg);
                }

                // validate types
                if (col_type != import_ti.get_type()) {
                  if (!PROMOTE_POLYGON_TO_MULTIPOLYGON ||
                      !(import_ti.get_type() == SQLTypes::kPOLYGON &&
                        col_type == SQLTypes::kMULTIPOLYGON)) {
                    throw std::runtime_error(
                        "Imported geometry doesn't match the type of column " +
                        cd->columnName);
                  }
                }

                if (columnIdToRenderGroupAnalyzerMap.size()) {
                  if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
                    if (ring_sizes.size()) {
                      // get a suitable render group for these poly coords
                      auto rga_it = columnIdToRenderGroupAnalyzerMap.find(cd->columnId);
                      CHECK(rga_it != columnIdToRenderGroupAnalyzerMap.end());
                      render_group =
                          (*rga_it).second->insertBoundsAndReturnRenderGroup(bounds);
                    } else {
                      // empty poly
                      render_group = -1;
                    }
                  }
                }
              }

              Importer::set_geo_physical_import_buffer(importer->getCatalog(),
                                                       cd,
                                                       import_buffers,
                                                       col_idx,
                                                       coords,
                                                       bounds,
                                                       ring_sizes,
                                                       poly_rings,
                                                       render_group);
              for (int i = 0; i < cd->columnType.get_physical_cols(); ++i) {
                ++cd_it;
              }
            }
          }
          import_status.rows_completed++;
        } catch (const std::exception& e) {
          for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
            import_buffers[col_idx_to_pop]->pop_value();
          }
          import_status.rows_rejected++;
          LOG(ERROR) << "Input exception thrown: " << e.what()
                     << ". Row discarded. Data: " << row;
        }
      });
      total_str_to_val_time_us += us;
    }
    if (import_status.rows_completed > 0) {
      load_ms = measure<>::execution(
          [&]() { importer->load(import_buffers, import_status.rows_completed); });
    }
  });
  if (DEBUG_TIMING && import_status.rows_completed > 0) {
    LOG(INFO) << "Thread" << std::this_thread::get_id() << ":"
              << import_status.rows_completed << " rows inserted in "
              << (double)ms / 1000.0 << "sec, Insert Time: " << (double)load_ms / 1000.0
              << "sec, get_row: " << (double)total_get_row_time_us / 1000000.0
              << "sec, str_to_val: " << (double)total_str_to_val_time_us / 1000000.0
              << "sec" << std::endl;
  }

  import_status.thread_id = thread_id;
  // LOG(INFO) << " return " << import_status.thread_id << std::endl;

  return import_status;
}

static ImportStatus import_thread_shapefile(
    int thread_id,
    Importer* importer,
    OGRSpatialReference* poGeographicSR,
    const FeaturePtrVector& features,
    size_t firstFeature,
    size_t numFeatures,
    const FieldNameToIndexMapType& fieldNameToIndexMap,
    const ColumnNameToSourceNameMapType& columnNameToSourceNameMap,
    const ColumnIdToRenderGroupAnalyzerMapType& columnIdToRenderGroupAnalyzerMap) {
  ImportStatus import_status;
  const CopyParams& copy_params = importer->get_copy_params();
  const std::list<const ColumnDescriptor*>& col_descs = importer->get_column_descs();
  std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers =
      importer->get_import_buffers(thread_id);

  for (const auto& p : import_buffers) {
    p->clear();
  }

  auto convert_timer = timer_start();

  for (size_t iFeature = 0; iFeature < numFeatures; iFeature++) {
    if (!features[iFeature]) {
      continue;
    }

    // get this feature's geometry
    OGRGeometry* pGeometry = features[iFeature]->GetGeometryRef();
    if (!pGeometry) {
      // Silently ignoring features with no geometry defined.
      // TODO: There's an argument to be made that such a case should be considered
      // NULL instead of ignoring.
      // Also, should we log a warning here? It seems like this code hit
      // the same feature several times over the coarse of an import, so we'd want
      // to minimize the amount of logging if the same feature is referenced many
      // times or if there can be many such empty geometries.
      continue;
    }

    // transform it
    // avoid GDAL error if not transformable
    if (pGeometry->getSpatialReference()) {
      pGeometry->transformTo(poGeographicSR);
    }

    size_t col_idx = 0;
    try {
      for (auto cd_it = col_descs.begin(); cd_it != col_descs.end(); cd_it++) {
        auto cd = *cd_it;

        // is this a geo column?
        const auto& col_ti = cd->columnType;
        if (col_ti.is_geometry()) {
          // Note that this assumes there is one and only one geo column in the table.
          // Currently, the importer only supports reading a single geospatial feature
          // from an input shapefile / geojson file, but this code will need to be
          // modified if that changes
          SQLTypes col_type = col_ti.get_type();

          // store null string in the base column
          import_buffers[col_idx]->add_value(cd, copy_params.null_str, true, copy_params);
          ++col_idx;

          // the data we now need to extract for the other columns
          std::vector<double> coords;
          std::vector<double> bounds;
          std::vector<int> ring_sizes;
          std::vector<int> poly_rings;
          int render_group = 0;

          // extract it
          SQLTypeInfo import_ti;

          if (!Geo_namespace::GeoTypesFactory::getGeoColumns(
                  pGeometry,
                  import_ti,
                  coords,
                  bounds,
                  ring_sizes,
                  poly_rings,
                  PROMOTE_POLYGON_TO_MULTIPOLYGON)) {
            std::string msg = "Failed to extract valid geometry from feature " +
                              std::to_string(firstFeature + iFeature + 1) +
                              " for column " + cd->columnName;
            throw std::runtime_error(msg);
          }

          // validate types
          if (col_type != import_ti.get_type()) {
            if (!PROMOTE_POLYGON_TO_MULTIPOLYGON ||
                !(import_ti.get_type() == SQLTypes::kPOLYGON &&
                  col_type == SQLTypes::kMULTIPOLYGON)) {
              throw std::runtime_error(
                  "Imported geometry doesn't match the type of column " + cd->columnName);
            }
          }

          if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
            if (ring_sizes.size()) {
              // get a suitable render group for these poly coords
              auto rga_it = columnIdToRenderGroupAnalyzerMap.find(cd->columnId);
              CHECK(rga_it != columnIdToRenderGroupAnalyzerMap.end());
              render_group = (*rga_it).second->insertBoundsAndReturnRenderGroup(bounds);
            } else {
              // empty poly
              render_group = -1;
            }
          }

          // create coords array value and add it to the physical column
          ++cd_it;
          auto cd_coords = *cd_it;
          std::vector<TDatum> td_coord_data;
          std::vector<uint8_t> compressed_coords = compress_coords(coords, col_ti);
          for (auto cc : compressed_coords) {
            TDatum td_byte;
            td_byte.val.int_val = cc;
            td_coord_data.push_back(td_byte);
          }
          TDatum tdd_coords;
          tdd_coords.val.arr_val = td_coord_data;
          tdd_coords.is_null = false;
          import_buffers[col_idx]->add_value(cd_coords, tdd_coords, false);
          ++col_idx;

          if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
            // Create ring_sizes array value and add it to the physical column
            ++cd_it;
            auto cd_ring_sizes = *cd_it;
            std::vector<TDatum> td_ring_sizes;
            for (auto ring_size : ring_sizes) {
              TDatum td_ring_size;
              td_ring_size.val.int_val = ring_size;
              td_ring_sizes.push_back(td_ring_size);
            }
            TDatum tdd_ring_sizes;
            tdd_ring_sizes.val.arr_val = td_ring_sizes;
            tdd_ring_sizes.is_null = false;
            import_buffers[col_idx]->add_value(cd_ring_sizes, tdd_ring_sizes, false);
            ++col_idx;
          }

          if (col_type == kMULTIPOLYGON) {
            // Create poly_rings array value and add it to the physical column
            ++cd_it;
            auto cd_poly_rings = *cd_it;
            std::vector<TDatum> td_poly_rings;
            for (auto num_rings : poly_rings) {
              TDatum td_num_rings;
              td_num_rings.val.int_val = num_rings;
              td_poly_rings.push_back(td_num_rings);
            }
            TDatum tdd_poly_rings;
            tdd_poly_rings.val.arr_val = td_poly_rings;
            tdd_poly_rings.is_null = false;
            import_buffers[col_idx]->add_value(cd_poly_rings, tdd_poly_rings, false);
            ++col_idx;
          }

          if (col_type == kLINESTRING || col_type == kPOLYGON ||
              col_type == kMULTIPOLYGON) {
            // Create bounds array value and add it to the physical column
            ++cd_it;
            auto cd_bounds = *cd_it;
            std::vector<TDatum> td_bounds_data;
            for (auto b : bounds) {
              TDatum td_double;
              td_double.val.real_val = b;
              td_bounds_data.push_back(td_double);
            }
            TDatum tdd_bounds;
            tdd_bounds.val.arr_val = td_bounds_data;
            tdd_bounds.is_null = false;
            import_buffers[col_idx]->add_value(cd_bounds, tdd_bounds, false);
            ++col_idx;
          }

          if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
            // Create render_group value and add it to the physical column
            ++cd_it;
            auto cd_render_group = *cd_it;
            TDatum td_render_group;
            td_render_group.val.int_val = render_group;
            td_render_group.is_null = false;
            import_buffers[col_idx]->add_value(cd_render_group, td_render_group, false);
            ++col_idx;
          }
        } else {
          // regular column
          // pull from GDAL metadata
          const auto cit = columnNameToSourceNameMap.find(cd->columnName);
          CHECK(cit != columnNameToSourceNameMap.end());
          const std::string& fieldName = cit->second;
          const auto fit = fieldNameToIndexMap.find(fieldName);
          CHECK(fit != fieldNameToIndexMap.end());
          size_t iField = fit->second;
          CHECK(iField < fieldNameToIndexMap.size());
          std::string fieldContents = features[iFeature]->GetFieldAsString(iField);
          import_buffers[col_idx]->add_value(cd, fieldContents, false, copy_params);
          ++col_idx;
        }
      }
      import_status.rows_completed++;
    } catch (const std::exception& e) {
      for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
        import_buffers[col_idx_to_pop]->pop_value();
      }
      import_status.rows_rejected++;
      LOG(ERROR) << "Input exception thrown: " << e.what() << ". Row discarded.";
    }
  }
  float convert_ms =
      float(timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
          convert_timer)) /
      1000.0f;

  float load_ms = 0.0f;
  if (import_status.rows_completed > 0) {
    auto load_timer = timer_start();
    importer->load(import_buffers, import_status.rows_completed);
    load_ms =
        float(
            timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
                load_timer)) /
        1000.0f;
  }

  if (DEBUG_TIMING && import_status.rows_completed > 0) {
    LOG(INFO) << "DEBUG:      Process " << convert_ms << "ms";
    LOG(INFO) << "DEBUG:      Load " << load_ms << "ms";
  }

  import_status.thread_id = thread_id;

  if (DEBUG_TIMING) {
    LOG(INFO) << "DEBUG:      Total "
              << float(timer_stop<std::chrono::steady_clock::time_point,
                                  std::chrono::microseconds>(convert_timer)) /
                     1000.0f
              << "ms";
  }

  return import_status;
}

static size_t find_end(const char* buffer, size_t size, const CopyParams& copy_params) {
  int i;
  // @TODO(wei) line_delim is in quotes note supported
  for (i = size - 1; i >= 0 && buffer[i] != copy_params.line_delim; i--) {
    ;
  }

  if (i < 0) {
    int slen = size < 50 ? size : 50;
    std::string showMsgStr(buffer, buffer + slen);
    LOG(ERROR) << "No line delimiter in block. Block was of size " << size
               << " bytes, first few characters " << showMsgStr;
    return size;
  }
  return i + 1;
}

bool Loader::loadNoCheckpoint(
    const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
    size_t row_count) {
  return loadImpl(import_buffers, row_count, false);
}

bool Loader::load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                  size_t row_count) {
  return loadImpl(import_buffers, row_count, true);
}

namespace {

int64_t int_value_at(const TypedImportBuffer& import_buffer, const size_t index) {
  const auto& ti = import_buffer.getTypeInfo();
  const int8_t* values_buffer{nullptr};
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    values_buffer = import_buffer.getStringDictBuffer();
  } else {
    values_buffer = import_buffer.getAsBytes();
  }
  CHECK(values_buffer);
  switch (ti.get_logical_size()) {
    case 1: {
      return values_buffer[index];
    }
    case 2: {
      return reinterpret_cast<const int16_t*>(values_buffer)[index];
    }
    case 4: {
      return reinterpret_cast<const int32_t*>(values_buffer)[index];
    }
    case 8: {
      return reinterpret_cast<const int64_t*>(values_buffer)[index];
    }
    default:
      CHECK(false);
  }
  UNREACHABLE();
  return 0;
}

float float_value_at(const TypedImportBuffer& import_buffer, const size_t index) {
  const auto& ti = import_buffer.getTypeInfo();
  CHECK_EQ(kFLOAT, ti.get_type());
  const auto values_buffer = import_buffer.getAsBytes();
  return reinterpret_cast<const float*>(may_alias_ptr(values_buffer))[index];
}

double double_value_at(const TypedImportBuffer& import_buffer, const size_t index) {
  const auto& ti = import_buffer.getTypeInfo();
  CHECK_EQ(kDOUBLE, ti.get_type());
  const auto values_buffer = import_buffer.getAsBytes();
  return reinterpret_cast<const double*>(may_alias_ptr(values_buffer))[index];
}

}  // namespace

void Loader::distributeToShards(std::vector<OneShardBuffers>& all_shard_import_buffers,
                                std::vector<size_t>& all_shard_row_counts,
                                const OneShardBuffers& import_buffers,
                                const size_t row_count,
                                const size_t shard_count) {
  all_shard_row_counts.resize(shard_count);
  for (size_t shard_idx = 0; shard_idx < shard_count; ++shard_idx) {
    all_shard_import_buffers.emplace_back();
    for (const auto& typed_import_buffer : import_buffers) {
      all_shard_import_buffers.back().emplace_back(
          new TypedImportBuffer(typed_import_buffer->getColumnDesc(),
                                typed_import_buffer->getStringDictionary()));
    }
  }
  CHECK_GT(table_desc->shardedColumnId, 0);
  int col_idx{0};
  const ColumnDescriptor* shard_col_desc{nullptr};
  for (const auto col_desc : column_descs) {
    ++col_idx;
    if (col_idx == table_desc->shardedColumnId) {
      shard_col_desc = col_desc;
      break;
    }
  }
  CHECK(shard_col_desc);
  CHECK_LE(static_cast<size_t>(table_desc->shardedColumnId), import_buffers.size());
  auto& shard_column_input_buffer = import_buffers[table_desc->shardedColumnId - 1];
  const auto& shard_col_ti = shard_col_desc->columnType;
  CHECK(shard_col_ti.is_integer() ||
        (shard_col_ti.is_string() && shard_col_ti.get_compression() == kENCODING_DICT));
  if (shard_col_ti.is_string()) {
    const auto payloads_ptr = shard_column_input_buffer->getStringBuffer();
    CHECK(payloads_ptr);
    shard_column_input_buffer->addDictEncodedString(*payloads_ptr);
  }

  int64_t rows_per_shard = (row_count + shard_count + 1) / shard_count;
  int64_t rows_left = row_count;

  for (size_t i = 0; i < row_count; ++i, rows_left -= rows_per_shard) {
    const auto val = get_replicating() ? i : int_value_at(*shard_column_input_buffer, i);
    const size_t shard = SHARD_FOR_KEY(val, shard_count);
    auto& shard_output_buffers = all_shard_import_buffers[shard];

    // when replicate a column, populate 'rows' to all shards only once
    if (get_replicating()) {
      if (i >= shard_count) {
        break;
      }
    }

    for (size_t col_idx = 0; col_idx < import_buffers.size(); ++col_idx) {
      const auto& input_buffer = import_buffers[col_idx];
      const auto& col_ti = input_buffer->getTypeInfo();
      const auto type =
          col_ti.is_decimal() ? decimal_to_int_type(col_ti) : col_ti.get_type();

      // for a replicated (added) column, populate rows_per_shard as per-shard replicate
      // count. and, bypass non-replicated column.
      if (get_replicating()) {
        if (input_buffer->get_replicate_count() > 0) {
          shard_output_buffers[col_idx]->set_replicate_count(
              std::min<int64_t>(rows_left, rows_per_shard));
        } else {
          continue;
        }
      }

      switch (type) {
        case kBOOLEAN:
          shard_output_buffers[col_idx]->addBoolean(int_value_at(*input_buffer, i));
          break;
        case kTINYINT:
          shard_output_buffers[col_idx]->addTinyint(int_value_at(*input_buffer, i));
          break;
        case kSMALLINT:
          shard_output_buffers[col_idx]->addSmallint(int_value_at(*input_buffer, i));
          break;
        case kINT:
          shard_output_buffers[col_idx]->addInt(int_value_at(*input_buffer, i));
          break;
        case kBIGINT:
          shard_output_buffers[col_idx]->addBigint(int_value_at(*input_buffer, i));
          break;
        case kFLOAT:
          shard_output_buffers[col_idx]->addFloat(float_value_at(*input_buffer, i));
          break;
        case kDOUBLE:
          shard_output_buffers[col_idx]->addDouble(double_value_at(*input_buffer, i));
          break;
        case kTEXT:
        case kVARCHAR:
        case kCHAR: {
          CHECK_LT(i, input_buffer->getStringBuffer()->size());
          shard_output_buffers[col_idx]->addString((*input_buffer->getStringBuffer())[i]);
          break;
        }
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          shard_output_buffers[col_idx]->addTime(int_value_at(*input_buffer, i));
          break;
        case kARRAY:
          if (IS_STRING(col_ti.get_subtype())) {
            CHECK(input_buffer->getStringArrayBuffer());
            CHECK_LT(i, input_buffer->getStringArrayBuffer()->size());
            const auto& input_arr = (*(input_buffer->getStringArrayBuffer()))[i];
            shard_output_buffers[col_idx]->addStringArray(input_arr);
          } else {
            shard_output_buffers[col_idx]->addArray((*input_buffer->getArrayBuffer())[i]);
          }
          break;
        case kPOINT:
        case kLINESTRING:
        case kPOLYGON:
        case kMULTIPOLYGON: {
          CHECK_LT(i, input_buffer->getGeoStringBuffer()->size());
          shard_output_buffers[col_idx]->addGeoString(
              (*input_buffer->getGeoStringBuffer())[i]);
          break;
        }
        default:
          CHECK(false);
      }
    }
    ++all_shard_row_counts[shard];
    // when replicating a column, row count of a shard == replicate count of the column on
    // the shard
    if (get_replicating()) {
      all_shard_row_counts[shard] = std::min<int64_t>(rows_left, rows_per_shard);
    }
  }
}

bool Loader::loadImpl(
    const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
    size_t row_count,
    bool checkpoint) {
  if (table_desc->nShards) {
    std::vector<OneShardBuffers> all_shard_import_buffers;
    std::vector<size_t> all_shard_row_counts;
    const auto shard_tables = catalog.getPhysicalTablesDescriptors(table_desc);
    distributeToShards(all_shard_import_buffers,
                       all_shard_row_counts,
                       import_buffers,
                       row_count,
                       shard_tables.size());
    bool success = true;
    for (size_t shard_idx = 0; shard_idx < shard_tables.size(); ++shard_idx) {
      if (!all_shard_row_counts[shard_idx]) {
        continue;
      }
      success = success && loadToShard(all_shard_import_buffers[shard_idx],
                                       all_shard_row_counts[shard_idx],
                                       shard_tables[shard_idx],
                                       checkpoint);
    }
    return success;
  }
  return loadToShard(import_buffers, row_count, table_desc, checkpoint);
}

std::vector<DataBlockPtr> Loader::get_data_block_pointers(
    const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers) {
  std::vector<DataBlockPtr> result(import_buffers.size());
  std::vector<std::pair<const size_t, std::future<int8_t*>>>
      encoded_data_block_ptrs_futures;
  // make all async calls to string dictionary here and then continue execution
  for (size_t buf_idx = 0; buf_idx < import_buffers.size(); buf_idx++) {
    if (import_buffers[buf_idx]->getTypeInfo().is_string() &&
        import_buffers[buf_idx]->getTypeInfo().get_compression() != kENCODING_NONE) {
      auto string_payload_ptr = import_buffers[buf_idx]->getStringBuffer();
      CHECK_EQ(kENCODING_DICT, import_buffers[buf_idx]->getTypeInfo().get_compression());

      encoded_data_block_ptrs_futures.emplace_back(std::make_pair(
          buf_idx,
          std::async(std::launch::async, [buf_idx, &import_buffers, string_payload_ptr] {
            import_buffers[buf_idx]->addDictEncodedString(*string_payload_ptr);
            return import_buffers[buf_idx]->getStringDictBuffer();
          })));
    }
  }

  for (size_t buf_idx = 0; buf_idx < import_buffers.size(); buf_idx++) {
    DataBlockPtr p;
    if (import_buffers[buf_idx]->getTypeInfo().is_number() ||
        import_buffers[buf_idx]->getTypeInfo().is_time() ||
        import_buffers[buf_idx]->getTypeInfo().get_type() == kBOOLEAN) {
      p.numbersPtr = import_buffers[buf_idx]->getAsBytes();
    } else if (import_buffers[buf_idx]->getTypeInfo().is_string()) {
      auto string_payload_ptr = import_buffers[buf_idx]->getStringBuffer();
      if (import_buffers[buf_idx]->getTypeInfo().get_compression() == kENCODING_NONE) {
        p.stringsPtr = string_payload_ptr;
      } else {
        // This condition means we have column which is ENCODED string. We already made
        // Async request to gain the encoded integer values above so we should skip this
        // iteration and continue.
        continue;
      }
    } else if (import_buffers[buf_idx]->getTypeInfo().is_geometry()) {
      auto geo_payload_ptr = import_buffers[buf_idx]->getGeoStringBuffer();
      p.stringsPtr = geo_payload_ptr;
    } else {
      CHECK(import_buffers[buf_idx]->getTypeInfo().get_type() == kARRAY);
      if (IS_STRING(import_buffers[buf_idx]->getTypeInfo().get_subtype())) {
        CHECK(import_buffers[buf_idx]->getTypeInfo().get_compression() == kENCODING_DICT);
        import_buffers[buf_idx]->addDictEncodedStringArray(
            *import_buffers[buf_idx]->getStringArrayBuffer());
        p.arraysPtr = import_buffers[buf_idx]->getStringArrayDictBuffer();
      } else {
        p.arraysPtr = import_buffers[buf_idx]->getArrayBuffer();
      }
    }
    result[buf_idx] = p;
  }

  // wait for the async requests we made for string dictionary
  for (auto& encoded_ptr_future : encoded_data_block_ptrs_futures) {
    result[encoded_ptr_future.first].numbersPtr = encoded_ptr_future.second.get();
  }
  return result;
}

bool Loader::loadToShard(
    const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
    size_t row_count,
    const TableDescriptor* shard_table,
    bool checkpoint) {
  std::lock_guard<std::mutex> loader_lock(loader_mutex_);
  // patch insert_data with new column
  if (this->get_replicating()) {
    for (const auto& import_buff : import_buffers) {
      insert_data.replicate_count = import_buff->get_replicate_count();
      insert_data.columnDescriptors[import_buff->getColumnDesc()->columnId] =
          import_buff->getColumnDesc();
    }
  }

  Fragmenter_Namespace::InsertData ins_data(insert_data);
  ins_data.numRows = row_count;
  bool success = true;

  ins_data.data = get_data_block_pointers(import_buffers);

  for (const auto& import_buffer : import_buffers) {
    ins_data.bypass.push_back(0 == import_buffer->get_replicate_count());
  }
  {
    try {
      if (checkpoint) {
        shard_table->fragmenter->insertData(ins_data);
      } else {
        shard_table->fragmenter->insertDataNoCheckpoint(ins_data);
      }
    } catch (std::exception& e) {
      LOG(ERROR) << "Fragmenter Insert Exception: " << e.what();
      success = false;
    }
  }
  return success;
}

void Loader::init() {
  insert_data.databaseId = catalog.getCurrentDB().dbId;
  insert_data.tableId = table_desc->tableId;
  for (auto cd : column_descs) {
    insert_data.columnIds.push_back(cd->columnId);
    if (cd->columnType.get_compression() == kENCODING_DICT) {
      CHECK(cd->columnType.is_string() || cd->columnType.is_string_array());
      const auto dd = catalog.getMetadataForDict(cd->columnType.get_comp_param());
      CHECK(dd);
      dict_map[cd->columnId] = dd->stringDict.get();
    }
  }
  insert_data.numRows = 0;
}

void Detector::init() {
  detect_row_delimiter();
  split_raw_data();
  find_best_sqltypes_and_headers();
}

ImportStatus Detector::importDelimited(const std::string& file_path,
                                       const bool decompressed) {
  if (!p_file) {
    p_file = fopen(file_path.c_str(), "rb");
  }
  if (!p_file) {
    throw std::runtime_error("failed to open file '" + file_path +
                             "': " + strerror(errno));
  }

  // somehow clang does not support ext/stdio_filebuf.h, so
  // need to diy readline with customized copy_params.line_delim...
  char line[1 << 20];
  auto end_time = std::chrono::steady_clock::now() +
                  timeout * (boost::istarts_with(file_path, "s3://") ? 3 : 1);
  try {
    while (!feof(p_file)) {
      int c;
      size_t n = 0;
      while (EOF != (c = fgetc(p_file)) && copy_params.line_delim != c) {
        line[n++] = c;
        if (n >= sizeof(line) - 1) {
          break;
        }
      }
      if (0 == n) {
        break;
      }
      line[n] = 0;
      // remember the first line, which is possibly a header line, to
      // ignore identical header line(s) in 2nd+ files of a archive;
      // otherwise, 2nd+ header may be mistaken as an all-string row
      // and so be final column types.
      if (line1.empty()) {
        line1 = line;
      } else if (line == line1) {
        continue;
      }

      raw_data += std::string(line, n);
      raw_data += copy_params.line_delim;
      ++import_status.rows_completed;
      if (std::chrono::steady_clock::now() > end_time) {
        if (import_status.rows_completed > 10000) {
          break;
        }
      }
    }
  } catch (std::exception& e) {
  }

  // as if load truncated
  import_status.load_truncated = true;
  load_failed = true;

  fclose(p_file);
  p_file = nullptr;
  return import_status;
}

void Detector::read_file() {
  // this becomes analogous to Importer::import()
  (void)DataStreamSink::archivePlumber();
}

void Detector::detect_row_delimiter() {
  if (copy_params.delimiter == '\0') {
    copy_params.delimiter = ',';
    if (boost::filesystem::extension(file_path) == ".tsv") {
      copy_params.delimiter = '\t';
    }
  }
}

void Detector::split_raw_data() {
  const char* buf = raw_data.c_str();
  const char* buf_end = buf + raw_data.size();
  bool try_single_thread = false;
  for (const char* p = buf; p < buf_end; p++) {
    std::vector<std::string> row;
    p = get_row(p, buf_end, buf_end, copy_params, true, nullptr, row, try_single_thread);
    raw_rows.push_back(row);
    if (try_single_thread) {
      break;
    }
  }
  if (try_single_thread) {
    copy_params.threads = 1;
    raw_rows.clear();
    for (const char* p = buf; p < buf_end; p++) {
      std::vector<std::string> row;
      p = get_row(
          p, buf_end, buf_end, copy_params, true, nullptr, row, try_single_thread);
      raw_rows.push_back(row);
    }
  }
}

template <class T>
bool try_cast(const std::string& str) {
  try {
    boost::lexical_cast<T>(str);
  } catch (const boost::bad_lexical_cast& e) {
    return false;
  }
  return true;
}

inline char* try_strptimes(const char* str, const std::vector<std::string>& formats) {
  std::tm tm_struct;
  char* buf;
  for (auto format : formats) {
    buf = strptime(str, format.c_str(), &tm_struct);
    if (buf) {
      return buf;
    }
  }
  return nullptr;
}

SQLTypes Detector::detect_sqltype(const std::string& str) {
  SQLTypes type = kTEXT;
  if (try_cast<double>(str)) {
    type = kDOUBLE;
    /*if (try_cast<bool>(str)) {
      type = kBOOLEAN;
    }*/
    if (try_cast<int16_t>(str)) {
      type = kSMALLINT;
    } else if (try_cast<int32_t>(str)) {
      type = kINT;
    } else if (try_cast<int64_t>(str)) {
      type = kBIGINT;
    } else if (try_cast<float>(str)) {
      type = kFLOAT;
    }
  }

  // check for geo types
  if (type == kTEXT) {
    // convert to upper case
    std::string str_upper_case = str;
    std::transform(
        str_upper_case.begin(), str_upper_case.end(), str_upper_case.begin(), ::toupper);

    // then test for leading words
    if (str_upper_case.find("POINT") == 0) {
      type = kPOINT;
    } else if (str_upper_case.find("LINESTRING") == 0) {
      type = kLINESTRING;
    } else if (str_upper_case.find("POLYGON") == 0) {
      if (PROMOTE_POLYGON_TO_MULTIPOLYGON) {
        type = kMULTIPOLYGON;
      } else {
        type = kPOLYGON;
      }
    } else if (str_upper_case.find("MULTIPOLYGON") == 0) {
      type = kMULTIPOLYGON;
    } else if (str_upper_case.find_first_not_of("0123456789ABCDEF") ==
                   std::string::npos &&
               (str_upper_case.size() % 2) == 0) {
      // could be a WKB hex blob
      // we can't handle these yet
      // leave as TEXT for now
      // deliberate return here, as otherwise this would get matched as TIME
      // @TODO
      // implement WKB import
      return type;
    }
  }

  // check for time types
  if (type == kTEXT) {
    // @TODO
    // make these tests more robust so they don't match stuff they should not
    char* buf;
    buf = try_strptimes(str.c_str(), {"%Y-%m-%d", "%m/%d/%Y", "%d-%b-%y", "%d/%b/%Y"});
    if (buf) {
      type = kDATE;
      if (*buf == 'T' || *buf == ' ' || *buf == ':') {
        buf++;
      }
    }
    buf = try_strptimes(buf == nullptr ? str.c_str() : buf,
                        {"%T %z", "%T", "%H%M%S", "%R"});
    if (buf) {
      if (type == kDATE) {
        type = kTIMESTAMP;
      } else {
        type = kTIME;
      }
    }
  }

  return type;
}

std::vector<SQLTypes> Detector::detect_column_types(const std::vector<std::string>& row) {
  std::vector<SQLTypes> types(row.size());
  for (size_t i = 0; i < row.size(); i++) {
    types[i] = detect_sqltype(row[i]);
  }
  return types;
}

bool Detector::more_restrictive_sqltype(const SQLTypes a, const SQLTypes b) {
  static std::array<int, kSQLTYPE_LAST> typeorder;
  typeorder[kCHAR] = 0;
  typeorder[kBOOLEAN] = 2;
  typeorder[kSMALLINT] = 3;
  typeorder[kINT] = 4;
  typeorder[kBIGINT] = 5;
  typeorder[kFLOAT] = 6;
  typeorder[kDOUBLE] = 7;
  typeorder[kTIMESTAMP] = 8;
  typeorder[kTIME] = 9;
  typeorder[kDATE] = 10;
  typeorder[kPOINT] = 11;
  typeorder[kLINESTRING] = 11;
  typeorder[kPOLYGON] = 11;
  typeorder[kMULTIPOLYGON] = 11;
  typeorder[kTEXT] = 12;

  // note: b < a instead of a < b because the map is ordered most to least restrictive
  return typeorder[b] < typeorder[a];
}

void Detector::find_best_sqltypes_and_headers() {
  best_sqltypes = find_best_sqltypes(raw_rows.begin() + 1, raw_rows.end(), copy_params);
  best_encodings =
      find_best_encodings(raw_rows.begin() + 1, raw_rows.end(), best_sqltypes);
  std::vector<SQLTypes> head_types = detect_column_types(raw_rows.at(0));
  has_headers = detect_headers(head_types, best_sqltypes);
  copy_params.has_header = has_headers;
}

void Detector::find_best_sqltypes() {
  best_sqltypes = find_best_sqltypes(raw_rows.begin(), raw_rows.end(), copy_params);
}

std::vector<SQLTypes> Detector::find_best_sqltypes(
    const std::vector<std::vector<std::string>>& raw_rows,
    const CopyParams& copy_params) {
  return find_best_sqltypes(raw_rows.begin(), raw_rows.end(), copy_params);
}

std::vector<SQLTypes> Detector::find_best_sqltypes(
    const std::vector<std::vector<std::string>>::const_iterator& row_begin,
    const std::vector<std::vector<std::string>>::const_iterator& row_end,
    const CopyParams& copy_params) {
  if (raw_rows.size() < 1) {
    throw std::runtime_error("No rows found in: " +
                             boost::filesystem::basename(file_path));
  }
  auto end_time = std::chrono::steady_clock::now() + timeout;
  size_t num_cols = raw_rows.front().size();
  std::vector<SQLTypes> best_types(num_cols, kCHAR);
  std::vector<size_t> non_null_col_counts(num_cols, 0);
  for (auto row = row_begin; row != row_end; row++) {
    while (best_types.size() < row->size() || non_null_col_counts.size() < row->size()) {
      best_types.push_back(kCHAR);
      non_null_col_counts.push_back(0);
    }
    for (size_t col_idx = 0; col_idx < row->size(); col_idx++) {
      // do not count nulls
      if (row->at(col_idx) == "" || !row->at(col_idx).compare(copy_params.null_str)) {
        continue;
      }
      SQLTypes t = detect_sqltype(row->at(col_idx));
      non_null_col_counts[col_idx]++;
      if (!more_restrictive_sqltype(best_types[col_idx], t)) {
        best_types[col_idx] = t;
      }
    }
    if (std::chrono::steady_clock::now() > end_time) {
      break;
    }
  }
  for (size_t col_idx = 0; col_idx < num_cols; col_idx++) {
    // if we don't have any non-null values for this column make it text to be
    // safe b/c that is least restrictive type
    if (non_null_col_counts[col_idx] == 0) {
      best_types[col_idx] = kTEXT;
    }
  }

  return best_types;
}

std::vector<EncodingType> Detector::find_best_encodings(
    const std::vector<std::vector<std::string>>::const_iterator& row_begin,
    const std::vector<std::vector<std::string>>::const_iterator& row_end,
    const std::vector<SQLTypes>& best_types) {
  if (raw_rows.size() < 1) {
    throw std::runtime_error("No rows found in: " +
                             boost::filesystem::basename(file_path));
  }
  size_t num_cols = best_types.size();
  std::vector<EncodingType> best_encodes(num_cols, kENCODING_NONE);
  std::vector<size_t> num_rows_per_col(num_cols, 1);
  std::vector<std::unordered_set<std::string>> count_set(num_cols);
  for (auto row = row_begin; row != row_end; row++) {
    for (size_t col_idx = 0; col_idx < row->size(); col_idx++) {
      if (IS_STRING(best_types[col_idx])) {
        count_set[col_idx].insert(row->at(col_idx));
        num_rows_per_col[col_idx]++;
      }
    }
  }
  for (size_t col_idx = 0; col_idx < num_cols; col_idx++) {
    if (IS_STRING(best_types[col_idx])) {
      float uniqueRatio =
          static_cast<float>(count_set[col_idx].size()) / num_rows_per_col[col_idx];
      if (uniqueRatio < 0.75) {
        best_encodes[col_idx] = kENCODING_DICT;
      }
    }
  }
  return best_encodes;
}

void Detector::detect_headers() {
  has_headers = detect_headers(raw_rows);
}

bool Detector::detect_headers(const std::vector<std::vector<std::string>>& raw_rows) {
  if (raw_rows.size() < 3) {
    return false;
  }
  std::vector<SQLTypes> head_types = detect_column_types(raw_rows.at(0));
  std::vector<SQLTypes> tail_types =
      find_best_sqltypes(raw_rows.begin() + 1, raw_rows.end(), copy_params);
  return detect_headers(head_types, tail_types);
}

// detect_headers returns true if:
// - all elements of the first argument are kTEXT
// - there is at least one instance where tail_types is more restrictive than head_types
// (ie, not kTEXT)
bool Detector::detect_headers(const std::vector<SQLTypes>& head_types,
                              const std::vector<SQLTypes>& tail_types) {
  if (head_types.size() != tail_types.size()) {
    return false;
  }
  bool has_headers = false;
  for (size_t col_idx = 0; col_idx < tail_types.size(); col_idx++) {
    if (head_types[col_idx] != kTEXT) {
      return false;
    }
    has_headers = has_headers || tail_types[col_idx] != kTEXT;
  }
  return has_headers;
}

std::vector<std::vector<std::string>> Detector::get_sample_rows(size_t n) {
  n = std::min(n, raw_rows.size());
  size_t offset = (has_headers && raw_rows.size() > 1) ? 1 : 0;
  std::vector<std::vector<std::string>> sample_rows(raw_rows.begin() + offset,
                                                    raw_rows.begin() + n);
  return sample_rows;
}

std::vector<std::string> Detector::get_headers() {
  std::vector<std::string> headers(best_sqltypes.size());
  for (size_t i = 0; i < best_sqltypes.size(); i++) {
    headers[i] = has_headers ? raw_rows[0][i] : "column_" + std::to_string(i + 1);
  }
  return headers;
}

void Importer::load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                    size_t row_count) {
  if (!loader->loadNoCheckpoint(import_buffers, row_count)) {
    load_failed = true;
  }
}

ImportStatus DataStreamSink::archivePlumber() {
  // in generalized importing scheme, reaching here file_path may
  // contain a file path, a url or a wildcard of file paths.
  // see CopyTableStmt::execute.
  auto file_paths = mapd_glob(file_path);
  if (file_paths.size() == 0) {
    file_paths.push_back(file_path);
  }

  // sum up sizes of all local files -- only for local files. if
  // file_path is a s3 url, sizes will be obtained via S3Archive.
  for (const auto& file_path : file_paths) {
    boost::filesystem::path boost_file_path{file_path};
    boost::system::error_code ec;
    const auto filesize = boost::filesystem::file_size(boost_file_path, ec);
    if (!ec) {
      total_file_size += filesize;
    }
  }

  // s3 parquet goes different route because the files do not use libarchive
  // but parquet api, and they need to landed like .7z files.
  //
  // note: parquet must be explicitly specified by a WITH parameter "parquet='true'".
  //       without the parameter, it means plain or compressed csv files.
  // note: .ORC and AVRO files should follow a similar path to Parquet?
  if (copy_params.is_parquet) {
    import_parquet(file_paths);
  } else {
    import_compressed(file_paths);
  }
  return import_status;
}

void DataStreamSink::import_local_parquet(const std::string& file_path) {
  // TODO: for now, this is skeleton only
  // note: for 'early-stop' purpose like that of Detector, this function
  // needs extra parameters, eg. timeout or maximum rows to scan, ...
}
void DataStreamSink::import_parquet(std::vector<std::string>& file_paths) {
  std::exception_ptr teptr;
  // file_paths may contain one local file path, a list of local file paths
  // or a s3/hdfs/... url that may translate to 1 or 1+ remote object keys.
  for (auto const& file_path : file_paths) {
    std::map<int, std::string> url_parts;
    Archive::parse_url(file_path, url_parts);

    // for a s3 url we need to know the obj keys that it comprises
    std::vector<std::string> objkeys;
    std::unique_ptr<S3ParquetArchive> us3arch;
    if ("s3" == url_parts[2]) {
#ifdef HAVE_AWS_S3
      us3arch.reset(new S3ParquetArchive(file_path,
                                         copy_params.s3_access_key,
                                         copy_params.s3_secret_key,
                                         copy_params.s3_region,
                                         copy_params.plain_text));
      us3arch->init_for_read();
      objkeys = us3arch->get_objkeys();
#else
      throw std::runtime_error("AWS S3 support not available");
#endif  // HAVE_AWS_S3
    } else {
      objkeys.emplace_back(file_path);
    }

    // for each obj key of a s3 url we need to land it before
    // importing it like doing with a 'local file'.
    for (auto const& objkey : objkeys) {
      try {
        auto file_path =
            us3arch
                ? us3arch->land(objkey, teptr, nullptr != dynamic_cast<Detector*>(this))
                : objkey;
        if (boost::filesystem::exists(file_path)) {
          import_local_parquet(file_path);
        }
        if (us3arch) {
          us3arch->vacuum(objkey);
        }
      } catch (...) {
        if (us3arch) {
          us3arch->vacuum(objkey);
        }
        throw;
      }
    }
  }
  // rethrow any exception happened herebefore
  if (teptr) {
    std::rethrow_exception(teptr);
  }
}

void DataStreamSink::import_compressed(std::vector<std::string>& file_paths) {
  // a new requirement is to have one single input stream into
  // Importer::importDelimited, so need to move pipe related
  // stuff to the outmost block.
  int fd[2];
  if (pipe(fd) < 0) {
    throw std::runtime_error(std::string("failed to create a pipe: ") + strerror(errno));
  }
  signal(SIGPIPE, SIG_IGN);

  std::exception_ptr teptr;
  // create a thread to read uncompressed byte stream out of pipe and
  // feed into importDelimited()
  ImportStatus ret;
  auto th_pipe_reader = std::thread([&]() {
    try {
      // importDelimited will read from FILE* p_file
      if (0 == (p_file = fdopen(fd[0], "r"))) {
        throw std::runtime_error(std::string("failed to open a pipe: ") +
                                 strerror(errno));
      }

      // in future, depending on data types of this uncompressed stream
      // it can be feed into other function such like importParquet, etc
      ret = importDelimited(file_path, true);
    } catch (...) {
      if (!teptr) {  // no replace
        teptr = std::current_exception();
      }
    }

    if (p_file) {
      fclose(p_file);
    }
    p_file = 0;
  });

  // create a thread to iterate all files (in all archives) and
  // forward the uncompressed byte stream to fd[1] which is
  // then feed into importDelimited, importParquet, and etc.
  auto th_pipe_writer = std::thread([&]() {
    std::unique_ptr<S3Archive> us3arch;
    bool stop = false;
    for (size_t fi = 0; !stop && fi < file_paths.size(); fi++) {
      try {
        auto file_path = file_paths[fi];
        std::unique_ptr<Archive> uarch;
        std::map<int, std::string> url_parts;
        Archive::parse_url(file_path, url_parts);
        const std::string S3_objkey_url_scheme = "s3ok";
        if ("file" == url_parts[2] || "" == url_parts[2]) {
          uarch.reset(new PosixFileArchive(file_path, copy_params.plain_text));
        } else if ("s3" == url_parts[2]) {
#ifdef HAVE_AWS_S3
          // new a S3Archive with a shared s3client.
          // should be safe b/c no wildcard with s3 url
          us3arch.reset(new S3Archive(file_path,
                                      copy_params.s3_access_key,
                                      copy_params.s3_secret_key,
                                      copy_params.s3_region,
                                      copy_params.plain_text));
          us3arch->init_for_read();
          total_file_size += us3arch->get_total_file_size();
          // not land all files here but one by one in following iterations
          for (const auto& objkey : us3arch->get_objkeys()) {
            file_paths.emplace_back(std::string(S3_objkey_url_scheme) + "://" + objkey);
          }
          continue;
#else
          throw std::runtime_error("AWS S3 support not available");
#endif  // HAVE_AWS_S3
        } else if (S3_objkey_url_scheme == url_parts[2]) {
#ifdef HAVE_AWS_S3
          auto objkey = file_path.substr(3 + S3_objkey_url_scheme.size());
          auto file_path =
              us3arch->land(objkey, teptr, nullptr != dynamic_cast<Detector*>(this));
          if (0 == file_path.size()) {
            throw std::runtime_error(std::string("failed to land s3 object: ") + objkey);
          }
          uarch.reset(new PosixFileArchive(file_path, copy_params.plain_text));
          // file not removed until file closed
          us3arch->vacuum(objkey);
#else
          throw std::runtime_error("AWS S3 support not available");
#endif  // HAVE_AWS_S3
        }
#if 0  // TODO(ppan): implement and enable any other archive class
        else
        if ("hdfs" == url_parts[2])
          uarch.reset(new HdfsArchive(file_path));
#endif
        else {
          throw std::runtime_error(std::string("unsupported archive url: ") + file_path);
        }

        // init the archive for read
        auto& arch = *uarch;

        // coming here, the archive of url should be ready to be read, unarchived
        // and uncompressed by libarchive into a byte stream (in csv) for the pipe
        const void* buf;
        size_t size;
        bool just_saw_archive_header;
        bool is_detecting = nullptr != dynamic_cast<Detector*>(this);
        bool first_text_header_skipped = false;
        // start reading uncompressed bytes of this archive from libarchive
        // note! this archive may contain more than one files!
        file_offsets.push_back(0);
        while (!stop && !!(just_saw_archive_header = arch.read_next_header())) {
          bool insert_line_delim_after_this_file = false;
          while (!stop) {
            int64_t offset{-1};
            auto ok = arch.read_data_block(&buf, &size, &offset);
            // can't use (uncompressed) size, so track (max) file offset.
            // also we want to capture offset even on e.o.f.
            if (offset > 0) {
              std::unique_lock<std::mutex> lock(file_offsets_mutex);
              file_offsets.back() = offset;
            }
            if (!ok) {
              break;
            }
            // one subtle point here is now we concatenate all files
            // to a single FILE stream with which we call importDelimited
            // only once. this would make it misunderstand that only one
            // header line is with this 'single' stream, while actually
            // we may have one header line for each of the files.
            // so we need to skip header lines here instead in importDelimited.
            const char* buf2 = (const char*)buf;
            int size2 = size;
            if (copy_params.has_header && just_saw_archive_header &&
                (first_text_header_skipped || !is_detecting)) {
              while (size2-- > 0) {
                if (*buf2++ == copy_params.line_delim) {
                  break;
                }
              }
              if (size2 <= 0) {
                LOG(WARNING) << "No line delimiter in block." << std::endl;
              }
              just_saw_archive_header = false;
              first_text_header_skipped = true;
            }
            // In very rare occasions the write pipe somehow operates in a mode similar to
            // non-blocking while pipe(fds) should behave like pipe2(fds, 0) which means
            // blocking mode. On such a unreliable blocking mode, a possible fix is to
            // loop reading till no bytes left, otherwise the annoying `failed to write
            // pipe: Success`...
            if (size2 > 0) {
              int nremaining = size2;
              while (nremaining > 0) {
                // try to write the entire remainder of the buffer to the pipe
                int nwritten = write(fd[1], buf2, nremaining);
                // how did we do?
                if (nwritten < 0) {
                  // something bad happened
                  if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                    // ignore these, assume nothing written, try again
                    nwritten = 0;
                  } else {
                    // a real error
                    throw std::runtime_error(
                        std::string("failed or interrupted write to pipe: ") +
                        strerror(errno));
                  }
                } else if (nwritten == nremaining) {
                  // we wrote everything; we're done
                  break;
                }
                // only wrote some (or nothing), try again
                nremaining -= nwritten;
                buf2 += nwritten;
                // no exception when too many rejected
                // @simon.eves how would this get set? from the other thread? mutex
                // needed?
                if (import_status.load_truncated) {
                  stop = true;
                  break;
                }
              }
              // check that this file (buf for size) ended with a line delim
              if (size > 0) {
                const char* plast = static_cast<const char*>(buf) + (size - 1);
                insert_line_delim_after_this_file = (*plast != copy_params.line_delim);
              }
            }
          }
          // if that file didn't end with a line delim, we insert one here to terminate
          // that file's stream use a loop for the same reason as above
          if (insert_line_delim_after_this_file) {
            while (true) {
              // write the delim char to the pipe
              int nwritten = write(fd[1], &copy_params.line_delim, 1);
              // how did we do?
              if (nwritten < 0) {
                // something bad happened
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                  // ignore these, assume nothing written, try again
                  nwritten = 0;
                } else {
                  // a real error
                  throw std::runtime_error(
                      std::string("failed or interrupted write to pipe: ") +
                      strerror(errno));
                }
              } else if (nwritten == 1) {
                // we wrote it; we're done
                break;
              }
            }
          }
        }
      } catch (...) {
        // when import is aborted because too many data errors or because end of a
        // detection, any exception thrown by s3 sdk or libarchive is okay and should be
        // suppressed.
        if (import_status.load_truncated) {
          break;
        }
        if (import_status.rows_completed > 0) {
          if (nullptr != dynamic_cast<Detector*>(this)) {
            break;
          }
        }
        if (!teptr) {  // no replace
          teptr = std::current_exception();
        }
        break;
      }
    }
    // close writer end
    close(fd[1]);
  });

  th_pipe_reader.join();
  th_pipe_writer.join();

  // rethrow any exception happened herebefore
  if (teptr) {
    std::rethrow_exception(teptr);
  }
}

ImportStatus Importer::import() {
  return DataStreamSink::archivePlumber();
}

#define IMPORT_FILE_BUFFER_SIZE \
  (1 << 23)  // not too big (need much memory) but not too small (many thread forks)
#define MIN_FILE_BUFFER_SIZE 50000  // 50K min buffer

ImportStatus Importer::importDelimited(const std::string& file_path,
                                       const bool decompressed) {
  bool load_truncated = false;
  set_import_status(import_id, import_status);

  if (!p_file) {
    p_file = fopen(file_path.c_str(), "rb");
  }
  if (!p_file) {
    throw std::runtime_error("failed to open file '" + file_path +
                             "': " + strerror(errno));
  }

  if (!decompressed) {
    (void)fseek(p_file, 0, SEEK_END);
    file_size = ftell(p_file);
  }

  if (copy_params.threads == 0) {
    max_threads = static_cast<size_t>(sysconf(_SC_NPROCESSORS_CONF));
  } else {
    max_threads = static_cast<size_t>(copy_params.threads);
  }

  // deal with small files
  size_t alloc_size = IMPORT_FILE_BUFFER_SIZE;
  if (!decompressed && file_size < alloc_size) {
    alloc_size = file_size;
  }

  for (size_t i = 0; i < max_threads; i++) {
    import_buffers_vec.emplace_back();
    for (const auto cd : loader->get_column_descs()) {
      import_buffers_vec[i].push_back(std::unique_ptr<TypedImportBuffer>(
          new TypedImportBuffer(cd, loader->get_string_dict(cd))));
    }
  }

  std::shared_ptr<char> sbuffer(new char[alloc_size]);
  size_t current_pos = 0;
  size_t end_pos;
  bool eof_reached = false;
  size_t begin_pos = 0;

  (void)fseek(p_file, current_pos, SEEK_SET);
  size_t size = fread((void*)sbuffer.get(), 1, alloc_size, p_file);

  // make render group analyzers for each poly column
  ColumnIdToRenderGroupAnalyzerMapType columnIdToRenderGroupAnalyzerMap;
  auto columnDescriptors = loader->getCatalog().getAllColumnMetadataForTable(
      loader->get_table_desc()->tableId, false, false, false);
  for (auto cd : columnDescriptors) {
    SQLTypes ct = cd->columnType.get_type();
    if (ct == kPOLYGON || ct == kMULTIPOLYGON) {
      auto rga = std::make_shared<RenderGroupAnalyzer>();
      rga->seedFromExistingTableContents(loader, cd->columnName);
      columnIdToRenderGroupAnalyzerMap[cd->columnId] = rga;
    }
  }

  ChunkKey chunkKey = {loader->getCatalog().getCurrentDB().dbId,
                       loader->get_table_desc()->tableId};
  {
    std::list<std::future<ImportStatus>> threads;

    // use a stack to track thread_ids which must not overlap among threads
    // because thread_id is used to index import_buffers_vec[]
    std::stack<size_t> stack_thread_ids;
    for (size_t i = 0; i < max_threads; i++) {
      stack_thread_ids.push(i);
    }

    size_t first_row_index_this_buffer = 0;

    auto start_epoch = loader->getTableEpoch();
    while (size > 0) {
      if (eof_reached) {
        end_pos = size;
      } else {
        end_pos = find_end(sbuffer.get(), size, copy_params);
      }
      // unput residual
      int nresidual = size - end_pos;
      std::unique_ptr<char> unbuf(nresidual > 0 ? new char[nresidual] : nullptr);
      if (unbuf) {
        memcpy(unbuf.get(), sbuffer.get() + end_pos, nresidual);
      }

      // added for true row index on error
      unsigned int num_rows_this_buffer = 0;
      {
        // we could multi-thread this, but not worth it
        // additional cost here is ~1.4ms per chunk and
        // probably free because this thread will spend
        // most of its time waiting for the child threads
        char* p = sbuffer.get() + begin_pos;
        char* pend = sbuffer.get() + end_pos;
        char d = copy_params.line_delim;
        while (p < pend) {
          if (*p++ == d) {
            num_rows_this_buffer++;
          }
        }
      }

      // get a thread_id not in use
      auto thread_id = stack_thread_ids.top();
      stack_thread_ids.pop();
      // LOG(INFO) << " stack_thread_ids.pop " << thread_id << std::endl;

      threads.push_back(std::async(std::launch::async,
                                   import_thread_delimited,
                                   thread_id,
                                   this,
                                   sbuffer,
                                   begin_pos,
                                   end_pos,
                                   end_pos,
                                   columnIdToRenderGroupAnalyzerMap,
                                   first_row_index_this_buffer));

      first_row_index_this_buffer += num_rows_this_buffer;

      current_pos += end_pos;
      sbuffer.reset(new char[alloc_size]);
      memcpy(sbuffer.get(), unbuf.get(), nresidual);
      size = nresidual + fread(sbuffer.get() + nresidual,
                               1,
                               IMPORT_FILE_BUFFER_SIZE - nresidual,
                               p_file);
      if (size < IMPORT_FILE_BUFFER_SIZE && feof(p_file)) {
        eof_reached = true;
      }

      begin_pos = 0;

      while (threads.size() > 0) {
        int nready = 0;
        for (std::list<std::future<ImportStatus>>::iterator it = threads.begin();
             it != threads.end();) {
          auto& p = *it;
          std::chrono::milliseconds span(
              0);  //(std::distance(it, threads.end()) == 1? 1: 0);
          if (p.wait_for(span) == std::future_status::ready) {
            auto ret_import_status = p.get();
            import_status += ret_import_status;
            // sum up current total file offsets
            size_t total_file_offset{0};
            if (decompressed) {
              std::unique_lock<std::mutex> lock(file_offsets_mutex);
              for (const auto file_offset : file_offsets) {
                total_file_offset += file_offset;
              }
            }
            // estimate number of rows per current total file offset
            if (decompressed ? total_file_offset : current_pos) {
              import_status.rows_estimated =
                  (decompressed ? (float)total_file_size / total_file_offset
                                : (float)file_size / current_pos) *
                  import_status.rows_completed;
            }
            VLOG(3) << "rows_completed " << import_status.rows_completed
                    << ", rows_estimated " << import_status.rows_estimated
                    << ", total_file_size " << total_file_size << ", total_file_offset "
                    << total_file_offset;
            set_import_status(import_id, import_status);
            // recall thread_id for reuse
            stack_thread_ids.push(ret_import_status.thread_id);
            threads.erase(it++);
            ++nready;
          } else {
            ++it;
          }
        }

        if (nready == 0) {
          std::this_thread::yield();
        }

        // on eof, wait all threads to finish
        if (0 == size) {
          continue;
        }

        // keep reading if any free thread slot
        // this is one of the major difference from old threading model !!
        if (threads.size() < max_threads) {
          break;
        }
      }

      if (import_status.rows_rejected > copy_params.max_reject) {
        load_truncated = true;
        load_failed = true;
        LOG(ERROR) << "Maximum rows rejected exceeded. Halting load";
        break;
      }
      if (load_failed) {
        load_truncated = true;
        LOG(ERROR) << "A call to the Loader::load failed, Please review the logs for "
                      "more details";
        break;
      }
    }

    // join dangling threads in case of LOG(ERROR) above
    for (auto& p : threads) {
      p.wait();
    }

    if (load_failed) {
      // rollback to starting epoch - undo all the added records
      loader->setTableEpoch(start_epoch);
    } else {
      loader->checkpoint();
    }
  }

  if (loader->get_table_desc()->persistenceLevel ==
      Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident tables
    auto ms = measure<>::execution([&]() {
      if (!load_failed) {
        for (auto& p : import_buffers_vec[0]) {
          if (!p->stringDictCheckpoint()) {
            LOG(ERROR) << "Checkpointing Dictionary for Column "
                       << p->getColumnDesc()->columnName << " failed.";
            load_failed = true;
            break;
          }
        }
      }
    });
    if (DEBUG_TIMING) {
      LOG(INFO) << "Dictionary Checkpointing took " << (double)ms / 1000.0 << " Seconds."
                << std::endl;
    }
  }
  // must set import_status.load_truncated before closing this end of pipe
  // otherwise, the thread on the other end would throw an unwanted 'write()'
  // exception
  import_status.load_truncated = load_truncated;

  fclose(p_file);
  p_file = nullptr;
  return import_status;
}

void Loader::checkpoint() {
  if (get_table_desc()->persistenceLevel ==
      Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident tables
    getCatalog().checkpoint(get_table_desc()->tableId);
  }
}

int32_t Loader::getTableEpoch() {
  return getCatalog().getTableEpoch(getCatalog().getCurrentDB().dbId,
                                    get_table_desc()->tableId);
}

void Loader::setTableEpoch(int32_t start_epoch) {
  getCatalog().setTableEpoch(
      getCatalog().getCurrentDB().dbId, get_table_desc()->tableId, start_epoch);
}

void GDALErrorHandler(CPLErr eErrClass, int err_no, const char* msg) {
  CHECK(eErrClass >= CE_None && eErrClass <= CE_Fatal);
  static const char* errClassStrings[5] = {
      "Info",
      "Debug",
      "Warning",
      "Failure",
      "Fatal",
  };
  std::string log_msg = std::string("GDAL ") + errClassStrings[eErrClass] +
                        std::string(": ") + msg + std::string(" (") +
                        std::to_string(err_no) + std::string(")");
  if (eErrClass >= CE_Failure) {
    throw std::runtime_error(log_msg);
  } else {
    LOG(INFO) << log_msg;
  }
}

/* static */
std::mutex Importer::init_gdal_mutex;

/* static */
void Importer::initGDAL() {
  // this should not be called from multiple threads, but...
  std::lock_guard<std::mutex> guard(Importer::init_gdal_mutex);
  // init under mutex
  static bool gdal_initialized = false;
  if (!gdal_initialized) {
    // FIXME(andrewseidl): investigate if CPLPushFinderLocation can be public
    setenv("GDAL_DATA",
           std::string(mapd_root_abs_path() + "/ThirdParty/gdal-data").c_str(),
           true);

    // configure SSL certificate path (per S3Archive::init_for_read)
    // in a production build, GDAL and Curl will have been built on
    // CentOS, so the baked-in system path will be wrong for Ubuntu
    // and other Linux distros. Unless the user is deliberately
    // overriding it by setting SSL_CERT_FILE explicitly in the server
    // environment, we set it to whichever CA bundle directory exists
    // on the machine we're running on
    std::list<std::string> v_known_ca_paths({
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/usr/share/ssl/certs/ca-bundle.crt",
        "/usr/local/share/certs/ca-root.crt",
        "/etc/ssl/cert.pem",
        "/etc/ssl/ca-bundle.pem",
    });
    for (const auto& known_ca_path : v_known_ca_paths) {
      if (boost::filesystem::exists(known_ca_path)) {
        LOG(INFO) << "GDAL SSL Certificate path: " << known_ca_path;
        setenv("SSL_CERT_FILE", known_ca_path.c_str(), false);  // no overwrite
        break;
      }
    }

    GDALAllRegister();
    OGRRegisterAll();
    CPLSetErrorHandler(*GDALErrorHandler);
    LOG(INFO) << "GDAL Initialized: " << GDALVersionInfo("--version");
    gdal_initialized = true;
  }
}

/* static */
void Importer::setGDALAuthorizationTokens(const CopyParams& copy_params) {
  // for now we only support S3
  // @TODO generalize CopyParams to have a dictionary of GDAL tokens
  // only set if non-empty to allow GDAL defaults to persist
  // explicitly clear if empty to revert to default and not reuse a previous session's
  // keys
  if (copy_params.s3_region.size()) {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Setting AWS_REGION to '" << copy_params.s3_region << "'";
#endif
    CPLSetConfigOption("AWS_REGION", copy_params.s3_region.c_str());
  } else {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Clearing AWS_REGION";
#endif
    CPLSetConfigOption("AWS_REGION", nullptr);
  }
  if (copy_params.s3_access_key.size()) {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Setting AWS_ACCESS_KEY_ID to '" << copy_params.s3_access_key
              << "'";
#endif
    CPLSetConfigOption("AWS_ACCESS_KEY_ID", copy_params.s3_access_key.c_str());
  } else {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Clearing AWS_ACCESS_KEY_ID";
#endif
    CPLSetConfigOption("AWS_ACCESS_KEY_ID", nullptr);
  }
  if (copy_params.s3_secret_key.size()) {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Setting AWS_SECRET_ACCESS_KEY to '" << copy_params.s3_secret_key
              << "'";
#endif
    CPLSetConfigOption("AWS_SECRET_ACCESS_KEY", copy_params.s3_secret_key.c_str());
  } else {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Clearing AWS_SECRET_ACCESS_KEY";
#endif
    CPLSetConfigOption("AWS_SECRET_ACCESS_KEY", nullptr);
  }

#if (GDAL_VERSION_MAJOR > 2) || (GDAL_VERSION_MAJOR == 2 && GDAL_VERSION_MINOR >= 3)
  // if we haven't set keys, we need to disable signed access
  if (copy_params.s3_access_key.size() || copy_params.s3_secret_key.size()) {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Clearing AWS_NO_SIGN_REQUEST";
#endif
    CPLSetConfigOption("AWS_NO_SIGN_REQUEST", nullptr);
  } else {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Setting AWS_NO_SIGN_REQUEST to 'YES'";
#endif
    CPLSetConfigOption("AWS_NO_SIGN_REQUEST", "YES");
  }
#endif
}

/* static */
OGRDataSource* Importer::openGDALDataset(const std::string& file_name,
                                         const CopyParams& copy_params) {
  // lazy init GDAL
  initGDAL();

  // set authorization tokens
  setGDALAuthorizationTokens(copy_params);

  // open the file
  OGRDataSource* poDS;
#if GDAL_VERSION_MAJOR == 1
  poDS = (OGRDataSource*)OGRSFDriverRegistrar::Open(file_name.c_str(), false);
#else
  poDS = (OGRDataSource*)GDALOpenEx(
      file_name.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
  if (poDS == nullptr) {
    poDS = (OGRDataSource*)GDALOpenEx(
        file_name.c_str(), GDAL_OF_READONLY | GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (poDS) {
      LOG(INFO) << "openGDALDataset had to open as read-only";
    }
  }
#endif
  if (poDS == nullptr) {
    LOG(ERROR) << "openGDALDataset Error: " << CPLGetLastErrorMsg();
  }
  // NOTE(adb): If extending this function, refactor to ensure any errors will not result
  // in a memory leak if GDAL successfully opened the input dataset.
  return poDS;
}

/* static */
void Importer::readMetadataSampleGDAL(
    const std::string& file_name,
    const std::string& geo_column_name,
    std::map<std::string, std::vector<std::string>>& metadata,
    int rowLimit,
    const CopyParams& copy_params) {
  OGRDataSourceUqPtr poDS(openGDALDataset(file_name, copy_params));
  if (poDS == nullptr) {
    throw std::runtime_error("openGDALDataset Error: Unable to open geo file " +
                             file_name);
  }
  OGRLayer* poLayer = poDS->GetLayer(0);
  if (poLayer == nullptr) {
    throw std::runtime_error("No layers found in " + file_name);
  }
  OGRFeatureDefn* poFDefn = poLayer->GetLayerDefn();

  // typeof GetFeatureCount() is different between GDAL 1.x (int32_t) and 2.x (int64_t)
  auto nFeats = poLayer->GetFeatureCount();
  size_t numFeatures =
      std::max(static_cast<decltype(nFeats)>(0),
               std::min(static_cast<decltype(nFeats)>(rowLimit), nFeats));
  for (auto iField = 0; iField < poFDefn->GetFieldCount(); iField++) {
    OGRFieldDefn* poFieldDefn = poFDefn->GetFieldDefn(iField);
    // FIXME(andrewseidl): change this to the faster one used by readVerticesFromGDAL
    metadata.emplace(poFieldDefn->GetNameRef(), std::vector<std::string>(numFeatures));
  }
  metadata.emplace(geo_column_name, std::vector<std::string>(numFeatures));
  poLayer->ResetReading();
  size_t iFeature = 0;
  while (iFeature < numFeatures) {
    OGRFeatureUqPtr poFeature(poLayer->GetNextFeature());
    if (!poFeature) {
      break;
    }

    OGRGeometry* poGeometry;
    poGeometry = poFeature->GetGeometryRef();
    if (poGeometry != nullptr) {
      // validate geom type (again?)
      switch (wkbFlatten(poGeometry->getGeometryType())) {
        case wkbPoint:
        case wkbLineString:
        case wkbPolygon:
        case wkbMultiPolygon:
          break;
        default:
          throw std::runtime_error("Unsupported geometry type: " +
                                   std::string(poGeometry->getGeometryName()));
      }

      // populate metadata for regular fields
      for (auto i : metadata) {
        auto iField = poFeature->GetFieldIndex(i.first.c_str());
        if (iField >= 0) {  // geom is -1
          metadata[i.first].at(iFeature) =
              std::string(poFeature->GetFieldAsString(iField));
        }
      }

      // populate metadata for geo column with WKT string
      char* wkts = nullptr;
      poGeometry->exportToWkt(&wkts);
      CHECK(wkts);
      metadata[geo_column_name].at(iFeature) = wkts;
      CPLFree(wkts);
    }
    iFeature++;
  }
}

std::pair<SQLTypes, bool> ogr_to_type(const OGRFieldType& ogr_type) {
  switch (ogr_type) {
    case OFTInteger:
      return std::make_pair(kINT, false);
    case OFTIntegerList:
      return std::make_pair(kINT, true);
#if GDAL_VERSION_MAJOR > 1
    case OFTInteger64:
      return std::make_pair(kBIGINT, false);
    case OFTInteger64List:
      return std::make_pair(kBIGINT, true);
#endif
    case OFTReal:
      return std::make_pair(kDOUBLE, false);
    case OFTRealList:
      return std::make_pair(kDOUBLE, true);
    case OFTString:
      return std::make_pair(kTEXT, false);
    case OFTStringList:
      return std::make_pair(kTEXT, true);
    case OFTDate:
      return std::make_pair(kDATE, false);
    case OFTTime:
      return std::make_pair(kTIME, false);
    case OFTDateTime:
      return std::make_pair(kTIMESTAMP, false);
    case OFTBinary:
    default:
      break;
  }
  throw std::runtime_error("Unknown OGR field type: " + std::to_string(ogr_type));
}

SQLTypes ogr_to_type(const OGRwkbGeometryType& ogr_type) {
  switch (ogr_type) {
    case wkbPoint:
      return kPOINT;
    case wkbLineString:
      return kLINESTRING;
    case wkbPolygon:
      return kPOLYGON;
    case wkbMultiPolygon:
      return kMULTIPOLYGON;
    default:
      break;
  }
  throw std::runtime_error("Unknown OGR geom type: " + std::to_string(ogr_type));
}

/* static */
const std::list<ColumnDescriptor> Importer::gdalToColumnDescriptors(
    const std::string& file_name,
    const std::string& geo_column_name,
    const CopyParams& copy_params) {
  std::list<ColumnDescriptor> cds;

  OGRDataSourceUqPtr poDS(openGDALDataset(file_name, copy_params));
  if (poDS == nullptr) {
    throw std::runtime_error("openGDALDataset Error: Unable to open geo file " +
                             file_name);
  }
  OGRLayer* poLayer = poDS->GetLayer(0);
  if (poLayer == nullptr) {
    throw std::runtime_error("No layers found in " + file_name);
  }
  poLayer->ResetReading();
  // TODO(andrewseidl): support multiple features
  OGRFeatureUqPtr poFeature(poLayer->GetNextFeature());
  if (poFeature == nullptr) {
    throw std::runtime_error("No features found in " + file_name);
  }
  // get fields as regular columns
  OGRFeatureDefn* poFDefn = poLayer->GetLayerDefn();
  int iField;
  for (iField = 0; iField < poFDefn->GetFieldCount(); iField++) {
    OGRFieldDefn* poFieldDefn = poFDefn->GetFieldDefn(iField);
    auto typePair = ogr_to_type(poFieldDefn->GetType());
    ColumnDescriptor cd;
    cd.columnName = poFieldDefn->GetNameRef();
    cd.sourceName = poFieldDefn->GetNameRef();
    SQLTypeInfo ti;
    if (typePair.second) {
      ti.set_type(kARRAY);
      ti.set_subtype(typePair.first);
    } else {
      ti.set_type(typePair.first);
    }
    if (typePair.first == kTEXT) {
      ti.set_compression(kENCODING_DICT);
      ti.set_comp_param(32);
    }
    ti.set_fixed_size();
    cd.columnType = ti;
    cds.push_back(cd);
  }
  // get geo column, if any
  OGRGeometry* poGeometry = poFeature->GetGeometryRef();
  if (poGeometry) {
    ColumnDescriptor cd;
    cd.columnName = geo_column_name;
    cd.sourceName = geo_column_name;
    SQLTypes geoType = ogr_to_type(wkbFlatten(poGeometry->getGeometryType()));
    if (PROMOTE_POLYGON_TO_MULTIPOLYGON) {
      geoType = (geoType == kPOLYGON) ? kMULTIPOLYGON : geoType;
    }
    SQLTypeInfo ti;
    ti.set_type(geoType);
    ti.set_subtype(copy_params.geo_coords_type);
    ti.set_input_srid(copy_params.geo_coords_srid);
    ti.set_output_srid(copy_params.geo_coords_srid);
    ti.set_compression(copy_params.geo_coords_encoding);
    ti.set_comp_param(copy_params.geo_coords_comp_param);
    cd.columnType = ti;
    cds.push_back(cd);
  }
  return cds;
}

bool Importer::gdalStatInternal(const std::string& path,
                                const CopyParams& copy_params,
                                bool also_dir) {
  // lazy init GDAL
  initGDAL();

  // set authorization tokens
  setGDALAuthorizationTokens(copy_params);

#if (GDAL_VERSION_MAJOR > 2) || (GDAL_VERSION_MAJOR == 2 && GDAL_VERSION_MINOR >= 3)
  // clear GDAL stat cache
  // without this, file existence will be cached, even if authentication changes
  // supposed to be available from GDAL 2.2.1 but our CentOS build disagrees
  VSICurlClearCache();
#endif

  // stat path
  VSIStatBufL sb;
  int result = VSIStatExL(path.c_str(), &sb, VSI_STAT_EXISTS_FLAG);
  if (result < 0) {
    return false;
  }

  // exists?
  if (also_dir && (VSI_ISREG(sb.st_mode) || VSI_ISDIR(sb.st_mode))) {
    return true;
  } else if (VSI_ISREG(sb.st_mode)) {
    return true;
  }
  return false;
}

/* static */
bool Importer::gdalFileExists(const std::string& path, const CopyParams& copy_params) {
  return gdalStatInternal(path, copy_params, false);
}

/* static */
bool Importer::gdalFileOrDirectoryExists(const std::string& path,
                                         const CopyParams& copy_params) {
  return gdalStatInternal(path, copy_params, true);
}

void gdalGatherFilesInArchiveRecursive(const std::string& archive_path,
                                       std::vector<std::string>& files) {
  // prepare to gather subdirectories
  std::vector<std::string> subdirectories;

  // get entries
  char** entries = VSIReadDir(archive_path.c_str());
  if (!entries) {
    LOG(WARNING) << "Failed to get file listing at archive: " << archive_path;
    return;
  }

  // force scope
  {
    // request clean-up
    ScopeGuard entries_guard = [&] { CSLDestroy(entries); };

    // check all the entries
    int index = 0;
    while (true) {
      // get next entry, or drop out if there isn't one
      char* entry_c = entries[index++];
      if (!entry_c) {
        break;
      }
      std::string entry(entry_c);

      // ignore '.' and '..'
      if (entry == "." || entry == "..") {
        continue;
      }

      // build the full path
      std::string entry_path = archive_path + std::string("/") + entry;

      // is it a file or a sub-folder
      VSIStatBufL sb;
      int result = VSIStatExL(entry_path.c_str(), &sb, VSI_STAT_NATURE_FLAG);
      if (result < 0) {
        break;
      }

      if (VSI_ISDIR(sb.st_mode)) {
        // add subdirectory to be recursed into
        subdirectories.push_back(entry_path);
      } else {
        // add this file
        files.push_back(entry_path);
      }
    }
  }

  // recurse into each subdirectories we found
  for (const auto& subdirectory : subdirectories) {
    gdalGatherFilesInArchiveRecursive(subdirectory, files);
  }
}

/* static */
std::vector<std::string> Importer::gdalGetAllFilesInArchive(
    const std::string& archive_path,
    const CopyParams& copy_params) {
  // lazy init GDAL
  initGDAL();

  // set authorization tokens
  setGDALAuthorizationTokens(copy_params);

  // prepare to gather files
  std::vector<std::string> files;

  // gather the files recursively
  gdalGatherFilesInArchiveRecursive(archive_path, files);

  // convert to relative paths inside archive
  for (auto& file : files) {
    file.erase(0, archive_path.size() + 1);  // remove archive_path and the slash
  }

  // done
  return files;
}

/* static */
bool Importer::gdalSupportsNetworkFileAccess() {
#if (GDAL_VERSION_MAJOR > 2) || (GDAL_VERSION_MAJOR == 2 && GDAL_VERSION_MINOR >= 2)
  return true;
#else
  return false;
#endif
}

ImportStatus Importer::importGDAL(
    ColumnNameToSourceNameMapType columnNameToSourceNameMap) {
  // initial status
  bool load_truncated = false;
  set_import_status(import_id, import_status);

  OGRDataSourceUqPtr poDS(openGDALDataset(file_path, copy_params));
  if (poDS == nullptr) {
    throw std::runtime_error("openGDALDataset Error: Unable to open geo file " +
                             file_path);
  }

  // get the first layer
  OGRLayer* poLayer = poDS->GetLayer(0);
  if (poLayer == nullptr) {
    throw std::runtime_error("No layers found in " + file_path);
  }

  // get the number of features in this layer
  size_t numFeatures = poLayer->GetFeatureCount();

  // build map of metadata field (additional columns) name to index
  // use shared_ptr since we need to pass it to the worker
  FieldNameToIndexMapType fieldNameToIndexMap;
  OGRFeatureDefn* poFDefn = poLayer->GetLayerDefn();
  size_t numFields = poFDefn->GetFieldCount();
  for (size_t iField = 0; iField < numFields; iField++) {
    OGRFieldDefn* poFieldDefn = poFDefn->GetFieldDefn(iField);
    fieldNameToIndexMap.emplace(std::make_pair(poFieldDefn->GetNameRef(), iField));
  }

  // the geographic spatial reference we want to put everything in
  OGRSpatialReferenceUqPtr poGeographicSR(new OGRSpatialReference());
  poGeographicSR->importFromEPSG(copy_params.geo_coords_srid);

#if DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT
  // just one "thread"
  size_t max_threads = 1;
#else
  // how many threads to use
  size_t max_threads = 0;
  if (copy_params.threads == 0) {
    max_threads = sysconf(_SC_NPROCESSORS_CONF);
  } else {
    max_threads = copy_params.threads;
  }
#endif

  // make an import buffer for each thread
  CHECK_EQ(import_buffers_vec.size(), 0);
  import_buffers_vec.resize(max_threads);
  for (size_t i = 0; i < max_threads; i++) {
    for (const auto cd : loader->get_column_descs()) {
      import_buffers_vec[i].emplace_back(
          new TypedImportBuffer(cd, loader->get_string_dict(cd)));
    }
  }

  // make render group analyzers for each poly column
  ColumnIdToRenderGroupAnalyzerMapType columnIdToRenderGroupAnalyzerMap;
  auto columnDescriptors = loader->getCatalog().getAllColumnMetadataForTable(
      loader->get_table_desc()->tableId, false, false, false);
  for (auto cd : columnDescriptors) {
    SQLTypes ct = cd->columnType.get_type();
    if (ct == kPOLYGON || ct == kMULTIPOLYGON) {
      auto rga = std::make_shared<RenderGroupAnalyzer>();
      rga->seedFromExistingTableContents(loader, cd->columnName);
      columnIdToRenderGroupAnalyzerMap[cd->columnId] = rga;
    }
  }

#if !DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT
  // threads
  std::list<std::future<ImportStatus>> threads;

  // use a stack to track thread_ids which must not overlap among threads
  // because thread_id is used to index import_buffers_vec[]
  std::stack<size_t> stack_thread_ids;
  for (size_t i = 0; i < max_threads; i++) {
    stack_thread_ids.push(i);
  }
#endif

  // checkpoint the table
  auto start_epoch = loader->getTableEpoch();

  // reset the layer
  poLayer->ResetReading();

  static const size_t MAX_FEATURES_PER_CHUNK = 1000;

  // make a features buffer for each thread
  std::vector<FeaturePtrVector> features(max_threads);

  // for each feature...
  size_t firstFeatureThisChunk = 0;
  while (firstFeatureThisChunk < numFeatures) {
    // how many features this chunk
    size_t numFeaturesThisChunk =
        std::min(MAX_FEATURES_PER_CHUNK, numFeatures - firstFeatureThisChunk);

// get a thread_id not in use
#if DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT
    size_t thread_id = 0;
#else
    auto thread_id = stack_thread_ids.top();
    stack_thread_ids.pop();
    CHECK(thread_id < max_threads);
#endif

    // fill features buffer for new thread
    for (size_t i = 0; i < numFeaturesThisChunk; i++) {
      features[thread_id].emplace_back(poLayer->GetNextFeature());
    }

#if DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT
    // call worker function directly
    auto ret_import_status = import_thread_shapefile(0,
                                                     this,
                                                     poGeographicSR.get(),
                                                     std::move(features[thread_id]),
                                                     firstFeatureThisChunk,
                                                     numFeaturesThisChunk,
                                                     fieldNameToIndexMap,
                                                     columnNameToSourceNameMap,
                                                     columnIdToRenderGroupAnalyzerMap);
    import_status += ret_import_status;
    import_status.rows_estimated = ((float)firstFeatureThisChunk / (float)numFeatures) *
                                   import_status.rows_completed;
    set_import_status(import_id, import_status);
#else
    // fire up that thread to import this geometry
    threads.push_back(std::async(std::launch::async,
                                 import_thread_shapefile,
                                 thread_id,
                                 this,
                                 poGeographicSR.get(),
                                 std::move(features[thread_id]),
                                 firstFeatureThisChunk,
                                 numFeaturesThisChunk,
                                 fieldNameToIndexMap,
                                 columnNameToSourceNameMap,
                                 columnIdToRenderGroupAnalyzerMap));

    // let the threads run
    while (threads.size() > 0) {
      int nready = 0;
      for (std::list<std::future<ImportStatus>>::iterator it = threads.begin();
           it != threads.end();) {
        auto& p = *it;
        std::chrono::milliseconds span(
            0);  //(std::distance(it, threads.end()) == 1? 1: 0);
        if (p.wait_for(span) == std::future_status::ready) {
          auto ret_import_status = p.get();
          import_status += ret_import_status;
          import_status.rows_estimated =
              ((float)firstFeatureThisChunk / (float)numFeatures) *
              import_status.rows_completed;
          set_import_status(import_id, import_status);

          // recall thread_id for reuse
          stack_thread_ids.push(ret_import_status.thread_id);

          threads.erase(it++);
          ++nready;
        } else {
          ++it;
        }
      }

      if (nready == 0) {
        std::this_thread::yield();
      }

      // keep reading if any free thread slot
      // this is one of the major difference from old threading model !!
      if (threads.size() < max_threads) {
        break;
      }
    }
#endif

    // out of rows?
    if (import_status.rows_rejected > copy_params.max_reject) {
      load_truncated = true;
      load_failed = true;
      LOG(ERROR) << "Maximum rows rejected exceeded. Halting load";
      break;
    }

    // failed?
    if (load_failed) {
      load_truncated = true;
      LOG(ERROR)
          << "A call to the Loader::load failed, Please review the logs for more details";
      break;
    }

    firstFeatureThisChunk += numFeaturesThisChunk;
  }

#if !DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT
  // wait for any remaining threads
  if (threads.size()) {
    for (auto& p : threads) {
      // wait for the thread
      p.wait();
      // get the result and update the final import status
      auto ret_import_status = p.get();
      import_status += ret_import_status;
      import_status.rows_estimated = import_status.rows_completed;
      set_import_status(import_id, import_status);
    }
  }
#endif

  if (load_failed) {
    // rollback to starting epoch - undo all the added records
    loader->setTableEpoch(start_epoch);
  } else {
    loader->checkpoint();
  }

  if (!load_failed &&
      loader->get_table_desc()->persistenceLevel ==
          Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident
                                                      // tables
    for (auto& p : import_buffers_vec[0]) {
      if (!p->stringDictCheckpoint()) {
        LOG(ERROR) << "Checkpointing Dictionary for Column "
                   << p->getColumnDesc()->columnName << " failed.";
        load_failed = true;
        break;
      }
    }
  }

  // must set import_status.load_truncated before closing this end of pipe
  // otherwise, the thread on the other end would throw an unwanted 'write()'
  // exception
  import_status.load_truncated = load_truncated;
  return import_status;
}

//
// class RenderGroupAnalyzer
//

void RenderGroupAnalyzer::seedFromExistingTableContents(
    const std::unique_ptr<Loader>& loader,
    const std::string& geoColumnBaseName) {
  // start timer
  auto seedTimer = timer_start();

  // get the table descriptor
  const auto& cat = loader->getCatalog();
  const std::string& tableName = loader->get_table_desc()->tableName;
  const auto td = cat.getMetadataForTable(tableName);
  CHECK(td);
  CHECK(td->fragmenter);

  // start with a fresh tree
  _rtree = nullptr;
  _numRenderGroups = 0;

  // if the table is empty, just make an empty tree
  if (td->fragmenter->getFragmentsForQuery().getPhysicalNumTuples() == 0) {
    if (DEBUG_RENDER_GROUP_ANALYZER) {
      LOG(INFO) << "DEBUG: Table is empty!";
    }
    _rtree = std::make_unique<RTree>();
    CHECK(_rtree);
    return;
  }

  // no seeding possible without these two columns
  const auto cd_bounds =
      cat.getMetadataForColumn(td->tableId, geoColumnBaseName + "_bounds");
  const auto cd_render_group =
      cat.getMetadataForColumn(td->tableId, geoColumnBaseName + "_render_group");
  if (!cd_bounds || !cd_render_group) {
    throw std::runtime_error("RenderGroupAnalyzer: Table " + tableName +
                             " doesn't have bounds or render_group columns!");
  }

  // and validate their types
  if (cd_bounds->columnType.get_type() != kARRAY ||
      cd_bounds->columnType.get_subtype() != kDOUBLE) {
    throw std::runtime_error("RenderGroupAnalyzer: Table " + tableName +
                             " bounds column is wrong type!");
  }
  if (cd_render_group->columnType.get_type() != kINT) {
    throw std::runtime_error("RenderGroupAnalyzer: Table " + tableName +
                             " render_group column is wrong type!");
  }

  // get chunk accessor table
  auto chunkAccessorTable = getChunkAccessorTable(
      cat, td, {geoColumnBaseName + "_bounds", geoColumnBaseName + "_render_group"});
  const auto table_count = std::get<0>(chunkAccessorTable.back());

  if (DEBUG_RENDER_GROUP_ANALYZER) {
    LOG(INFO) << "DEBUG: Scanning existing table geo column set '" << geoColumnBaseName
              << "'";
  }

  std::vector<Node> nodes;
  try {
    nodes.resize(table_count);
  } catch (const std::exception& e) {
    throw std::runtime_error("RenderGroupAnalyzer failed to reserve memory for " +
                             std::to_string(table_count) + " rows");
  }

  for (size_t row = 0; row < table_count; row++) {
    ArrayDatum ad;
    VarlenDatum vd;
    bool is_end;

    // get ChunkIters and fragment row offset
    size_t rowOffset = 0;
    auto& chunkIters = getChunkItersAndRowOffset(chunkAccessorTable, row, rowOffset);
    auto& boundsChunkIter = chunkIters[0];
    auto& renderGroupChunkIter = chunkIters[1];

    // get bounds values
    ChunkIter_get_nth(&boundsChunkIter, row - rowOffset, &ad, &is_end);
    CHECK(!is_end);
    CHECK(ad.pointer);
    int numBounds = (int)(ad.length / sizeof(double));
    CHECK(numBounds == 4);

    // convert to bounding box
    double* bounds = reinterpret_cast<double*>(ad.pointer);
    BoundingBox bounding_box;
    boost::geometry::assign_inverse(bounding_box);
    boost::geometry::expand(bounding_box, Point(bounds[0], bounds[1]));
    boost::geometry::expand(bounding_box, Point(bounds[2], bounds[3]));

    // get render group
    ChunkIter_get_nth(&renderGroupChunkIter, row - rowOffset, false, &vd, &is_end);
    CHECK(!is_end);
    CHECK(vd.pointer);
    int renderGroup = *reinterpret_cast<int32_t*>(vd.pointer);
    CHECK_GE(renderGroup, 0);

    // store
    nodes[row] = std::make_pair(bounding_box, renderGroup);

    // how many render groups do we have now?
    if (renderGroup >= _numRenderGroups) {
      _numRenderGroups = renderGroup + 1;
    }

    if (DEBUG_RENDER_GROUP_ANALYZER) {
      LOG(INFO) << "DEBUG:   Existing row " << row << " has Render Group " << renderGroup;
    }
  }

  // bulk-load the tree
  auto bulk_load_timer = timer_start();
  _rtree = std::make_unique<RTree>(nodes);
  CHECK(_rtree);
  LOG(INFO) << "Scanning render groups of poly column '" << geoColumnBaseName
            << "' of table '" << tableName << "' took " << timer_stop(seedTimer) << "ms ("
            << timer_stop(bulk_load_timer) << " ms for tree)";

  if (DEBUG_RENDER_GROUP_ANALYZER) {
    LOG(INFO) << "DEBUG: Done! Now have " << _numRenderGroups << " Render Groups";
  }
}

int RenderGroupAnalyzer::insertBoundsAndReturnRenderGroup(
    const std::vector<double>& bounds) {
  // validate
  CHECK(bounds.size() == 4);

  // get bounds
  BoundingBox bounding_box;
  boost::geometry::assign_inverse(bounding_box);
  boost::geometry::expand(bounding_box, Point(bounds[0], bounds[1]));
  boost::geometry::expand(bounding_box, Point(bounds[2], bounds[3]));

  // remainder under mutex to allow this to be multi-threaded
  std::lock_guard<std::mutex> guard(_rtreeMutex);

  // get the intersecting nodes
  std::vector<Node> intersects;
  _rtree->query(boost::geometry::index::intersects(bounding_box),
                std::back_inserter(intersects));

  // build bitset of render groups of the intersecting rectangles
  // clear bit means available, allows use of find_first()
  boost::dynamic_bitset<> bits(_numRenderGroups);
  bits.set();
  for (const auto& intersection : intersects) {
    CHECK(intersection.second < _numRenderGroups);
    bits.reset(intersection.second);
  }

  // find first available group
  int firstAvailableRenderGroup = 0;
  size_t firstSetBit = bits.find_first();
  if (firstSetBit == boost::dynamic_bitset<>::npos) {
    // all known groups represented, add a new one
    firstAvailableRenderGroup = _numRenderGroups;
    _numRenderGroups++;
  } else {
    firstAvailableRenderGroup = (int)firstSetBit;
  }

  // insert new node
  _rtree->insert(std::make_pair(bounding_box, firstAvailableRenderGroup));

  // return it
  return firstAvailableRenderGroup;
}

void ImportDriver::import_geo_table(const std::string& file_path,
                                    const std::string& table_name,
                                    const bool compression,
                                    const bool create_table) {
  const std::string geo_column_name(OMNISCI_GEO_PREFIX);

  CopyParams copy_params;
  if (compression) {
    copy_params.geo_coords_encoding = EncodingType::kENCODING_GEOINT;
    copy_params.geo_coords_comp_param = 32;
  } else {
    copy_params.geo_coords_encoding = EncodingType::kENCODING_NONE;
    copy_params.geo_coords_comp_param = 0;
  }

  const auto cds =
      Importer::gdalToColumnDescriptors(file_path, geo_column_name, copy_params);
  std::map<std::string, std::string> colname_to_src;
  for (auto cd : cds) {
    const auto col_name_sanitized = ImportHelpers::sanitize_name(cd.columnName);
    const auto ret =
        colname_to_src.insert(std::make_pair(cd.columnName, col_name_sanitized));
    CHECK(ret.second);
    cd.columnName = col_name_sanitized;
  }

  auto& cat = session_->getCatalog();

  if (create_table) {
    const auto td = cat.getMetadataForTable(table_name);
    if (td != nullptr) {
      throw std::runtime_error("Error: Table " + table_name +
                               " already exists. Possible failure to correctly re-create "
                               "mapd_data directory.");
    }
    if (table_name != ImportHelpers::sanitize_name(table_name)) {
      throw std::runtime_error("Invalid characters in table name: " + table_name);
    }

    std::string stmt{"CREATE TABLE " + table_name};
    std::vector<std::string> col_stmts;

    for (auto col : cds) {
      if (col.columnType.get_type() == SQLTypes::kINTERVAL_DAY_TIME ||
          col.columnType.get_type() == SQLTypes::kINTERVAL_YEAR_MONTH) {
        throw std::runtime_error(
            "Unsupported type: INTERVAL_DAY_TIME or INTERVAL_YEAR_MONTH for col " +
            col.columnName + " (table: " + table_name + ")");
      }

      if (col.columnType.get_type() == SQLTypes::kDECIMAL) {
        if (col.columnType.get_precision() == 0 && col.columnType.get_scale() == 0) {
          col.columnType.set_precision(14);
          col.columnType.set_scale(7);
        }
      }

      std::string col_stmt;
      col_stmt.append(col.columnName + " " + col.columnType.get_type_name() + " ");

      if (col.columnType.get_compression() != EncodingType::kENCODING_NONE) {
        col_stmt.append("ENCODING " + col.columnType.get_compression_name() + " ");
      } else {
        if (col.columnType.is_string()) {
          col_stmt.append("ENCODING NONE");
        } else if (col.columnType.is_geometry()) {
          if (col.columnType.get_output_srid() == 4326) {
            col_stmt.append("ENCODING NONE");
          }
        }
      }
      col_stmts.push_back(col_stmt);
    }

    stmt.append(" (" + boost::algorithm::join(col_stmts, ",") + ");");
    QueryRunner::run_ddl_statement(stmt, session_);

    LOG(INFO) << "Created table: " << table_name;
  } else {
    LOG(INFO) << "Not creating table: " << table_name;
  }

  const auto td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    throw std::runtime_error("Error: Failed to create table " + table_name);
  }

  Importer_NS::Importer importer(cat, td, file_path, copy_params);
  auto ms = measure<>::execution([&]() { importer.importGDAL(colname_to_src); });
  LOG(INFO) << "Import Time for " << table_name << ": " << (double)ms / 1000.0 << " s";
}

}  // namespace Importer_NS
