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

#include "ImportExport/Importer.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <gdal.h>
#include <ogrsf_frmts.h>
#include <boost/algorithm/string.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/filesystem.hpp>
#include <boost/geometry.hpp>
#include <boost/variant.hpp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <stack>
#include <stdexcept>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Archive/PosixFileArchive.h"
#include "Archive/S3Archive.h"
#include "ArrowImporter.h"
#include "Geospatial/Compression.h"
#include "Geospatial/GDAL.h"
#include "Geospatial/Transforms.h"
#include "Geospatial/Types.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "Logger/Logger.h"
#include "OSDependent/omnisci_glob.h"
#include "QueryEngine/TypePunning.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/import_helpers.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/scope.h"
#include "Shared/shard_key.h"
#include "Shared/thread_count.h"
#include "Utils/ChunkAccessorTable.h"

#include "gen-cpp/OmniSci.h"

size_t g_max_import_threads =
    32;  // Max number of default import threads to use (num hardware threads will be used
         // if lower, and can also be explicitly overriden in copy statement with threads
         // option)
size_t g_archive_read_buf_size = 1 << 20;

inline auto get_filesize(const std::string& file_path) {
  boost::filesystem::path boost_file_path{file_path};
  boost::system::error_code ec;
  const auto filesize = boost::filesystem::file_size(boost_file_path, ec);
  return ec ? 0 : filesize;
}

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

// For logging std::vector<std::string> row.
namespace boost {
namespace log {
formatting_ostream& operator<<(formatting_ostream& out, std::vector<std::string>& row) {
  out << '[';
  for (size_t i = 0; i < row.size(); ++i) {
    out << (i ? ", " : "") << row[i];
  }
  out << ']';
  return out;
}
}  // namespace log
}  // namespace boost

namespace import_export {

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
      d.bigintval = inline_fixed_encoding_null_val(ti);
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

Datum NullArrayDatum(SQLTypeInfo& ti) {
  Datum d;
  const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  switch (type) {
    case kBOOLEAN:
      d.boolval = inline_fixed_encoding_null_array_val(ti);
      break;
    case kBIGINT:
      d.bigintval = inline_fixed_encoding_null_array_val(ti);
      break;
    case kINT:
      d.intval = inline_fixed_encoding_null_array_val(ti);
      break;
    case kSMALLINT:
      d.smallintval = inline_fixed_encoding_null_array_val(ti);
      break;
    case kTINYINT:
      d.tinyintval = inline_fixed_encoding_null_array_val(ti);
      break;
    case kFLOAT:
      d.floatval = NULL_ARRAY_FLOAT;
      break;
    case kDOUBLE:
      d.doubleval = NULL_ARRAY_DOUBLE;
      break;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      d.bigintval = inline_fixed_encoding_null_array_val(ti);
      break;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      throw std::runtime_error("Internal error: geometry type in NullArrayDatum.");
    default:
      throw std::runtime_error("Internal error: invalid type in NullArrayDatum.");
  }
  return d;
}

ArrayDatum StringToArray(const std::string& s,
                         const SQLTypeInfo& ti,
                         const CopyParams& copy_params) {
  SQLTypeInfo elem_ti = ti.get_elem_type();
  if (s == copy_params.null_str || s == "NULL" || s.empty()) {
    return ArrayDatum(0, NULL, true);
  }
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
  if (elem_strs.size() == 1) {
    auto str = elem_strs.front();
    auto str_trimmed = trim_space(str.c_str(), str.length());
    if (str_trimmed == "") {
      elem_strs.clear();  // Empty array
    }
  }
  if (!elem_ti.is_string()) {
    size_t len = elem_strs.size() * elem_ti.get_size();
    int8_t* buf = (int8_t*)checked_malloc(len);
    int8_t* p = buf;
    for (auto& es : elem_strs) {
      auto e = trim_space(es.c_str(), es.length());
      bool is_null = (e == copy_params.null_str) || e == "NULL";
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
    return ArrayDatum(len, buf, false);
  }
  // must not be called for array of strings
  CHECK(false);
  return ArrayDatum(0, NULL, true);
}

ArrayDatum NullArray(const SQLTypeInfo& ti) {
  SQLTypeInfo elem_ti = ti.get_elem_type();
  auto len = ti.get_size();

  if (elem_ti.is_string()) {
    // must not be called for array of strings
    CHECK(false);
    return ArrayDatum(0, NULL, true);
  }

  if (len > 0) {
    // Compose a NULL fixlen array
    int8_t* buf = (int8_t*)checked_malloc(len);
    // First scalar is a NULL_ARRAY sentinel
    Datum d = NullArrayDatum(elem_ti);
    int8_t* p = appendDatum(buf, d, elem_ti);
    // Rest is filled with normal NULL sentinels
    Datum d0 = NullDatum(elem_ti);
    while ((p - buf) < len) {
      p = appendDatum(p, d0, elem_ti);
    }
    CHECK((p - buf) == len);
    return ArrayDatum(len, buf, true);
  }
  // NULL varlen array
  return ArrayDatum(0, NULL, true);
}

ArrayDatum ImporterUtils::composeNullArray(const SQLTypeInfo& ti) {
  return NullArray(ti);
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
      d.bigintval =
          datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
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

  if (datum.is_null) {
    return NullArray(ti);
  }

  size_t len = datum.val.arr_val.size() * elem_ti.get_size();
  int8_t* buf = (int8_t*)checked_malloc(len);
  int8_t* p = buf;
  for (auto& e : datum.val.arr_val) {
    p = appendDatum(p, TDatumToDatum(e, elem_ti), elem_ti);
  }

  return ArrayDatum(len, buf, false);
}

void TypedImportBuffer::addDictEncodedString(const std::vector<std::string>& string_vec) {
  CHECK(string_dict_);
  std::vector<std::string_view> string_view_vec;
  string_view_vec.reserve(string_vec.size());
  for (const auto& str : string_vec) {
    if (str.size() > StringDictionary::MAX_STRLEN) {
      throw std::runtime_error("String too long for dictionary encoding.");
    }
    string_view_vec.push_back(str);
  }
  switch (column_desc_->columnType.get_size()) {
    case 1:
      string_dict_i8_buffer_->resize(string_view_vec.size());
      string_dict_->getOrAddBulk(string_view_vec, string_dict_i8_buffer_->data());
      break;
    case 2:
      string_dict_i16_buffer_->resize(string_view_vec.size());
      string_dict_->getOrAddBulk(string_view_vec, string_dict_i16_buffer_->data());
      break;
    case 4:
      string_dict_i32_buffer_->resize(string_view_vec.size());
      string_dict_->getOrAddBulk(string_view_vec, string_dict_i32_buffer_->data());
      break;
    default:
      CHECK(false);
  }
}

void TypedImportBuffer::add_value(const ColumnDescriptor* cd,
                                  const std::string_view val,
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
        auto ti = cd->columnType;
        Datum d = StringToDatum(val, ti);
        addBoolean(static_cast<int8_t>(d.boolval));
      }
      break;
    }
    case kTINYINT: {
      if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
        auto ti = cd->columnType;
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
        auto ti = cd->columnType;
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
        auto ti = cd->columnType;
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
        auto ti = cd->columnType;
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
        addFloat(static_cast<float>(std::atof(std::string(val).c_str())));
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addFloat(NULL_FLOAT);
      }
      break;
    case kDOUBLE:
      if (!is_null && (val[0] == '.' || isdigit(val[0]) || val[0] == '-')) {
        addDouble(std::atof(std::string(val).c_str()));
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
        addBigint(d.bigintval);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addBigint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kARRAY: {
      if (is_null && cd->columnType.get_notnull()) {
        throw std::runtime_error("NULL for column " + cd->columnName);
      }
      SQLTypeInfo ti = cd->columnType;
      if (IS_STRING(ti.get_subtype())) {
        std::vector<std::string> string_vec;
        // Just parse string array, don't push it to buffer yet as we might throw
        import_export::delimited_parser::parse_string_array(
            std::string(val), copy_params, string_vec);
        if (!is_null) {
          // TODO: add support for NULL string arrays
          if (ti.get_size() > 0) {
            auto sti = ti.get_elem_type();
            size_t expected_size = ti.get_size() / sti.get_size();
            size_t actual_size = string_vec.size();
            if (actual_size != expected_size) {
              throw std::runtime_error("Fixed length array column " + cd->columnName +
                                       " expects " + std::to_string(expected_size) +
                                       " values, received " +
                                       std::to_string(actual_size));
            }
          }
          addStringArray(string_vec);
        } else {
          if (ti.get_size() > 0) {
            // TODO: remove once NULL fixlen arrays are allowed
            throw std::runtime_error("Fixed length array column " + cd->columnName +
                                     " currently cannot accept NULL arrays");
          }
          // TODO: add support for NULL string arrays, replace with addStringArray(),
          //       for now add whatever parseStringArray() outputs for NULLs ("NULL")
          addStringArray(string_vec);
        }
      } else {
        if (!is_null) {
          ArrayDatum d = StringToArray(std::string(val), ti, copy_params);
          if (d.is_null) {  // val could be "NULL"
            addArray(NullArray(ti));
          } else {
            if (ti.get_size() > 0 && static_cast<size_t>(ti.get_size()) != d.length) {
              throw std::runtime_error("Fixed length array for column " + cd->columnName +
                                       " has incorrect length: " + std::string(val));
            }
            addArray(d);
          }
        } else {
          addArray(NullArray(ti));
        }
      }
      break;
    }
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
      bigint_buffer_->pop_back();
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

struct GeoImportException : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// appends (streams) a slice of Arrow array of values (RHS) to TypedImportBuffer (LHS)
template <typename DATA_TYPE>
size_t TypedImportBuffer::convert_arrow_val_to_import_buffer(
    const ColumnDescriptor* cd,
    const Array& array,
    std::vector<DATA_TYPE>& buffer,
    const ArraySliceRange& slice_range,
    import_export::BadRowsTracker* const bad_rows_tracker) {
  auto data =
      std::make_unique<DataBuffer<DATA_TYPE>>(cd, array, buffer, bad_rows_tracker);
  auto f_value_getter = value_getter(array, cd, bad_rows_tracker);
  std::function<void(const int64_t)> f_add_geo_phy_cols = [&](const int64_t row) {};
  if (bad_rows_tracker && cd->columnType.is_geometry()) {
    f_add_geo_phy_cols = [&](const int64_t row) {
      // Populate physical columns (ref. DBHandler::load_table)
      std::vector<double> coords, bounds;
      std::vector<int> ring_sizes, poly_rings;
      int render_group = 0;
      SQLTypeInfo ti;
      // replace any unexpected exception from getGeoColumns or other
      // on this path with a GeoImportException so that we wont over
      // push a null to the logical column...
      try {
        SQLTypeInfo import_ti{ti};
        if (array.IsNull(row)) {
          Geospatial::GeoTypesFactory::getNullGeoColumns(
              import_ti, coords, bounds, ring_sizes, poly_rings, false);
        } else {
          arrow_throw_if<GeoImportException>(
              !Geospatial::GeoTypesFactory::getGeoColumns(geo_string_buffer_->back(),
                                                          ti,
                                                          coords,
                                                          bounds,
                                                          ring_sizes,
                                                          poly_rings,
                                                          false),
              error_context(cd, bad_rows_tracker) + "Invalid geometry");
          arrow_throw_if<GeoImportException>(
              cd->columnType.get_type() != ti.get_type(),
              error_context(cd, bad_rows_tracker) + "Geometry type mismatch");
        }
        auto col_idx_workpad = col_idx;  // what a pitfall!!
        import_export::Importer::set_geo_physical_import_buffer(
            bad_rows_tracker->importer->getCatalog(),
            cd,
            *import_buffers,
            col_idx_workpad,
            coords,
            bounds,
            ring_sizes,
            poly_rings,
            render_group);
      } catch (GeoImportException&) {
        throw;
      } catch (std::runtime_error& e) {
        throw GeoImportException(e.what());
      } catch (const std::exception& e) {
        throw GeoImportException(e.what());
      } catch (...) {
        throw GeoImportException("unknown exception");
      }
    };
  }
  auto f_mark_a_bad_row = [&](const auto row) {
    std::unique_lock<std::mutex> lck(bad_rows_tracker->mutex);
    bad_rows_tracker->rows.insert(row - slice_range.first);
  };
  buffer.reserve(slice_range.second - slice_range.first);
  for (size_t row = slice_range.first; row < slice_range.second; ++row) {
    try {
      *data << (array.IsNull(row) ? nullptr : f_value_getter(array, row));
      f_add_geo_phy_cols(row);
    } catch (GeoImportException&) {
      f_mark_a_bad_row(row);
    } catch (ArrowImporterException&) {
      // trace bad rows of each column; otherwise rethrow.
      if (bad_rows_tracker) {
        *data << nullptr;
        f_mark_a_bad_row(row);
      } else {
        throw;
      }
    }
  }
  return buffer.size();
}

size_t TypedImportBuffer::add_arrow_values(const ColumnDescriptor* cd,
                                           const Array& col,
                                           const bool exact_type_match,
                                           const ArraySliceRange& slice_range,
                                           BadRowsTracker* const bad_rows_tracker) {
  const auto type = cd->columnType.get_type();
  if (cd->columnType.get_notnull()) {
    // We can't have any null values for this column; to have them is an error
    arrow_throw_if(col.null_count() > 0, "NULL not allowed for column " + cd->columnName);
  }

  switch (type) {
    case kBOOLEAN:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::BOOL, "Expected boolean type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *bool_buffer_, slice_range, bad_rows_tracker);
    case kTINYINT:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::INT8, "Expected int8 type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *tinyint_buffer_, slice_range, bad_rows_tracker);
    case kSMALLINT:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::INT16, "Expected int16 type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *smallint_buffer_, slice_range, bad_rows_tracker);
    case kINT:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::INT32, "Expected int32 type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *int_buffer_, slice_range, bad_rows_tracker);
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::INT64, "Expected int64 type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *bigint_buffer_, slice_range, bad_rows_tracker);
    case kFLOAT:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::FLOAT, "Expected float type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *float_buffer_, slice_range, bad_rows_tracker);
    case kDOUBLE:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::DOUBLE, "Expected double type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *double_buffer_, slice_range, bad_rows_tracker);
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::BINARY && col.type_id() != Type::STRING,
                       "Expected string type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *string_buffer_, slice_range, bad_rows_tracker);
    case kTIME:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::TIME32 && col.type_id() != Type::TIME64,
                       "Expected time32 or time64 type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *bigint_buffer_, slice_range, bad_rows_tracker);
    case kTIMESTAMP:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::TIMESTAMP, "Expected timestamp type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *bigint_buffer_, slice_range, bad_rows_tracker);
    case kDATE:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::DATE32 && col.type_id() != Type::DATE64,
                       "Expected date32 or date64 type");
      }
      return convert_arrow_val_to_import_buffer(
          cd, col, *bigint_buffer_, slice_range, bad_rows_tracker);
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      arrow_throw_if(col.type_id() != Type::BINARY && col.type_id() != Type::STRING,
                     "Expected string type");
      return convert_arrow_val_to_import_buffer(
          cd, col, *geo_string_buffer_, slice_range, bad_rows_tracker);
    case kARRAY:
      throw std::runtime_error("Arrow array appends not yet supported");
    default:
      throw std::runtime_error("Invalid Type");
  }
}

// this is exclusively used by load_table_binary_columnar
size_t TypedImportBuffer::add_values(const ColumnDescriptor* cd, const TColumn& col) {
  size_t dataSize = 0;
  if (cd->columnType.get_notnull()) {
    // We can't have any null values for this column; to have them is an error
    if (std::any_of(col.nulls.begin(), col.nulls.end(), [](int i) { return i != 0; })) {
      throw std::runtime_error("NULL for column " + cd->columnName);
    }
  }

  switch (cd->columnType.get_type()) {
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
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL: {
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
      bigint_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i]) {
          bigint_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        } else {
          bigint_buffer_->push_back(static_cast<int64_t>(col.data.int_col[i]));
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
                addArray(NullArray(cd->columnType));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int8_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(bool*)p = static_cast<bool>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(bool);
                }
                addArray(ArrayDatum(byteSize, buf, false));
              }
            }
            break;
          }
          case kTINYINT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(NullArray(cd->columnType));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int8_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(int8_t*)p = static_cast<int8_t>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(int8_t);
                }
                addArray(ArrayDatum(byteSize, buf, false));
              }
            }
            break;
          }
          case kSMALLINT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(NullArray(cd->columnType));
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
                addArray(ArrayDatum(byteSize, buf, false));
              }
            }
            break;
          }
          case kINT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(NullArray(cd->columnType));
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
                addArray(ArrayDatum(byteSize, buf, false));
              }
            }
            break;
          }
          case kBIGINT:
          case kNUMERIC:
          case kDECIMAL: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(NullArray(cd->columnType));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int64_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(int64_t*)p =
                      static_cast<int64_t>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(int64_t);
                }
                addArray(ArrayDatum(byteSize, buf, false));
              }
            }
            break;
          }
          case kFLOAT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(NullArray(cd->columnType));
              } else {
                size_t len = col.data.arr_col[i].data.real_col.size();
                size_t byteSize = len * sizeof(float);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(float*)p = static_cast<float>(col.data.arr_col[i].data.real_col[j]);
                  p += sizeof(float);
                }
                addArray(ArrayDatum(byteSize, buf, false));
              }
            }
            break;
          }
          case kDOUBLE: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(NullArray(cd->columnType));
              } else {
                size_t len = col.data.arr_col[i].data.real_col.size();
                size_t byteSize = len * sizeof(double);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(double*)p = static_cast<double>(col.data.arr_col[i].data.real_col[j]);
                  p += sizeof(double);
                }
                addArray(ArrayDatum(byteSize, buf, false));
              }
            }
            break;
          }
          case kTIME:
          case kTIMESTAMP:
          case kDATE: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i]) {
                addArray(NullArray(cd->columnType));
              } else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteWidth = sizeof(int64_t);
                size_t byteSize = len * byteWidth;
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *reinterpret_cast<int64_t*>(p) =
                      static_cast<int64_t>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(int64_t);
                }
                addArray(ArrayDatum(byteSize, buf, false));
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
        addBigint(datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addBigint(inline_fixed_encoding_null_val(cd->columnType));
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
          addArray(NullArray(cd->columnType));
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

bool importGeoFromLonLat(double lon,
                         double lat,
                         std::vector<double>& coords,
                         SQLTypeInfo& ti) {
  if (std::isinf(lat) || std::isnan(lat) || std::isinf(lon) || std::isnan(lon)) {
    return false;
  }
  if (ti.transforms()) {
    Geospatial::GeoPoint pt{std::vector<double>{lon, lat}};
    if (!pt.transform(ti)) {
      return false;
    }
    pt.getColumns(coords);
    return true;
  }
  coords.push_back(lon);
  coords.push_back(lat);
  return true;
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
  bool is_null_geo = false;
  bool is_null_point = false;
  if (!col_ti.get_notnull()) {
    // Check for NULL geo
    if (col_type == kPOINT && (coords.empty() || coords[0] == NULL_ARRAY_DOUBLE)) {
      is_null_point = true;
      coords.clear();
    }
    is_null_geo = coords.empty();
    if (is_null_point) {
      coords.push_back(NULL_ARRAY_DOUBLE);
      coords.push_back(NULL_DOUBLE);
      // Treating POINT coords as notnull, need to store actual encoding
      // [un]compressed+[not]null
      is_null_geo = false;
    }
  }
  TDatum tdd_coords;
  // Get the raw data representing [optionally compressed] non-NULL geo's coords.
  // One exception - NULL POINT geo: coords need to be processed to encode nullness
  // in a fixlen array, compressed and uncompressed.
  if (!is_null_geo) {
    std::vector<uint8_t> compressed_coords = Geospatial::compress_coords(coords, col_ti);
    tdd_coords.val.arr_val.reserve(compressed_coords.size());
    for (auto cc : compressed_coords) {
      tdd_coords.val.arr_val.emplace_back();
      tdd_coords.val.arr_val.back().val.int_val = cc;
    }
  }
  tdd_coords.is_null = is_null_geo;
  import_buffers[col_idx++]->add_value(cd_coords, tdd_coords, false, replicate_count);

  if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    // Create ring_sizes array value and add it to the physical column
    auto cd_ring_sizes = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    TDatum tdd_ring_sizes;
    tdd_ring_sizes.val.arr_val.reserve(ring_sizes.size());
    if (!is_null_geo) {
      for (auto ring_size : ring_sizes) {
        tdd_ring_sizes.val.arr_val.emplace_back();
        tdd_ring_sizes.val.arr_val.back().val.int_val = ring_size;
      }
    }
    tdd_ring_sizes.is_null = is_null_geo;
    import_buffers[col_idx++]->add_value(
        cd_ring_sizes, tdd_ring_sizes, false, replicate_count);
  }

  if (col_type == kMULTIPOLYGON) {
    // Create poly_rings array value and add it to the physical column
    auto cd_poly_rings = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    TDatum tdd_poly_rings;
    tdd_poly_rings.val.arr_val.reserve(poly_rings.size());
    if (!is_null_geo) {
      for (auto num_rings : poly_rings) {
        tdd_poly_rings.val.arr_val.emplace_back();
        tdd_poly_rings.val.arr_val.back().val.int_val = num_rings;
      }
    }
    tdd_poly_rings.is_null = is_null_geo;
    import_buffers[col_idx++]->add_value(
        cd_poly_rings, tdd_poly_rings, false, replicate_count);
  }

  if (col_type == kLINESTRING || col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    auto cd_bounds = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    TDatum tdd_bounds;
    tdd_bounds.val.arr_val.reserve(bounds.size());
    if (!is_null_geo) {
      for (auto b : bounds) {
        tdd_bounds.val.arr_val.emplace_back();
        tdd_bounds.val.arr_val.back().val.real_val = b;
      }
    }
    tdd_bounds.is_null = is_null_geo;
    import_buffers[col_idx++]->add_value(cd_bounds, tdd_bounds, false, replicate_count);
  }

  if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    // Create render_group value and add it to the physical column
    auto cd_render_group = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    TDatum td_render_group;
    td_render_group.val.int_val = render_group;
    td_render_group.is_null = is_null_geo;
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
    bool is_null_geo = false;
    bool is_null_point = false;
    if (!col_ti.get_notnull()) {
      // Check for NULL geo
      if (col_type == kPOINT && (coords.empty() || coords[0] == NULL_ARRAY_DOUBLE)) {
        is_null_point = true;
        coords.clear();
      }
      is_null_geo = coords.empty();
      if (is_null_point) {
        coords.push_back(NULL_ARRAY_DOUBLE);
        coords.push_back(NULL_DOUBLE);
        // Treating POINT coords as notnull, need to store actual encoding
        // [un]compressed+[not]null
        is_null_geo = false;
      }
    }
    std::vector<TDatum> td_coords_data;
    if (!is_null_geo) {
      std::vector<uint8_t> compressed_coords =
          Geospatial::compress_coords(coords, col_ti);
      for (auto cc : compressed_coords) {
        TDatum td_byte;
        td_byte.val.int_val = cc;
        td_coords_data.push_back(td_byte);
      }
    }
    TDatum tdd_coords;
    tdd_coords.val.arr_val = td_coords_data;
    tdd_coords.is_null = is_null_geo;
    import_buffers[col_idx]->add_value(cd_coords, tdd_coords, false, replicate_count);
  }
  col_idx++;

  if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    if (ring_sizes_column.size() != coords_row_count) {
      CHECK(false) << "Geometry import columnar: ring sizes column size mismatch";
    }
    // Create ring_sizes array value and add it to the physical column
    auto cd_ring_sizes = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    for (auto ring_sizes : ring_sizes_column) {
      bool is_null_geo = false;
      if (!col_ti.get_notnull()) {
        // Check for NULL geo
        is_null_geo = ring_sizes.empty();
      }
      std::vector<TDatum> td_ring_sizes;
      for (auto ring_size : ring_sizes) {
        TDatum td_ring_size;
        td_ring_size.val.int_val = ring_size;
        td_ring_sizes.push_back(td_ring_size);
      }
      TDatum tdd_ring_sizes;
      tdd_ring_sizes.val.arr_val = td_ring_sizes;
      tdd_ring_sizes.is_null = is_null_geo;
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
    auto cd_poly_rings = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    for (auto poly_rings : poly_rings_column) {
      bool is_null_geo = false;
      if (!col_ti.get_notnull()) {
        // Check for NULL geo
        is_null_geo = poly_rings.empty();
      }
      std::vector<TDatum> td_poly_rings;
      for (auto num_rings : poly_rings) {
        TDatum td_num_rings;
        td_num_rings.val.int_val = num_rings;
        td_poly_rings.push_back(td_num_rings);
      }
      TDatum tdd_poly_rings;
      tdd_poly_rings.val.arr_val = td_poly_rings;
      tdd_poly_rings.is_null = is_null_geo;
      import_buffers[col_idx]->add_value(
          cd_poly_rings, tdd_poly_rings, false, replicate_count);
    }
    col_idx++;
  }

  if (col_type == kLINESTRING || col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    if (bounds_column.size() != coords_row_count) {
      CHECK(false) << "Geometry import columnar: bounds column size mismatch";
    }
    auto cd_bounds = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    for (auto bounds : bounds_column) {
      bool is_null_geo = false;
      if (!col_ti.get_notnull()) {
        // Check for NULL geo
        is_null_geo = (bounds.empty() || bounds[0] == NULL_ARRAY_DOUBLE);
      }
      std::vector<TDatum> td_bounds_data;
      for (auto b : bounds) {
        TDatum td_double;
        td_double.val.real_val = b;
        td_bounds_data.push_back(td_double);
      }
      TDatum tdd_bounds;
      tdd_bounds.val.arr_val = td_bounds_data;
      tdd_bounds.is_null = is_null_geo;
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

namespace {

std::tuple<int, SQLTypes, std::string> explode_collections_step1(
    const std::list<const ColumnDescriptor*>& col_descs) {
  // validate the columns
  // for now we can only explode into a single destination column
  // which must be of the child type (POLYGON, LINESTRING, POINT)
  int collection_col_idx = -1;
  int col_idx = 0;
  std::string collection_col_name;
  SQLTypes collection_child_type = kNULLT;
  for (auto cd_it = col_descs.begin(); cd_it != col_descs.end(); cd_it++) {
    auto const& cd = *cd_it;
    auto const col_type = cd->columnType.get_type();
    if (col_type == kPOLYGON || col_type == kLINESTRING || col_type == kPOINT) {
      if (collection_col_idx >= 0) {
        throw std::runtime_error(
            "Explode Collections: Found more than one destination column");
      }
      collection_col_idx = col_idx;
      collection_child_type = col_type;
      collection_col_name = cd->columnName;
    }
    for (int i = 0; i < cd->columnType.get_physical_cols(); ++i) {
      ++cd_it;
    }
    col_idx++;
  }
  if (collection_col_idx < 0) {
    throw std::runtime_error(
        "Explode Collections: Failed to find a supported column type to explode "
        "into");
  }
  return std::make_tuple(collection_col_idx, collection_child_type, collection_col_name);
}

int64_t explode_collections_step2(
    OGRGeometry* ogr_geometry,
    SQLTypes collection_child_type,
    const std::string& collection_col_name,
    size_t row_or_feature_idx,
    std::function<void(OGRGeometry*)> execute_import_lambda) {
  auto ogr_geometry_type = wkbFlatten(ogr_geometry->getGeometryType());
  bool is_collection = false;
  switch (collection_child_type) {
    case kPOINT:
      switch (ogr_geometry_type) {
        case wkbMultiPoint:
          is_collection = true;
          break;
        case wkbPoint:
          break;
        default:
          throw std::runtime_error(
              "Explode Collections: Source geo type must be MULTIPOINT or POINT");
      }
      break;
    case kLINESTRING:
      switch (ogr_geometry_type) {
        case wkbMultiLineString:
          is_collection = true;
          break;
        case wkbLineString:
          break;
        default:
          throw std::runtime_error(
              "Explode Collections: Source geo type must be MULTILINESTRING or "
              "LINESTRING");
      }
      break;
    case kPOLYGON:
      switch (ogr_geometry_type) {
        case wkbMultiPolygon:
          is_collection = true;
          break;
        case wkbPolygon:
          break;
        default:
          throw std::runtime_error(
              "Explode Collections: Source geo type must be MULTIPOLYGON or POLYGON");
      }
      break;
    default:
      CHECK(false) << "Unsupported geo child type " << collection_child_type;
  }

  int64_t us = 0LL;

  // explode or just import
  if (is_collection) {
    // cast to collection
    OGRGeometryCollection* collection_geometry = ogr_geometry->toGeometryCollection();
    CHECK(collection_geometry);

#if LOG_EXPLODE_COLLECTIONS
    // log number of children
    LOG(INFO) << "Exploding row/feature " << row_or_feature_idx << " for column '"
              << explode_col_name << "' into " << collection_geometry->getNumGeometries()
              << " child rows";
#endif

    // loop over children
    uint32_t child_geometry_count = 0;
    auto child_geometry_it = collection_geometry->begin();
    while (child_geometry_it != collection_geometry->end()) {
      // get and import this child
      OGRGeometry* import_geometry = *child_geometry_it;
      us += measure<std::chrono::microseconds>::execution(
          [&] { execute_import_lambda(import_geometry); });

      // next child
      child_geometry_it++;
      child_geometry_count++;
    }
  } else {
    // import non-collection row just once
    us = measure<std::chrono::microseconds>::execution(
        [&] { execute_import_lambda(ogr_geometry); });
  }

  // done
  return us;
}

}  // namespace

static ImportStatus import_thread_delimited(
    int thread_id,
    Importer* importer,
    std::unique_ptr<char[]> scratch_buffer,
    size_t begin_pos,
    size_t end_pos,
    size_t total_size,
    const ColumnIdToRenderGroupAnalyzerMapType& columnIdToRenderGroupAnalyzerMap,
    size_t first_row_index_this_buffer) {
  ImportStatus import_status;
  int64_t total_get_row_time_us = 0;
  int64_t total_str_to_val_time_us = 0;
  CHECK(scratch_buffer);
  auto buffer = scratch_buffer.get();
  auto load_ms = measure<>::execution([]() {});
  auto ms = measure<>::execution([&]() {
    const CopyParams& copy_params = importer->get_copy_params();
    const std::list<const ColumnDescriptor*>& col_descs = importer->get_column_descs();
    size_t begin =
        delimited_parser::find_beginning(buffer, begin_pos, end_pos, copy_params);
    const char* thread_buf = buffer + begin_pos + begin;
    const char* thread_buf_end = buffer + end_pos;
    const char* buf_end = buffer + total_size;
    bool try_single_thread = false;
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers =
        importer->get_import_buffers(thread_id);
    auto us = measure<std::chrono::microseconds>::execution([&]() {});
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
    for (const auto& p : import_buffers) {
      p->clear();
    }
    std::vector<std::string_view> row;
    size_t row_index_plus_one = 0;
    for (const char* p = thread_buf; p < thread_buf_end; p++) {
      row.clear();
      std::vector<std::unique_ptr<char[]>>
          tmp_buffers;  // holds string w/ removed escape chars, etc
      if (DEBUG_TIMING) {
        us = measure<std::chrono::microseconds>::execution([&]() {
          p = import_export::delimited_parser::get_row(p,
                                                       thread_buf_end,
                                                       buf_end,
                                                       copy_params,
                                                       importer->get_is_array(),
                                                       row,
                                                       tmp_buffers,
                                                       try_single_thread);
        });
        total_get_row_time_us += us;
      } else {
        p = import_export::delimited_parser::get_row(p,
                                                     thread_buf_end,
                                                     buf_end,
                                                     copy_params,
                                                     importer->get_is_array(),
                                                     row,
                                                     tmp_buffers,
                                                     try_single_thread);
      }
      row_index_plus_one++;
      // Each POINT could consume two separate coords instead of a single WKT
      if (row.size() < num_cols || (num_cols + point_cols) < row.size()) {
        import_status.rows_rejected++;
        LOG(ERROR) << "Incorrect Row (expected " << num_cols << " columns, has "
                   << row.size() << "): " << shared::printContainer(row);
        if (import_status.rows_rejected > copy_params.max_reject) {
          break;
        }
        continue;
      }

      //
      // lambda for importing a row (perhaps multiple times if exploding a collection)
      //

      auto execute_import_row = [&](OGRGeometry* import_geometry) {
        size_t import_idx = 0;
        size_t col_idx = 0;
        try {
          for (auto cd_it = col_descs.begin(); cd_it != col_descs.end(); cd_it++) {
            auto cd = *cd_it;
            const auto& col_ti = cd->columnType;

            bool is_null =
                (row[import_idx] == copy_params.null_str || row[import_idx] == "NULL");
            // Note: default copy_params.null_str is "\N", but everyone uses "NULL".
            // So initially nullness may be missed and not passed to add_value,
            // which then might also check and still decide it's actually a NULL, e.g.
            // if kINT doesn't start with a digit or a '-' then it's considered NULL.
            // So "NULL" is not recognized as NULL but then it's not recognized as
            // a valid kINT, so it's a NULL after all.
            // Checking for "NULL" here too, as a widely accepted notation for NULL.

            // Treating empty as NULL
            if (!cd->columnType.is_string() && row[import_idx].empty()) {
              is_null = true;
            }

            if (col_ti.get_physical_cols() == 0) {
              // not geo

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
              auto const& geo_string = row[import_idx];

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

              // if this is a POINT column, and the field is not null, and
              // looks like a scalar numeric value (and not a hex blob)
              // attempt to import two columns as lon/lat (or lat/lon)
              if (col_type == kPOINT && !is_null && geo_string.size() > 0 &&
                  (geo_string[0] == '.' || isdigit(geo_string[0]) ||
                   geo_string[0] == '-') &&
                  geo_string.find_first_of("ABCDEFabcdef") == std::string::npos) {
                double lon = std::atof(std::string(geo_string).c_str());
                double lat = NAN;
                auto lat_str = row[import_idx];
                ++import_idx;
                if (lat_str.size() > 0 &&
                    (lat_str[0] == '.' || isdigit(lat_str[0]) || lat_str[0] == '-')) {
                  lat = std::atof(std::string(lat_str).c_str());
                }
                // Swap coordinates if this table uses a reverse order: lat/lon
                if (!copy_params.lonlat) {
                  std::swap(lat, lon);
                }
                // TODO: should check if POINT column should have been declared with
                // SRID WGS 84, EPSG 4326 ? if (col_ti.get_dimension() != 4326) {
                //  throw std::runtime_error("POINT column " + cd->columnName + " is
                //  not WGS84, cannot insert lon/lat");
                // }
                SQLTypeInfo import_ti{col_ti};
                if (copy_params.file_type == FileType::DELIMITED &&
                    import_ti.get_output_srid() == 4326) {
                  auto srid0 = copy_params.source_srid;
                  if (srid0 > 0) {
                    // srid0 -> 4326 transform is requested on import
                    import_ti.set_input_srid(srid0);
                  }
                }
                if (!importGeoFromLonLat(lon, lat, coords, import_ti)) {
                  throw std::runtime_error(
                      "Cannot read lon/lat to insert into POINT column " +
                      cd->columnName);
                }
              } else {
                // import it
                SQLTypeInfo import_ti{col_ti};
                if (copy_params.file_type == FileType::DELIMITED &&
                    import_ti.get_output_srid() == 4326) {
                  auto srid0 = copy_params.source_srid;
                  if (srid0 > 0) {
                    // srid0 -> 4326 transform is requested on import
                    import_ti.set_input_srid(srid0);
                  }
                }
                if (is_null) {
                  if (col_ti.get_notnull()) {
                    throw std::runtime_error("NULL geo for column " + cd->columnName);
                  }
                  Geospatial::GeoTypesFactory::getNullGeoColumns(
                      import_ti,
                      coords,
                      bounds,
                      ring_sizes,
                      poly_rings,
                      PROMOTE_POLYGON_TO_MULTIPOLYGON);
                } else {
                  if (import_geometry) {
                    // geometry already exploded
                    if (!Geospatial::GeoTypesFactory::getGeoColumns(
                            import_geometry,
                            import_ti,
                            coords,
                            bounds,
                            ring_sizes,
                            poly_rings,
                            PROMOTE_POLYGON_TO_MULTIPOLYGON)) {
                      std::string msg =
                          "Failed to extract valid geometry from exploded row " +
                          std::to_string(first_row_index_this_buffer +
                                         row_index_plus_one) +
                          " for column " + cd->columnName;
                      throw std::runtime_error(msg);
                    }
                  } else {
                    // extract geometry directly from WKT
                    if (!Geospatial::GeoTypesFactory::getGeoColumns(
                            std::string(geo_string),
                            import_ti,
                            coords,
                            bounds,
                            ring_sizes,
                            poly_rings,
                            PROMOTE_POLYGON_TO_MULTIPOLYGON)) {
                      std::string msg = "Failed to extract valid geometry from row " +
                                        std::to_string(first_row_index_this_buffer +
                                                       row_index_plus_one) +
                                        " for column " + cd->columnName;
                      throw std::runtime_error(msg);
                    }
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
                }

                // assign render group?
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

              // import extracted geo
              Importer::set_geo_physical_import_buffer(importer->getCatalog(),
                                                       cd,
                                                       import_buffers,
                                                       col_idx,
                                                       coords,
                                                       bounds,
                                                       ring_sizes,
                                                       poly_rings,
                                                       render_group);

              // skip remaining physical columns
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
                     << ". Row discarded. Data: " << shared::printContainer(row);
        }
      };

      if (copy_params.geo_explode_collections) {
        // explode and import
        auto const [collection_col_idx, collection_child_type, collection_col_name] =
            explode_collections_step1(col_descs);
        // pull out the collection WKT or WKB hex
        CHECK_LT(collection_col_idx, (int)row.size()) << "column index out of range";
        auto const& collection_geo_string = row[collection_col_idx];
        // convert to OGR
        OGRGeometry* ogr_geometry = nullptr;
        ScopeGuard destroy_ogr_geometry = [&] {
          if (ogr_geometry) {
            OGRGeometryFactory::destroyGeometry(ogr_geometry);
          }
        };
        ogr_geometry = Geospatial::GeoTypesFactory::createOGRGeometry(
            std::string(collection_geo_string));
        // do the explode and import
        us = explode_collections_step2(ogr_geometry,
                                       collection_child_type,
                                       collection_col_name,
                                       first_row_index_this_buffer + row_index_plus_one,
                                       execute_import_row);
      } else {
        // import non-collection row just once
        us = measure<std::chrono::microseconds>::execution(
            [&] { execute_import_row(nullptr); });
      }
      total_str_to_val_time_us += us;
    }  // end thread
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

  // we create this on the fly based on the first feature's SR
  std::unique_ptr<OGRCoordinateTransformation> coordinate_transformation;

  for (size_t iFeature = 0; iFeature < numFeatures; iFeature++) {
    if (!features[iFeature]) {
      continue;
    }

    // get this feature's geometry
    OGRGeometry* pGeometry = features[iFeature]->GetGeometryRef();
    if (pGeometry) {
      // for geodatabase, we need to consider features with no geometry
      // as we still want to create a table, even if it has no geo column

      // transform it
      // avoid GDAL error if not transformable
      auto geometry_sr = pGeometry->getSpatialReference();
      if (geometry_sr) {
        // create an OGRCoordinateTransformation (CT) on the fly
        // we must assume that all geo in this file will have
        // the same source SR, so the CT will be valid for all
        // transforming to a reusable CT is faster than to an SR
        if (coordinate_transformation == nullptr) {
          coordinate_transformation.reset(
              OGRCreateCoordinateTransformation(geometry_sr, poGeographicSR));
          if (coordinate_transformation == nullptr) {
            throw std::runtime_error(
                "Failed to create a GDAL CoordinateTransformation for incoming geo");
          }
        }
        pGeometry->transform(coordinate_transformation.get());
      }
    }

    //
    // lambda for importing a feature (perhaps multiple times if exploding a collection)
    //

    auto execute_import_feature = [&](OGRGeometry* import_geometry) {
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
            import_buffers[col_idx]->add_value(
                cd, copy_params.null_str, true, copy_params);
            ++col_idx;

            // the data we now need to extract for the other columns
            std::vector<double> coords;
            std::vector<double> bounds;
            std::vector<int> ring_sizes;
            std::vector<int> poly_rings;
            int render_group = 0;

            // extract it
            SQLTypeInfo import_ti{col_ti};
            bool is_null_geo = !import_geometry;
            if (is_null_geo) {
              if (col_ti.get_notnull()) {
                throw std::runtime_error("NULL geo for column " + cd->columnName);
              }
              Geospatial::GeoTypesFactory::getNullGeoColumns(
                  import_ti,
                  coords,
                  bounds,
                  ring_sizes,
                  poly_rings,
                  PROMOTE_POLYGON_TO_MULTIPOLYGON);
            } else {
              if (!Geospatial::GeoTypesFactory::getGeoColumns(
                      import_geometry,
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
                      "Imported geometry doesn't match the type of column " +
                      cd->columnName);
                }
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
            if (!is_null_geo) {
              std::vector<uint8_t> compressed_coords =
                  Geospatial::compress_coords(coords, col_ti);
              for (auto cc : compressed_coords) {
                TDatum td_byte;
                td_byte.val.int_val = cc;
                td_coord_data.push_back(td_byte);
              }
            }
            TDatum tdd_coords;
            tdd_coords.val.arr_val = td_coord_data;
            tdd_coords.is_null = is_null_geo;
            import_buffers[col_idx]->add_value(cd_coords, tdd_coords, false);
            ++col_idx;

            if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
              // Create ring_sizes array value and add it to the physical column
              ++cd_it;
              auto cd_ring_sizes = *cd_it;
              std::vector<TDatum> td_ring_sizes;
              if (!is_null_geo) {
                for (auto ring_size : ring_sizes) {
                  TDatum td_ring_size;
                  td_ring_size.val.int_val = ring_size;
                  td_ring_sizes.push_back(td_ring_size);
                }
              }
              TDatum tdd_ring_sizes;
              tdd_ring_sizes.val.arr_val = td_ring_sizes;
              tdd_ring_sizes.is_null = is_null_geo;
              import_buffers[col_idx]->add_value(cd_ring_sizes, tdd_ring_sizes, false);
              ++col_idx;
            }

            if (col_type == kMULTIPOLYGON) {
              // Create poly_rings array value and add it to the physical column
              ++cd_it;
              auto cd_poly_rings = *cd_it;
              std::vector<TDatum> td_poly_rings;
              if (!is_null_geo) {
                for (auto num_rings : poly_rings) {
                  TDatum td_num_rings;
                  td_num_rings.val.int_val = num_rings;
                  td_poly_rings.push_back(td_num_rings);
                }
              }
              TDatum tdd_poly_rings;
              tdd_poly_rings.val.arr_val = td_poly_rings;
              tdd_poly_rings.is_null = is_null_geo;
              import_buffers[col_idx]->add_value(cd_poly_rings, tdd_poly_rings, false);
              ++col_idx;
            }

            if (col_type == kLINESTRING || col_type == kPOLYGON ||
                col_type == kMULTIPOLYGON) {
              // Create bounds array value and add it to the physical column
              ++cd_it;
              auto cd_bounds = *cd_it;
              std::vector<TDatum> td_bounds_data;
              if (!is_null_geo) {
                for (auto b : bounds) {
                  TDatum td_double;
                  td_double.val.real_val = b;
                  td_bounds_data.push_back(td_double);
                }
              }
              TDatum tdd_bounds;
              tdd_bounds.val.arr_val = td_bounds_data;
              tdd_bounds.is_null = is_null_geo;
              import_buffers[col_idx]->add_value(cd_bounds, tdd_bounds, false);
              ++col_idx;
            }

            if (col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
              // Create render_group value and add it to the physical column
              ++cd_it;
              auto cd_render_group = *cd_it;
              TDatum td_render_group;
              td_render_group.val.int_val = render_group;
              td_render_group.is_null = is_null_geo;
              import_buffers[col_idx]->add_value(cd_render_group, td_render_group, false);
              ++col_idx;
            }
          } else {
            // regular column
            // pull from GDAL metadata
            auto const cit = columnNameToSourceNameMap.find(cd->columnName);
            CHECK(cit != columnNameToSourceNameMap.end());
            auto const& field_name = cit->second;

            auto const fit = fieldNameToIndexMap.find(field_name);
            CHECK(fit != fieldNameToIndexMap.end());
            auto const& field_index = fit->second;
            CHECK(field_index < fieldNameToIndexMap.size());

            auto const& feature = features[iFeature];

            auto field_defn = feature->GetFieldDefnRef(field_index);
            CHECK(field_defn);

            // OGRFeature::GetFieldAsString() can only return 80 characters
            // so for array columns, we are obliged to fetch the actual values
            // and construct the concatenated string ourselves

            std::string value_string;
            int array_index = 0, array_size = 0;

            auto stringify_numeric_list = [&](auto* values) {
              value_string = "{";
              while (array_index < array_size) {
                auto separator = (array_index > 0) ? "," : "";
                value_string += separator + std::to_string(values[array_index]);
                array_index++;
              }
              value_string += "}";
            };

            auto field_type = field_defn->GetType();
            switch (field_type) {
              case OFTInteger:
              case OFTInteger64:
              case OFTReal:
              case OFTString:
              case OFTBinary:
              case OFTDate:
              case OFTTime:
              case OFTDateTime: {
                value_string = feature->GetFieldAsString(field_index);
              } break;
              case OFTIntegerList: {
                auto* values = feature->GetFieldAsIntegerList(field_index, &array_size);
                stringify_numeric_list(values);
              } break;
              case OFTInteger64List: {
                auto* values = feature->GetFieldAsInteger64List(field_index, &array_size);
                stringify_numeric_list(values);
              } break;
              case OFTRealList: {
                auto* values = feature->GetFieldAsDoubleList(field_index, &array_size);
                stringify_numeric_list(values);
              } break;
              case OFTStringList: {
                auto** array_of_strings = feature->GetFieldAsStringList(field_index);
                value_string = "{";
                if (array_of_strings) {
                  while (auto* this_string = array_of_strings[array_index]) {
                    auto separator = (array_index > 0) ? "," : "";
                    value_string += separator + std::string(this_string);
                    array_index++;
                  }
                }
                value_string += "}";
              } break;
              default:
                throw std::runtime_error("Unsupported geo file field type (" +
                                         std::to_string(static_cast<int>(field_type)) +
                                         ")");
            }

            static CopyParams default_copy_params;
            import_buffers[col_idx]->add_value(
                cd, value_string, false, default_copy_params);
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
    };

    if (pGeometry && copy_params.geo_explode_collections) {
      // explode and import
      auto const [collection_idx_type_name, collection_child_type, collection_col_name] =
          explode_collections_step1(col_descs);
      explode_collections_step2(pGeometry,
                                collection_child_type,
                                collection_col_name,
                                firstFeature + iFeature + 1,
                                execute_import_feature);
    } else {
      // import non-collection or null feature just once
      execute_import_feature(pGeometry);
    }
  }  // end features

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
  const int logical_size = ti.is_string() ? ti.get_size() : ti.get_logical_size();
  switch (logical_size) {
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
      LOG(FATAL) << "Unexpected size for shard key: " << logical_size;
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
  CHECK_GT(table_desc_->shardedColumnId, 0);
  int col_idx{0};
  const ColumnDescriptor* shard_col_desc{nullptr};
  for (const auto col_desc : column_descs_) {
    ++col_idx;
    if (col_idx == table_desc_->shardedColumnId) {
      shard_col_desc = col_desc;
      break;
    }
  }
  CHECK(shard_col_desc);
  CHECK_LE(static_cast<size_t>(table_desc_->shardedColumnId), import_buffers.size());
  auto& shard_column_input_buffer = import_buffers[table_desc_->shardedColumnId - 1];
  const auto& shard_col_ti = shard_col_desc->columnType;
  CHECK(shard_col_ti.is_integer() ||
        (shard_col_ti.is_string() && shard_col_ti.get_compression() == kENCODING_DICT) ||
        shard_col_ti.is_time());
  if (shard_col_ti.is_string()) {
    const auto payloads_ptr = shard_column_input_buffer->getStringBuffer();
    CHECK(payloads_ptr);
    shard_column_input_buffer->addDictEncodedString(*payloads_ptr);
  }

  // for each replicated (alter added) columns, number of rows in a shard is
  // inferred from that of the sharding column, not simply evenly distributed.
  const auto shard_tds = catalog_.getPhysicalTablesDescriptors(table_desc_);
  // Here the loop count is overloaded. For normal imports, we loop thru all
  // input values (rows), so the loop count is the number of input rows.
  // For ALTER ADD COLUMN, we replicate one default value to existing rows in
  // all shards, so the loop count is the number of shards.
  const auto loop_count = getReplicating() ? table_desc_->nShards : row_count;
  for (size_t i = 0; i < loop_count; ++i) {
    const size_t shard =
        getReplicating()
            ? i
            : SHARD_FOR_KEY(int_value_at(*shard_column_input_buffer, i), shard_count);
    auto& shard_output_buffers = all_shard_import_buffers[shard];

    // when replicate a column, populate 'rows' to all shards only once
    // and its value is fetch from the first and the single row
    const auto row_index = getReplicating() ? 0 : i;

    for (size_t col_idx = 0; col_idx < import_buffers.size(); ++col_idx) {
      const auto& input_buffer = import_buffers[col_idx];
      const auto& col_ti = input_buffer->getTypeInfo();
      const auto type =
          col_ti.is_decimal() ? decimal_to_int_type(col_ti) : col_ti.get_type();

      // for a replicated (added) column, populate rows_per_shard as per-shard replicate
      // count. and, bypass non-replicated column.
      if (getReplicating()) {
        if (input_buffer->get_replicate_count() > 0) {
          shard_output_buffers[col_idx]->set_replicate_count(
              shard_tds[shard]->fragmenter->getNumRows());
        } else {
          continue;
        }
      }

      switch (type) {
        case kBOOLEAN:
          shard_output_buffers[col_idx]->addBoolean(
              int_value_at(*input_buffer, row_index));
          break;
        case kTINYINT:
          shard_output_buffers[col_idx]->addTinyint(
              int_value_at(*input_buffer, row_index));
          break;
        case kSMALLINT:
          shard_output_buffers[col_idx]->addSmallint(
              int_value_at(*input_buffer, row_index));
          break;
        case kINT:
          shard_output_buffers[col_idx]->addInt(int_value_at(*input_buffer, row_index));
          break;
        case kBIGINT:
          shard_output_buffers[col_idx]->addBigint(
              int_value_at(*input_buffer, row_index));
          break;
        case kFLOAT:
          shard_output_buffers[col_idx]->addFloat(
              float_value_at(*input_buffer, row_index));
          break;
        case kDOUBLE:
          shard_output_buffers[col_idx]->addDouble(
              double_value_at(*input_buffer, row_index));
          break;
        case kTEXT:
        case kVARCHAR:
        case kCHAR: {
          CHECK_LT(row_index, input_buffer->getStringBuffer()->size());
          shard_output_buffers[col_idx]->addString(
              (*input_buffer->getStringBuffer())[row_index]);
          break;
        }
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          shard_output_buffers[col_idx]->addBigint(
              int_value_at(*input_buffer, row_index));
          break;
        case kARRAY:
          if (IS_STRING(col_ti.get_subtype())) {
            CHECK(input_buffer->getStringArrayBuffer());
            CHECK_LT(row_index, input_buffer->getStringArrayBuffer()->size());
            const auto& input_arr = (*(input_buffer->getStringArrayBuffer()))[row_index];
            shard_output_buffers[col_idx]->addStringArray(input_arr);
          } else {
            shard_output_buffers[col_idx]->addArray(
                (*input_buffer->getArrayBuffer())[row_index]);
          }
          break;
        case kPOINT:
        case kLINESTRING:
        case kPOLYGON:
        case kMULTIPOLYGON: {
          CHECK_LT(row_index, input_buffer->getGeoStringBuffer()->size());
          shard_output_buffers[col_idx]->addGeoString(
              (*input_buffer->getGeoStringBuffer())[row_index]);
          break;
        }
        default:
          CHECK(false);
      }
    }
    ++all_shard_row_counts[shard];
    // when replicating a column, row count of a shard == replicate count of the column on
    // the shard
    if (getReplicating()) {
      all_shard_row_counts[shard] = shard_tds[shard]->fragmenter->getNumRows();
    }
  }
}

bool Loader::loadImpl(
    const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
    size_t row_count,
    bool checkpoint) {
  if (load_callback_) {
    auto data_blocks = get_data_block_pointers(import_buffers);
    return load_callback_(import_buffers, data_blocks, row_count);
  }
  if (table_desc_->nShards) {
    std::vector<OneShardBuffers> all_shard_import_buffers;
    std::vector<size_t> all_shard_row_counts;
    const auto shard_tables = catalog_.getPhysicalTablesDescriptors(table_desc_);
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
  return loadToShard(import_buffers, row_count, table_desc_, checkpoint);
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
  std::unique_lock<std::mutex> loader_lock(loader_mutex_);
  // patch insert_data with new column
  if (this->getReplicating()) {
    for (const auto& import_buff : import_buffers) {
      insert_data_.replicate_count = import_buff->get_replicate_count();
    }
  }

  Fragmenter_Namespace::InsertData ins_data(insert_data_);
  ins_data.numRows = row_count;
  bool success = true;

  ins_data.data = get_data_block_pointers(import_buffers);

  for (const auto& import_buffer : import_buffers) {
    ins_data.bypass.push_back(0 == import_buffer->get_replicate_count());
  }

  // release loader_lock so that in InsertOrderFragmenter::insertDat
  // we can have multiple threads sort/shuffle InsertData
  loader_lock.unlock();

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

void Loader::dropColumns(const std::vector<int>& columnIds) {
  std::vector<const TableDescriptor*> table_descs(1, table_desc_);
  if (table_desc_->nShards) {
    table_descs = catalog_.getPhysicalTablesDescriptors(table_desc_);
  }
  for (auto table_desc : table_descs) {
    table_desc->fragmenter->dropColumns(columnIds);
  }
}

void Loader::init(const bool use_catalog_locks) {
  insert_data_.databaseId = catalog_.getCurrentDB().dbId;
  insert_data_.tableId = table_desc_->tableId;
  for (auto cd : column_descs_) {
    insert_data_.columnIds.push_back(cd->columnId);
    if (cd->columnType.get_compression() == kENCODING_DICT) {
      CHECK(cd->columnType.is_string() || cd->columnType.is_string_array());
      const auto dd = use_catalog_locks
                          ? catalog_.getMetadataForDict(cd->columnType.get_comp_param())
                          : catalog_.getMetadataForDictUnlocked(
                                cd->columnType.get_comp_param(), true);
      CHECK(dd);
      dict_map_[cd->columnId] = dd->stringDict.get();
    }
  }
  insert_data_.numRows = 0;
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
  std::string line;
  line.reserve(1 * 1024 * 1024);
  auto end_time = std::chrono::steady_clock::now() +
                  timeout * (boost::istarts_with(file_path, "s3://") ? 3 : 1);
  try {
    while (!feof(p_file)) {
      int c;
      size_t n = 0;
      while (EOF != (c = fgetc(p_file)) && copy_params.line_delim != c) {
        if (n++ >= line.capacity()) {
          break;
        }
        line += c;
      }
      if (0 == n) {
        break;
      }
      // remember the first line, which is possibly a header line, to
      // ignore identical header line(s) in 2nd+ files of a archive;
      // otherwise, 2nd+ header may be mistaken as an all-string row
      // and so be final column types.
      if (line1.empty()) {
        line1 = line;
      } else if (line == line1) {
        line.clear();
        continue;
      }

      raw_data += line;
      raw_data += copy_params.line_delim;
      line.clear();
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
    std::vector<std::unique_ptr<char[]>> tmp_buffers;
    p = import_export::delimited_parser::get_row(
        p, buf_end, buf_end, copy_params, nullptr, row, tmp_buffers, try_single_thread);
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
      std::vector<std::unique_ptr<char[]>> tmp_buffers;
      p = import_export::delimited_parser::get_row(
          p, buf_end, buf_end, copy_params, nullptr, row, tmp_buffers, try_single_thread);
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
      // simple hex blob (two characters per byte, not uu-encode or base64)
      if (str_upper_case.size() >= 10) {
        // match WKB blobs for supported geometry types
        // the first byte specifies if the data is big-endian or little-endian
        // the next four bytes are the geometry type (1 = POINT etc.)
        // @TODO support eWKB, which has extra bits set in the geometry type
        auto first_five_bytes = str_upper_case.substr(0, 10);
        if (first_five_bytes == "0000000001" || first_five_bytes == "0101000000") {
          type = kPOINT;
        } else if (first_five_bytes == "0000000002" || first_five_bytes == "0102000000") {
          type = kLINESTRING;
        } else if (first_five_bytes == "0000000003" || first_five_bytes == "0103000000") {
          type = kPOLYGON;
        } else if (first_five_bytes == "0000000006" || first_five_bytes == "0106000000") {
          type = kMULTIPOLYGON;
        } else {
          // unsupported WKB type
          return type;
        }
      } else {
        // too short to be WKB
        return type;
      }
    }
  }

  // check for time types
  if (type == kTEXT) {
    // @TODO
    // make these tests more robust so they don't match stuff they should not
    char* buf;
    buf = try_strptimes(str.c_str(),
                        {"%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%y", "%d/%b/%Y"});
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
  switch (copy_params.has_header) {
    case import_export::ImportHeaderRow::AUTODETECT:
      has_headers = detect_headers(head_types, best_sqltypes);
      if (has_headers) {
        copy_params.has_header = import_export::ImportHeaderRow::HAS_HEADER;
      } else {
        copy_params.has_header = import_export::ImportHeaderRow::NO_HEADER;
      }
      break;
    case import_export::ImportHeaderRow::NO_HEADER:
      has_headers = false;
      break;
    case import_export::ImportHeaderRow::HAS_HEADER:
      has_headers = true;
      break;
  }
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
    for (size_t col_idx = 0; col_idx < row->size() && col_idx < num_cols; col_idx++) {
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
    if (has_headers && i < raw_rows[0].size()) {
      headers[i] = raw_rows[0][i];
    } else {
      headers[i] = "column_" + std::to_string(i + 1);
    }
  }
  return headers;
}

void Importer::load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                    size_t row_count) {
  if (!loader->loadNoCheckpoint(import_buffers, row_count)) {
    load_failed = true;
  }
}

void Importer::checkpoint(
    const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs) {
  if (loader->getTableDesc()->storageType != StorageType::FOREIGN_TABLE) {
    if (load_failed) {
      // rollback to starting epoch - undo all the added records
      loader->setTableEpochs(table_epochs);
    } else {
      loader->checkpoint();
    }
  }

  if (loader->getTableDesc()->persistenceLevel ==
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
}

ImportStatus DataStreamSink::archivePlumber() {
  // in generalized importing scheme, reaching here file_path may
  // contain a file path, a url or a wildcard of file paths.
  // see CopyTableStmt::execute.
  auto file_paths = omnisci::glob(file_path);
  if (file_paths.size() == 0) {
    file_paths.push_back(file_path);
  }

  // sum up sizes of all local files -- only for local files. if
  // file_path is a s3 url, sizes will be obtained via S3Archive.
  for (const auto& file_path : file_paths) {
    total_file_size += get_filesize(file_path);
  }

#ifdef ENABLE_IMPORT_PARQUET
  // s3 parquet goes different route because the files do not use libarchive
  // but parquet api, and they need to landed like .7z files.
  //
  // note: parquet must be explicitly specified by a WITH parameter "parquet='true'",
  //       because for example spark sql users may specify a output url w/o file
  //       extension like this:
  //                df.write
  //                  .mode("overwrite")
  //                  .parquet("s3://bucket/folder/parquet/mydata")
  //       without the parameter, it means plain or compressed csv files.
  // note: .ORC and AVRO files should follow a similar path to Parquet?
  if (copy_params.file_type == FileType::PARQUET) {
    import_parquet(file_paths);
  } else
#endif
  {
    import_compressed(file_paths);
  }
  return import_status;
}

#ifdef ENABLE_IMPORT_PARQUET
inline auto open_parquet_table(const std::string& file_path,
                               std::shared_ptr<arrow::io::ReadableFile>& infile,
                               std::unique_ptr<parquet::arrow::FileReader>& reader,
                               std::shared_ptr<arrow::Table>& table) {
  using namespace parquet::arrow;
  auto file_result = arrow::io::ReadableFile::Open(file_path);
  PARQUET_THROW_NOT_OK(file_result.status());
  infile = file_result.ValueOrDie();

  PARQUET_THROW_NOT_OK(OpenFile(infile, arrow::default_memory_pool(), &reader));
  PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
  const auto num_row_groups = reader->num_row_groups();
  const auto num_columns = table->num_columns();
  const auto num_rows = table->num_rows();
  LOG(INFO) << "File " << file_path << " has " << num_rows << " rows and " << num_columns
            << " columns in " << num_row_groups << " groups.";
  return std::make_tuple(num_row_groups, num_columns, num_rows);
}

void Detector::import_local_parquet(const std::string& file_path) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  std::unique_ptr<parquet::arrow::FileReader> reader;
  std::shared_ptr<arrow::Table> table;
  int num_row_groups, num_columns;
  int64_t num_rows;
  std::tie(num_row_groups, num_columns, num_rows) =
      open_parquet_table(file_path, infile, reader, table);
  // make up header line if not yet
  if (0 == raw_data.size()) {
    copy_params.has_header = ImportHeaderRow::HAS_HEADER;
    copy_params.line_delim = '\n';
    copy_params.delimiter = ',';
    // must quote values to skip any embedded delimiter
    copy_params.quoted = true;
    copy_params.quote = '"';
    copy_params.escape = '"';
    for (int c = 0; c < num_columns; ++c) {
      if (c) {
        raw_data += copy_params.delimiter;
      }
      raw_data += table->ColumnNames().at(c);
    }
    raw_data += copy_params.line_delim;
  }
  // make up raw data... rowwize...
  const ColumnDescriptor cd;
  for (int g = 0; g < num_row_groups; ++g) {
    // data is columnwise
    std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
    std::vector<VarValue (*)(const Array&, const int64_t)> getters;
    arrays.resize(num_columns);
    for (int c = 0; c < num_columns; ++c) {
      PARQUET_THROW_NOT_OK(reader->RowGroup(g)->Column(c)->Read(&arrays[c]));
      for (auto chunk : arrays[c]->chunks()) {
        getters.push_back(value_getter(*chunk, nullptr, nullptr));
      }
    }
    for (int r = 0; r < num_rows; ++r) {
      for (int c = 0; c < num_columns; ++c) {
        std::vector<std::string> buffer;
        for (auto chunk : arrays[c]->chunks()) {
          DataBuffer<std::string> data(&cd, *chunk, buffer, nullptr);
          if (c) {
            raw_data += copy_params.delimiter;
          }
          if (!chunk->IsNull(r)) {
            raw_data += copy_params.quote;
            raw_data += boost::replace_all_copy(
                (data << getters[c](*chunk, r)).buffer.front(), "\"", "\"\"");
            raw_data += copy_params.quote;
          }
        }
      }
      raw_data += copy_params.line_delim;
      if (++import_status.rows_completed >= 10000) {
        // as if load truncated
        import_status.load_truncated = true;
        load_failed = true;
        return;
      }
    }
  }
}

template <typename DATA_TYPE>
auto TypedImportBuffer::del_values(
    std::vector<DATA_TYPE>& buffer,
    import_export::BadRowsTracker* const bad_rows_tracker) {
  const auto old_size = buffer.size();
  // erase backward to minimize memory movement overhead
  for (auto rit = bad_rows_tracker->rows.crbegin(); rit != bad_rows_tracker->rows.crend();
       ++rit) {
    buffer.erase(buffer.begin() + *rit);
  }
  return std::make_tuple(old_size, buffer.size());
}

auto TypedImportBuffer::del_values(const SQLTypes type,
                                   BadRowsTracker* const bad_rows_tracker) {
  switch (type) {
    case kBOOLEAN:
      return del_values(*bool_buffer_, bad_rows_tracker);
    case kTINYINT:
      return del_values(*tinyint_buffer_, bad_rows_tracker);
    case kSMALLINT:
      return del_values(*smallint_buffer_, bad_rows_tracker);
    case kINT:
      return del_values(*int_buffer_, bad_rows_tracker);
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return del_values(*bigint_buffer_, bad_rows_tracker);
    case kFLOAT:
      return del_values(*float_buffer_, bad_rows_tracker);
    case kDOUBLE:
      return del_values(*double_buffer_, bad_rows_tracker);
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return del_values(*string_buffer_, bad_rows_tracker);
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      return del_values(*geo_string_buffer_, bad_rows_tracker);
    case kARRAY:
      return del_values(*array_buffer_, bad_rows_tracker);
    default:
      throw std::runtime_error("Invalid Type");
  }
}

void Importer::import_local_parquet(const std::string& file_path) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  std::unique_ptr<parquet::arrow::FileReader> reader;
  std::shared_ptr<arrow::Table> table;
  int num_row_groups, num_columns;
  int64_t nrow_in_file;
  std::tie(num_row_groups, num_columns, nrow_in_file) =
      open_parquet_table(file_path, infile, reader, table);
  // column_list has no $deleted
  const auto& column_list = get_column_descs();
  // for now geo columns expect a wkt or wkb hex string
  std::vector<const ColumnDescriptor*> cds;
  int num_physical_cols = 0;
  for (auto& cd : column_list) {
    cds.push_back(cd);
    num_physical_cols += cd->columnType.get_physical_cols();
  }
  arrow_throw_if(num_columns != (int)(column_list.size() - num_physical_cols),
                 "Unmatched numbers of columns in parquet file " + file_path + ": " +
                     std::to_string(num_columns) + " columns in file vs " +
                     std::to_string(column_list.size() - num_physical_cols) +
                     " columns in table.");
  // slice each group to import slower columns faster, eg. geo or string
  max_threads = copy_params.threads
                    ? copy_params.threads
                    : std::min(static_cast<size_t>(cpu_threads()), g_max_import_threads);
  VLOG(1) << "Parquet import # threads: " << max_threads;
  const int num_slices = std::max<decltype(max_threads)>(max_threads, num_columns);
  // init row estimate for this file
  const auto filesize = get_filesize(file_path);
  size_t nrow_completed{0};
  file_offsets.push_back(0);
  // map logic column index to physical column index
  auto get_physical_col_idx = [&cds](const int logic_col_idx) -> auto {
    int physical_col_idx = 0;
    for (int i = 0; i < logic_col_idx; ++i) {
      physical_col_idx += 1 + cds[physical_col_idx]->columnType.get_physical_cols();
    }
    return physical_col_idx;
  };
  // load a file = nested iteration of row groups, row slices and logical columns
  auto ms_load_a_file = measure<>::execution([&]() {
    for (int row_group = 0; row_group < num_row_groups && !load_failed; ++row_group) {
      // a sliced row group will be handled like a (logic) parquet file, with
      // a entirely clean set of bad_rows_tracker, import_buffers_vec, ... etc
      import_buffers_vec.resize(num_slices);
      for (int slice = 0; slice < num_slices; slice++) {
        import_buffers_vec[slice].clear();
        for (const auto cd : cds) {
          import_buffers_vec[slice].emplace_back(
              new TypedImportBuffer(cd, loader->getStringDict(cd)));
        }
      }
      /*
       * A caveat here is: Parquet files or arrow data is imported column wise.
       * Unlike importing row-wise csv files, a error on any row of any column
       * forces to give up entire row group of all columns, unless there is a
       * sophisticated method to trace erroneous rows in individual columns so
       * that we can union bad rows and drop them from corresponding
       * import_buffers_vec; otherwise, we may exceed maximum number of
       * truncated rows easily even with very sparse errors in the files.
       */
      std::vector<BadRowsTracker> bad_rows_trackers(num_slices);
      for (size_t slice = 0; slice < bad_rows_trackers.size(); ++slice) {
        auto& bad_rows_tracker = bad_rows_trackers[slice];
        bad_rows_tracker.file_name = file_path;
        bad_rows_tracker.row_group = slice;
        bad_rows_tracker.importer = this;
      }
      // process arrow arrays to import buffers
      for (int logic_col_idx = 0; logic_col_idx < num_columns; ++logic_col_idx) {
        const auto physical_col_idx = get_physical_col_idx(logic_col_idx);
        const auto cd = cds[physical_col_idx];
        std::shared_ptr<arrow::ChunkedArray> array;
        PARQUET_THROW_NOT_OK(
            reader->RowGroup(row_group)->Column(logic_col_idx)->Read(&array));
        const size_t array_size = array->length();
        const size_t slice_size = (array_size + num_slices - 1) / num_slices;
        ThreadController_NS::SimpleThreadController<void> thread_controller(num_slices);
        for (int slice = 0; slice < num_slices; ++slice) {
          thread_controller.startThread([&, slice] {
            const auto slice_offset = slice % num_slices;
            ArraySliceRange slice_range(
                std::min<size_t>((slice_offset + 0) * slice_size, array_size),
                std::min<size_t>((slice_offset + 1) * slice_size, array_size));
            auto& bad_rows_tracker = bad_rows_trackers[slice];
            auto& import_buffer = import_buffers_vec[slice][physical_col_idx];
            import_buffer->import_buffers = &import_buffers_vec[slice];
            import_buffer->col_idx = physical_col_idx + 1;
            for (auto chunk : array->chunks()) {
              import_buffer->add_arrow_values(
                  cd, *chunk, false, slice_range, &bad_rows_tracker);
            }
          });
        }
        thread_controller.finish();
      }
      std::vector<size_t> nrow_in_slice_raw(num_slices);
      std::vector<size_t> nrow_in_slice_successfully_loaded(num_slices);
      // trim bad rows from import buffers
      for (int logic_col_idx = 0; logic_col_idx < num_columns; ++logic_col_idx) {
        const auto physical_col_idx = get_physical_col_idx(logic_col_idx);
        const auto cd = cds[physical_col_idx];
        for (int slice = 0; slice < num_slices; ++slice) {
          auto& bad_rows_tracker = bad_rows_trackers[slice];
          auto& import_buffer = import_buffers_vec[slice][physical_col_idx];
          std::tie(nrow_in_slice_raw[slice], nrow_in_slice_successfully_loaded[slice]) =
              import_buffer->del_values(cd->columnType.get_type(), &bad_rows_tracker);
        }
      }
      // flush slices of this row group to chunks
      for (int slice = 0; slice < num_slices; ++slice) {
        load(import_buffers_vec[slice], nrow_in_slice_successfully_loaded[slice]);
      }
      // update import stats
      const auto nrow_original =
          std::accumulate(nrow_in_slice_raw.begin(), nrow_in_slice_raw.end(), 0);
      const auto nrow_imported =
          std::accumulate(nrow_in_slice_successfully_loaded.begin(),
                          nrow_in_slice_successfully_loaded.end(),
                          0);
      const auto nrow_dropped = nrow_original - nrow_imported;
      LOG(INFO) << "row group " << row_group << ": add " << nrow_imported
                << " rows, drop " << nrow_dropped << " rows.";
      mapd_lock_guard<mapd_shared_mutex> write_lock(status_mutex);
      import_status.rows_completed += nrow_imported;
      import_status.rows_rejected += nrow_dropped;
      if (import_status.rows_rejected > copy_params.max_reject) {
        import_status.load_truncated = true;
        load_failed = true;
        LOG(ERROR) << "Maximum (" << copy_params.max_reject
                   << ") rows rejected exceeded. Halting load.";
      }
      // row estimate
      std::unique_lock<std::mutex> lock(file_offsets_mutex);
      nrow_completed += nrow_imported;
      file_offsets.back() =
          nrow_in_file ? (float)filesize * nrow_completed / nrow_in_file : 0;
      // sum up current total file offsets
      const auto total_file_offset =
          std::accumulate(file_offsets.begin(), file_offsets.end(), 0);
      // estimate number of rows per current total file offset
      if (total_file_offset) {
        import_status.rows_estimated =
            (float)total_file_size / total_file_offset * import_status.rows_completed;
        VLOG(3) << "rows_completed " << import_status.rows_completed
                << ", rows_estimated " << import_status.rows_estimated
                << ", total_file_size " << total_file_size << ", total_file_offset "
                << total_file_offset;
      }
    }
  });
  LOG(INFO) << "Import " << nrow_in_file << " rows of parquet file " << file_path
            << " took " << (double)ms_load_a_file / 1000.0 << " secs";
}

void DataStreamSink::import_parquet(std::vector<std::string>& file_paths) {
  auto importer = dynamic_cast<Importer*>(this);
  auto table_epochs = importer ? importer->getLoader()->getTableEpochs()
                               : std::vector<Catalog_Namespace::TableEpochInfo>{};
  try {
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
                                           copy_params.s3_endpoint,
                                           copy_params.plain_text));
        us3arch->init_for_read();
        total_file_size += us3arch->get_total_file_size();
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
          import_local_parquet(file_path);
          if (us3arch) {
            us3arch->vacuum(objkey);
          }
        } catch (...) {
          if (us3arch) {
            us3arch->vacuum(objkey);
          }
          throw;
        }
        if (import_status.load_truncated) {
          break;
        }
      }
    }
    // rethrow any exception happened herebefore
    if (teptr) {
      std::rethrow_exception(teptr);
    }
  } catch (...) {
    load_failed = true;
    if (!import_status.load_truncated) {
      throw;
    }
  }

  if (importer) {
    importer->checkpoint(table_epochs);
  }
}
#endif  // ENABLE_IMPORT_PARQUET

void DataStreamSink::import_compressed(std::vector<std::string>& file_paths) {
#ifdef _MSC_VER
  throw std::runtime_error("CSV Import not yet supported on Windows.");
#else
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
                                      copy_params.s3_endpoint,
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
#if 0   // TODO(ppan): implement and enable any other archive class
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
            if (copy_params.has_header != import_export::ImportHeaderRow::NO_HEADER &&
                just_saw_archive_header && (first_text_header_skipped || !is_detecting)) {
              while (size2-- > 0) {
                if (*buf2++ == copy_params.line_delim) {
                  break;
                }
              }
              if (size2 <= 0) {
                LOG(WARNING) << "No line delimiter in block." << std::endl;
              } else {
                just_saw_archive_header = false;
                first_text_header_skipped = true;
              }
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
        mapd_shared_lock<mapd_shared_mutex> read_lock(status_mutex);
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
#endif
}

ImportStatus Importer::import() {
  return DataStreamSink::archivePlumber();
}

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
    max_threads = std::min(static_cast<size_t>(sysconf(_SC_NPROCESSORS_CONF)),
                           g_max_import_threads);
  } else {
    max_threads = static_cast<size_t>(copy_params.threads);
  }
  VLOG(1) << "Delimited import # threads: " << max_threads;

  // deal with small files
  size_t alloc_size = copy_params.buffer_size;
  if (!decompressed && file_size < alloc_size) {
    alloc_size = file_size;
  }

  for (size_t i = 0; i < max_threads; i++) {
    import_buffers_vec.emplace_back();
    for (const auto cd : loader->get_column_descs()) {
      import_buffers_vec[i].emplace_back(
          std::make_unique<TypedImportBuffer>(cd, loader->getStringDict(cd)));
    }
  }

  auto scratch_buffer = std::make_unique<char[]>(alloc_size);
  size_t current_pos = 0;
  size_t end_pos;
  size_t begin_pos = 0;

  (void)fseek(p_file, current_pos, SEEK_SET);
  size_t size =
      fread(reinterpret_cast<void*>(scratch_buffer.get()), 1, alloc_size, p_file);

  // make render group analyzers for each poly column
  ColumnIdToRenderGroupAnalyzerMapType columnIdToRenderGroupAnalyzerMap;
  if (copy_params.geo_assign_render_groups) {
    auto columnDescriptors = loader->getCatalog().getAllColumnMetadataForTable(
        loader->getTableDesc()->tableId, false, false, false);
    for (auto cd : columnDescriptors) {
      SQLTypes ct = cd->columnType.get_type();
      if (ct == kPOLYGON || ct == kMULTIPOLYGON) {
        auto rga = std::make_shared<RenderGroupAnalyzer>();
        rga->seedFromExistingTableContents(loader, cd->columnName);
        columnIdToRenderGroupAnalyzerMap[cd->columnId] = rga;
      }
    }
  }

  ChunkKey chunkKey = {loader->getCatalog().getCurrentDB().dbId,
                       loader->getTableDesc()->tableId};
  auto table_epochs = loader->getTableEpochs();
  {
    std::list<std::future<ImportStatus>> threads;

    // use a stack to track thread_ids which must not overlap among threads
    // because thread_id is used to index import_buffers_vec[]
    std::stack<size_t> stack_thread_ids;
    for (size_t i = 0; i < max_threads; i++) {
      stack_thread_ids.push(i);
    }
    // added for true row index on error
    size_t first_row_index_this_buffer = 0;

    while (size > 0) {
      unsigned int num_rows_this_buffer = 0;
      CHECK(scratch_buffer);
      end_pos = delimited_parser::find_row_end_pos(alloc_size,
                                                   scratch_buffer,
                                                   size,
                                                   copy_params,
                                                   first_row_index_this_buffer,
                                                   num_rows_this_buffer,
                                                   p_file);

      // unput residual
      int nresidual = size - end_pos;
      std::unique_ptr<char[]> unbuf;
      if (nresidual > 0) {
        unbuf = std::make_unique<char[]>(nresidual);
        memcpy(unbuf.get(), scratch_buffer.get() + end_pos, nresidual);
      }

      // get a thread_id not in use
      auto thread_id = stack_thread_ids.top();
      stack_thread_ids.pop();
      // LOG(INFO) << " stack_thread_ids.pop " << thread_id << std::endl;

      threads.push_back(std::async(std::launch::async,
                                   import_thread_delimited,
                                   thread_id,
                                   this,
                                   std::move(scratch_buffer),
                                   begin_pos,
                                   end_pos,
                                   end_pos,
                                   columnIdToRenderGroupAnalyzerMap,
                                   first_row_index_this_buffer));

      first_row_index_this_buffer += num_rows_this_buffer;

      current_pos += end_pos;
      scratch_buffer = std::make_unique<char[]>(alloc_size);
      CHECK(scratch_buffer);
      memcpy(scratch_buffer.get(), unbuf.get(), nresidual);
      size = nresidual +
             fread(scratch_buffer.get() + nresidual, 1, alloc_size - nresidual, p_file);

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
  }

  checkpoint(table_epochs);

  // must set import_status.load_truncated before closing this end of pipe
  // otherwise, the thread on the other end would throw an unwanted 'write()'
  // exception
  mapd_lock_guard<mapd_shared_mutex> write_lock(status_mutex);
  import_status.load_truncated = load_truncated;

  fclose(p_file);
  p_file = nullptr;
  return import_status;
}

void Loader::checkpoint() {
  if (getTableDesc()->persistenceLevel ==
      Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident tables
    getCatalog().checkpointWithAutoRollback(getTableDesc()->tableId);
  }
}

std::vector<Catalog_Namespace::TableEpochInfo> Loader::getTableEpochs() const {
  return getCatalog().getTableEpochs(getCatalog().getCurrentDB().dbId,
                                     getTableDesc()->tableId);
}

void Loader::setTableEpochs(
    const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs) {
  getCatalog().setTableEpochs(getCatalog().getCurrentDB().dbId, table_epochs);
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
  if (copy_params.s3_endpoint.size()) {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Setting AWS_S3_ENDPOINT to '" << copy_params.s3_endpoint << "'";
#endif
    CPLSetConfigOption("AWS_S3_ENDPOINT", copy_params.s3_endpoint.c_str());
  } else {
#if DEBUG_AWS_AUTHENTICATION
    LOG(INFO) << "GDAL: Clearing AWS_S3_ENDPOINT";
#endif
    CPLSetConfigOption("AWS_S3_ENDPOINT", nullptr);
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
  Geospatial::GDAL::init();

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

namespace {

OGRLayer& getLayerWithSpecifiedName(const std::string& geo_layer_name,
                                    const OGRDataSourceUqPtr& poDS,
                                    const std::string& file_name) {
  // get layer with specified name, or default to first layer
  OGRLayer* poLayer = nullptr;
  if (geo_layer_name.size()) {
    poLayer = poDS->GetLayerByName(geo_layer_name.c_str());
    if (poLayer == nullptr) {
      throw std::runtime_error("Layer '" + geo_layer_name + "' not found in " +
                               file_name);
    }
  } else {
    poLayer = poDS->GetLayer(0);
    if (poLayer == nullptr) {
      throw std::runtime_error("No layers found in " + file_name);
    }
  }
  return *poLayer;
}

}  // namespace

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

  OGRLayer& layer =
      getLayerWithSpecifiedName(copy_params.geo_layer_name, poDS, file_name);

  OGRFeatureDefn* poFDefn = layer.GetLayerDefn();
  CHECK(poFDefn);

  // typeof GetFeatureCount() is different between GDAL 1.x (int32_t) and 2.x (int64_t)
  auto nFeats = layer.GetFeatureCount();
  size_t numFeatures =
      std::max(static_cast<decltype(nFeats)>(0),
               std::min(static_cast<decltype(nFeats)>(rowLimit), nFeats));
  for (auto iField = 0; iField < poFDefn->GetFieldCount(); iField++) {
    OGRFieldDefn* poFieldDefn = poFDefn->GetFieldDefn(iField);
    // FIXME(andrewseidl): change this to the faster one used by readVerticesFromGDAL
    metadata.emplace(poFieldDefn->GetNameRef(), std::vector<std::string>(numFeatures));
  }
  metadata.emplace(geo_column_name, std::vector<std::string>(numFeatures));
  layer.ResetReading();
  size_t iFeature = 0;
  while (iFeature < numFeatures) {
    OGRFeatureUqPtr poFeature(layer.GetNextFeature());
    if (!poFeature) {
      break;
    }

    OGRGeometry* poGeometry = poFeature->GetGeometryRef();
    if (poGeometry != nullptr) {
      // validate geom type (again?)
      switch (wkbFlatten(poGeometry->getGeometryType())) {
        case wkbPoint:
        case wkbLineString:
        case wkbPolygon:
        case wkbMultiPolygon:
          break;
        case wkbMultiPoint:
        case wkbMultiLineString:
          // supported if geo_explode_collections is specified
          if (!copy_params.geo_explode_collections) {
            throw std::runtime_error("Unsupported geometry type: " +
                                     std::string(poGeometry->getGeometryName()));
          }
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
      // Interpret binary blobs as byte arrays here
      // but actual import will store NULL as GDAL will not
      // extract the blob (OGRFeature::GetFieldAsString will
      // result in the import buffers having an empty string)
      return std::make_pair(kTINYINT, true);
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

  OGRLayer& layer =
      getLayerWithSpecifiedName(copy_params.geo_layer_name, poDS, file_name);

  layer.ResetReading();
  // TODO(andrewseidl): support multiple features
  OGRFeatureUqPtr poFeature(layer.GetNextFeature());
  if (poFeature == nullptr) {
    throw std::runtime_error("No features found in " + file_name);
  }
  // get fields as regular columns
  OGRFeatureDefn* poFDefn = layer.GetLayerDefn();
  CHECK(poFDefn);
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

    // get GDAL type
    auto ogr_type = wkbFlatten(poGeometry->getGeometryType());

    // if exploding, override any collection type to child type
    if (copy_params.geo_explode_collections) {
      if (ogr_type == wkbMultiPolygon) {
        ogr_type = wkbPolygon;
      } else if (ogr_type == wkbMultiLineString) {
        ogr_type = wkbLineString;
      } else if (ogr_type == wkbMultiPoint) {
        ogr_type = wkbPoint;
      }
    }

    // convert to internal type
    SQLTypes geoType = ogr_to_type(ogr_type);

    // for now, we promote POLYGON to MULTIPOLYGON (unless exploding)
    if (PROMOTE_POLYGON_TO_MULTIPOLYGON && !copy_params.geo_explode_collections) {
      geoType = (geoType == kPOLYGON) ? kMULTIPOLYGON : geoType;
    }

    // build full internal type
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
  Geospatial::GDAL::init();

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
        // a directory that ends with .gdb could be a Geodatabase bundle
        // arguably dangerous to decide this purely by name, but any further
        // validation would be very complex especially at this scope
        if (boost::iends_with(entry_path, ".gdb")) {
          // add the directory as if it was a file and don't recurse into it
          files.push_back(entry_path);
        } else {
          // add subdirectory to be recursed into
          subdirectories.push_back(entry_path);
        }
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
  Geospatial::GDAL::init();

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
std::vector<Importer::GeoFileLayerInfo> Importer::gdalGetLayersInGeoFile(
    const std::string& file_name,
    const CopyParams& copy_params) {
  // lazy init GDAL
  Geospatial::GDAL::init();

  // set authorization tokens
  setGDALAuthorizationTokens(copy_params);

  // prepare to gather layer info
  std::vector<GeoFileLayerInfo> layer_info;

  // open the data set
  OGRDataSourceUqPtr poDS(openGDALDataset(file_name, copy_params));
  if (poDS == nullptr) {
    throw std::runtime_error("openGDALDataset Error: Unable to open geo file " +
                             file_name);
  }

  // enumerate the layers
  for (auto&& poLayer : poDS->GetLayers()) {
    GeoFileLayerContents contents = GeoFileLayerContents::EMPTY;
    // prepare to read this layer
    poLayer->ResetReading();
    // skip layer if empty
    if (poLayer->GetFeatureCount() > 0) {
      // get first feature
      OGRFeatureUqPtr first_feature(poLayer->GetNextFeature());
      CHECK(first_feature);
      // check feature for geometry
      const OGRGeometry* geometry = first_feature->GetGeometryRef();
      if (!geometry) {
        // layer has no geometry
        contents = GeoFileLayerContents::NON_GEO;
      } else {
        // check the geometry type
        const OGRwkbGeometryType geometry_type = geometry->getGeometryType();
        switch (wkbFlatten(geometry_type)) {
          case wkbPoint:
          case wkbLineString:
          case wkbPolygon:
          case wkbMultiPolygon:
            // layer has supported geo
            contents = GeoFileLayerContents::GEO;
            break;
          case wkbMultiPoint:
          case wkbMultiLineString:
            // supported if geo_explode_collections is specified
            contents = copy_params.geo_explode_collections
                           ? GeoFileLayerContents::GEO
                           : GeoFileLayerContents::UNSUPPORTED_GEO;
            break;
          default:
            // layer has unsupported geometry
            contents = GeoFileLayerContents::UNSUPPORTED_GEO;
            break;
        }
      }
    }
    // store info for this layer
    layer_info.emplace_back(poLayer->GetName(), contents);
  }

  // done
  return layer_info;
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

  OGRLayer& layer =
      getLayerWithSpecifiedName(copy_params.geo_layer_name, poDS, file_path);

  // get the number of features in this layer
  size_t numFeatures = layer.GetFeatureCount();

  // build map of metadata field (additional columns) name to index
  // use shared_ptr since we need to pass it to the worker
  FieldNameToIndexMapType fieldNameToIndexMap;
  OGRFeatureDefn* poFDefn = layer.GetLayerDefn();
  CHECK(poFDefn);
  size_t numFields = poFDefn->GetFieldCount();
  for (size_t iField = 0; iField < numFields; iField++) {
    OGRFieldDefn* poFieldDefn = poFDefn->GetFieldDefn(iField);
    fieldNameToIndexMap.emplace(std::make_pair(poFieldDefn->GetNameRef(), iField));
  }

  // the geographic spatial reference we want to put everything in
  OGRSpatialReferenceUqPtr poGeographicSR(new OGRSpatialReference());
  poGeographicSR->importFromEPSG(copy_params.geo_coords_srid);

#if GDAL_VERSION_MAJOR >= 3
  // GDAL 3.x (really Proj.4 6.x) now enforces lat, lon order
  // this results in X and Y being transposed for angle-based
  // coordinate systems. This restores the previous behavior.
  poGeographicSR->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

#if DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT
  // just one "thread"
  max_threads = 1;
#else
  // how many threads to use
  if (copy_params.threads == 0) {
    max_threads = std::min(static_cast<size_t>(sysconf(_SC_NPROCESSORS_CONF)),
                           g_max_import_threads);
  } else {
    max_threads = copy_params.threads;
  }
#endif

  VLOG(1) << "GDAL import # threads: " << max_threads;

  // make an import buffer for each thread
  CHECK_EQ(import_buffers_vec.size(), 0u);
  import_buffers_vec.resize(max_threads);
  for (size_t i = 0; i < max_threads; i++) {
    for (const auto cd : loader->get_column_descs()) {
      import_buffers_vec[i].emplace_back(
          new TypedImportBuffer(cd, loader->getStringDict(cd)));
    }
  }

  // make render group analyzers for each poly column
  ColumnIdToRenderGroupAnalyzerMapType columnIdToRenderGroupAnalyzerMap;
  if (copy_params.geo_assign_render_groups) {
    auto columnDescriptors = loader->getCatalog().getAllColumnMetadataForTable(
        loader->getTableDesc()->tableId, false, false, false);
    for (auto cd : columnDescriptors) {
      SQLTypes ct = cd->columnType.get_type();
      if (ct == kPOLYGON || ct == kMULTIPOLYGON) {
        auto rga = std::make_shared<RenderGroupAnalyzer>();
        rga->seedFromExistingTableContents(loader, cd->columnName);
        columnIdToRenderGroupAnalyzerMap[cd->columnId] = rga;
      }
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
  auto table_epochs = loader->getTableEpochs();

  // reset the layer
  layer.ResetReading();

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
      features[thread_id].emplace_back(layer.GetNextFeature());
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

  checkpoint(table_epochs);

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

  // start with a fresh tree
  _rtree = nullptr;
  _numRenderGroups = 0;

  // get the table descriptor
  const auto& cat = loader->getCatalog();
  if (loader->getTableDesc()->storageType == StorageType::FOREIGN_TABLE) {
    if (DEBUG_RENDER_GROUP_ANALYZER) {
      LOG(INFO) << "DEBUG: Table is a foreign table";
    }
    _rtree = std::make_unique<RTree>();
    CHECK(_rtree);
    return;
  }

  const std::string& tableName = loader->getTableDesc()->tableName;
  const auto td = cat.getMetadataForTable(tableName);
  CHECK(td);
  CHECK(td->fragmenter);

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

    // skip rows with invalid render groups (e.g. EMPTY geometry)
    if (renderGroup < 0) {
      continue;
    }

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

std::vector<std::unique_ptr<TypedImportBuffer>> setup_column_loaders(
    const TableDescriptor* td,
    Loader* loader) {
  CHECK(td);
  auto col_descs = loader->get_column_descs();

  std::vector<std::unique_ptr<TypedImportBuffer>> import_buffers;
  for (auto cd : col_descs) {
    import_buffers.emplace_back(
        std::make_unique<TypedImportBuffer>(cd, loader->getStringDict(cd)));
  }

  return import_buffers;
}

}  // namespace import_export
