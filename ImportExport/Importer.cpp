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
#include "ImportExport/DelimitedParserUtils.h"
#include "Logger/Logger.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/TypePunning.h"
#include "RenderGroupAnalyzer.h"
#include "Shared/DateTimeParser.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/file_path_util.h"
#include "Shared/import_helpers.h"
#include "Shared/likely.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/scope.h"
#include "Shared/thread_count.h"
#include "Utils/ChunkAccessorTable.h"

#include "gen-cpp/OmniSci.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

size_t g_max_import_threads =
    32;  // Max number of default import threads to use (num hardware threads will be used
         // if lower, and can also be explicitly overriden in copy statement with threads
         // option)
size_t g_archive_read_buf_size = 1 << 20;

extern bool g_enable_non_kernel_time_query_interrupt;

inline auto get_filesize(const std::string& file_path) {
  boost::filesystem::path boost_file_path{file_path};
  boost::system::error_code ec;
  const auto filesize = boost::filesystem::file_size(boost_file_path, ec);
  return ec ? 0 : filesize;
}

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

#define DEBUG_TIMING false
#define DEBUG_RENDER_GROUP_ANALYZER 0
#define DEBUG_AWS_AUTHENTICATION 0

#define DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT 0

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
    if (p->isVirtualCol) {
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

ArrayDatum ImporterUtils::composeNullPointCoords(const SQLTypeInfo& coords_ti,
                                                 const SQLTypeInfo& geo_ti) {
  UNREACHABLE();  
  auto modified_ti = coords_ti;
  modified_ti.set_subtype(kDOUBLE);
  return import_export::ImporterUtils::composeNullArray(modified_ti);
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
      std::ostringstream oss;
      oss << "while processing dictionary for column " << getColumnDesc()->columnName
          << " a string was detected too long for encoding, string length = "
          << str.size() << ", first 100 characters are '" << str.substr(0, 100) << "'";
      throw std::runtime_error(oss.str());
    }
    string_view_vec.push_back(str);
  }
  try {
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
  } catch (std::exception& e) {
    std::ostringstream oss;
    oss << "while processing dictionary for column " << getColumnDesc()->columnName
        << " : " << e.what();
    LOG(ERROR) << oss.str();
    throw std::runtime_error(oss.str());
  }
}

void TypedImportBuffer::add_value(const ColumnDescriptor* cd,
                                  const std::string_view val,
                                  const bool is_null,
                                  const CopyParams& copy_params) {
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
    default:
      CHECK(false) << "TypedImportBuffer::pop_value() does not support type " << type;
  }
}

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
  auto f_mark_a_bad_row = [&](const auto row) {
    std::unique_lock<std::mutex> lck(bad_rows_tracker->mutex);
    bad_rows_tracker->rows.insert(row - slice_range.first);
  };
  buffer.reserve(slice_range.second - slice_range.first);
  for (size_t row = slice_range.first; row < slice_range.second; ++row) {
    try {
      *data << (array.IsNull(row) ? nullptr : f_value_getter(array, row));
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
                  // Explicitly checking the item for null because
                  // casting null value (-128) to bool results
                  // incorrect value 1.
                  if (col.data.arr_col[i].nulls[j]) {
                    *p = static_cast<int8_t>(
                        inline_fixed_encoding_null_val(cd->columnType.get_elem_type()));
                  } else {
                    *(bool*)p = static_cast<bool>(col.data.arr_col[i].data.int_col[j]);
                  }
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
                                  const bool is_null) {
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
    default:
      CHECK(false) << "TypedImportBuffer::add_value() does not support type " << type;
  }
}

void TypedImportBuffer::addDefaultValues(const ColumnDescriptor* cd, size_t num_rows) {
  bool is_null = !cd->default_value.has_value();
  CHECK(!(is_null && cd->columnType.get_notnull()));
  const auto type = cd->columnType.get_type();
  auto ti = cd->columnType;
  auto val = cd->default_value.value_or("NULL");
  CopyParams cp;
  switch (type) {
    case kBOOLEAN: {
      if (!is_null) {
        bool_buffer_->resize(num_rows, StringToDatum(val, ti).boolval);
      } else {
        bool_buffer_->resize(num_rows, inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kTINYINT: {
      if (!is_null) {
        tinyint_buffer_->resize(num_rows, StringToDatum(val, ti).tinyintval);
      } else {
        tinyint_buffer_->resize(num_rows, inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kSMALLINT: {
      if (!is_null) {
        smallint_buffer_->resize(num_rows, StringToDatum(val, ti).smallintval);
      } else {
        smallint_buffer_->resize(num_rows,
                                 inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kINT: {
      if (!is_null) {
        int_buffer_->resize(num_rows, StringToDatum(val, ti).intval);
      } else {
        int_buffer_->resize(num_rows, inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kBIGINT: {
      if (!is_null) {
        bigint_buffer_->resize(num_rows, StringToDatum(val, ti).bigintval);
      } else {
        bigint_buffer_->resize(num_rows, inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kDECIMAL:
    case kNUMERIC: {
      if (!is_null) {
        const auto converted_decimal_value = convert_decimal_value_to_scale(
            StringToDatum(val, ti).bigintval, ti, cd->columnType);
        bigint_buffer_->resize(num_rows, converted_decimal_value);
      } else {
        bigint_buffer_->resize(num_rows, inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kFLOAT:
      if (!is_null) {
        float_buffer_->resize(num_rows,
                              static_cast<float>(std::atof(std::string(val).c_str())));
      } else {
        float_buffer_->resize(num_rows, NULL_FLOAT);
      }
      break;
    case kDOUBLE:
      if (!is_null) {
        double_buffer_->resize(num_rows, std::atof(std::string(val).c_str()));
      } else {
        double_buffer_->resize(num_rows, NULL_DOUBLE);
      }
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      if (is_null) {
        string_buffer_->resize(num_rows, "");
      } else {
        if (val.length() > StringDictionary::MAX_STRLEN) {
          throw std::runtime_error("String too long for column " + cd->columnName +
                                   " was " + std::to_string(val.length()) + " max is " +
                                   std::to_string(StringDictionary::MAX_STRLEN));
        }
        string_buffer_->resize(num_rows, val);
      }
      break;
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      if (!is_null) {
        bigint_buffer_->resize(num_rows, StringToDatum(val, ti).bigintval);
      } else {
        bigint_buffer_->resize(num_rows, inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kARRAY: {
      if (IS_STRING(ti.get_subtype())) {
        std::vector<std::string> string_vec;
        // Just parse string array, don't push it to buffer yet as we might throw
        import_export::delimited_parser::parse_string_array(
            std::string(val), cp, string_vec);
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
          string_array_buffer_->resize(num_rows, string_vec);
        } else {
          if (ti.get_size() > 0) {
            // TODO: remove once NULL fixlen arrays are allowed
            throw std::runtime_error("Fixed length array column " + cd->columnName +
                                     " currently cannot accept NULL arrays");
          }
          // TODO: add support for NULL string arrays, replace with addStringArray(),
          //       for now add whatever parseStringArray() outputs for NULLs ("NULL")
          string_array_buffer_->resize(num_rows, string_vec);
        }
      } else {
        if (!is_null) {
          ArrayDatum d = StringToArray(std::string(val), ti, cp);
          if (d.is_null) {  // val could be "NULL"
            array_buffer_->resize(num_rows, NullArray(ti));
          } else {
            if (ti.get_size() > 0 && static_cast<size_t>(ti.get_size()) != d.length) {
              throw std::runtime_error("Fixed length array for column " + cd->columnName +
                                       " has incorrect length: " + std::string(val));
            }
            array_buffer_->resize(num_rows, d);
          }
        } else {
          array_buffer_->resize(num_rows, NullArray(ti));
        }
      }
      break;
    }
    default:
      CHECK(false) << "TypedImportBuffer::addDefaultValues() does not support type "
                   << type;
  }
}

static ImportStatus import_thread_delimited(
    int thread_id,
    Importer* importer,
    std::unique_ptr<char[]> scratch_buffer,
    size_t begin_pos,
    size_t end_pos,
    size_t total_size,
    const ColumnIdToRenderGroupAnalyzerMapType& columnIdToRenderGroupAnalyzerMap,
    size_t first_row_index_this_buffer,
    const Catalog_Namespace::SessionInfo* session_info) {
  ImportStatus thread_import_status;
  int64_t total_get_row_time_us = 0;
  int64_t total_str_to_val_time_us = 0;
  auto query_session = session_info ? session_info->get_session_id() : "";
  CHECK(scratch_buffer);
  auto buffer = scratch_buffer.get();
  auto load_ms = measure<>::execution([]() {});

  thread_import_status.thread_id = thread_id;

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
                                                       try_single_thread,
                                                       true);
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
                                                     try_single_thread,
                                                     true);
      }
      row_index_plus_one++;
      // Each POINT could consume two separate coords instead of a single WKT
      if (row.size() < num_cols || (num_cols + point_cols) < row.size()) {
        thread_import_status.rows_rejected++;
        LOG(ERROR) << "Incorrect Row (expected " << num_cols << " columns, has "
                   << row.size() << "): " << shared::printContainer(row);
        if (thread_import_status.rows_rejected > copy_params.max_reject) {
          break;
        }
        continue;
      }

      //
      // lambda for importing a row (perhaps multiple times if exploding a collection)
      //

      auto execute_import_row = [&]() {
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

            CHECK_EQ(col_ti.get_physical_cols(), 0);
            // not geo

            import_buffers[col_idx]->add_value(cd, row[import_idx], is_null, copy_params);

            // next
            ++import_idx;
            ++col_idx;
          }
          thread_import_status.rows_completed++;
        } catch (const std::exception& e) {
          for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
            import_buffers[col_idx_to_pop]->pop_value();
          }
          thread_import_status.rows_rejected++;
          LOG(ERROR) << "Input exception thrown: " << e.what()
                     << ". Row discarded. Data: " << shared::printContainer(row);
          if (thread_import_status.rows_rejected > copy_params.max_reject) {
            LOG(ERROR) << "Load was cancelled due to max reject rows being reached";
            thread_import_status.load_failed = true;
            thread_import_status.load_msg =
                "Load was cancelled due to max reject rows being reached";
          }
        }
      };  // End of lambda

      // import non-collection row just once
      us = measure<std::chrono::microseconds>::execution(
          [&] { execute_import_row(); });

      if (thread_import_status.load_failed) {
        break;
      }
    }  // end thread
    total_str_to_val_time_us += us;
    if (!thread_import_status.load_failed && thread_import_status.rows_completed > 0) {
      load_ms = measure<>::execution([&]() {
        importer->load(import_buffers, thread_import_status.rows_completed, session_info);
      });
    }
  });  // end execution

  if (DEBUG_TIMING && !thread_import_status.load_failed &&
      thread_import_status.rows_completed > 0) {
    LOG(INFO) << "Thread" << std::this_thread::get_id() << ":"
              << thread_import_status.rows_completed << " rows inserted in "
              << (double)ms / 1000.0 << "sec, Insert Time: " << (double)load_ms / 1000.0
              << "sec, get_row: " << (double)total_get_row_time_us / 1000000.0
              << "sec, str_to_val: " << (double)total_str_to_val_time_us / 1000000.0
              << "sec" << std::endl;
  }

  return thread_import_status;
}

bool Loader::loadNoCheckpoint(
    const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
    size_t row_count,
    const Catalog_Namespace::SessionInfo* session_info) {
  return loadImpl(import_buffers, row_count, false, session_info);
}

bool Loader::load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                  size_t row_count,
                  const Catalog_Namespace::SessionInfo* session_info) {
  return loadImpl(import_buffers, row_count, true, session_info);
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

void Loader::fillShardRow(const size_t row_index,
                          OneShardBuffers& shard_output_buffers,
                          const OneShardBuffers& import_buffers) {
  for (size_t col_idx = 0; col_idx < import_buffers.size(); ++col_idx) {
    const auto& input_buffer = import_buffers[col_idx];
    const auto& col_ti = input_buffer->getTypeInfo();
    const auto type =
        col_ti.is_decimal() ? decimal_to_int_type(col_ti) : col_ti.get_type();

    switch (type) {
      case kBOOLEAN:
        shard_output_buffers[col_idx]->addBoolean(int_value_at(*input_buffer, row_index));
        break;
      case kTINYINT:
        shard_output_buffers[col_idx]->addTinyint(int_value_at(*input_buffer, row_index));
        break;
      case kSMALLINT:
        shard_output_buffers[col_idx]->addSmallint(
            int_value_at(*input_buffer, row_index));
        break;
      case kINT:
        shard_output_buffers[col_idx]->addInt(int_value_at(*input_buffer, row_index));
        break;
      case kBIGINT:
        shard_output_buffers[col_idx]->addBigint(int_value_at(*input_buffer, row_index));
        break;
      case kFLOAT:
        shard_output_buffers[col_idx]->addFloat(float_value_at(*input_buffer, row_index));
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
        shard_output_buffers[col_idx]->addBigint(int_value_at(*input_buffer, row_index));
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
      default:
        CHECK(false);
    }
  }
}

bool Loader::loadImpl(
    const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
    size_t row_count,
    bool checkpoint,
    const Catalog_Namespace::SessionInfo* session_info) {
  if (load_callback_) {
    auto data_blocks = TypedImportBuffer::get_data_block_pointers(import_buffers);
    return load_callback_(import_buffers, data_blocks, row_count);
  }
  return loadToShard(import_buffers, row_count, table_desc_, checkpoint, session_info);
}

std::vector<DataBlockPtr> TypedImportBuffer::get_data_block_pointers(
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
    bool checkpoint,
    const Catalog_Namespace::SessionInfo* session_info) {
  std::unique_lock<std::mutex> loader_lock(loader_mutex_);
  Fragmenter_Namespace::InsertData ins_data(insert_data_);
  ins_data.numRows = row_count;
  bool success = false;
  try {
    ins_data.data = TypedImportBuffer::get_data_block_pointers(import_buffers);
  } catch (std::exception& e) {
    std::ostringstream oss;
    oss << "Exception when loading Table " << shard_table->tableName << ", issue was "
        << e.what();

    LOG(ERROR) << oss.str();
    error_msg_ = oss.str();
    return success;
  }
  if (isAddingColumns()) {
    // when Adding columns we omit any columns except the ones being added
    ins_data.columnIds.clear();
    ins_data.is_default.clear();
    for (auto& buffer : import_buffers) {
      ins_data.columnIds.push_back(buffer->getColumnDesc()->columnId);
      ins_data.is_default.push_back(true);
    }
  } else {
    ins_data.is_default.resize(ins_data.columnIds.size(), false);
  }
  // release loader_lock so that in InsertOrderFragmenter::insertDat
  // we can have multiple threads sort/shuffle InsertData
  loader_lock.unlock();
  success = true;
  {
    try {
      if (checkpoint) {
        shard_table->fragmenter->insertData(ins_data);
      } else {
        shard_table->fragmenter->insertDataNoCheckpoint(ins_data);
      }
    } catch (std::exception& e) {
      std::ostringstream oss;
      oss << "Fragmenter Insert Exception when processing Table  "
          << shard_table->tableName << " issue was " << e.what();

      LOG(ERROR) << oss.str();
      loader_lock.lock();
      error_msg_ = oss.str();
      success = false;
    }
  }
  return success;
}

void Loader::dropColumns(const std::vector<int>& columnIds) {
  table_desc_->fragmenter->dropColumns(columnIds);
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

ImportStatus Detector::importDelimited(
    const std::string& file_path,
    const bool decompressed,
    const Catalog_Namespace::SessionInfo* session_info) {
  // we do not check interrupt status for this detection
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
      ++import_status_.rows_completed;
      if (std::chrono::steady_clock::now() > end_time) {
        if (import_status_.rows_completed > 10000) {
          break;
        }
      }
    }
  } catch (std::exception& e) {
  }

  mapd_lock_guard<mapd_shared_mutex> write_lock(import_mutex_);
  import_status_.load_failed = true;

  fclose(p_file);
  p_file = nullptr;
  return import_status_;
}

void Detector::read_file() {
  // this becomes analogous to Importer::import()
  (void)DataStreamSink::archivePlumber(nullptr);
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
    p = import_export::delimited_parser::get_row(p,
                                                 buf_end,
                                                 buf_end,
                                                 copy_params,
                                                 nullptr,
                                                 row,
                                                 tmp_buffers,
                                                 try_single_thread,
                                                 true);
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
      p = import_export::delimited_parser::get_row(p,
                                                   buf_end,
                                                   buf_end,
                                                   copy_params,
                                                   nullptr,
                                                   row,
                                                   tmp_buffers,
                                                   try_single_thread,
                                                   true);
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


  // check for time types
  if (type == kTEXT) {
    // This won't match unix timestamp, since floats and ints were checked above.
    if (dateTimeParseOptional<kTIME>(str, 0)) {
      type = kTIME;
    } else if (dateTimeParseOptional<kTIMESTAMP>(str, 0)) {
      type = kTIMESTAMP;
    } else if (dateTimeParseOptional<kDATE>(str, 0)) {
      type = kDATE;
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
                    size_t row_count,
                    const Catalog_Namespace::SessionInfo* session_info) {
  if (!loader->loadNoCheckpoint(import_buffers, row_count, session_info)) {
    mapd_lock_guard<mapd_shared_mutex> write_lock(import_mutex_);
    import_status_.load_failed = true;
    import_status_.load_msg = loader->getErrorMessage();
  }
}

void Importer::checkpoint(
    const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs) {
  {
    mapd_lock_guard<mapd_shared_mutex> read_lock(import_mutex_);
    if (import_status_.load_failed) {
      // rollback to starting epoch - undo all the added records
      loader->setTableEpochs(table_epochs);
    } else {
      loader->checkpoint();
    }
  }

  if (loader->getTableDesc()->persistenceLevel ==
      Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident
                                                  // tables
    auto ms = measure<>::execution([&]() {
      mapd_lock_guard<mapd_shared_mutex> write_lock(import_mutex_);
      if (!import_status_.load_failed) {
        for (auto& p : import_buffers_vec[0]) {
          if (!p->stringDictCheckpoint()) {
            LOG(ERROR) << "Checkpointing Dictionary for Column "
                       << p->getColumnDesc()->columnName << " failed.";
            import_status_.load_failed = true;
            import_status_.load_msg = "Dictionary checkpoint failed";
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

ImportStatus DataStreamSink::archivePlumber(
    const Catalog_Namespace::SessionInfo* session_info) {
  // in generalized importing scheme, reaching here file_path may
  // contain a file path, a url or a wildcard of file paths.
  // see CopyTableStmt::execute.

  std::vector<std::string> file_paths;
  try {
    shared::validate_sort_options(copy_params.file_sort_order_by,
                                  copy_params.file_sort_regex);
    file_paths = shared::local_glob_filter_sort_files(file_path,
                                                      copy_params.regex_path_filter,
                                                      copy_params.file_sort_order_by,
                                                      copy_params.file_sort_regex);
  } catch (const shared::FileNotFoundException& e) {
    // After finding no matching files locally, file_path may still be an s3 url
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
    import_parquet(file_paths, session_info);
  } else
#endif
  {
    import_compressed(file_paths, session_info);
  }

  return import_status_;
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

void Detector::import_local_parquet(const std::string& file_path,
                                    const Catalog_Namespace::SessionInfo* session_info) {
  /*Skip interrupt checking in detector*/
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
      mapd_lock_guard<mapd_shared_mutex> write_lock(import_mutex_);
      if (++import_status_.rows_completed >= 10000) {
        // as if load truncated
        import_status_.load_failed = true;
        import_status_.load_msg = "Detector processed 10000 records";
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
    case kARRAY:
      return del_values(*array_buffer_, bad_rows_tracker);
    default:
      throw std::runtime_error("Invalid Type");
  }
}

void Importer::import_local_parquet(const std::string& file_path,
                                    const Catalog_Namespace::SessionInfo* session_info) {
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
  auto query_session = session_info ? session_info->get_session_id() : "";
  auto ms_load_a_file = measure<>::execution([&]() {
    for (int row_group = 0; row_group < num_row_groups; ++row_group) {
      {
        mapd_shared_lock<mapd_shared_mutex> read_lock(import_mutex_);
        if (import_status_.load_failed) {
          break;
        }
      }
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
        load(import_buffers_vec[slice],
             nrow_in_slice_successfully_loaded[slice],
             session_info);
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
      {
        mapd_lock_guard<mapd_shared_mutex> write_lock(import_mutex_);
        import_status_.rows_completed += nrow_imported;
        import_status_.rows_rejected += nrow_dropped;
        if (import_status_.rows_rejected > copy_params.max_reject) {
          import_status_.load_failed = true;
          import_status_.load_msg = "Maximum (" + std::to_string(copy_params.max_reject) +
                                    ") rows rejected exceeded. Halting load.";
          LOG(ERROR) << "Maximum (" << copy_params.max_reject
                     << ") rows rejected exceeded. Halting load.";
        }
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
        import_status_.rows_estimated =
            (float)total_file_size / total_file_offset * import_status_.rows_completed;
        VLOG(3) << "rows_completed " << import_status_.rows_completed
                << ", rows_estimated " << import_status_.rows_estimated
                << ", total_file_size " << total_file_size << ", total_file_offset "
                << total_file_offset;
      }
    }
  });
  LOG(INFO) << "Import " << nrow_in_file << " rows of parquet file " << file_path
            << " took " << (double)ms_load_a_file / 1000.0 << " secs";
}

void DataStreamSink::import_parquet(std::vector<std::string>& file_paths,
                                    const Catalog_Namespace::SessionInfo* session_info) {
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
                                           copy_params.s3_session_token,
                                           copy_params.s3_region,
                                           copy_params.s3_endpoint,
                                           copy_params.plain_text,
                                           copy_params.regex_path_filter,
                                           copy_params.file_sort_order_by,
                                           copy_params.file_sort_regex));
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
          import_local_parquet(file_path, session_info);
          if (us3arch) {
            us3arch->vacuum(objkey);
          }
        } catch (...) {
          if (us3arch) {
            us3arch->vacuum(objkey);
          }
          throw;
        }
        mapd_shared_lock<mapd_shared_mutex> read_lock(import_mutex_);
        if (import_status_.load_failed) {
          break;
        }
      }
    }
    // rethrow any exception happened herebefore
    if (teptr) {
      std::rethrow_exception(teptr);
    }
  } catch (const shared::NoRegexFilterMatchException& e) {
    mapd_lock_guard<mapd_shared_mutex> write_lock(import_mutex_);
    import_status_.load_failed = true;
    import_status_.load_msg = e.what();
    throw e;
  } catch (const std::exception& e) {
    mapd_lock_guard<mapd_shared_mutex> write_lock(import_mutex_);
    import_status_.load_failed = true;
    import_status_.load_msg = e.what();
  }

  if (importer) {
    importer->checkpoint(table_epochs);
  }
}
#endif  // ENABLE_IMPORT_PARQUET

void DataStreamSink::import_compressed(
    std::vector<std::string>& file_paths,
    const Catalog_Namespace::SessionInfo* session_info) {
  // a new requirement is to have one single input stream into
  // Importer::importDelimited, so need to move pipe related
  // stuff to the outmost block.
  int fd[2];
#ifdef _WIN32
  // For some reason when folly is used to create the pipe, reader can
  // read nothing.
  auto pipe_res =
      _pipe(fd, static_cast<unsigned int>(copy_params.buffer_size), _O_BINARY);
#else
  auto pipe_res = pipe(fd);
#endif
  if (pipe_res < 0) {
    throw std::runtime_error(std::string("failed to create a pipe: ") + strerror(errno));
  }
#ifndef _WIN32
  signal(SIGPIPE, SIG_IGN);
#endif

  std::exception_ptr teptr;
  // create a thread to read uncompressed byte stream out of pipe and
  // feed into importDelimited()
  ImportStatus ret1;
  auto th_pipe_reader = std::thread([&]() {
    try {
      // importDelimited will read from FILE* p_file
      if (0 == (p_file = fdopen(fd[0], "r"))) {
        throw std::runtime_error(std::string("failed to open a pipe: ") +
                                 strerror(errno));
      }

      // in future, depending on data types of this uncompressed stream
      // it can be feed into other function such like importParquet, etc
      ret1 = importDelimited(file_path, true, session_info);

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
                                      copy_params.s3_session_token,
                                      copy_params.s3_region,
                                      copy_params.s3_endpoint,
                                      copy_params.plain_text,
                                      copy_params.regex_path_filter,
                                      copy_params.file_sort_order_by,
                                      copy_params.file_sort_regex));
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
        size_t num_block_read = 0;
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
            // In very rare occasions the write pipe somehow operates in a mode similar
            // to non-blocking while pipe(fds) should behave like pipe2(fds, 0) which
            // means blocking mode. On such a unreliable blocking mode, a possible fix
            // is to loop reading till no bytes left, otherwise the annoying `failed to
            // write pipe: Success`...
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
                mapd_shared_lock<mapd_shared_mutex> read_lock(import_mutex_);
                if (import_status_.load_failed) {
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
            ++num_block_read;
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
        mapd_shared_lock<mapd_shared_mutex> read_lock(import_mutex_);
        if (import_status_.load_failed) {
          break;
        }
        if (import_status_.rows_completed > 0) {
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

ImportStatus Importer::import(const Catalog_Namespace::SessionInfo* session_info) {
  return DataStreamSink::archivePlumber(session_info);
}

ImportStatus Importer::importDelimited(
    const std::string& file_path,
    const bool decompressed,
    const Catalog_Namespace::SessionInfo* session_info) {
  set_import_status(import_id, import_status_);
  auto query_session = session_info ? session_info->get_session_id() : "";

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
                                   first_row_index_this_buffer,
                                   session_info));

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
          std::chrono::milliseconds span(0);
          if (p.wait_for(span) == std::future_status::ready) {
            auto ret_import_status = p.get();
            {
              mapd_lock_guard<mapd_shared_mutex> write_lock(import_mutex_);
              import_status_ += ret_import_status;
              if (ret_import_status.load_failed) {
                set_import_status(import_id, import_status_);
              }
            }
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
              import_status_.rows_estimated =
                  (decompressed ? (float)total_file_size / total_file_offset
                                : (float)file_size / current_pos) *
                  import_status_.rows_completed;
            }
            VLOG(3) << "rows_completed " << import_status_.rows_completed
                    << ", rows_estimated " << import_status_.rows_estimated
                    << ", total_file_size " << total_file_size << ", total_file_offset "
                    << total_file_offset;
            set_import_status(import_id, import_status_);
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
        mapd_shared_lock<mapd_shared_mutex> read_lock(import_mutex_);
        if (import_status_.load_failed) {
          break;
        }
      }
      mapd_unique_lock<mapd_shared_mutex> write_lock(import_mutex_);
      if (import_status_.rows_rejected > copy_params.max_reject) {
        import_status_.load_failed = true;
        // todo use better message
        import_status_.load_msg = "Maximum rows rejected exceeded. Halting load";
        LOG(ERROR) << "Maximum rows rejected exceeded. Halting load";
        break;
      }
      if (import_status_.load_failed) {
        LOG(ERROR) << "Load failed, the issue was: " + import_status_.load_msg;
        break;
      }
    }

    // join dangling threads in case of LOG(ERROR) above
    for (auto& p : threads) {
      p.wait();
    }
  }

  checkpoint(table_epochs);

  fclose(p_file);
  p_file = nullptr;
  return import_status_;
}

void Loader::checkpoint() {
  if (getTableDesc()->persistenceLevel ==
      Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident
                                                  // tables
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

[[nodiscard]] std::vector<std::unique_ptr<TypedImportBuffer>> fill_missing_columns(
    const Catalog_Namespace::Catalog* cat,
    Fragmenter_Namespace::InsertData& insert_data) {
  std::vector<std::unique_ptr<import_export::TypedImportBuffer>> defaults_buffers;
  if (insert_data.is_default.size() == 0) {
    insert_data.is_default.resize(insert_data.columnIds.size(), false);
  }
  CHECK(insert_data.is_default.size() == insert_data.is_default.size());
  auto cds = cat->getAllColumnMetadataForTable(insert_data.tableId, false, false, true);
  if (cds.size() == insert_data.columnIds.size()) {
    // all columns specified
    return defaults_buffers;
  }
  for (auto cd : cds) {
    if (std::find(insert_data.columnIds.begin(),
                  insert_data.columnIds.end(),
                  cd->columnId) == insert_data.columnIds.end()) {
      StringDictionary* dict = nullptr;
      if (cd->columnType.get_type() == kARRAY &&
          IS_STRING(cd->columnType.get_subtype()) && !cd->default_value.has_value()) {
        throw std::runtime_error("Cannot omit column \"" + cd->columnName +
                                 "\": omitting TEXT arrays is not supported yet");
      }
      if (cd->columnType.get_compression() == kENCODING_DICT) {
        dict = cat->getMetadataForDict(cd->columnType.get_comp_param())->stringDict.get();
      }
      defaults_buffers.emplace_back(std::make_unique<TypedImportBuffer>(cd, dict));
    }
  }
  // put buffers in order to fill geo sub-columns properly
  std::sort(defaults_buffers.begin(),
            defaults_buffers.end(),
            [](decltype(defaults_buffers[0])& a, decltype(defaults_buffers[0])& b) {
              return a->getColumnDesc()->columnId < b->getColumnDesc()->columnId;
            });
  for (size_t i = 0; i < defaults_buffers.size(); ++i) {
    auto cd = defaults_buffers[i]->getColumnDesc();
    std::string default_value = cd->default_value.value_or("NULL");
    defaults_buffers[i]->add_value(
        cd, default_value, !cd->default_value.has_value(), import_export::CopyParams());
  }
  auto data = import_export::TypedImportBuffer::get_data_block_pointers(defaults_buffers);
  CHECK(data.size() == defaults_buffers.size());
  for (size_t i = 0; i < defaults_buffers.size(); ++i) {
    insert_data.data.push_back(data[i]);
    insert_data.columnIds.push_back(defaults_buffers[i]->getColumnDesc()->columnId);
    insert_data.is_default.push_back(true);
  }
  return defaults_buffers;
}

}  // namespace import_export
