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
 * @file Importer.cpp
 * @brief Functions for Importer class
 *
 */

#include "ImportExport/Importer.h"

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
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
#include "Shared/LonLatBoundingBox.h"
#include "Catalog/os/UserMapping.h"
#ifdef ENABLE_IMPORT_PARQUET
#include "DataMgr/ForeignStorage/ParquetDataWrapper.h"
#endif
#if defined(ENABLE_IMPORT_PARQUET)
#include "Catalog/ForeignTable.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapperFactory.h"
#endif
#ifdef ENABLE_IMPORT_PARQUET
#include "DataMgr/ForeignStorage/ParquetS3DetectFileSystem.h"
#endif
#include "Geospatial/Compression.h"
#include "Geospatial/GDAL.h"
#include "Geospatial/Transforms.h"
#include "Geospatial/Types.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "ImportExport/ForeignDataImporter.h"
#include "ImportExport/MetadataColumn.h"
#include "ImportExport/RasterImporter.h"
#include "Logger/Logger.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/TypePunning.h"
#include "Shared/DateTimeParser.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/SysDefinitions.h"
#include "Shared/file_path_util.h"
#include "Shared/import_helpers.h"
#include "Shared/likely.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/scope.h"
#include "Shared/shard_key.h"
#include "Shared/thread_count.h"
#include "Utils/ChunkAccessorTable.h"

#include "gen-cpp/Heavy.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define TIMER_STOP(t)                                                                  \
  (float(timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>( \
       t)) /                                                                           \
   1.0E6f)

size_t g_max_import_threads =
    32;  // Max number of default import threads to use (num hardware threads will be used
// if lower, and can also be explicitly overriden in copy statement with threads
// option)
size_t g_archive_read_buf_size = 1 << 20;

std::optional<size_t> g_detect_test_sample_size = std::nullopt;

static constexpr int kMaxRasterScanlinesPerThread = 32;

inline auto get_filesize(const std::string& file_path) {
  boost::filesystem::path boost_file_path{file_path};
  boost::system::error_code ec;
  const auto filesize = boost::filesystem::file_size(boost_file_path, ec);
  return ec ? 0 : filesize;
}

namespace {

bool check_session_interrupted(const QuerySessionId& query_session, Executor* executor) {
  if (g_enable_non_kernel_time_query_interrupt && !query_session.empty()) {
    heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
        executor->getSessionLock());
    return executor->checkIsQuerySessionInterrupted(query_session, session_read_lock);
  }
  return false;
}
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
using FeaturePtrVector = std::vector<Geospatial::GDAL::FeatureUqPtr>;

#define DEBUG_TIMING false
#define DEBUG_RENDER_GROUP_ANALYZER 0
#define DEBUG_AWS_AUTHENTICATION 0

#define DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT 0

// only auto-promote polygon column type
// @TODO auto-promote linestring
static constexpr bool PROMOTE_POINT_TO_MULTIPOINT = false;
static constexpr bool PROMOTE_LINESTRING_TO_MULTILINESTRING = false;
static constexpr bool PROMOTE_POLYGON_TO_MULTIPOLYGON = true;

static heavyai::shared_mutex status_mutex;
static std::map<std::string, ImportStatus> import_status_map;

// max # rows to import
static const size_t kImportRowLimit = 10000;

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
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(status_mutex);
  auto it = import_status_map.find(import_id);
  if (it == import_status_map.end()) {
    throw std::runtime_error("Import status not found for id: " + import_id);
  }
  return it->second;
}

void Importer::set_import_status(const std::string& import_id, ImportStatus is) {
  heavyai::lock_guard<heavyai::shared_mutex> write_lock(status_mutex);
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

namespace {
inline SQLTypes get_type_for_datum(const SQLTypeInfo& ti) {
  SQLTypes type;
  if (ti.is_decimal()) {
    type = decimal_to_int_type(ti);
  } else if (ti.is_dict_encoded_string()) {
    type = string_dict_to_int_type(ti);
  } else {
    type = ti.get_type();
  }
  return type;
}
}  // namespace

Datum NullArrayDatum(SQLTypeInfo& ti) {
  Datum d;
  const auto type = get_type_for_datum(ti);
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
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
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
    std::unique_ptr<int8_t, FreeDeleter> buf(
        reinterpret_cast<int8_t*>(checked_malloc(len)));
    int8_t* p = buf.get();
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
      p = append_datum(p, d, elem_ti);
      CHECK(p);
    }
    return ArrayDatum(len, buf.release(), false);
  }
  // must not be called for array of strings
  CHECK(false);
  return ArrayDatum(0, NULL, true);
}

ArrayDatum NullArray(const SQLTypeInfo& ti) {
  SQLTypeInfo elem_ti = ti.get_elem_type();
  auto len = ti.get_size();

  if (len > 0) {
    // Compose a NULL fixlen array
    int8_t* buf = (int8_t*)checked_malloc(len);
    // First scalar is a NULL_ARRAY sentinel
    Datum d = NullArrayDatum(elem_ti);
    int8_t* p = append_datum(buf, d, elem_ti);
    CHECK(p);
    // Rest is filled with normal NULL sentinels
    Datum d0 = NullDatum(elem_ti);
    while ((p - buf) < len) {
      p = append_datum(p, d0, elem_ti);
      CHECK(p);
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
  if (geo_ti.get_compression() == kENCODING_GEOINT) {
    CHECK(geo_ti.get_comp_param() == 32);
    std::vector<double> null_point_coords = {NULL_ARRAY_DOUBLE, NULL_DOUBLE};
    auto compressed_null_coords = Geospatial::compress_coords(null_point_coords, geo_ti);
    const size_t len = compressed_null_coords.size();
    int8_t* buf = (int8_t*)checked_malloc(len);
    memcpy(buf, compressed_null_coords.data(), len);
    return ArrayDatum(len, buf, false);
  }
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
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
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
    p = append_datum(p, TDatumToDatum(e, elem_ti), elem_ti);
    CHECK(p);
  }

  return ArrayDatum(len, buf, false);
}

template <const bool managed_memory>
template <typename VectorType>
void OptionallyMemoryManagedTypedImportBuffer<managed_memory>::addDictEncodedString(
    const VectorType& string_vec) {
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

template void OptionallyMemoryManagedTypedImportBuffer<true>::addDictEncodedString<
    std::vector<std::string>>(const std::vector<std::string>& string_vec);
template void OptionallyMemoryManagedTypedImportBuffer<false>::addDictEncodedString<
    std::vector<std::string>>(const std::vector<std::string>& string_vec);

template void OptionallyMemoryManagedTypedImportBuffer<true>::addDictEncodedString(
    const import_export::vector<std::string>& string_vec);

template void OptionallyMemoryManagedTypedImportBuffer<false>::addDictEncodedString(
    const import_export::vector<std::string>& string_vec);

template <const bool managed_memory>
void OptionallyMemoryManagedTypedImportBuffer<managed_memory>::add_value(
    const ColumnDescriptor* cd,
    const std::string_view val,
    const bool is_null,
    const CopyParams& copy_params,
    const bool check_not_null) {
  const auto type = cd->columnType.get_type();
  switch (type) {
    case kBOOLEAN: {
      if (is_null) {
        if (check_not_null && cd->columnType.get_notnull()) {
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
        if (check_not_null && cd->columnType.get_notnull()) {
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
        if (check_not_null && cd->columnType.get_notnull()) {
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
        if (check_not_null && cd->columnType.get_notnull()) {
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
        if (check_not_null && cd->columnType.get_notnull()) {
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
        DecimalOverflowValidator validator(ti);
        validator.validate(d.bigintval);
        addBigint(d.bigintval);
      } else {
        if (check_not_null && cd->columnType.get_notnull()) {
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
        if (check_not_null && cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addFloat(NULL_FLOAT);
      }
      break;
    case kDOUBLE:
      if (!is_null && (val[0] == '.' || isdigit(val[0]) || val[0] == '-')) {
        addDouble(std::atof(std::string(val).c_str()));
      } else {
        if (check_not_null && cd->columnType.get_notnull()) {
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
        if (check_not_null && cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addString(std::string());
      } else {
        if (val.length() > cd->columnType.get_max_strlen()) {
          throw std::runtime_error("String too long for column " + cd->columnName +
                                   " was " + std::to_string(val.length()) + " max is " +
                                   std::to_string(cd->columnType.get_max_strlen()));
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
        if (check_not_null && cd->columnType.get_notnull()) {
          throw std::runtime_error("NULL for column " + cd->columnName);
        }
        addBigint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kARRAY: {
      if (check_not_null && is_null && cd->columnType.get_notnull()) {
        throw std::runtime_error("NULL for column " + cd->columnName);
      }
      SQLTypeInfo ti = cd->columnType;
      if (IS_STRING(ti.get_subtype())) {
        std::vector<std::string> string_vec;
        // Just parse string array, don't push it to buffer yet as we might throw
        import_export::delimited_parser::parse_string_array(
            std::string(val), copy_params, string_vec);
        if (!is_null) {
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
          addStringArray(std::nullopt);
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
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      addGeoString(val);
      break;
    default:
      CHECK(false) << "TypedImportBuffer::add_value() does not support type " << type;
  }
}

template <const bool managed_memory>
void OptionallyMemoryManagedTypedImportBuffer<managed_memory>::pop_value() {
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
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
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
template <const bool managed_memory>
template <typename DATA_TYPE>
size_t OptionallyMemoryManagedTypedImportBuffer<managed_memory>::
    convert_arrow_val_to_import_buffer(
        const ColumnDescriptor* cd,
        const Array& array,
        vector<DATA_TYPE>& buffer,
        const ArraySliceRange& slice_range,
        import_export::BadRowsTracker* const bad_rows_tracker) {
  auto data =
      std::make_unique<DataBuffer<DATA_TYPE, typename vector<DATA_TYPE>::allocator_type>>(
          cd, array, buffer, bad_rows_tracker);
  auto f_value_getter = value_getter(array, cd, bad_rows_tracker);
  std::function<void(const int64_t)> f_add_geo_phy_cols = [&](const int64_t row) {};
  if (bad_rows_tracker && cd->columnType.is_geometry()) {
    f_add_geo_phy_cols = [&](const int64_t row) {
      // Populate physical columns (ref. DBHandler::load_table)
      std::vector<double> coords, bounds;
      std::vector<int> ring_sizes, poly_rings;
      SQLTypeInfo ti;
      // replace any unexpected exception from getGeoColumns or other
      // on this path with a GeoImportException so that we wont over
      // push a null to the logical column...
      try {
        SQLTypeInfo import_ti{ti};
        if (array.IsNull(row)) {
          Geospatial::GeoTypesFactory::getNullGeoColumns(
              import_ti, coords, bounds, ring_sizes, poly_rings);
        } else {
          const bool validate_with_geos_if_available = false;
          arrow_throw_if<GeoImportException>(
              !Geospatial::GeoTypesFactory::getGeoColumns(
                  geo_string_buffer_->back(),
                  ti,
                  coords,
                  bounds,
                  ring_sizes,
                  poly_rings,
                  validate_with_geos_if_available),
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
            poly_rings);
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

template <const bool managed_memory>
size_t OptionallyMemoryManagedTypedImportBuffer<managed_memory>::add_arrow_values(
    const ColumnDescriptor* cd,
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
      return convert_arrow_val_to_import_buffer<int8_t>(
          cd, col, *bool_buffer_, slice_range, bad_rows_tracker);
    case kTINYINT:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::INT8, "Expected int8 type");
      }
      return convert_arrow_val_to_import_buffer<int8_t>(
          cd, col, *tinyint_buffer_, slice_range, bad_rows_tracker);
    case kSMALLINT:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::INT16, "Expected int16 type");
      }
      return convert_arrow_val_to_import_buffer<int16_t>(
          cd, col, *smallint_buffer_, slice_range, bad_rows_tracker);
    case kINT:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::INT32, "Expected int32 type");
      }
      return convert_arrow_val_to_import_buffer<int32_t>(
          cd, col, *int_buffer_, slice_range, bad_rows_tracker);
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::INT64, "Expected int64 type");
      }
      return convert_arrow_val_to_import_buffer<int64_t>(
          cd, col, *bigint_buffer_, slice_range, bad_rows_tracker);
    case kFLOAT:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::FLOAT, "Expected float type");
      }
      return convert_arrow_val_to_import_buffer<float>(
          cd, col, *float_buffer_, slice_range, bad_rows_tracker);
    case kDOUBLE:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::DOUBLE, "Expected double type");
      }
      return convert_arrow_val_to_import_buffer<double>(
          cd, col, *double_buffer_, slice_range, bad_rows_tracker);
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::BINARY && col.type_id() != Type::STRING,
                       "Expected string type");
      }
      return convert_arrow_val_to_import_buffer<std::string>(
          cd, col, *string_buffer_, slice_range, bad_rows_tracker);
    case kTIME:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::TIME32 && col.type_id() != Type::TIME64,
                       "Expected time32 or time64 type");
      }
      return convert_arrow_val_to_import_buffer<int64_t>(
          cd, col, *bigint_buffer_, slice_range, bad_rows_tracker);
    case kTIMESTAMP:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::TIMESTAMP, "Expected timestamp type");
      }
      return convert_arrow_val_to_import_buffer<int64_t>(
          cd, col, *bigint_buffer_, slice_range, bad_rows_tracker);
    case kDATE:
      if (exact_type_match) {
        arrow_throw_if(col.type_id() != Type::DATE32 && col.type_id() != Type::DATE64,
                       "Expected date32 or date64 type");
      }
      return convert_arrow_val_to_import_buffer<int64_t>(
          cd, col, *bigint_buffer_, slice_range, bad_rows_tracker);
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      arrow_throw_if(col.type_id() != Type::BINARY && col.type_id() != Type::STRING,
                     "Expected string type");
      return convert_arrow_val_to_import_buffer<std::string>(
          cd, col, *geo_string_buffer_, slice_range, bad_rows_tracker);
    case kARRAY:
      throw std::runtime_error("Arrow array appends not yet supported");
    default:
      throw std::runtime_error("Invalid Type");
  }
}

namespace {
size_t validate_and_get_column_data_size(const SQLTypeInfo& type,
                                         const TColumnData& data) {
  size_t data_size;
  if (type.is_boolean() || type.is_integer() || type.is_decimal() || type.is_time()) {
    data_size = data.int_col.size();
  } else if (type.is_fp()) {
    data_size = data.real_col.size();
  } else if (type.is_string() || type.is_geometry()) {
    data_size = data.str_col.size();
  } else if (type.is_array()) {
    data_size = data.arr_col.size();
  } else {
    UNREACHABLE() << "Unexpected column type: " << type;
    data_size = 0;
  }
  auto total_data_size = data.int_col.size() + data.real_col.size() +
                         data.str_col.size() + data.arr_col.size();
  if (data_size != total_data_size) {
    throw std::runtime_error("Column data set for the wrong type.");
  }
  return data_size;
}
}  // namespace

// this is exclusively used by load_table_binary_columnar
template <const bool managed_memory>
size_t OptionallyMemoryManagedTypedImportBuffer<managed_memory>::add_values(
    const ColumnDescriptor* cd,
    const TColumn& col) {
  CHECK(cd);
  if (col.nulls.size() != validate_and_get_column_data_size(cd->columnType, col.data)) {
    throw std::runtime_error("Column nulls vector and data vector sizes do not match.");
  }

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
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
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
          OptionalStringVector& string_vec = addStringArray();
          if (!col.nulls[i]) {
            size_t stringArrSize = col.data.arr_col[i].data.str_col.size();
            for (size_t str_idx = 0; str_idx != stringArrSize; ++str_idx) {
              string_vec->push_back(col.data.arr_col[i].data.str_col[str_idx]);
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

template <const bool managed_memory>
void OptionallyMemoryManagedTypedImportBuffer<managed_memory>::add_value(
    const ColumnDescriptor* cd,
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
        OptionalStringVector& string_vec = addStringArray();
        addBinaryStringArray(datum, *string_vec);
      } else {
        if (!is_null) {
          addArray(TDatumToArrayDatum(datum, cd->columnType));
        } else {
          addArray(NullArray(cd->columnType));
        }
      }
      break;
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
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

template <const bool managed_memory>
void OptionallyMemoryManagedTypedImportBuffer<managed_memory>::addDefaultValues(
    const ColumnDescriptor* cd,
    size_t num_rows) {
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
        if (val.length() > ti.get_max_strlen()) {
          throw std::runtime_error("String too long for column " + cd->columnName +
                                   " was " + std::to_string(val.length()) + " max is " +
                                   std::to_string(ti.get_max_strlen()));
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
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      geo_string_buffer_->resize(num_rows, val);
      break;
    default:
      CHECK(false) << "TypedImportBuffer::addDefaultValues() does not support type "
                   << type;
  }
}

template <const bool managed_memory>
template <typename DATA_TYPE>
auto OptionallyMemoryManagedTypedImportBuffer<managed_memory>::del_values(
    vector<DATA_TYPE>& buffer,
    import_export::BadRowsTracker* const bad_rows_tracker) {
  const auto old_size = buffer.size();
  // erase backward to minimize memory movement overhead
  for (auto rit = bad_rows_tracker->rows.crbegin(); rit != bad_rows_tracker->rows.crend();
       ++rit) {
    buffer.erase(buffer.begin() + *rit);
  }
  return std::make_tuple(old_size, buffer.size());
}

template <const bool managed_memory>
auto OptionallyMemoryManagedTypedImportBuffer<managed_memory>::del_values(
    const SQLTypes type,
    BadRowsTracker* const bad_rows_tracker) {
  switch (type) {
    case kBOOLEAN:
      return del_values<int8_t>(*bool_buffer_, bad_rows_tracker);
    case kTINYINT:
      return del_values<int8_t>(*tinyint_buffer_, bad_rows_tracker);
    case kSMALLINT:
      return del_values<int16_t>(*smallint_buffer_, bad_rows_tracker);
    case kINT:
      return del_values<int32_t>(*int_buffer_, bad_rows_tracker);
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return del_values<int64_t>(*bigint_buffer_, bad_rows_tracker);
    case kFLOAT:
      return del_values<float>(*float_buffer_, bad_rows_tracker);
    case kDOUBLE:
      return del_values<double>(*double_buffer_, bad_rows_tracker);
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return del_values<std::string>(*string_buffer_, bad_rows_tracker);
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      return del_values<std::string>(*geo_string_buffer_, bad_rows_tracker);
    case kARRAY:
      return del_values<ArrayDatum>(*array_buffer_, bad_rows_tracker);
    default:
      throw std::runtime_error("Invalid Type");
  }
}

bool importGeoFromLonLat(double lon,
                         double lat,
                         std::vector<double>& coords,
                         std::vector<double>& bounds,
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
  } else {
    coords.push_back(lon);
    coords.push_back(lat);
  }
  // in case of promotion to MULTIPOINT
  CHECK_EQ(coords.size(), 2u);
  bounds.reserve(4);
  bounds.push_back(coords[0]);
  bounds.push_back(coords[1]);
  bounds.push_back(coords[0]);
  bounds.push_back(coords[1]);
  return true;
}

template <typename TypedImportBufferType>
void Importer::set_geo_physical_import_buffer(
    const Catalog_Namespace::Catalog& catalog,
    const ColumnDescriptor* cd,
    std::vector<std::unique_ptr<TypedImportBufferType>>& import_buffers,
    size_t& col_idx,
    std::vector<double>& coords,
    std::vector<double>& bounds,
    std::vector<int>& ring_sizes,
    std::vector<int>& poly_rings,
    const bool force_null) {
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
  if (force_null) {
    is_null_geo = true;
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
  import_buffers[col_idx++]->add_value(cd_coords, tdd_coords, false);

  if (col_type == kMULTILINESTRING || col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    // Create [linest]ring_sizes array value and add it to the physical column
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
    import_buffers[col_idx++]->add_value(cd_ring_sizes, tdd_ring_sizes, false);
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
    import_buffers[col_idx++]->add_value(cd_poly_rings, tdd_poly_rings, false);
  }

  if (col_type == kLINESTRING || col_type == kMULTILINESTRING || col_type == kPOLYGON ||
      col_type == kMULTIPOLYGON || col_type == kMULTIPOINT) {
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
    import_buffers[col_idx++]->add_value(cd_bounds, tdd_bounds, false);
  }
}

template void Importer::set_geo_physical_import_buffer(
    const Catalog_Namespace::Catalog& catalog,
    const ColumnDescriptor* cd,
    std::vector<std::unique_ptr<OptionallyMemoryManagedTypedImportBuffer<true>>>&
        import_buffers,
    size_t& col_idx,
    std::vector<double>& coords,
    std::vector<double>& bounds,
    std::vector<int>& ring_sizes,
    std::vector<int>& poly_rings,
    const bool force_null);
template void Importer::set_geo_physical_import_buffer(
    const Catalog_Namespace::Catalog& catalog,
    const ColumnDescriptor* cd,
    std::vector<std::unique_ptr<OptionallyMemoryManagedTypedImportBuffer<false>>>&
        import_buffers,
    size_t& col_idx,
    std::vector<double>& coords,
    std::vector<double>& bounds,
    std::vector<int>& ring_sizes,
    std::vector<int>& poly_rings,
    const bool force_null);

template <typename TypedImportBufferType>
void Importer::set_geo_physical_import_buffer_columnar(
    const Catalog_Namespace::Catalog& catalog,
    const ColumnDescriptor* cd,
    std::vector<std::unique_ptr<TypedImportBufferType>>& import_buffers,
    size_t& col_idx,
    std::vector<std::vector<double>>& coords_column,
    std::vector<std::vector<double>>& bounds_column,
    std::vector<std::vector<int>>& ring_sizes_column,
    std::vector<std::vector<int>>& poly_rings_column) {
  const auto col_ti = cd->columnType;
  const auto col_type = col_ti.get_type();
  auto columnId = cd->columnId;

  auto coords_row_count = coords_column.size();
  auto cd_coords = catalog.getMetadataForColumn(cd->tableId, ++columnId);
  for (auto& coords : coords_column) {
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
      for (auto const& cc : compressed_coords) {
        TDatum td_byte;
        td_byte.val.int_val = cc;
        td_coords_data.push_back(td_byte);
      }
    }
    TDatum tdd_coords;
    tdd_coords.val.arr_val = td_coords_data;
    tdd_coords.is_null = is_null_geo;
    import_buffers[col_idx]->add_value(cd_coords, tdd_coords, false);
  }
  col_idx++;

  if (col_type == kMULTILINESTRING || col_type == kPOLYGON || col_type == kMULTIPOLYGON) {
    if (ring_sizes_column.size() != coords_row_count) {
      CHECK(false) << "Geometry import columnar: ring sizes column size mismatch";
    }
    // Create [linest[ring_sizes array value and add it to the physical column
    auto cd_ring_sizes = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    for (auto const& ring_sizes : ring_sizes_column) {
      bool is_null_geo = false;
      if (!col_ti.get_notnull()) {
        // Check for NULL geo
        is_null_geo = ring_sizes.empty();
      }
      std::vector<TDatum> td_ring_sizes;
      for (auto const& ring_size : ring_sizes) {
        TDatum td_ring_size;
        td_ring_size.val.int_val = ring_size;
        td_ring_sizes.push_back(td_ring_size);
      }
      TDatum tdd_ring_sizes;
      tdd_ring_sizes.val.arr_val = td_ring_sizes;
      tdd_ring_sizes.is_null = is_null_geo;
      import_buffers[col_idx]->add_value(cd_ring_sizes, tdd_ring_sizes, false);
    }
    col_idx++;
  }

  if (col_type == kMULTIPOLYGON) {
    if (poly_rings_column.size() != coords_row_count) {
      CHECK(false) << "Geometry import columnar: poly rings column size mismatch";
    }
    // Create poly_rings array value and add it to the physical column
    auto cd_poly_rings = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    for (auto const& poly_rings : poly_rings_column) {
      bool is_null_geo = false;
      if (!col_ti.get_notnull()) {
        // Check for NULL geo
        is_null_geo = poly_rings.empty();
      }
      std::vector<TDatum> td_poly_rings;
      for (auto const& num_rings : poly_rings) {
        TDatum td_num_rings;
        td_num_rings.val.int_val = num_rings;
        td_poly_rings.push_back(td_num_rings);
      }
      TDatum tdd_poly_rings;
      tdd_poly_rings.val.arr_val = td_poly_rings;
      tdd_poly_rings.is_null = is_null_geo;
      import_buffers[col_idx]->add_value(cd_poly_rings, tdd_poly_rings, false);
    }
    col_idx++;
  }

  if (col_type == kLINESTRING || col_type == kMULTILINESTRING || col_type == kPOLYGON ||
      col_type == kMULTIPOLYGON || col_type == kMULTIPOINT) {
    if (bounds_column.size() != coords_row_count) {
      CHECK(false) << "Geometry import columnar: bounds column size mismatch";
    }
    auto cd_bounds = catalog.getMetadataForColumn(cd->tableId, ++columnId);
    for (auto const& bounds : bounds_column) {
      bool is_null_geo = false;
      if (!col_ti.get_notnull()) {
        // Check for NULL geo
        is_null_geo = (bounds.empty() || bounds[0] == NULL_ARRAY_DOUBLE);
      }
      std::vector<TDatum> td_bounds_data;
      for (auto const& b : bounds) {
        TDatum td_double;
        td_double.val.real_val = b;
        td_bounds_data.push_back(td_double);
      }
      TDatum tdd_bounds;
      tdd_bounds.val.arr_val = td_bounds_data;
      tdd_bounds.is_null = is_null_geo;
      import_buffers[col_idx]->add_value(cd_bounds, tdd_bounds, false);
    }
    col_idx++;
  }
}

template void Importer::set_geo_physical_import_buffer_columnar(
    const Catalog_Namespace::Catalog& catalog,
    const ColumnDescriptor* cd,
    std::vector<std::unique_ptr<OptionallyMemoryManagedTypedImportBuffer<true>>>&
        import_buffers,
    size_t& col_idx,
    std::vector<std::vector<double>>& coords_column,
    std::vector<std::vector<double>>& bounds_column,
    std::vector<std::vector<int>>& ring_sizes_column,
    std::vector<std::vector<int>>& poly_rings_column);
template void Importer::set_geo_physical_import_buffer_columnar(
    const Catalog_Namespace::Catalog& catalog,
    const ColumnDescriptor* cd,
    std::vector<std::unique_ptr<OptionallyMemoryManagedTypedImportBuffer<false>>>&
        import_buffers,
    size_t& col_idx,
    std::vector<std::vector<double>>& coords_column,
    std::vector<std::vector<double>>& bounds_column,
    std::vector<std::vector<int>>& ring_sizes_column,
    std::vector<std::vector<int>>& poly_rings_column);

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
    size_t first_row_index_this_buffer,
    const Catalog_Namespace::SessionInfo* session_info,
    Executor* executor) {
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
      if (cd->columnType.get_type() == kPOINT ||
          cd->columnType.get_type() == kMULTIPOINT) {
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
      row_index_plus_one++;
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

      auto execute_import_row = [&](OGRGeometry* import_geometry) {
        size_t import_idx = 0;
        size_t col_idx = 0;
        try {
          for (auto cd_it = col_descs.begin(); cd_it != col_descs.end(); cd_it++) {
            auto cd = *cd_it;
            const auto& col_ti = cd->columnType;

            bool is_null =
                ImportHelpers::is_null_datum(row[import_idx], copy_params.null_str);
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
            if (!cd->columnType.is_string() && !copy_params.trim_spaces) {
              // everything but strings should be always trimmed
              row[import_idx] = sv_strip(row[import_idx]);
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

              // if this is a POINT or MULTIPOINT column, and the field is not null, and
              // looks like a scalar numeric value (and not a hex blob) attempt to import
              // two columns as lon/lat (or lat/lon)
              if ((col_type == kPOINT || col_type == kMULTIPOINT) && !is_null &&
                  geo_string.size() > 0 &&
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
                // TODO: should check if POINT/MULTIPOINT column should have been declared
                // with SRID WGS 84, EPSG 4326 ? if (col_ti.get_dimension() != 4326) {
                //  throw std::runtime_error("POINT column " + cd->columnName + " is
                //  not WGS84, cannot insert lon/lat");
                // }
                SQLTypeInfo import_ti{col_ti};
                if (copy_params.source_type ==
                        import_export::SourceType::kDelimitedFile &&
                    import_ti.get_output_srid() == 4326) {
                  auto srid0 = copy_params.source_srid;
                  if (srid0 > 0) {
                    // srid0 -> 4326 transform is requested on import
                    import_ti.set_input_srid(srid0);
                  }
                }
                if (!importGeoFromLonLat(lon, lat, coords, bounds, import_ti)) {
                  throw std::runtime_error(
                      "Cannot read lon/lat to insert into POINT/MULTIPOINT column " +
                      cd->columnName);
                }
              } else {
                // import it
                SQLTypeInfo import_ti{col_ti};
                if (copy_params.source_type ==
                        import_export::SourceType::kDelimitedFile &&
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
                      import_ti, coords, bounds, ring_sizes, poly_rings);
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
                            copy_params.geo_validate_geometry)) {
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
                            copy_params.geo_validate_geometry)) {
                      std::string msg = "Failed to extract valid geometry from row " +
                                        std::to_string(first_row_index_this_buffer +
                                                       row_index_plus_one) +
                                        " for column " + cd->columnName;
                      throw std::runtime_error(msg);
                    }
                  }

                  // validate types
                  if (!geo_promoted_type_match(import_ti.get_type(), col_type)) {
                    throw std::runtime_error(
                        "Imported geometry doesn't match the type of column " +
                        cd->columnName);
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
                                                       poly_rings);

              // skip remaining physical columns
              for (int i = 0; i < cd->columnType.get_physical_cols(); ++i) {
                ++cd_it;
              }
            }
          }
          if (UNLIKELY((thread_import_status.rows_completed & 0xFFFF) == 0 &&
                       check_session_interrupted(query_session, executor))) {
            thread_import_status.load_failed = true;
            thread_import_status.load_msg =
                "Table load was cancelled via Query Interrupt";
            return;
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
            std::string(collection_geo_string), copy_params.geo_validate_geometry);
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

class ColumnNotGeoError : public std::runtime_error {
 public:
  ColumnNotGeoError(const std::string& column_name)
      : std::runtime_error("Column '" + column_name + "' is not a geo column") {}
};

static ImportStatus import_thread_shapefile(
    int thread_id,
    Importer* importer,
    OGRCoordinateTransformation* coordinate_transformation,
    const FeaturePtrVector& features,
    size_t firstFeature,
    size_t numFeatures,
    const FieldNameToIndexMapType& fieldNameToIndexMap,
    const ColumnNameToSourceNameMapType& columnNameToSourceNameMap,
    const Catalog_Namespace::SessionInfo* session_info,
    Executor* executor,
    const MetadataColumnInfos& metadata_column_infos,
    const std::optional<shared::LonLatBoundingBox>& bounding_box_clip) {
  ImportStatus thread_import_status;
  const CopyParams& copy_params = importer->get_copy_params();
  const std::list<const ColumnDescriptor*>& col_descs = importer->get_column_descs();
  std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers =
      importer->get_import_buffers(thread_id);
  auto query_session = session_info ? session_info->get_session_id() : "";
  for (const auto& p : import_buffers) {
    p->clear();
  }

  auto convert_timer = timer_start();

  // for all the features in this chunk...
  for (size_t iFeature = 0; iFeature < numFeatures; iFeature++) {
    // ignore null features
    if (!features[iFeature]) {
      continue;
    }

    // get this feature's geometry
    // for geodatabase, we need to consider features with no geometry
    // as we still want to create a table, even if it has no geo column
    OGRGeometry* pGeometry = features[iFeature]->GetGeometryRef();
    if (pGeometry && coordinate_transformation) {
      pGeometry->transform(coordinate_transformation);
    }

    //
    // lambda for importing a feature (perhaps multiple times if exploding a collection)
    //

    auto execute_import_feature = [&](OGRGeometry* import_geometry) {
      size_t col_idx = 0;
      try {
        if (UNLIKELY((thread_import_status.rows_completed & 0xFFFF) == 0 &&
                     check_session_interrupted(query_session, executor))) {
          thread_import_status.load_failed = true;
          thread_import_status.load_msg = "Table load was cancelled via Query Interrupt";
          throw QueryExecutionError(ErrorCode::INTERRUPTED);
        }

        // skip if outside import bounding box clip
        // getEnvelope() just scans all points of the geo, which is not cheap
        // but GDAL does not provide the bounding box in the general case
        // we only suffer this expense when the option is enabled
        if (bounding_box_clip.has_value()) {
          OGREnvelope envelope;
          import_geometry->getEnvelope(&envelope);
          auto const& bb = *bounding_box_clip;
          if (envelope.MaxX < bb.min_lon || envelope.MinX > bb.max_lon ||
              envelope.MaxY < bb.min_lat || envelope.MinY > bb.max_lat) {
            return;
          }
        }

        uint32_t field_column_count{0u};
        uint32_t metadata_column_count{0u};

        for (auto cd_it = col_descs.begin(); cd_it != col_descs.end(); cd_it++) {
          auto cd = *cd_it;

          // is this a geo column?
          const auto& col_ti = cd->columnType;
          if (col_ti.is_geometry()) {
            // Note that this assumes there is one and only one geo column in the
            // table. Currently, the importer only supports reading a single
            // geospatial feature from an input shapefile / geojson file, but this
            // code will need to be modified if that changes
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

            // extract it
            SQLTypeInfo import_ti{col_ti};
            bool is_null_geo = !import_geometry;
            if (is_null_geo) {
              if (col_ti.get_notnull()) {
                throw std::runtime_error("NULL geo for column " + cd->columnName);
              }
              Geospatial::GeoTypesFactory::getNullGeoColumns(
                  import_ti, coords, bounds, ring_sizes, poly_rings);
            } else {
              if (!Geospatial::GeoTypesFactory::getGeoColumns(
                      import_geometry,
                      import_ti,
                      coords,
                      bounds,
                      ring_sizes,
                      poly_rings,
                      copy_params.geo_validate_geometry)) {
                std::string msg = "Failed to extract valid geometry from feature " +
                                  std::to_string(firstFeature + iFeature + 1) +
                                  " for column " + cd->columnName;
                throw std::runtime_error(msg);
              }

              // validate types
              if (!geo_promoted_type_match(import_ti.get_type(), col_type)) {
                throw std::runtime_error(
                    "Imported geometry doesn't match the type of column " +
                    cd->columnName);
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

            if (col_type == kMULTILINESTRING || col_type == kPOLYGON ||
                col_type == kMULTIPOLYGON) {
              // Create [linest]ring_sizes array value and add it to the physical column
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

            if (col_type == kLINESTRING || col_type == kMULTILINESTRING ||
                col_type == kPOLYGON || col_type == kMULTIPOLYGON ||
                col_type == kMULTIPOINT) {
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
          } else if (field_column_count < fieldNameToIndexMap.size()) {
            //
            // field column
            //
            auto const cit = columnNameToSourceNameMap.find(cd->columnName);
            CHECK(cit != columnNameToSourceNameMap.end());
            auto const& field_name = cit->second;

            auto const fit = fieldNameToIndexMap.find(field_name);
            if (fit == fieldNameToIndexMap.end()) {
              throw ColumnNotGeoError(cd->columnName);
            }

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

            import_buffers[col_idx]->add_value(cd, value_string, false, copy_params);
            ++col_idx;
            field_column_count++;
          } else if (metadata_column_count < metadata_column_infos.size()) {
            //
            // metadata column
            //
            auto const& mci = metadata_column_infos[metadata_column_count];
            if (mci.column_descriptor.columnName != cd->columnName) {
              throw std::runtime_error("Metadata column name mismatch");
            }
            import_buffers[col_idx]->add_value(cd, mci.value, false, copy_params);
            ++col_idx;
            metadata_column_count++;
          } else {
            throw std::runtime_error("Column count mismatch");
          }
        }
        thread_import_status.rows_completed++;
      } catch (QueryExecutionError& e) {
        if (e.hasErrorCode(ErrorCode::INTERRUPTED)) {
          throw e;
        }
      } catch (ColumnNotGeoError& e) {
        LOG(ERROR) << "Input exception thrown: " << e.what() << ". Aborting import.";
        throw std::runtime_error(e.what());
      } catch (const std::exception& e) {
        for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
          import_buffers[col_idx_to_pop]->pop_value();
        }
        thread_import_status.rows_rejected++;
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

  float convert_s = TIMER_STOP(convert_timer);

  float load_s = 0.0f;
  if (thread_import_status.rows_completed > 0) {
    auto load_timer = timer_start();
    importer->load(import_buffers, thread_import_status.rows_completed, session_info);
    load_s = TIMER_STOP(load_timer);
  }

  if (DEBUG_TIMING && thread_import_status.rows_completed > 0) {
    LOG(INFO) << "DEBUG:      Process " << convert_s << "s";
    LOG(INFO) << "DEBUG:      Load " << load_s << "s";
    LOG(INFO) << "DEBUG:      Total " << (convert_s + load_s) << "s";
  }

  thread_import_status.thread_id = thread_id;

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
      case kPOINT:
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
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
}

void Loader::distributeToShardsExistingColumns(
    std::vector<OneShardBuffers>& all_shard_import_buffers,
    std::vector<size_t>& all_shard_row_counts,
    const OneShardBuffers& import_buffers,
    const size_t row_count,
    const size_t shard_count,
    const Catalog_Namespace::SessionInfo* session_info) {
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

  for (size_t i = 0; i < row_count; ++i) {
    const size_t shard =
        SHARD_FOR_KEY(int_value_at(*shard_column_input_buffer, i), shard_count);
    auto& shard_output_buffers = all_shard_import_buffers[shard];
    fillShardRow(i, shard_output_buffers, import_buffers);
    ++all_shard_row_counts[shard];
  }
}

void Loader::distributeToShardsNewColumns(
    std::vector<OneShardBuffers>& all_shard_import_buffers,
    std::vector<size_t>& all_shard_row_counts,
    const OneShardBuffers& import_buffers,
    const size_t row_count,
    const size_t shard_count,
    const Catalog_Namespace::SessionInfo* session_info) {
  const auto shard_tds = catalog_.getPhysicalTablesDescriptors(table_desc_);
  CHECK(shard_tds.size() == shard_count);

  for (size_t shard = 0; shard < shard_count; ++shard) {
    auto& shard_output_buffers = all_shard_import_buffers[shard];
    if (row_count != 0) {
      fillShardRow(0, shard_output_buffers, import_buffers);
    }
    // when replicating a column, row count of a shard == replicate count of the column
    // on the shard
    all_shard_row_counts[shard] = shard_tds[shard]->fragmenter->getNumRows();
  }
}

void Loader::distributeToShards(std::vector<OneShardBuffers>& all_shard_import_buffers,
                                std::vector<size_t>& all_shard_row_counts,
                                const OneShardBuffers& import_buffers,
                                const size_t row_count,
                                const size_t shard_count,
                                const Catalog_Namespace::SessionInfo* session_info) {
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
  if (isAddingColumns()) {
    distributeToShardsNewColumns(all_shard_import_buffers,
                                 all_shard_row_counts,
                                 import_buffers,
                                 row_count,
                                 shard_count,
                                 session_info);
  } else {
    distributeToShardsExistingColumns(all_shard_import_buffers,
                                      all_shard_row_counts,
                                      import_buffers,
                                      row_count,
                                      shard_count,
                                      session_info);
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
  if (table_desc_->nShards) {
    std::vector<OneShardBuffers> all_shard_import_buffers;
    std::vector<size_t> all_shard_row_counts;
    const auto shard_tables = catalog_.getPhysicalTablesDescriptors(table_desc_);
    distributeToShards(all_shard_import_buffers,
                       all_shard_row_counts,
                       import_buffers,
                       row_count,
                       shard_tables.size(),
                       session_info);
    bool success = true;
    for (size_t shard_idx = 0; shard_idx < shard_tables.size(); ++shard_idx) {
      success = success && loadToShard(all_shard_import_buffers[shard_idx],
                                       all_shard_row_counts[shard_idx],
                                       shard_tables[shard_idx],
                                       checkpoint,
                                       session_info);
    }
    return success;
  }
  return loadToShard(import_buffers, row_count, table_desc_, checkpoint, session_info);
}

template <const bool managed_memory>
std::vector<DataBlockPtr>
OptionallyMemoryManagedTypedImportBuffer<managed_memory>::get_data_block_pointers(
    const std::vector<
        std::unique_ptr<OptionallyMemoryManagedTypedImportBuffer<managed_memory>>>&
        import_buffers) {
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
        p.setStringsPtr(*string_payload_ptr);
      } else {
        // This condition means we have column which is ENCODED string. We already made
        // Async request to gain the encoded integer values above so we should skip this
        // iteration and continue.
        continue;
      }
    } else if (import_buffers[buf_idx]->getTypeInfo().is_geometry()) {
      auto geo_payload_ptr = import_buffers[buf_idx]->getGeoStringBuffer();
      p.setStringsPtr(*geo_payload_ptr);
    } else {
      CHECK(import_buffers[buf_idx]->getTypeInfo().get_type() == kARRAY);
      if (IS_STRING(import_buffers[buf_idx]->getTypeInfo().get_subtype())) {
        CHECK(import_buffers[buf_idx]->getTypeInfo().get_compression() == kENCODING_DICT);
        import_buffers[buf_idx]->addDictEncodedStringArray(
            *import_buffers[buf_idx]->getStringArrayBuffer());
        p.setArraysPtr(*import_buffers[buf_idx]->getStringArrayDictBuffer());
      } else {
        p.setArraysPtr(*import_buffers[buf_idx]->getArrayBuffer());
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
  std::vector<const TableDescriptor*> table_descs(1, table_desc_);
  if (table_desc_->nShards) {
    table_descs = catalog_.getPhysicalTablesDescriptors(table_desc_);
  }
  for (auto table_desc : table_descs) {
    table_desc->fragmenter->dropColumns(columnIds);
  }
}

void Loader::init() {
  insert_data_.databaseId = catalog_.getCurrentDB().dbId;
  insert_data_.tableId = table_desc_->tableId;
  for (auto cd : column_descs_) {
    insert_data_.columnIds.push_back(cd->columnId);
    if (cd->columnType.get_compression() == kENCODING_DICT) {
      CHECK(cd->columnType.is_string() || cd->columnType.is_string_array());
      const auto dd = catalog_.getMetadataForDict(cd->columnType.get_comp_param());
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
        if (import_status_.rows_completed >= kImportRowLimit) {
          // stop import when row limit reached
          break;
        }
      }
    }
  } catch (std::exception& e) {
  }

  heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
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
    if (boost::filesystem::path(file_path).extension() == ".tsv") {
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

  // check for geo types
  if (type == kTEXT) {
    // convert to upper case
    std::string str_upper_case = str;
    std::transform(
        str_upper_case.begin(), str_upper_case.end(), str_upper_case.begin(), ::toupper);

    // then test for leading words
    if (str_upper_case.find("POINT") == 0) {
      // promote column type?
      type = PROMOTE_POINT_TO_MULTIPOINT ? kMULTIPOINT : kPOINT;
    } else if (str_upper_case.find("MULTIPOINT") == 0) {
      type = kMULTIPOINT;
    } else if (str_upper_case.find("LINESTRING") == 0) {
      // promote column type?
      type = PROMOTE_LINESTRING_TO_MULTILINESTRING ? kMULTILINESTRING : kLINESTRING;
    } else if (str_upper_case.find("MULTILINESTRING") == 0) {
      type = kMULTILINESTRING;
    } else if (str_upper_case.find("POLYGON") == 0) {
      // promote column type?
      type = PROMOTE_POLYGON_TO_MULTIPOLYGON ? kMULTIPOLYGON : kPOLYGON;
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
        } else if (first_five_bytes == "0000000004" || first_five_bytes == "0104000000") {
          type = kMULTIPOINT;
        } else if (first_five_bytes == "0000000002" || first_five_bytes == "0102000000") {
          type = kLINESTRING;
        } else if (first_five_bytes == "0000000005" || first_five_bytes == "0105000000") {
          type = kMULTILINESTRING;
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
  typeorder[kPOINT] = 11;
  typeorder[kMULTIPOINT] = 11;
  typeorder[kLINESTRING] = 11;
  typeorder[kMULTILINESTRING] = 11;
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
    case import_export::ImportHeaderRow::kAutoDetect:
      has_headers = detect_headers(head_types, best_sqltypes);
      if (has_headers) {
        copy_params.has_header = import_export::ImportHeaderRow::kHasHeader;
      } else {
        copy_params.has_header = import_export::ImportHeaderRow::kNoHeader;
      }
      break;
    case import_export::ImportHeaderRow::kNoHeader:
      has_headers = false;
      break;
    case import_export::ImportHeaderRow::kHasHeader:
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
                             boost::filesystem::path(file_path).stem().string());
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
                             boost::filesystem::path(file_path).stem().string());
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
#if defined(ENABLE_IMPORT_PARQUET)
  if (data_preview_.has_value()) {
    return data_preview_.value().sample_rows;
  } else
#endif
  {
    n = std::min(n, raw_rows.size());
    size_t offset = (has_headers && raw_rows.size() > 1) ? 1 : 0;
    std::vector<std::vector<std::string>> sample_rows(raw_rows.begin() + offset,
                                                      raw_rows.begin() + n);
    return sample_rows;
  }
}

std::vector<std::string> Detector::get_headers() {
#if defined(ENABLE_IMPORT_PARQUET)
  if (data_preview_.has_value()) {
    return data_preview_.value().column_names;
  } else
#endif
  {
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
}

std::vector<SQLTypeInfo> Detector::getBestColumnTypes() const {
#if defined(ENABLE_IMPORT_PARQUET)
  if (data_preview_.has_value()) {
    return data_preview_.value().column_types;
  } else
#endif
  {
    std::vector<SQLTypeInfo> types;
    CHECK_EQ(best_sqltypes.size(), best_encodings.size());
    for (size_t i = 0; i < best_sqltypes.size(); i++) {
      types.emplace_back(best_sqltypes[i], false, best_encodings[i]);
    }
    return types;
  }
}

void Importer::load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                    size_t row_count,
                    const Catalog_Namespace::SessionInfo* session_info) {
  if (!loader->loadNoCheckpoint(import_buffers, row_count, session_info)) {
    heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
    import_status_.load_failed = true;
    import_status_.load_msg = loader->getErrorMessage();
  }
}

void Importer::checkpoint(
    const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs) {
  if (loader->getTableDesc()->storageType != StorageType::FOREIGN_TABLE) {
    heavyai::lock_guard<heavyai::shared_mutex> read_lock(import_mutex_);
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
      heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
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
    const shared::FilePathOptions options{copy_params.regex_path_filter,
                                          copy_params.file_sort_order_by,
                                          copy_params.file_sort_regex};
    shared::validate_sort_options(options);
    file_paths = shared::local_glob_filter_sort_files(file_path, options);
  } catch (const shared::FileNotFoundException& e) {
    // After finding no matching files locally, file_path may still be an s3 url
    file_paths.push_back(file_path);
  }

  // sum up sizes of all local files -- only for local files. if
  // file_path is a s3 url, sizes will be obtained via S3Archive.
  for (const auto& file_path : file_paths) {
    total_file_size += get_filesize(file_path);
  }

  // s3 parquet goes different route because the files do not use libarchive
  // but parquet api, and they need to landed like .7z files.
  //
  // note: parquet must be explicitly specified by a WITH parameter
  // "source_type='parquet_file'", because for example spark sql users may specify a
  // output url w/o file extension like this:
  //                df.write
  //                  .mode("overwrite")
  //                  .parquet("s3://bucket/folder/parquet/mydata")
  // without the parameter, it means plain or compressed csv files.
  // note: .ORC and AVRO files should follow a similar path to Parquet?
  if (copy_params.source_type == import_export::SourceType::kParquetFile) {
#ifdef ENABLE_IMPORT_PARQUET
    import_parquet(file_paths, session_info);
#else
    throw std::runtime_error("Parquet not supported!");
#endif
  } else {
    import_compressed(file_paths, session_info);
  }

  return import_status_;
}

namespace {
#ifdef ENABLE_IMPORT_PARQUET

#ifdef HAVE_AWS_S3
foreign_storage::ParquetS3DetectFileSystem::ParquetS3DetectFileSystemConfiguration
create_parquet_s3_detect_filesystem_config(const foreign_storage::ForeignServer* server,
                                           const CopyParams& copy_params) {
  foreign_storage::ParquetS3DetectFileSystem::ParquetS3DetectFileSystemConfiguration
      config;

  if (!copy_params.s3_config.access_key.empty()) {
    config.s3_access_key = copy_params.s3_config.access_key;
  }
  if (!copy_params.s3_config.secret_key.empty()) {
    config.s3_secret_key = copy_params.s3_config.secret_key;
  }
  if (!copy_params.s3_config.session_token.empty()) {
    config.s3_session_token = copy_params.s3_config.session_token;
  }

  return config;
}
#endif

foreign_storage::DataPreview get_parquet_data_preview(const std::string& file_name,
                                                      const CopyParams& copy_params) {
  TableDescriptor td;
  td.tableName = "parquet-detect-table";
  td.tableId = -1;
  td.maxFragRows = shared::kDefaultSampleRowsCount;
  auto [foreign_server, user_mapping, foreign_table] =
      foreign_storage::create_proxy_fsi_objects(file_name, copy_params, &td);

  std::shared_ptr<arrow::fs::FileSystem> file_system;
  auto& server_options = foreign_server->options;
  if (server_options
          .find(foreign_storage::AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY)
          ->second ==
      foreign_storage::AbstractFileStorageDataWrapper::LOCAL_FILE_STORAGE_TYPE) {
    file_system = std::make_shared<arrow::fs::LocalFileSystem>();
#ifdef HAVE_AWS_S3
  } else if (server_options
                 .find(foreign_storage::AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY)
                 ->second ==
             foreign_storage::AbstractFileStorageDataWrapper::S3_STORAGE_TYPE) {
    file_system = foreign_storage::ParquetS3DetectFileSystem::create(
        create_parquet_s3_detect_filesystem_config(foreign_server.get(), copy_params));
#endif
  } else {
    UNREACHABLE();
  }

  auto parquet_data_wrapper = std::make_unique<foreign_storage::ParquetDataWrapper>(
      foreign_table.get(), file_system);
  return parquet_data_wrapper->getDataPreview(g_detect_test_sample_size.has_value()
                                                  ? g_detect_test_sample_size.value()
                                                  : shared::kDefaultSampleRowsCount);
}
#endif

}  // namespace

Detector::Detector(const boost::filesystem::path& fp, CopyParams& cp)
    : DataStreamSink(cp, fp.string()), file_path(fp) {
#ifdef ENABLE_IMPORT_PARQUET
  if (cp.source_type == import_export::SourceType::kParquetFile && g_enable_fsi &&
      !g_enable_legacy_parquet_import) {
    data_preview_ = get_parquet_data_preview(fp.string(), cp);
  } else
#endif
  {
    read_file();
    init();
  }
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
}  // namespace import_export

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
    copy_params.has_header = ImportHeaderRow::kHasHeader;
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
        import_export::vector<std::string> buffer;
        for (auto chunk : arrays[c]->chunks()) {
          DataBuffer<std::string, import_export::vector<std::string>::allocator_type>
              data(&cd, *chunk, buffer, nullptr);
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
      heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
      if (++import_status_.rows_completed >= 10000) {
        // as if load truncated
        import_status_.load_failed = true;
        import_status_.load_msg = "Detector processed 10000 records";
        return;
      }
    }
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
  Executor* executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
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
  max_threads = num_import_threads(copy_params.threads);
  VLOG(1) << "Parquet import # threads: " << max_threads;
  const int num_slices = std::max<decltype(max_threads)>(max_threads, num_columns);
  // init row estimate for this file
  const auto filesize = get_filesize(file_path);
  size_t nrow_completed{0};
  file_offsets.push_back(0);
  // map logic column index to physical column index
  auto get_physical_col_idx = [&cds](const int logic_col_idx) -> auto{
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
        heavyai::shared_lock<heavyai::shared_mutex> read_lock(import_mutex_);
        if (import_status_.load_failed) {
          break;
        }
      }
      // a sliced row group will be handled like a (logic) parquet file, with
      // a entirely clean set of bad_rows_tracker, import_buffers_vec, ... etc
      if (UNLIKELY(check_session_interrupted(query_session, executor))) {
        heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
        import_status_.load_failed = true;
        import_status_.load_msg = "Table load was cancelled via Query Interrupt";
      }
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
          if (UNLIKELY(check_session_interrupted(query_session, executor))) {
            heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
            import_status_.load_failed = true;
            import_status_.load_msg = "Table load was cancelled via Query Interrupt";
          }
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
        heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
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
        LOG(INFO) << "rows_completed " << import_status_.rows_completed
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
                                           copy_params.s3_config,
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
        heavyai::shared_lock<heavyai::shared_mutex> read_lock(import_mutex_);
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
    heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
    import_status_.load_failed = true;
    import_status_.load_msg = e.what();
    throw e;
  } catch (const std::exception& e) {
    heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
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
                                      copy_params.s3_config,
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
            if (copy_params.has_header != import_export::ImportHeaderRow::kNoHeader &&
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
                  } else if (errno == EPIPE &&
                             import_status_.rows_completed >= kImportRowLimit) {
                    // the reader thread has shut down the pipe from the read end
                    stop = true;
                    break;
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
                heavyai::shared_lock<heavyai::shared_mutex> read_lock(import_mutex_);
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
                } else if (errno == EPIPE &&
                           import_status_.rows_completed >= kImportRowLimit) {
                  // the reader thread has shut down the pipe from the read end
                  stop = true;
                  break;
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
        heavyai::shared_lock<heavyai::shared_mutex> read_lock(import_mutex_);
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

  max_threads = num_import_threads(copy_params.threads);
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

  ChunkKey chunkKey = {loader->getCatalog().getCurrentDB().dbId,
                       loader->getTableDesc()->tableId};
  auto table_epochs = loader->getTableEpochs();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
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
                                   first_row_index_this_buffer,
                                   session_info,
                                   executor));

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
              heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
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

            LOG(INFO) << "rows_completed " << import_status_.rows_completed
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
        heavyai::shared_lock<heavyai::shared_mutex> read_lock(import_mutex_);
        if (import_status_.load_failed) {
          break;
        }
      }
      heavyai::unique_lock<heavyai::shared_mutex> write_lock(import_mutex_);
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

/* static */
Geospatial::GDAL::DataSourceUqPtr Importer::openGDALDataSource(
    const std::string& file_name,
    const CopyParams& copy_params) {
  Geospatial::GDAL::init();
  Geospatial::GDAL::setAuthorizationTokens(copy_params.s3_config);
  if (copy_params.source_type != import_export::SourceType::kGeoFile) {
    throw std::runtime_error("Unexpected CopyParams.source_type (" +
                             std::to_string(static_cast<int>(copy_params.source_type)) +
                             ")");
  }
  return Geospatial::GDAL::openDataSource(file_name, import_export::SourceType::kGeoFile);
}

namespace {

OGRLayer& getLayerWithSpecifiedName(const std::string& geo_layer_name,
                                    const Geospatial::GDAL::DataSourceUqPtr& poDS,
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
    std::map<std::string, std::vector<std::string>>& sample_data,
    int row_limit,
    const CopyParams& copy_params) {
  Geospatial::GDAL::DataSourceUqPtr datasource(
      openGDALDataSource(file_name, copy_params));
  if (datasource == nullptr) {
    throw std::runtime_error("openGDALDataSource Error: Unable to open geo file " +
                             file_name);
  }

  OGRLayer& layer =
      getLayerWithSpecifiedName(copy_params.geo_layer_name, datasource, file_name);

  auto const* feature_defn = layer.GetLayerDefn();
  CHECK(feature_defn);

  // metadata columns?
  auto const metadata_column_infos =
      parse_add_metadata_columns(copy_params.add_metadata_columns, file_name);

  // get limited feature count
  // typeof GetFeatureCount() is different between GDAL 1.x (int32_t) and 2.x (int64_t)
  auto const feature_count = static_cast<uint64_t>(layer.GetFeatureCount());
  auto const num_features = std::min(static_cast<uint64_t>(row_limit), feature_count);

  // prepare sample data map
  for (int field_index = 0; field_index < feature_defn->GetFieldCount(); field_index++) {
    auto const* column_name = feature_defn->GetFieldDefn(field_index)->GetNameRef();
    CHECK(column_name);
    sample_data[column_name] = {};
  }
  sample_data[geo_column_name] = {};
  for (auto const& mci : metadata_column_infos) {
    sample_data[mci.column_descriptor.columnName] = {};
  }

  // prepare to read
  layer.ResetReading();

  // read features (up to limited count)
  uint64_t feature_index{0u};
  while (feature_index < num_features) {
    // get (and take ownership of) feature
    Geospatial::GDAL::FeatureUqPtr feature(layer.GetNextFeature());
    if (!feature) {
      break;
    }

    // get feature geometry
    auto const* geometry = feature->GetGeometryRef();
    if (geometry == nullptr) {
      break;
    }

    // validate geom type (again?)
    switch (wkbFlatten(geometry->getGeometryType())) {
      case wkbPoint:
      case wkbMultiPoint:
      case wkbLineString:
      case wkbMultiLineString:
      case wkbPolygon:
      case wkbMultiPolygon:
        break;
      default:
        throw std::runtime_error("Unsupported geometry type: " +
                                 std::string(geometry->getGeometryName()));
    }

    // populate sample data for regular field columns
    for (int field_index = 0; field_index < feature->GetFieldCount(); field_index++) {
      auto const* column_name = feature_defn->GetFieldDefn(field_index)->GetNameRef();
      sample_data[column_name].push_back(feature->GetFieldAsString(field_index));
    }

    // populate sample data for metadata columns?
    for (auto const& mci : metadata_column_infos) {
      sample_data[mci.column_descriptor.columnName].push_back(mci.value);
    }

    // populate sample data for geo column with WKT string
    char* wkts = nullptr;
    geometry->exportToWkt(&wkts);
    CHECK(wkts);
    sample_data[geo_column_name].push_back(wkts);
    CPLFree(wkts);

    // next feature
    feature_index++;
  }
}

namespace {

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
    case wkbMultiPoint:
      return kMULTIPOINT;
    case wkbLineString:
      return kLINESTRING;
    case wkbMultiLineString:
      return kMULTILINESTRING;
    case wkbPolygon:
      return kPOLYGON;
    case wkbMultiPolygon:
      return kMULTIPOLYGON;
    default:
      break;
  }
  throw std::runtime_error("Unknown OGR geom type: " + std::to_string(ogr_type));
}

RasterImporter::PointType convert_raster_point_type(
    const import_export::RasterPointType raster_point_type) {
  switch (raster_point_type) {
    case import_export::RasterPointType::kNone:
      return RasterImporter::PointType::kNone;
    case import_export::RasterPointType::kAuto:
      return RasterImporter::PointType::kAuto;
    case import_export::RasterPointType::kSmallInt:
      return RasterImporter::PointType::kSmallInt;
    case import_export::RasterPointType::kInt:
      return RasterImporter::PointType::kInt;
    case import_export::RasterPointType::kFloat:
      return RasterImporter::PointType::kFloat;
    case import_export::RasterPointType::kDouble:
      return RasterImporter::PointType::kDouble;
    case import_export::RasterPointType::kPoint:
      return RasterImporter::PointType::kPoint;
  }
  UNREACHABLE();
  return RasterImporter::PointType::kNone;
}

RasterImporter::PointTransform convert_raster_point_transform(
    const import_export::RasterPointTransform raster_point_transform) {
  switch (raster_point_transform) {
    case import_export::RasterPointTransform::kNone:
      return RasterImporter::PointTransform::kNone;
    case import_export::RasterPointTransform::kAuto:
      return RasterImporter::PointTransform::kAuto;
    case import_export::RasterPointTransform::kFile:
      return RasterImporter::PointTransform::kFile;
    case import_export::RasterPointTransform::kWorld:
      return RasterImporter::PointTransform::kWorld;
  }
  UNREACHABLE();
  return RasterImporter::PointTransform::kNone;
}

}  // namespace

/* static */
const std::list<ColumnDescriptor> Importer::gdalToColumnDescriptors(
    const std::string& file_name,
    const bool is_raster,
    const std::string& geo_column_name,
    const CopyParams& copy_params) {
  if (is_raster) {
    return gdalToColumnDescriptorsRaster(file_name, geo_column_name, copy_params);
  }
  return gdalToColumnDescriptorsGeo(file_name, geo_column_name, copy_params);
}

/* static */
const std::list<ColumnDescriptor> Importer::gdalToColumnDescriptorsRaster(
    const std::string& file_name,
    const std::string& geo_column_name,
    const CopyParams& copy_params) {
  // lazy init GDAL
  Geospatial::GDAL::init();
  Geospatial::GDAL::setAuthorizationTokens(copy_params.s3_config);

  // check for unsupported options
  if (!copy_params.bounding_box_clip.empty()) {
    throw std::runtime_error(
        "Bounding Box Clip option not supported by Legacy Raster Importer");
  }
  if (copy_params.raster_import_bands.find("/") != std::string::npos) {
    throw std::runtime_error(
        "Advanced Raster Import Bands syntax not supported by Legacy Raster Importer");
  }

  // prepare for metadata column
  auto metadata_column_infos =
      parse_add_metadata_columns(copy_params.add_metadata_columns, file_name);

  // create a raster importer and do the detect
  RasterImporter raster_importer;
  raster_importer.detect(
      file_name,
      copy_params.raster_import_bands,
      copy_params.raster_import_dimensions,
      convert_raster_point_type(copy_params.raster_point_type),
      convert_raster_point_transform(copy_params.raster_point_transform),
      false,
      metadata_column_infos);

  // prepare to capture column descriptors
  std::list<ColumnDescriptor> cds;

  // get the point column info
  auto const point_names_and_sql_types = raster_importer.getPointNamesAndSQLTypes();

  // create the columns for the point in the specified type
  for (auto const& [col_name, sql_type] : point_names_and_sql_types) {
    ColumnDescriptor cd;
    cd.columnName = cd.sourceName = col_name;
    cd.columnType.set_type(sql_type);
    // hardwire other POINT attributes for now
    if (sql_type == kPOINT) {
      cd.columnType.set_subtype(kGEOMETRY);
      cd.columnType.set_input_srid(4326);
      cd.columnType.set_output_srid(4326);
      cd.columnType.set_compression(kENCODING_GEOINT);
      cd.columnType.set_comp_param(32);
    }
    cds.push_back(cd);
  }

  // get the names and types for the band column(s)
  auto const band_names_and_types = raster_importer.getBandNamesAndSQLTypes();

  // add column descriptors for each band
  for (auto const& [band_name, sql_type] : band_names_and_types) {
    ColumnDescriptor cd;
    cd.columnName = cd.sourceName = band_name;
    cd.columnType.set_type(sql_type);
    cd.columnType.set_fixed_size();
    cds.push_back(cd);
  }

  // metadata columns?
  for (auto& mci : metadata_column_infos) {
    cds.push_back(std::move(mci.column_descriptor));
  }

  // return the results
  return cds;
}

/* static */
const std::list<ColumnDescriptor> Importer::gdalToColumnDescriptorsGeo(
    const std::string& file_name,
    const std::string& geo_column_name,
    const CopyParams& copy_params) {
  std::list<ColumnDescriptor> cds;

  Geospatial::GDAL::DataSourceUqPtr poDS(openGDALDataSource(file_name, copy_params));
  if (poDS == nullptr) {
    throw std::runtime_error("openGDALDataSource Error: Unable to open geo file " +
                             file_name);
  }
  if (poDS->GetLayerCount() == 0) {
    throw std::runtime_error("gdalToColumnDescriptors Error: Geo file " + file_name +
                             " has no layers");
  }

  OGRLayer& layer =
      getLayerWithSpecifiedName(copy_params.geo_layer_name, poDS, file_name);

  layer.ResetReading();
  // TODO(andrewseidl): support multiple features
  Geospatial::GDAL::FeatureUqPtr poFeature(layer.GetNextFeature());
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
      ti.set_comp_param(0);
    }
    ti.set_fixed_size();
    cd.columnType = ti;
    cds.push_back(cd);
  }
  // try getting the geo column type from the layer
  auto ogr_type = wkbFlatten(layer.GetGeomType());
  if (ogr_type == wkbUnknown) {
    // layer geo type unknown, so try the feature (that we already got)
    auto const* ogr_geometry = poFeature->GetGeometryRef();
    if (ogr_geometry) {
      ogr_type = wkbFlatten(ogr_geometry->getGeometryType());
    }
  }
  // do we have a geo column?
  if (ogr_type != wkbNone) {
    ColumnDescriptor cd;
    cd.columnName = geo_column_name;
    cd.sourceName = geo_column_name;

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
    // this will throw if the type is unsupported
    SQLTypes geoType = ogr_to_type(ogr_type);

    // promote column type? (unless exploding)
    if (!copy_params.geo_explode_collections) {
      if (PROMOTE_POINT_TO_MULTIPOINT && geoType == kPOINT) {
        geoType = kMULTIPOINT;
      } else if (PROMOTE_LINESTRING_TO_MULTILINESTRING && geoType == kLINESTRING) {
        geoType = kMULTILINESTRING;
      } else if (PROMOTE_POLYGON_TO_MULTIPOLYGON && geoType == kPOLYGON) {
        geoType = kMULTIPOLYGON;
      }
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

  // metadata columns?
  auto metadata_column_infos =
      parse_add_metadata_columns(copy_params.add_metadata_columns, file_name);
  for (auto& mci : metadata_column_infos) {
    cds.push_back(std::move(mci.column_descriptor));
  }

  return cds;
}

bool Importer::gdalStatInternal(const std::string& path,
                                const CopyParams& copy_params,
                                bool also_dir) {
  // lazy init GDAL
  Geospatial::GDAL::init();
  Geospatial::GDAL::setAuthorizationTokens(copy_params.s3_config);

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
  Geospatial::GDAL::setAuthorizationTokens(copy_params.s3_config);

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
  Geospatial::GDAL::setAuthorizationTokens(copy_params.s3_config);

  // prepare to gather layer info
  std::vector<GeoFileLayerInfo> layer_info;

  // open the data set
  Geospatial::GDAL::DataSourceUqPtr poDS(openGDALDataSource(file_name, copy_params));
  if (poDS == nullptr) {
    throw std::runtime_error("openGDALDataSource Error: Unable to open geo file " +
                             file_name);
  }

  // enumerate the layers
  for (auto&& poLayer : poDS->GetLayers()) {
    GeoFileLayerContents contents = GeoFileLayerContents::EMPTY;
    // prepare to read this layer
    poLayer->ResetReading();
    // skip layer if empty
    if (poLayer->GetFeatureCount() > 0) {
      // first read layer geo type
      auto ogr_type = wkbFlatten(poLayer->GetGeomType());
      if (ogr_type == wkbUnknown) {
        // layer geo type unknown, so try reading from the first feature
        Geospatial::GDAL::FeatureUqPtr first_feature(poLayer->GetNextFeature());
        CHECK(first_feature);
        auto const* ogr_geometry = first_feature->GetGeometryRef();
        if (ogr_geometry) {
          ogr_type = wkbFlatten(ogr_geometry->getGeometryType());
        } else {
          ogr_type = wkbNone;
        }
      }
      switch (ogr_type) {
        case wkbNone:
          // no geo
          contents = GeoFileLayerContents::NON_GEO;
          break;
        case wkbPoint:
        case wkbMultiPoint:
        case wkbLineString:
        case wkbMultiLineString:
        case wkbPolygon:
        case wkbMultiPolygon:
          // layer has supported geo
          contents = GeoFileLayerContents::GEO;
          break;
        default:
          // layer has unsupported geometry
          contents = GeoFileLayerContents::UNSUPPORTED_GEO;
          break;
      }
    }
    // store info for this layer
    layer_info.emplace_back(poLayer->GetName(), contents);
  }

  // done
  return layer_info;
}

ImportStatus Importer::importGDAL(
    const ColumnNameToSourceNameMapType& columnNameToSourceNameMap,
    const Catalog_Namespace::SessionInfo* session_info,
    const bool is_raster) {
  if (is_raster) {
    return importGDALRaster(session_info);
  }
  return importGDALGeo(columnNameToSourceNameMap, session_info);
}

ImportStatus Importer::importGDALGeo(
    const ColumnNameToSourceNameMapType& columnNameToSourceNameMap,
    const Catalog_Namespace::SessionInfo* session_info) {
  // initial status
  set_import_status(import_id, import_status_);
  Geospatial::GDAL::DataSourceUqPtr poDS(openGDALDataSource(file_path, copy_params));
  if (poDS == nullptr) {
    throw std::runtime_error("openGDALDataSource Error: Unable to open geo file " +
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
  Geospatial::GDAL::SpatialReferenceUqPtr poGeographicSR(new OGRSpatialReference());
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
  max_threads = num_import_threads(copy_params.threads);
#endif
  VLOG(1) << "GDAL import # threads: " << max_threads;

  // metadata columns?
  auto const metadata_column_infos =
      parse_add_metadata_columns(copy_params.add_metadata_columns, file_path);

  // import geo table is specifically handled in both DBHandler and QueryRunner
  // that is separate path against a normal SQL execution
  // so we here explicitly enroll the import session to allow interruption
  // while importing geo table
  auto query_session = session_info ? session_info->get_session_id() : "";
  auto query_submitted_time = ::toString(std::chrono::system_clock::now());
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto is_session_already_registered = false;
  {
    heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
        executor->getSessionLock());
    is_session_already_registered =
        executor->checkIsQuerySessionEnrolled(query_session, session_read_lock);
  }
  if (g_enable_non_kernel_time_query_interrupt && !query_session.empty() &&
      !is_session_already_registered) {
    executor->enrollQuerySession(query_session,
                                 "IMPORT_GEO_TABLE",
                                 query_submitted_time,
                                 Executor::UNITARY_EXECUTOR_ID,
                                 QuerySessionStatus::QueryStatus::RUNNING_IMPORTER);
  }
  ScopeGuard clearInterruptStatus = [executor, &query_session, &query_submitted_time] {
    // reset the runtime query interrupt status
    if (g_enable_non_kernel_time_query_interrupt && !query_session.empty()) {
      executor->clearQuerySessionStatus(query_session, query_submitted_time);
    }
  };

  // make an import buffer for each thread
  CHECK_EQ(import_buffers_vec.size(), 0u);
  import_buffers_vec.resize(max_threads);
  for (size_t i = 0; i < max_threads; i++) {
    for (const auto cd : loader->get_column_descs()) {
      import_buffers_vec[i].emplace_back(
          new TypedImportBuffer(cd, loader->getStringDict(cd)));
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

  // make one of these for each thread, based on the first feature's SR
  std::vector<std::unique_ptr<OGRCoordinateTransformation>> coordinate_transformations(
      max_threads);

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

    // construct a coordinate transformation for each thread, if needed
    // some features may not have geometry, so look for the first one that does
    if (coordinate_transformations[thread_id] == nullptr) {
      for (auto const& feature : features[thread_id]) {
        auto const* geometry = feature->GetGeometryRef();
        if (geometry) {
          auto const* geometry_sr = geometry->getSpatialReference();
          // if the SR is non-null and non-empty and different from what we want
          // we need to make a reusable CoordinateTransformation
          if (geometry_sr &&
#if GDAL_VERSION_MAJOR >= 3
              !geometry_sr->IsEmpty() &&
#endif
              !geometry_sr->IsSame(poGeographicSR.get())) {
            // create the OGRCoordinateTransformation that will be used for
            // all the features in this chunk
            coordinate_transformations[thread_id].reset(
                OGRCreateCoordinateTransformation(geometry_sr, poGeographicSR.get()));
            if (coordinate_transformations[thread_id] == nullptr) {
              throw std::runtime_error(
                  "Failed to create a GDAL CoordinateTransformation for incoming geo");
            }
          }
          // once we find at least one geometry with an SR, we're done
          break;
        }
      }
    }

    // parse bounding box clip string
    auto const bounding_box_clip =
        shared::LonLatBoundingBox::parse(copy_params.bounding_box_clip);

#if DISABLE_MULTI_THREADED_SHAPEFILE_IMPORT
    // call worker function directly
    auto ret_import_status =
        import_thread_shapefile(0,
                                this,
                                coordinate_transformations[thread_id].get(),
                                std::move(features[thread_id]),
                                firstFeatureThisChunk,
                                numFeaturesThisChunk,
                                fieldNameToIndexMap,
                                columnNameToSourceNameMap,
                                session_info,
                                executor.get(),
                                metadata_column_infos,
                                bounding_box_clip);
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
                                 coordinate_transformations[thread_id].get(),
                                 std::move(features[thread_id]),
                                 firstFeatureThisChunk,
                                 numFeaturesThisChunk,
                                 fieldNameToIndexMap,
                                 columnNameToSourceNameMap,
                                 session_info,
                                 executor.get(),
                                 metadata_column_infos,
                                 bounding_box_clip));

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
          {
            heavyai::lock_guard<heavyai::shared_mutex> write_lock(import_mutex_);
            import_status_ += ret_import_status;
            import_status_.rows_estimated =
                ((float)firstFeatureThisChunk / (float)numFeatures) *
                import_status_.rows_completed;
            set_import_status(import_id, import_status_);
            if (import_status_.load_failed) {
              break;
            }
          }
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

    heavyai::unique_lock<heavyai::shared_mutex> write_lock(import_mutex_);
    if (import_status_.rows_rejected > copy_params.max_reject) {
      import_status_.load_failed = true;
      // todo use better message
      import_status_.load_msg = "Maximum rows rejected exceeded. Halting load";
      LOG(ERROR) << "Maximum rows rejected exceeded. Halting load";
      break;
    }
    if (import_status_.load_failed) {
      LOG(ERROR) << "A call to the Loader failed in GDAL, Please review the logs for "
                    "more details";
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
      import_status_ += ret_import_status;
      import_status_.rows_estimated = import_status_.rows_completed;
      set_import_status(import_id, import_status_);
    }
  }
#endif

  checkpoint(table_epochs);

  return import_status_;
}

ImportStatus Importer::importGDALRaster(
    const Catalog_Namespace::SessionInfo* session_info) {
  // initial status
  set_import_status(import_id, import_status_);

  // check for unsupported options
  if (!copy_params.bounding_box_clip.empty()) {
    throw std::runtime_error(
        "Bounding Box Clip option not supported by Legacy Raster Importer");
  }
  if (copy_params.raster_import_bands.find("/") != std::string::npos) {
    throw std::runtime_error(
        "Advanced Raster Import Bands syntax not supported by Legacy Raster Importer");
  }

  // metadata columns?
  auto const metadata_column_infos =
      parse_add_metadata_columns(copy_params.add_metadata_columns, file_path);

  // create a raster importer and do the detect
  RasterImporter raster_importer;
  raster_importer.detect(
      file_path,
      copy_params.raster_import_bands,
      copy_params.raster_import_dimensions,
      convert_raster_point_type(copy_params.raster_point_type),
      convert_raster_point_transform(copy_params.raster_point_transform),
      true,
      metadata_column_infos);

  // get the table columns and count actual columns
  auto const& column_descs = loader->get_column_descs();
  uint32_t num_table_cols{0u};
  for (auto const* cd : column_descs) {
    if (!cd->isGeoPhyCol) {
      num_table_cols++;
    }
  }

  // how many bands do we have?
  auto num_bands = raster_importer.getNumBands();

  // get point columns info
  auto const point_names_and_sql_types = raster_importer.getPointNamesAndSQLTypes();

  // validate that the table column count matches
  auto num_expected_cols = num_bands;
  num_expected_cols += point_names_and_sql_types.size();
  num_expected_cols += metadata_column_infos.size();
  if (num_expected_cols != num_table_cols) {
    throw std::runtime_error(
        "Raster Import aborted. Band/Column count mismatch (file requires " +
        std::to_string(num_expected_cols) + ", table has " +
        std::to_string(num_table_cols) + ")");
  }

  // validate the point column names and types
  // if we're importing the coords as a POINT, then the first column
  // must be a POINT (two physical columns, POINT and TINYINT[])
  // if we're not, the first two columns must be the matching type
  auto cd_itr = column_descs.begin();
  for (auto const& [col_name, sql_type] : point_names_and_sql_types) {
    if (sql_type == kPOINT) {
      // POINT column
      {
        auto const* cd = *cd_itr++;
        if (cd->columnName != col_name) {
          throw std::runtime_error("Column '" + cd->columnName +
                                   "' overridden name invalid (must be '" + col_name +
                                   "')");
        }
        auto const cd_type = cd->columnType.get_type();
        if (cd_type != kPOINT) {
          throw std::runtime_error("Column '" + cd->columnName +
                                   "' overridden type invalid (must be POINT)");
        }
        if (cd->columnType.get_output_srid() != 4326) {
          throw std::runtime_error("Column '" + cd->columnName +
                                   "' overridden SRID invalid (must be 4326)");
        }
      }
      // TINYINT[] coords sub-column
      {
        // if the above is true, this must be true
        auto const* cd = *cd_itr++;
        CHECK(cd->columnType.get_type() == kARRAY);
        CHECK(cd->columnType.get_subtype() == kTINYINT);
      }
    } else {
      // column of the matching name and type
      auto const* cd = *cd_itr++;
      if (cd->columnName != col_name) {
        throw std::runtime_error("Column '" + cd->columnName +
                                 "' overridden name invalid (must be '" + col_name +
                                 "')");
      }
      auto const cd_type = cd->columnType.get_type();
      if (cd_type != sql_type) {
        throw std::runtime_error("Column '" + cd->columnName +
                                 "' overridden type invalid (must be " +
                                 to_string(sql_type) + ")");
      }
    }
  }

  // validate the band column types
  // any Immerse overriding to other types will currently be rejected
  auto const band_names_and_types = raster_importer.getBandNamesAndSQLTypes();
  if (band_names_and_types.size() != num_bands) {
    throw std::runtime_error("Column/Band count mismatch when validating types");
  }
  for (uint32_t i = 0; i < num_bands; i++) {
    auto const* cd = *cd_itr++;
    auto const cd_type = cd->columnType.get_type();
    auto const sql_type = band_names_and_types[i].second;
    if (cd_type != sql_type) {
      throw std::runtime_error("Band Column '" + cd->columnName +
                               "' overridden type invalid (must be " +
                               to_string(sql_type) + ")");
    }
  }

  // validate metadata column
  for (auto const& mci : metadata_column_infos) {
    auto const* cd = *cd_itr++;
    if (mci.column_descriptor.columnName != cd->columnName) {
      throw std::runtime_error("Metadata Column '" + cd->columnName +
                               "' overridden name invalid (must be '" +
                               mci.column_descriptor.columnName + "')");
    }
    auto const cd_type = cd->columnType.get_type();
    auto const md_type = mci.column_descriptor.columnType.get_type();
    if (cd_type != md_type) {
      throw std::runtime_error("Metadata Column '" + cd->columnName +
                               "' overridden type invalid (must be " +
                               to_string(md_type) + ")");
    }
  }

  // import geo table is specifically handled in both DBHandler and QueryRunner
  // that is separate path against a normal SQL execution
  // so we here explicitly enroll the import session to allow interruption
  // while importing geo table
  auto query_session = session_info ? session_info->get_session_id() : "";
  auto query_submitted_time = ::toString(std::chrono::system_clock::now());
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto is_session_already_registered = false;
  {
    heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
        executor->getSessionLock());
    is_session_already_registered =
        executor->checkIsQuerySessionEnrolled(query_session, session_read_lock);
  }
  if (g_enable_non_kernel_time_query_interrupt && !query_session.empty() &&
      !is_session_already_registered) {
    executor->enrollQuerySession(query_session,
                                 "IMPORT_GEO_TABLE",
                                 query_submitted_time,
                                 Executor::UNITARY_EXECUTOR_ID,
                                 QuerySessionStatus::QueryStatus::RUNNING_IMPORTER);
  }
  ScopeGuard clearInterruptStatus = [executor, &query_session, &query_submitted_time] {
    // reset the runtime query interrupt status
    if (g_enable_non_kernel_time_query_interrupt && !query_session.empty()) {
      executor->clearQuerySessionStatus(query_session, query_submitted_time);
    }
  };

  // how many threads are we gonna use?
  max_threads = num_import_threads(copy_params.threads);
  VLOG(1) << "GDAL import # threads: " << max_threads;

  if (copy_params.raster_scanlines_per_thread < 0) {
    throw std::runtime_error("Invalid CopyParams.raster_scanlines_per_thread! (" +
                             std::to_string(copy_params.raster_scanlines_per_thread) +
                             ")");
  }
  const int max_scanlines_per_thread =
      copy_params.raster_scanlines_per_thread == 0
          ? kMaxRasterScanlinesPerThread
          : std::min(copy_params.raster_scanlines_per_thread,
                     kMaxRasterScanlinesPerThread);
  VLOG(1) << "Raster Importer: Max scanlines per thread: " << max_scanlines_per_thread;

  // prepare to checkpoint the table
  auto table_epochs = loader->getTableEpochs();

  // start wall clock
  auto wall_timer = timer_start();

  // start the import
  raster_importer.import(
      max_threads,
      copy_params.threads == 0);  // NOTE: `max_threads` may change after this call

  // make an import buffer for each thread
  CHECK_EQ(import_buffers_vec.size(), 0u);
  import_buffers_vec.resize(max_threads);
  for (size_t i = 0; i < max_threads; i++) {
    for (auto const& cd : loader->get_column_descs()) {
      import_buffers_vec[i].emplace_back(
          new TypedImportBuffer(cd, loader->getStringDict(cd)));
    }
  }

  // status and times
  using ThreadReturn = std::tuple<ImportStatus, std::array<float, 3>>;

  // get the band dimensions
  auto const band_size_x = raster_importer.getBandsWidth();
  auto const band_size_y = raster_importer.getBandsHeight();

  // allocate raw pixel buffers per thread
  std::vector<RasterImporter::RawPixels> raw_pixel_bytes_per_thread(max_threads);
  for (size_t i = 0; i < max_threads; i++) {
    raw_pixel_bytes_per_thread[i].resize(band_size_x * max_scanlines_per_thread *
                                         sizeof(double));
  }

  // just the sql type of the first point column (if any)
  auto const point_sql_type = point_names_and_sql_types.size()
                                  ? point_names_and_sql_types.begin()->second
                                  : kNULLT;

  // lambda for importing to raw data buffers (threadable)
  auto import_rows =
      [&](const size_t thread_idx, const int y_start, const int y_end) -> ThreadReturn {
    // this threads's import buffers
    auto& import_buffers = import_buffers_vec[thread_idx];

    // this thread's raw pixel bytes
    auto& raw_pixel_bytes = raw_pixel_bytes_per_thread[thread_idx];

    // clear the buffers
    for (auto& col_buffer : import_buffers) {
      col_buffer->clear();
    }

    // prepare to iterate columns
    auto col_itr = column_descs.begin();
    int col_idx{0};

    float proj_s{0.0f};
    if (point_sql_type != kNULLT) {
      // the first two columns (either lon/lat or POINT/coords)
      auto const* cd_col0 = *col_itr++;
      auto const* cd_col1 = *col_itr++;

      // compute and add x and y
      auto proj_timer = timer_start();
      for (int y = y_start; y < y_end; y++) {
        // get projected pixel coords for this scan-line
        auto const coords = raster_importer.getProjectedPixelCoords(thread_idx, y);

        // add to buffers
        for (int x = 0; x < band_size_x; x++) {
          // this point
          auto const& [dx, dy] = coords[x];

          // store the point
          switch (point_sql_type) {
            case kPOINT: {
              // add empty value to POINT buffer
              TDatum td_point;
              import_buffers[0]->add_value(cd_col0, td_point, false);

              // convert lon/lat to bytes (compressed or not) and add to POINT coords
              // buffer
              auto const compressed_coords =
                  Geospatial::compress_coords({dx, dy}, cd_col0->columnType);
              std::vector<TDatum> td_coords_data;
              for (auto const& cc : compressed_coords) {
                TDatum td_byte;
                td_byte.val.int_val = cc;
                td_coords_data.push_back(td_byte);
              }
              TDatum td_coords;
              td_coords.val.arr_val = td_coords_data;
              td_coords.is_null = false;
              import_buffers[1]->add_value(cd_col1, td_coords, false);
            } break;
            case kFLOAT:
            case kDOUBLE: {
              TDatum td;
              td.is_null = false;
              td.val.real_val = dx;
              import_buffers[0]->add_value(cd_col0, td, false);
              td.val.real_val = dy;
              import_buffers[1]->add_value(cd_col1, td, false);
            } break;
            case kSMALLINT:
            case kINT: {
              TDatum td;
              td.is_null = false;
              td.val.int_val = static_cast<int64_t>(x);
              import_buffers[0]->add_value(cd_col0, td, false);
              td.val.int_val = static_cast<int64_t>(y);
              import_buffers[1]->add_value(cd_col1, td, false);
            } break;
            default:
              CHECK(false);
          }
        }
      }
      proj_s = TIMER_STOP(proj_timer);
      col_idx += 2;
    }

    // prepare to accumulate read and conv times
    float read_s{0.0f};
    float conv_s{0.0f};

    // y_end is one past the actual end, so don't add 1
    auto const num_rows = y_end - y_start;
    auto const num_elems = band_size_x * num_rows;

    ImportStatus thread_import_status;

    bool read_block_failed = false;

    // prepare to store which band values in which rows are null
    boost::dynamic_bitset<> row_band_nulls;
    if (copy_params.raster_drop_if_all_null) {
      row_band_nulls.resize(num_elems * num_bands);
    }

    auto set_row_band_null = [&](const int row, const uint32_t band) {
      auto const bit_index = (row * num_bands) + band;
      row_band_nulls.set(bit_index);
    };
    auto all_row_bands_null = [&](const int row) -> bool {
      auto const first_bit_index = row * num_bands;
      bool all_null = true;
      for (auto i = first_bit_index; i < first_bit_index + num_bands; i++) {
        all_null = all_null && row_band_nulls.test(i);
      }
      return all_null;
    };

    // for each band/column
    for (uint32_t band_idx = 0; band_idx < num_bands; band_idx++) {
      // the corresponding column
      auto const* cd_band = *col_itr;
      CHECK(cd_band);

      // data type to read as
      auto const cd_type = cd_band->columnType.get_type();

      // read the scanlines (will do a data type conversion if necessary)
      try {
        auto read_timer = timer_start();
        raster_importer.getRawPixels(
            thread_idx, band_idx, y_start, num_rows, cd_type, raw_pixel_bytes);
        read_s += TIMER_STOP(read_timer);
      } catch (std::runtime_error& e) {
        // report error
        LOG(ERROR) << e.what();
        // abort this block
        read_block_failed = true;
        break;
      }

      // null value?
      auto const [null_value, null_value_valid] =
          raster_importer.getBandNullValue(band_idx);

      // copy to this thread's import buffers
      // convert any nulls we find
      auto conv_timer = timer_start();
      TDatum td;
      switch (cd_type) {
        case kSMALLINT: {
          const int16_t* values =
              reinterpret_cast<const int16_t*>(raw_pixel_bytes.data());
          for (int idx = 0; idx < num_elems; idx++) {
            auto const& value = values[idx];
            if (null_value_valid && value == static_cast<int16_t>(null_value)) {
              td.is_null = true;
              td.val.int_val = NULL_SMALLINT;
              if (copy_params.raster_drop_if_all_null) {
                set_row_band_null(idx, band_idx);
              }
            } else {
              td.is_null = false;
              td.val.int_val = static_cast<int64_t>(value);
            }
            import_buffers[col_idx]->add_value(cd_band, td, false);
          }
        } break;
        case kINT: {
          const int32_t* values =
              reinterpret_cast<const int32_t*>(raw_pixel_bytes.data());
          for (int idx = 0; idx < num_elems; idx++) {
            auto const& value = values[idx];
            if (null_value_valid && value == static_cast<int32_t>(null_value)) {
              td.is_null = true;
              td.val.int_val = NULL_INT;
              if (copy_params.raster_drop_if_all_null) {
                set_row_band_null(idx, band_idx);
              }
            } else {
              td.is_null = false;
              td.val.int_val = static_cast<int64_t>(value);
            }
            import_buffers[col_idx]->add_value(cd_band, td, false);
          }
        } break;
        case kBIGINT: {
          const uint32_t* values =
              reinterpret_cast<const uint32_t*>(raw_pixel_bytes.data());
          for (int idx = 0; idx < num_elems; idx++) {
            auto const& value = values[idx];
            if (null_value_valid && value == static_cast<uint32_t>(null_value)) {
              td.is_null = true;
              td.val.int_val = NULL_INT;
              if (copy_params.raster_drop_if_all_null) {
                set_row_band_null(idx, band_idx);
              }
            } else {
              td.is_null = false;
              td.val.int_val = static_cast<int64_t>(value);
            }
            import_buffers[col_idx]->add_value(cd_band, td, false);
          }
        } break;
        case kFLOAT: {
          const float* values = reinterpret_cast<const float*>(raw_pixel_bytes.data());
          for (int idx = 0; idx < num_elems; idx++) {
            auto const& value = values[idx];
            if (null_value_valid && value == static_cast<float>(null_value)) {
              td.is_null = true;
              td.val.real_val = NULL_FLOAT;
              if (copy_params.raster_drop_if_all_null) {
                set_row_band_null(idx, band_idx);
              }
            } else {
              td.is_null = false;
              td.val.real_val = static_cast<double>(value);
            }
            import_buffers[col_idx]->add_value(cd_band, td, false);
          }
        } break;
        case kDOUBLE: {
          const double* values = reinterpret_cast<const double*>(raw_pixel_bytes.data());
          for (int idx = 0; idx < num_elems; idx++) {
            auto const& value = values[idx];
            if (null_value_valid && value == null_value) {
              td.is_null = true;
              td.val.real_val = NULL_DOUBLE;
              if (copy_params.raster_drop_if_all_null) {
                set_row_band_null(idx, band_idx);
              }
            } else {
              td.is_null = false;
              td.val.real_val = value;
            }
            import_buffers[col_idx]->add_value(cd_band, td, false);
          }
        } break;
        default:
          CHECK(false);
      }
      conv_s += TIMER_STOP(conv_timer);

      // next column
      col_idx++;
      col_itr++;
    }

    if (read_block_failed) {
      // discard block data
      for (auto& col_buffer : import_buffers) {
        col_buffer->clear();
      }
      thread_import_status.rows_estimated = 0;
      thread_import_status.rows_completed = 0;
      thread_import_status.rows_rejected = num_elems;
    } else {
      // metadata columns?
      for (auto const& mci : metadata_column_infos) {
        auto const* cd_band = *col_itr++;
        CHECK(cd_band);
        for (int i = 0; i < num_elems; i++) {
          import_buffers[col_idx]->add_value(cd_band, mci.value, false, copy_params);
        }
        col_idx++;
      }

      // drop rows where all band columns are null?
      int num_dropped_as_all_null = 0;
      if (copy_params.raster_drop_if_all_null) {
        // capture rows where ALL the band values (only) were NULL
        // count rows first (implies two passes on the bitset but
        // still quicker than building the row set if not needed,
        // in the case where ALL rows are to be dropped)
        for (int row = 0; row < num_elems; row++) {
          if (all_row_bands_null(row)) {
            num_dropped_as_all_null++;
          }
        }
        // delete those rows from ALL column buffers (including coords and metadata)
        if (num_dropped_as_all_null == num_elems) {
          // all rows need dropping, just clear (fast)
          for (auto& col_buffer : import_buffers) {
            col_buffer->clear();
          }
        } else if (num_dropped_as_all_null > 0) {
          // drop "bad" rows selectively (slower)
          // build row set to drop
          BadRowsTracker bad_rows_tracker;
          for (int row = 0; row < num_elems; row++) {
            if (all_row_bands_null(row)) {
              bad_rows_tracker.rows.emplace(static_cast<int64_t>(row));
            }
          }
          // then delete rows
          for (auto& col_buffer : import_buffers) {
            auto const* cd = col_buffer->getColumnDesc();
            CHECK(cd);
            auto const col_type = cd->columnType.get_type();
            col_buffer->del_values(col_type, &bad_rows_tracker);
          }
        }
      }

      // final count
      CHECK_LE(num_dropped_as_all_null, num_elems);
      auto const actual_num_elems = num_elems - num_dropped_as_all_null;
      thread_import_status.rows_estimated = actual_num_elems;
      thread_import_status.rows_completed = actual_num_elems;
      thread_import_status.rows_rejected = 0;
    }

    // done
    return {std::move(thread_import_status), {proj_s, read_s, conv_s}};
  };

  // time the phases
  float total_proj_s{0.0f};
  float total_read_s{0.0f};
  float total_conv_s{0.0f};
  float total_load_s{0.0f};

  const int min_scanlines_per_thread = 8;
  const int max_scanlines_per_block = max_scanlines_per_thread * max_threads;
  for (int block_y = 0; block_y < band_size_y;
       block_y += (max_threads * max_scanlines_per_thread)) {
    using Future = std::future<ThreadReturn>;
    std::vector<Future> futures;
    const int scanlines_in_block =
        std::min(band_size_y - block_y, max_scanlines_per_block);
    const int pixels_in_block = scanlines_in_block * band_size_x;
    const int block_max_scanlines_per_thread =
        std::max((scanlines_in_block + static_cast<int>(max_threads) - 1) /
                     static_cast<int>(max_threads),
                 min_scanlines_per_thread);
    VLOG(1) << "Raster Importer: scanlines_in_block: " << scanlines_in_block
            << ", block_max_scanlines_per_thread:  " << block_max_scanlines_per_thread;

    auto block_wall_timer = timer_start();
    // run max_threads scanlines at once
    for (size_t thread_id = 0; thread_id < max_threads; thread_id++) {
      const int y_start = block_y + thread_id * block_max_scanlines_per_thread;
      if (y_start < band_size_y) {
        const int y_end = std::min(y_start + block_max_scanlines_per_thread, band_size_y);
        if (y_start < y_end) {
          futures.emplace_back(
              std::async(std::launch::async, import_rows, thread_id, y_start, y_end));
        }
      }
    }

    // wait for the threads to finish and
    // accumulate the results and times
    float proj_s{0.0f}, read_s{0.0f}, conv_s{0.0f}, load_s{0.0f};
    size_t thread_idx = 0;
    for (auto& future : futures) {
      auto const [import_status, times] = future.get();
      import_status_ += import_status;
      proj_s += times[0];
      read_s += times[1];
      conv_s += times[2];
      // We load the data in thread order so we can get deterministic row-major raster
      // ordering
      // Todo: We should consider invoking the load on another thread in a ping-pong
      // fashion so we can simultaneously read the next batch of data
      auto thread_load_timer = timer_start();
      // only try to load this thread's data if valid
      if (import_status.rows_completed > 0) {
        load(import_buffers_vec[thread_idx], import_status.rows_completed, session_info);
      }
      load_s += TIMER_STOP(thread_load_timer);
      ++thread_idx;
    }

    // average times over all threads (except for load which is single-threaded)
    total_proj_s += (proj_s / float(futures.size()));
    total_read_s += (read_s / float(futures.size()));
    total_conv_s += (conv_s / float(futures.size()));
    total_load_s += load_s;

    // update the status
    set_import_status(import_id, import_status_);

    // more debug
    auto const block_wall_s = TIMER_STOP(block_wall_timer);
    auto const scanlines_per_second = scanlines_in_block / block_wall_s;
    auto const rows_per_second = pixels_in_block / block_wall_s;
    LOG(INFO) << "Raster Importer: Loaded " << scanlines_in_block
              << " scanlines starting at " << block_y << " out of " << band_size_y
              << " in " << block_wall_s << "s at " << scanlines_per_second
              << " scanlines/s and " << rows_per_second << " rows/s";

    // check for interrupt
    if (UNLIKELY(check_session_interrupted(query_session, executor.get()))) {
      import_status_.load_failed = true;
      import_status_.load_msg = "Raster Import interrupted";
      throw QueryExecutionError(ErrorCode::INTERRUPTED);
    }

    // hit max_reject?
    if (import_status_.rows_rejected > copy_params.max_reject) {
      break;
    }
  }

  // checkpoint
  auto checkpoint_timer = timer_start();
  checkpoint(table_epochs);
  auto const checkpoint_s = TIMER_STOP(checkpoint_timer);

  // stop wall clock
  auto const total_wall_s = TIMER_STOP(wall_timer);

  // report
  auto const total_scanlines_per_second = float(band_size_y) / total_wall_s;
  auto const total_rows_per_second =
      float(band_size_x) * float(band_size_y) / total_wall_s;
  LOG(INFO) << "Raster Importer: Imported "
            << static_cast<uint64_t>(band_size_x) * static_cast<uint64_t>(band_size_y)
            << " rows";
  LOG(INFO) << "Raster Importer: Total Import Time " << total_wall_s << "s at "
            << total_scanlines_per_second << " scanlines/s and " << total_rows_per_second
            << " rows/s";

  // if we hit max_reject, throw an exception now to report the error and abort any
  // multi-file loop
  if (import_status_.rows_rejected > copy_params.max_reject) {
    std::string msg = "Raster Importer: Import aborted after failing to read " +
                      std::to_string(import_status_.rows_rejected) +
                      " rows/pixels (limit " + std::to_string(copy_params.max_reject) +
                      ")";
    import_status_.load_msg = msg;
    set_import_status(import_id, import_status_);
    throw std::runtime_error(msg);
  }

  // phase times (with proportions)
  auto proj_pct = float(int(total_proj_s / total_wall_s * 1000.0f) * 0.1f);
  auto read_pct = float(int(total_read_s / total_wall_s * 1000.0f) * 0.1f);
  auto conv_pct = float(int(total_conv_s / total_wall_s * 1000.0f) * 0.1f);
  auto load_pct = float(int(total_load_s / total_wall_s * 1000.0f) * 0.1f);
  auto cpnt_pct = float(int(checkpoint_s / total_wall_s * 1000.0f) * 0.1f);

  VLOG(1) << "Raster Importer: Import timing breakdown:";
  VLOG(1) << "  Project    " << total_proj_s << "s (" << proj_pct << "%)";
  VLOG(1) << "  Read       " << total_read_s << "s (" << read_pct << "%)";
  VLOG(1) << "  Convert    " << total_conv_s << "s (" << conv_pct << "%)";
  VLOG(1) << "  Load       " << total_load_s << "s (" << load_pct << "%)";
  VLOG(1) << "  Checkpoint " << checkpoint_s << "s (" << cpnt_pct << "%)";

  // all done
  return import_status_;
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
    if (cd->columnType.is_geometry()) {
      std::vector<double> coords, bounds;
      std::vector<int> ring_sizes, poly_rings;
      SQLTypeInfo tinfo{cd->columnType};
      const bool validate_with_geos_if_available = false;
      CHECK(Geospatial::GeoTypesFactory::getGeoColumns(default_value,
                                                       tinfo,
                                                       coords,
                                                       bounds,
                                                       ring_sizes,
                                                       poly_rings,
                                                       validate_with_geos_if_available));
      // set physical columns starting with the following ID
      auto next_col = i + 1;
      import_export::Importer::set_geo_physical_import_buffer(
          *cat, cd, defaults_buffers, next_col, coords, bounds, ring_sizes, poly_rings);
      // skip physical columns filled with the call above
      i += cd->columnType.get_physical_cols();
    }
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

std::unique_ptr<AbstractImporter> create_importer(
    Catalog_Namespace::Catalog& catalog,
    const TableDescriptor* td,
    const std::string& copy_from_source,
    const import_export::CopyParams& copy_params) {
  if (copy_params.source_type == import_export::SourceType::kParquetFile) {
#ifdef ENABLE_IMPORT_PARQUET
    if (!g_enable_legacy_parquet_import) {
      return std::make_unique<import_export::ForeignDataImporter>(
          copy_from_source, copy_params, td);
    }
#else
    throw std::runtime_error("Parquet not supported!");
#endif
  }

  if (copy_params.source_type == import_export::SourceType::kDelimitedFile &&
      !g_enable_legacy_delimited_import) {
    return std::make_unique<import_export::ForeignDataImporter>(
        copy_from_source, copy_params, td);
  }

  if (copy_params.source_type == import_export::SourceType::kRegexParsedFile) {
    if (g_enable_fsi_regex_import) {
      return std::make_unique<import_export::ForeignDataImporter>(
          copy_from_source, copy_params, td);
    } else {
      throw std::runtime_error(
          "Regex parsed import only supported using 'fsi-regex-import' flag");
    }
  }

  if (copy_params.source_type == import_export::SourceType::kRasterFile) {
    throw std::runtime_error(
        "HeavyConnect-based Raster file import only supported in Enterprise Edition.  "
        "For legacy raster import, use the '--enable-legacy-raster-import' option.");
  }

  return std::make_unique<import_export::Importer>(
      catalog, td, copy_from_source, copy_params);
}

template class OptionallyMemoryManagedTypedImportBuffer<true>;
template class OptionallyMemoryManagedTypedImportBuffer<false>;

}  // namespace import_export
