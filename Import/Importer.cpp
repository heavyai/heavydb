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

#include <csignal>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <stdexcept>
#include <list>
#include <vector>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <future>
#include <mutex>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <ogrsf_frmts.h>
#include <gdal.h>
#include "../QueryEngine/SqlTypesLayout.h"
#include "../QueryEngine/TypePunning.h"
#include "../Shared/mapdpath.h"
#include "../Shared/measure.h"
#include "../Shared/unreachable.h"
#include "../Shared/geosupport.h"
#include "../Shared/mapd_glob.h"
#include "../Shared/scope.h"
#include "Importer.h"
#include "DataMgr/LockMgr.h"
#include "gen-cpp/MapD.h"
#include <vector>
#include <iostream>

#include <arrow/api.h>

#include "../Archive/PosixFileArchive.h"

#include "../Archive/S3Archive.h"

using std::ostream;

namespace Importer_NS {

bool debug_timing = false;

static mapd_shared_mutex status_mutex;
static std::map<std::string, ImportStatus> import_status_map;

Importer::Importer(Catalog_Namespace::Catalog& c, const TableDescriptor* t, const std::string& f, const CopyParams& p)
    : Importer(new Loader(c, t), f, p) {}

Importer::Importer(Loader* providedLoader, const std::string& f, const CopyParams& p)
    : DataStreamSink(p, f), loader(providedLoader) {
  import_id = boost::filesystem::path(file_path).filename().string();
  file_size = 0;
  max_threads = 0;
  p_file = nullptr;
  buffer[0] = nullptr;
  buffer[1] = nullptr;
  auto is_array = std::unique_ptr<bool[]>(new bool[loader->get_column_descs().size()]);
  int i = 0;
  bool has_array = false;
  for (auto& p : loader->get_column_descs()) {
    if (p->columnType.get_type() == kARRAY) {
      is_array.get()[i] = true;
      has_array = true;
    } else
      is_array.get()[i] = false;
    ++i;
  }
  if (has_array)
    is_array_a = std::unique_ptr<bool[]>(is_array.release());
  else
    is_array_a = std::unique_ptr<bool[]>(nullptr);
}

Importer::~Importer() {
  if (p_file != nullptr)
    fclose(p_file);
  if (buffer[0] != nullptr)
    free(buffer[0]);
  if (buffer[1] != nullptr)
    free(buffer[1]);
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
    if (*p == copy_params.escape && p < entire_buf_end - 1 && *(p + 1) == copy_params.quote) {
      p++;
      has_escape = true;
    } else if (copy_params.quoted && *p == copy_params.quote) {
      in_quote = !in_quote;
      if (in_quote)
        strip_quotes = true;
    } else if (!in_quote && is_array != nullptr && *p == copy_params.array_begin && is_array[row.size()]) {
      in_array = true;
    } else if (!in_quote && is_array != nullptr && *p == copy_params.array_end && is_array[row.size()]) {
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
            if (has_escape && field[i] == copy_params.escape && field[i + 1] == copy_params.quote) {
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
      if (is_eol(*p, line_endings) && ((!in_quote && !in_array) || copy_params.threads != 1)) {
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

ArrayDatum StringToArray(const std::string& s, const SQLTypeInfo& ti, const CopyParams& copy_params) {
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
  if (last + 1 < s.size()) {
    elem_strs.push_back(s.substr(last, s.size() - 1 - last));
  }
  if (!elem_ti.is_string()) {
    size_t len = elem_strs.size() * elem_ti.get_size();
    int8_t* buf = (int8_t*)checked_malloc(len);
    int8_t* p = buf;
    for (auto& e : elem_strs) {
      Datum d = StringToDatum(e, elem_ti);
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
      d.bigintval = datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
      break;
    case kINT:
      d.intval = datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
      break;
    case kSMALLINT:
      d.smallintval = datum.is_null ? inline_fixed_encoding_null_val(ti) : datum.val.int_val;
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
    default:
      throw std::runtime_error("Internal error: invalid type in StringToDatum.");
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

static size_t find_beginning(const char* buffer, size_t begin, size_t end, const CopyParams& copy_params) {
  // @TODO(wei) line_delim is in quotes note supported
  if (begin == 0 || (begin > 0 && buffer[begin - 1] == copy_params.line_delim))
    return 0;
  size_t i;
  const char* buf = buffer + begin;
  for (i = 0; i < end - begin; i++)
    if (buf[i] == copy_params.line_delim)
      return i + 1;
  return i;
}

void TypedImportBuffer::add_value(const ColumnDescriptor* cd,
                                  const std::string& val,
                                  const bool is_null,
                                  const CopyParams& copy_params) {
  const auto type = cd->columnType.get_type();
  switch (type) {
    case kBOOLEAN: {
      if (is_null) {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addBoolean(inline_fixed_encoding_null_val(cd->columnType));
      } else {
        SQLTypeInfo ti = cd->columnType;
        Datum d = StringToDatum(val, ti);
        addBoolean((int8_t)d.boolval);
      }
      break;
    }
    case kSMALLINT: {
      if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
        SQLTypeInfo ti = cd->columnType;
        Datum d = StringToDatum(val, ti);
        addSmallint(d.smallintval);
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
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
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
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
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addBigint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kDECIMAL:
    case kNUMERIC: {
      if (!is_null) {
        SQLTypeInfo ti(kNUMERIC, 0, 0, false);
        Datum d = StringToDatum(val, ti);
        const auto converted_decimal_value = convert_decimal_value_to_scale(d.bigintval, ti, cd->columnType);
        addBigint(converted_decimal_value);
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addBigint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    }
    case kFLOAT:
      if (!is_null && (val[0] == '.' || isdigit(val[0]) || val[0] == '-')) {
        addFloat((float)std::atof(val.c_str()));
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addFloat(NULL_FLOAT);
      }
      break;
    case kDOUBLE:
      if (!is_null && (val[0] == '.' || isdigit(val[0]) || val[0] == '-')) {
        addDouble(std::atof(val.c_str()));
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addDouble(NULL_DOUBLE);
      }
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      // @TODO(wei) for now, use empty string for nulls
      if (is_null) {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addString(std::string());
      } else {
        if (val.length() > StringDictionary::MAX_STRLEN)
          throw std::runtime_error("String too long for column " + cd->columnName + " was " +
                                   std::to_string(val.length()) + " max is " +
                                   std::to_string(StringDictionary::MAX_STRLEN));
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
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
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
          ArrayDatum d = StringToArray(val, cd->columnType, copy_params);
          addArray(d);
        } else {
          addArray(ArrayDatum(0, NULL, true));
        }
      }
      break;
    default:
      CHECK(false);
  }
}

void TypedImportBuffer::pop_value() {
  const auto type = column_desc_->columnType.is_decimal() ? decimal_to_int_type(column_desc_->columnType)
                                                          : column_desc_->columnType.get_type();
  switch (type) {
    case kBOOLEAN:
      bool_buffer_->pop_back();
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
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      time_buffer_->pop_back();
      break;
    case kARRAY:
      if (IS_STRING(column_desc_->columnType.get_subtype())) {
        string_array_buffer_->pop_back();
      } else {
        array_buffer_->pop_back();
      }
      break;
    default:
      CHECK(false);
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
inline void append_arrow_primitive(const Array& values, const T null_sentinel, std::vector<T>* buffer) {
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

void append_arrow_boolean(const ColumnDescriptor* cd, const Array& values, std::vector<int8_t>* buffer) {
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
void append_arrow_integer(const ColumnDescriptor* cd, const Array& values, std::vector<T>* buffer) {
  using ArrayType = typename TypeTraits<ArrowType>::ArrayType;
  const T null_sentinel = inline_fixed_encoding_null_val(cd->columnType);
  append_arrow_primitive<ArrayType, T>(values, null_sentinel, buffer);
}

void append_arrow_float(const ColumnDescriptor* cd, const Array& values, std::vector<float>* buffer) {
  ARROW_THROW_IF(values.type_id() != Type::FLOAT, "Expected float col");
  append_arrow_primitive<FloatArray, float>(values, NULL_FLOAT, buffer);
}

void append_arrow_double(const ColumnDescriptor* cd, const Array& values, std::vector<double>* buffer) {
  ARROW_THROW_IF(values.type_id() != Type::DOUBLE, "Expected double col");
  append_arrow_primitive<DoubleArray, double>(values, NULL_DOUBLE, buffer);
}

constexpr int64_t kMillisecondsInSecond = 1000L;
constexpr int64_t kMicrosecondsInSecond = 1000L * 1000L;
constexpr int64_t kNanosecondsinSecond = 1000L * 1000L * 1000L;
constexpr int32_t kSecondsInDay = 86400;

void append_arrow_time(const ColumnDescriptor* cd, const Array& values, std::vector<time_t>* buffer) {
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

void append_arrow_timestamp(const ColumnDescriptor* cd, const Array& values, std::vector<time_t>* buffer) {
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

void append_arrow_date(const ColumnDescriptor* cd, const Array& values, std::vector<time_t>* buffer) {
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

void append_arrow_binary(const ColumnDescriptor* cd, const Array& values, std::vector<std::string>* buffer) {
  ARROW_THROW_IF(values.type_id() != Type::BINARY && values.type_id() != Type::STRING, "Expected binary col");

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

size_t TypedImportBuffer::add_arrow_values(const ColumnDescriptor* cd, const arrow::Array& col) {
  const auto type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType) : cd->columnType.get_type();
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
  const auto type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType) : cd->columnType.get_type();
  if (cd->columnType.get_notnull()) {
    // We can't have any null values for this column; to have them is an error
    if (std::any_of(col.nulls.begin(), col.nulls.end(), [](int i) { return i != 0; }))
      throw std::runtime_error("NULL for column " + cd->columnName);
  }

  switch (type) {
    case kBOOLEAN: {
      dataSize = col.data.int_col.size();
      bool_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i])
          bool_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        else
          bool_buffer_->push_back((int8_t)col.data.int_col[i]);
      }
      break;
    }
    case kSMALLINT: {
      dataSize = col.data.int_col.size();
      smallint_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i])
          smallint_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        else
          smallint_buffer_->push_back((int16_t)col.data.int_col[i]);
      }
      break;
    }
    case kINT: {
      dataSize = col.data.int_col.size();
      int_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i])
          int_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        else
          int_buffer_->push_back((int32_t)col.data.int_col[i]);
      }
      break;
    }
    case kBIGINT: {
      dataSize = col.data.int_col.size();
      bigint_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i])
          bigint_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        else
          bigint_buffer_->push_back((int64_t)col.data.int_col[i]);
      }
      break;
    }
    case kFLOAT: {
      dataSize = col.data.real_col.size();
      float_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i])
          float_buffer_->push_back(NULL_FLOAT);
        else
          float_buffer_->push_back((float)col.data.real_col[i]);
      }
      break;
    }
    case kDOUBLE: {
      dataSize = col.data.real_col.size();
      double_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i])
          double_buffer_->push_back(NULL_DOUBLE);
        else
          double_buffer_->push_back((double)col.data.real_col[i]);
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
        if (col.nulls[i])
          string_buffer_->push_back(std::string());
        else
          string_buffer_->push_back(col.data.str_col[i]);
      }
      break;
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      dataSize = col.data.int_col.size();
      time_buffer_->reserve(dataSize);
      for (size_t i = 0; i < dataSize; i++) {
        if (col.nulls[i])
          time_buffer_->push_back(inline_fixed_encoding_null_val(cd->columnType));
        else
          time_buffer_->push_back((time_t)col.data.int_col[i]);
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
            for (size_t str_idx = 0; str_idx != stringArrSize; ++str_idx)
              string_vec.push_back(col.data.arr_col[i].data.str_col[str_idx]);
          }
        }
      } else {
        auto elem_ti = cd->columnType.get_subtype();
        switch (elem_ti) {
          case kBOOLEAN: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i])
                addArray(ArrayDatum(0, NULL, true));
              else {
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
          case kSMALLINT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i])
                addArray(ArrayDatum(0, NULL, true));
              else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int16_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(int16_t*)p = static_cast<int16_t>(col.data.arr_col[i].data.int_col[j]);
                  p += sizeof(int16_t);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          case kINT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i])
                addArray(ArrayDatum(0, NULL, true));
              else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int32_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(int32_t*)p = static_cast<int32_t>(col.data.arr_col[i].data.int_col[j]);
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
              if (col.nulls[i])
                addArray(ArrayDatum(0, NULL, true));
              else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(int64_t);
                int8_t* buf = (int8_t*)checked_malloc(len * byteSize);
                int8_t* p = buf;
                for (size_t j = 0; j < len; ++j) {
                  *(int64_t*)p = static_cast<int64_t>(col.data.arr_col[j].data.int_col[j]);
                  p += sizeof(int64_t);
                }
                addArray(ArrayDatum(byteSize, buf, len == 0));
              }
            }
            break;
          }
          case kFLOAT: {
            for (size_t i = 0; i < dataSize; i++) {
              if (col.nulls[i])
                addArray(ArrayDatum(0, NULL, true));
              else {
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
              if (col.nulls[i])
                addArray(ArrayDatum(0, NULL, true));
              else {
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
              if (col.nulls[i])
                addArray(ArrayDatum(0, NULL, true));
              else {
                size_t len = col.data.arr_col[i].data.int_col.size();
                size_t byteSize = len * sizeof(time_t);
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

void TypedImportBuffer::add_value(const ColumnDescriptor* cd, const TDatum& datum, const bool is_null) {
  const auto type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType) : cd->columnType.get_type();
  switch (type) {
    case kBOOLEAN: {
      if (is_null) {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addBoolean(inline_fixed_encoding_null_val(cd->columnType));
      } else {
        addBoolean((int8_t)datum.val.int_val);
      }
      break;
    }
    case kSMALLINT:
      if (!is_null) {
        addSmallint((int16_t)datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addSmallint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kINT:
      if (!is_null) {
        addInt((int32_t)datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addInt(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kBIGINT:
      if (!is_null) {
        addBigint(datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addBigint(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
    case kFLOAT:
      if (!is_null) {
        addFloat((float)datum.val.real_val);
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addFloat(NULL_FLOAT);
      }
      break;
    case kDOUBLE:
      if (!is_null) {
        addDouble(datum.val.real_val);
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addDouble(NULL_DOUBLE);
      }
      break;
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      // @TODO(wei) for now, use empty string for nulls
      if (is_null) {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addString(std::string());
      } else
        addString(datum.val.str_val);
      break;
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      if (!is_null) {
        addTime((time_t)datum.val.int_val);
      } else {
        if (cd->columnType.get_notnull())
          throw std::runtime_error("NULL for column " + cd->columnName);
        addTime(inline_fixed_encoding_null_val(cd->columnType));
      }
      break;
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
    default:
      CHECK(false);
  }
}

template <typename T>
ostream& operator<<(ostream& out, const std::vector<T>& v) {
  out << "[";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last)
      out << ", ";
  }
  out << "]";
  return out;
}

static ImportStatus import_thread(int thread_id,
                                  Importer* importer,
                                  std::shared_ptr<const char> sbuffer,
                                  size_t begin_pos,
                                  size_t end_pos,
                                  size_t total_size) {
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
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers = importer->get_import_buffers(thread_id);
    auto us = measure<std::chrono::microseconds>::execution([&]() {});
    for (const auto& p : import_buffers)
      p->clear();
    std::vector<std::string> row;
    for (const char* p = thread_buf; p < thread_buf_end; p++) {
      row.clear();
      if (debug_timing) {
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
      } else
        p = get_row(
            p, thread_buf_end, buf_end, copy_params, p == thread_buf, importer->get_is_array(), row, try_single_thread);
      if (row.size() != col_descs.size()) {
        import_status.rows_rejected++;
        LOG(ERROR) << "Incorrect Row (expected " << col_descs.size() << " columns, has " << row.size() << "): " << row;
        if (import_status.rows_rejected > copy_params.max_reject)
          break;
        continue;
      }
      us = measure<std::chrono::microseconds>::execution([&]() {
        size_t col_idx = 0;
        try {
          for (const auto cd : col_descs) {
            bool is_null = (row[col_idx] == copy_params.null_str);
            if (!cd->columnType.is_string() && row[col_idx].empty())
              is_null = true;
            import_buffers[col_idx]->add_value(cd, row[col_idx], is_null, copy_params);
            ++col_idx;
          }
          import_status.rows_completed++;
        } catch (const std::exception& e) {
          for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
            import_buffers[col_idx_to_pop]->pop_value();
          }
          import_status.rows_rejected++;
          LOG(ERROR) << "Input exception thrown: " << e.what() << ". Row discarded, issue at column : " << (col_idx + 1)
                     << " data :" << row;
        }
      });
      total_str_to_val_time_us += us;
    }
    if (import_status.rows_completed > 0) {
      load_ms = measure<>::execution([&]() { importer->load(import_buffers, import_status.rows_completed); });
    }
  });
  if (debug_timing && import_status.rows_completed > 0) {
    LOG(INFO) << "Thread" << std::this_thread::get_id() << ":" << import_status.rows_completed << " rows inserted in "
              << (double)ms / 1000.0 << "sec, Insert Time: " << (double)load_ms / 1000.0
              << "sec, get_row: " << (double)total_get_row_time_us / 1000000.0
              << "sec, str_to_val: " << (double)total_str_to_val_time_us / 1000000.0 << "sec" << std::endl;
  }

  import_status.thread_id = thread_id;
  // LOG(INFO) << " return " << import_status.thread_id << std::endl;

  return import_status;
}

static size_t find_end(const char* buffer, size_t size, const CopyParams& copy_params) {
  int i;
  // @TODO(wei) line_delim is in quotes note supported
  for (i = size - 1; i >= 0 && buffer[i] != copy_params.line_delim; i--)
    ;

  if (i < 0)
    LOG(ERROR) << "No line delimiter in block.";
  return i + 1;
}

bool Loader::loadNoCheckpoint(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers, size_t row_count) {
  return loadImpl(import_buffers, row_count, false);
}

bool Loader::load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers, size_t row_count) {
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
          new TypedImportBuffer(typed_import_buffer->getColumnDesc(), typed_import_buffer->getStringDictionary()));
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
  CHECK(shard_col_ti.is_integer() || (shard_col_ti.is_string() && shard_col_ti.get_compression() == kENCODING_DICT));
  if (shard_col_ti.is_string()) {
    const auto payloads_ptr = shard_column_input_buffer->getStringBuffer();
    CHECK(payloads_ptr);
    shard_column_input_buffer->addDictEncodedString(*payloads_ptr);
  }
  for (size_t i = 0; i < row_count; ++i) {
    const auto val = int_value_at(*shard_column_input_buffer, i);
    const auto shard = val % shard_count;
    auto& shard_output_buffers = all_shard_import_buffers[shard];
    for (size_t col_idx = 0; col_idx < import_buffers.size(); ++col_idx) {
      const auto& input_buffer = import_buffers[col_idx];
      const auto& col_ti = input_buffer->getTypeInfo();
      const auto type = col_ti.is_decimal() ? decimal_to_int_type(col_ti) : col_ti.get_type();
      switch (type) {
        case kBOOLEAN:
          shard_output_buffers[col_idx]->addBoolean(int_value_at(*input_buffer, i));
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
        default:
          CHECK(false);
      }
    }
    ++all_shard_row_counts[shard];
  }
}

bool Loader::loadImpl(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                      size_t row_count,
                      bool checkpoint) {
  if (table_desc->nShards) {
    std::vector<OneShardBuffers> all_shard_import_buffers;
    std::vector<size_t> all_shard_row_counts;
    const auto shard_tables = catalog.getPhysicalTablesDescriptors(table_desc);
    distributeToShards(all_shard_import_buffers, all_shard_row_counts, import_buffers, row_count, shard_tables.size());
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

bool Loader::loadToShard(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                         size_t row_count,
                         const TableDescriptor* shard_table,
                         bool checkpoint) {
  Fragmenter_Namespace::InsertData ins_data(insert_data);
  ins_data.numRows = row_count;
  bool success = true;
  for (const auto& import_buff : import_buffers) {
    DataBlockPtr p;
    if (import_buff->getTypeInfo().is_number() || import_buff->getTypeInfo().is_time() ||
        import_buff->getTypeInfo().get_type() == kBOOLEAN) {
      p.numbersPtr = import_buff->getAsBytes();
    } else if (import_buff->getTypeInfo().is_string()) {
      auto string_payload_ptr = import_buff->getStringBuffer();
      if (import_buff->getTypeInfo().get_compression() == kENCODING_NONE) {
        p.stringsPtr = string_payload_ptr;
      } else {
        CHECK_EQ(kENCODING_DICT, import_buff->getTypeInfo().get_compression());
        import_buff->addDictEncodedString(*string_payload_ptr);
        p.numbersPtr = import_buff->getStringDictBuffer();
      }
    } else {
      CHECK(import_buff->getTypeInfo().get_type() == kARRAY);
      if (IS_STRING(import_buff->getTypeInfo().get_subtype())) {
        CHECK(import_buff->getTypeInfo().get_compression() == kENCODING_DICT);
        import_buff->addDictEncodedStringArray(*import_buff->getStringArrayBuffer());
        p.arraysPtr = import_buff->getStringArrayDictBuffer();
      } else
        p.arraysPtr = import_buff->getArrayBuffer();
    }
    ins_data.data.push_back(p);
  }
  {
    try {
      if (checkpoint)
        shard_table->fragmenter->insertData(ins_data);
      else
        shard_table->fragmenter->insertDataNoCheckpoint(ins_data);
    } catch (std::exception& e) {
      LOG(ERROR) << "Fragmenter Insert Exception: " << e.what();
      success = false;
    }
  }
  return success;
}

void Loader::init() {
  insert_data.databaseId = catalog.get_currentDB().dbId;
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

ImportStatus Detector::importDelimited(const std::string& file_path, const bool decompressed) {
  if (!p_file)
    p_file = fopen(file_path.c_str(), "rb");
  if (!p_file)
    throw std::runtime_error("failed to open file '" + file_path + "': " + strerror(errno));

  // somehow clang does not support ext/stdio_filebuf.h, so
  // need to diy readline with customized copy_params.line_delim...
  char line[1 << 20];
  auto end_time = std::chrono::steady_clock::now() + timeout;
  try {
    while (!feof(p_file)) {
      int c;
      size_t n = 0;
      while (EOF != (c = fgetc(p_file)) && copy_params.line_delim != c) {
        line[n++] = c;
        if (n >= sizeof(line) - 1)
          break;
      }
      if (0 == n)
        break;
      line[n] = 0;
      // remember the first line, which is possibly a header line, to
      // ignore identical header line(s) in 2nd+ files of a archive;
      // otherwise, 2nd+ header may be mistaken as an all-string row
      // and so be final column types.
      if (line1.empty())
        line1 = line;
      else if (line == line1)
        continue;

      raw_data += std::string(line, n);
      raw_data += copy_params.line_delim;
      if (std::chrono::steady_clock::now() > end_time) {
        break;
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
      p = get_row(p, buf_end, buf_end, copy_params, true, nullptr, row, try_single_thread);
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
    }*/ if (try_cast<int16_t>(str)) {
      type = kSMALLINT;
    } else if (try_cast<int32_t>(str)) {
      type = kINT;
    } else if (try_cast<int64_t>(str)) {
      type = kBIGINT;
    } else if (try_cast<float>(str)) {
      type = kFLOAT;
    }
  }

  // see StringToDatum in Shared/Datum.cpp
  if (type == kTEXT) {
    char* buf;
    buf = try_strptimes(str.c_str(), {"%Y-%m-%d", "%m/%d/%Y", "%d-%b-%y", "%d/%b/%Y"});
    if (buf) {
      type = kDATE;
      if (*buf == 'T' || *buf == ' ' || *buf == ':') {
        buf++;
      }
    }
    buf = try_strptimes(buf == nullptr ? str.c_str() : buf, {"%T %z", "%T", "%H%M%S", "%R"});
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
  typeorder[kTEXT] = 11;

  // note: b < a instead of a < b because the map is ordered most to least restrictive
  return typeorder[b] < typeorder[a];
}

void Detector::find_best_sqltypes_and_headers() {
  best_sqltypes = find_best_sqltypes(raw_rows.begin() + 1, raw_rows.end(), copy_params);
  best_encodings = find_best_encodings(raw_rows.begin() + 1, raw_rows.end(), best_sqltypes);
  std::vector<SQLTypes> head_types = detect_column_types(raw_rows.at(0));
  has_headers = detect_headers(head_types, best_sqltypes);
  copy_params.has_header = has_headers;
}

void Detector::find_best_sqltypes() {
  best_sqltypes = find_best_sqltypes(raw_rows.begin(), raw_rows.end(), copy_params);
}

std::vector<SQLTypes> Detector::find_best_sqltypes(const std::vector<std::vector<std::string>>& raw_rows,
                                                   const CopyParams& copy_params) {
  return find_best_sqltypes(raw_rows.begin(), raw_rows.end(), copy_params);
}

std::vector<SQLTypes> Detector::find_best_sqltypes(
    const std::vector<std::vector<std::string>>::const_iterator& row_begin,
    const std::vector<std::vector<std::string>>::const_iterator& row_end,
    const CopyParams& copy_params) {
  if (raw_rows.size() < 1) {
    throw std::runtime_error("No rows found in: " + boost::filesystem::basename(file_path));
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
      if (row->at(col_idx) == "" || !row->at(col_idx).compare(copy_params.null_str))
        continue;
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
    if (non_null_col_counts[col_idx] == 0)
      best_types[col_idx] = kTEXT;
  }

  return best_types;
}

std::vector<EncodingType> Detector::find_best_encodings(
    const std::vector<std::vector<std::string>>::const_iterator& row_begin,
    const std::vector<std::vector<std::string>>::const_iterator& row_end,
    const std::vector<SQLTypes>& best_types) {
  if (raw_rows.size() < 1) {
    throw std::runtime_error("No rows found in: " + boost::filesystem::basename(file_path));
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
      float uniqueRatio = static_cast<float>(count_set[col_idx].size()) / num_rows_per_col[col_idx];
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
  std::vector<SQLTypes> tail_types = find_best_sqltypes(raw_rows.begin() + 1, raw_rows.end(), copy_params);
  return detect_headers(head_types, tail_types);
}

// detect_headers returns true if:
// - all elements of the first argument are kTEXT
// - there is at least one instance where tail_types is more restrictive than head_types (ie, not kTEXT)
bool Detector::detect_headers(const std::vector<SQLTypes>& head_types, const std::vector<SQLTypes>& tail_types) {
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
  std::vector<std::vector<std::string>> sample_rows(raw_rows.begin() + offset, raw_rows.begin() + n);
  return sample_rows;
}

std::vector<std::string> Detector::get_headers() {
  std::vector<std::string> headers(best_sqltypes.size());
  for (size_t i = 0; i < best_sqltypes.size(); i++) {
    headers[i] = has_headers ? raw_rows[0][i] : "column_" + std::to_string(i + 1);
  }
  return headers;
}

void Importer::load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers, size_t row_count) {
  if (!loader->loadNoCheckpoint(import_buffers, row_count))
    load_failed = true;
}

ImportStatus DataStreamSink::archivePlumber() {
  // in generalized importing scheme, reaching here file_path may
  // contain a file path, a url or a wildcard of file paths.
  // see CopyTableStmt::execute.
  auto file_paths = mapd_glob(file_path);
  if (file_paths.size() == 0)
    file_paths.push_back(file_path);

  // s3 parquet goes different route because the files do not use libarchive
  // but parquet api, and they need to landed like .7z files.
  //
  // note: parquet must be explicitly specified by a WITH parameter "parquet='true'".
  //       without the parameter, it means plain or compressed csv files.
  // note: .ORC and AVRO files should follow a similar path to Parquet?
  if (copy_params.is_parquet)
    import_parquet(file_paths);
  else
    import_compressed(file_paths);
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
    } else
      objkeys.emplace_back(file_path);

    // for each obj key of a s3 url we need to land it before
    // importing it like doing with a 'local file'.
    for (auto const& objkey : objkeys)
      try {
        auto file_path = us3arch ? us3arch->land(objkey, teptr) : objkey;
        if (boost::filesystem::exists(file_path))
          import_local_parquet(file_path);
        if (us3arch)
          us3arch->vacuum(objkey);
      } catch (...) {
        if (us3arch)
          us3arch->vacuum(objkey);
        throw;
      }
  }
  // rethrow any exception happened herebefore
  if (teptr)
    std::rethrow_exception(teptr);
}

void DataStreamSink::import_compressed(std::vector<std::string>& file_paths) {
  // a new requirement is to have one single input stream into
  // Importer::importDelimited, so need to move pipe related
  // stuff to the outmost block.
  int fd[2];
  if (pipe(fd) < 0)
    throw std::runtime_error(std::string("failed to create a pipe: ") + strerror(errno));
  signal(SIGPIPE, SIG_IGN);

  std::exception_ptr teptr;
  // create a thread to read uncompressed byte stream out of pipe and
  // feed into importDelimited()
  ImportStatus ret;
  auto th_pipe_reader = std::thread([&]() {
    try {
      // importDelimited will read from FILE* p_file
      if (0 == (p_file = fdopen(fd[0], "r")))
        throw std::runtime_error(std::string("failed to open a pipe: ") + strerror(errno));

      // in future, depending on data types of this uncompressed stream
      // it can be feed into other function such like importParquet, etc
      ret = importDelimited(file_path, true);
    } catch (...) {
      if (!teptr)  // no replace
        teptr = std::current_exception();
    }

    if (p_file)
      fclose(p_file);
    p_file = 0;
  });

  // create a thread to iterate all files (in all archives) and
  // forward the uncompressed byte stream to fd[1] which is
  // then feed into importDelimited, importParquet, and etc.
  auto th_pipe_writer = std::thread([&]() {
    std::unique_ptr<S3Archive> us3arch;
    for (size_t fi = 0; fi < file_paths.size(); fi++) {
      try {
        auto file_path = file_paths[fi];
        std::unique_ptr<Archive> uarch;
        std::map<int, std::string> url_parts;
        Archive::parse_url(file_path, url_parts);
        const std::string S3_objkey_url_scheme = "s3ok";
        if ("file" == url_parts[2] || "" == url_parts[2])
          uarch.reset(new PosixFileArchive(file_path, copy_params.plain_text));
        else if ("s3" == url_parts[2]) {
#ifdef HAVE_AWS_S3
          // new a S3Archive with a shared s3client.
          // should be safe b/c no wildcard with s3 url
          us3arch.reset(new S3Archive(file_path,
                                      copy_params.s3_access_key,
                                      copy_params.s3_secret_key,
                                      copy_params.s3_region,
                                      copy_params.plain_text));
          us3arch->init_for_read();
          // not land all files here but one by one in following iterations
          for (const auto& objkey : us3arch->get_objkeys())
            file_paths.emplace_back(std::string(S3_objkey_url_scheme) + "://" + objkey);
          continue;
#else
          throw std::runtime_error("AWS S3 support not available");
#endif  // HAVE_AWS_S3
        } else if (S3_objkey_url_scheme == url_parts[2]) {
#ifdef HAVE_AWS_S3
          auto objkey = file_path.substr(3 + S3_objkey_url_scheme.size());
          auto file_path = us3arch->land(objkey, teptr);
          if (0 == file_path.size())
            throw std::runtime_error(std::string("failed to land s3 object: ") + objkey);
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
        else
          throw std::runtime_error(std::string("unsupported archive url: ") + file_path);

        // init the archive for read
        auto& arch = *uarch;

        // coming here, the archive of url should be ready to be read, unarchived
        // and uncompressed by libarchive into a byte stream (in csv) for the pipe
        const void* buf;
        size_t size;
        int64_t offset;
        bool just_saw_header;
        bool stop = false;
        // start reading uncompressed bytes of this archive from libarchive
        // note! this archive may contain more than one files!
        while (!stop && !!(just_saw_header = arch.read_next_header()))
          while (!stop && arch.read_data_block(&buf, &size, &offset)) {
            // one subtle point here is now we concatenate all files
            // to a single FILE stream with which we call importDelimited
            // only once. this would make it misunderstand that only one
            // header line is with this 'single' stream, while actually
            // we may have one header line for each of the files.
            // so we need to skip header lines here instead in importDelimited.
            const char* buf2 = (const char*)buf;
            int size2 = size;
            if (copy_params.has_header && just_saw_header) {
              while (size2-- > 0)
                if (*buf2++ == copy_params.line_delim)
                  break;
              if (size2 <= 0)
                LOG(WARNING) << "No line delimiter in block." << std::endl;
              just_saw_header = false;
            }
            // In very rare occasions the write pipe somehow operates in a mode similar to non-blocking
            // while pipe(fds) should behave like pipe2(fds, 0) which means blocking mode. On such a
            // unreliable blocking mode, a possible fix is to loop reading till no bytes left, otherwise
            // the annoying `failed to write pipe: Success`...
            if (size2 > 0)
              for (int nread = 0, nleft = size2; nleft > 0; nleft -= (nread > 0 ? nread : 0)) {
                nread = write(fd[1], buf2, nleft);
                if (nread == nleft)
                  break;  // done
                // no exception when too many rejected
                if (import_status.load_truncated) {
                  stop = true;
                  break;
                }
                // not to overwrite original error
                if (nread < 0 && !(errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
                  throw std::runtime_error(std::string("failed or interrupted write to pipe: ") + strerror(errno));
              }
          }
      } catch (...) {
        if (!teptr)  // no replace
          teptr = std::current_exception();
        break;
      }
    }
    // close writer end
    close(fd[1]);
  });

  th_pipe_reader.join();
  th_pipe_writer.join();

  // rethrow any exception happened herebefore
  if (teptr)
    std::rethrow_exception(teptr);
}

ImportStatus Importer::import() {
  return DataStreamSink::archivePlumber();
}

#define IMPORT_FILE_BUFFER_SIZE (1 << 23)  // not too big (need much memory) but not too small (many thread forks)
#define MIN_FILE_BUFFER_SIZE 50000         // 50K min buffer

ImportStatus Importer::importDelimited(const std::string& file_path, const bool decompressed) {
  bool load_truncated = false;
  set_import_status(import_id, import_status);

  if (!p_file)
    p_file = fopen(file_path.c_str(), "rb");
  if (!p_file) {
    throw std::runtime_error("failed to open file '" + file_path + "': " + strerror(errno));
  }

  if (!decompressed) {
    (void)fseek(p_file, 0, SEEK_END);
    file_size = ftell(p_file);
  }

  if (copy_params.threads == 0)
    max_threads = sysconf(_SC_NPROCESSORS_CONF);
  else
    max_threads = copy_params.threads;

  // deal with small files
  size_t alloc_size = IMPORT_FILE_BUFFER_SIZE;
  if (!decompressed && file_size < alloc_size)
    alloc_size = file_size;

  for (int i = 0; i < max_threads; i++) {
    import_buffers_vec.push_back(std::vector<std::unique_ptr<TypedImportBuffer>>());
    for (const auto cd : loader->get_column_descs())
      import_buffers_vec[i].push_back(
          std::unique_ptr<TypedImportBuffer>(new TypedImportBuffer(cd, loader->get_string_dict(cd))));
  }

  std::shared_ptr<char> sbuffer(new char[alloc_size]);
  size_t current_pos = 0;
  size_t end_pos;
  bool eof_reached = false;
  size_t begin_pos = 0;

  (void)fseek(p_file, current_pos, SEEK_SET);
  size_t size = fread((void*)sbuffer.get(), 1, alloc_size, p_file);

  ChunkKey chunkKey = {loader->get_catalog().get_currentDB().dbId, loader->get_table_desc()->tableId};
  {
    std::list<std::future<ImportStatus>> threads;

    // use a stack to track thread_ids which must not overlap among threads
    // because thread_id is used to index import_buffers_vec[]
    std::stack<int> stack_thread_ids;
    for (int i = 0; i < max_threads; i++)
      stack_thread_ids.push(i);
    auto start_epoch = loader->getTableEpoch();
    while (size > 0) {
      if (eof_reached)
        end_pos = size;
      else
        end_pos = find_end(sbuffer.get(), size, copy_params);
      // unput residual
      int nresidual = size - end_pos;
      std::unique_ptr<char> unbuf(nresidual > 0 ? new char[nresidual] : nullptr);
      if (unbuf)
        memcpy(unbuf.get(), sbuffer.get() + end_pos, nresidual);

      // get a thread_id not in use
      auto thread_id = stack_thread_ids.top();
      stack_thread_ids.pop();
      // LOG(INFO) << " stack_thread_ids.pop " << thread_id << std::endl;

      threads.push_back(
          std::async(std::launch::async, import_thread, thread_id, this, sbuffer, begin_pos, end_pos, end_pos));

      current_pos += end_pos;
      sbuffer.reset(new char[alloc_size]);
      memcpy(sbuffer.get(), unbuf.get(), nresidual);
      size = nresidual + fread(sbuffer.get() + nresidual, 1, IMPORT_FILE_BUFFER_SIZE - nresidual, p_file);
      if (size < IMPORT_FILE_BUFFER_SIZE && feof(p_file))
        eof_reached = true;

      begin_pos = 0;

      while (threads.size() > 0) {
        int nready = 0;
        for (std::list<std::future<ImportStatus>>::iterator it = threads.begin(); it != threads.end(); it = it) {
          auto& p = *it;
          std::chrono::milliseconds span(0);  //(std::distance(it, threads.end()) == 1? 1: 0);
          if (p.wait_for(span) == std::future_status::ready) {
            auto ret_import_status = p.get();
            import_status += ret_import_status;
            import_status.rows_estimated = ((float)file_size / current_pos) * import_status.rows_completed;
            set_import_status(import_id, import_status);

            // recall thread_id for reuse
            stack_thread_ids.push(ret_import_status.thread_id);
            // LOG(INFO) << " stack_thread_ids.push " << ret_import_status.thread_id << std::endl;

            threads.erase(it++);
            ++nready;
          } else
            ++it;
        }

        if (nready == 0)
          std::this_thread::yield();

        // on eof, wait all threads to finish
        if (0 == size)
          continue;

        // keep reading if any free thread slot
        // this is one of the major difference from old threading model !!
        if ((int)threads.size() < max_threads)
          break;
      }

      if (import_status.rows_rejected > copy_params.max_reject) {
        load_truncated = true;
        load_failed = true;
        LOG(ERROR) << "Maximum rows rejected exceeded. Halting load";
        break;
      }
      if (load_failed) {
        load_truncated = true;
        LOG(ERROR) << "A call to the Loader::load failed, Please review the logs for more details";
        break;
      }
    }

    // join dangling threads in case of LOG(ERROR) above
    for (auto& p : threads)
      p.wait();

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
            LOG(ERROR) << "Checkpointing Dictionary for Column " << p->getColumnDesc()->columnName << " failed.";
            load_failed = true;
            break;
          }
        }
      }
    });
    if (debug_timing)
      LOG(INFO) << "Dictionary Checkpointing took " << (double)ms / 1000.0 << " Seconds." << std::endl;
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
    const auto shard_tables = get_catalog().getPhysicalTablesDescriptors(get_table_desc());
    for (const auto shard_table : shard_tables) {
      get_catalog().get_dataMgr().checkpoint(get_catalog().get_currentDB().dbId, shard_table->tableId);
    }
  }
}

int32_t Loader::getTableEpoch() {
  return get_catalog().getTableEpoch(get_catalog().get_currentDB().dbId, get_table_desc()->tableId);
}

void Loader::setTableEpoch(int32_t start_epoch) {
  get_catalog().setTableEpoch(get_catalog().get_currentDB().dbId, get_table_desc()->tableId, start_epoch);
}

void GDALErrorHandler(CPLErr eErrClass, int err_no, const char* msg) {
  throw std::runtime_error("GDAL error: " + std::string(msg));
}

namespace GeomConvexHull {
// The code in this namespace is based on the code found here:
// http://www.geeksforgeeks.org/convex-hull-set-2-graham-scan/
// using Graham's scan algorithm: https://en.wikipedia.org/wiki/Graham_scan

// TODO(croot): add this to a geom utility library

// squared distance between points p1 and p2
double distSq(const p2t::Point* p1, const p2t::Point* p2) {
  return (p1->x - p2->x) * (p1->x - p2->x) + (p1->y - p2->y) * (p1->y - p2->y);
}

// Finds the orientation of a triplet of points (p, q, r).
// 0 --> All collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(const p2t::Point* p, const p2t::Point* q, const p2t::Point* r) {
  auto val = (q->y - p->y) * (r->x - q->x) - (q->x - p->x) * (r->y - q->y);

  if (val == 0)
    return 0;                // colinear
  return (val > 0) ? 1 : 2;  // clock or counterclock wise
}

// Returns the convex hull of a set of points
std::vector<p2t::Point*> buildConvexHull(const std::vector<p2t::Point*>& pts) {
  // Find the bottommost point
  std::vector<p2t::Point*> mypts = pts;

  double ymin = mypts[0]->y;
  int min = 0;
  for (int i = 1; i < static_cast<int>(mypts.size()); i++) {
    double y = mypts[i]->y;

    // Pick the bottom-most or chose the left
    // most point in case of tie
    if ((y < ymin) || (ymin == y && mypts[i]->x < mypts[min]->x)) {
      ymin = mypts[i]->y;
      min = i;
    }
  }

  // Place the bottom-most point at first position
  if (min != 0) {
    auto ptr = mypts[0];
    mypts[0] = mypts[min];
    mypts[min] = ptr;
  }

  // Sort pts.
  // A point p1 comes before p2 if p2
  // has larger polar angle (in counterclockwise direction) than p1
  auto p0 = mypts[0];
  std::sort(mypts.begin() + 1, mypts.end(), [&p0](const p2t::Point* p1, const p2t::Point* p2) {
    // Find orientation
    auto o = orientation(p0, p1, p2);
    if (o == 0) {
      return (distSq(p0, p2) >= distSq(p0, p1)) ? true : false;
    }

    return (o == 2) ? true : false;
  });

  // If two or more pts make same angle with p0,
  // Remove all but the one that is farthest from p0
  int m = 1;  // Initialize size of modified array
  int n = static_cast<int>(mypts.size());
  for (int i = 1; i < n; i++) {
    // Keep removing i while angle of i and i+1 is same
    // with respect to p0
    while (i < n - 1 && orientation(p0, mypts[i], mypts[i + 1]) == 0)
      i++;

    mypts[m] = mypts[i];
    m++;  // Update size of modified array
  }

  // If modified array of pts has less than 3 pts,
  // convex hull is not possible
  if (m < 3) {
    throw std::runtime_error("Cannot compute convex hull. All points are collinear.");
  }

  // Create an empty stack and push first three pts
  // to it.
  std::vector<p2t::Point*> hullPts;
  hullPts.push_back(mypts[0]);
  hullPts.push_back(mypts[1]);
  hullPts.push_back(mypts[2]);

  // Process remaining n-3 pts
  for (int i = 3; i < m; i++) {
    // Keep removing top while the angle formed by
    // pts next-to-top, top, and pts[i] makes
    // a non-left turn
    while (orientation(hullPts.at(hullPts.size() - 2), hullPts.back(), mypts[i]) != 2) {
      hullPts.pop_back();
    }
    hullPts.push_back(mypts[i]);
  }

  return hullPts;
}

}  // namespace GeomConvexHull

namespace GeomSimplify {
// This code in this namespace is taken from https://github.com/mourner/simplify-js, translated to C++
// and re-purposed for poly simplification to handle problematic polys on import.
// It makes use of the Douglas-Peuker algorithm for simplification:
// https://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm

// TODO(croot): add this to a geom utility library

// returns a value proportional to the distance
// between the point p and the line joining the
// points p1 and p2
double lineDist(const p2t::Point* p1, const p2t::Point* p2, const p2t::Point* p) {
  return std::abs((p->y - p1->y) * (p2->x - p1->x) - (p2->y - p1->y) * (p->x - p1->x));
}

// square distance between 2 points
double getSqDist(const p2t::Point* p1, const p2t::Point* p2) {
  auto dx = p1->x - p2->x;
  auto dy = p1->y - p2->y;
  return dx * dx + dy * dy;
}

// square distance from a point to a segment
double getSqSegDist(const p2t::Point* p, const p2t::Point* p1, const p2t::Point* p2) {
  auto x = p1->x, y = p1->y, dx = p2->x - x, dy = p2->y - y;

  if (dx != 0 || dy != 0) {
    auto t = ((p->x - x) * dx + (p->y - y) * dy) / (dx * dx + dy * dy);
    if (t > 1) {
      x = p2->x;
      y = p2->y;
    } else if (t > 0) {
      x += dx * t;
      y += dy * t;
    }
  }

  dx = p->x - x;
  dy = p->y - y;

  return dx * dx + dy * dy;
}

// basic distance-based simplification
std::vector<p2t::Point*> simplifyRadialDist(const std::vector<p2t::Point*>& points, const double sqTolerance) {
  p2t::Point* prevPoint = points[0];
  std::vector<p2t::Point*> newPoints = {prevPoint};
  p2t::Point* point;

  for (size_t i = 1; i < points.size(); ++i) {
    point = points[i];
    if (getSqDist(point, prevPoint) > sqTolerance) {
      newPoints.push_back(points[i]);
      prevPoint = point;
    }
  }

  if (prevPoint != point) {
    newPoints.push_back(point);
  }

  return newPoints;
}

void simplifyDPStep(const std::vector<p2t::Point*>& points,
                    const int first,
                    const int last,
                    const double sqTolerance,
                    std::vector<p2t::Point*>& simplified) {
  double maxSqDist = sqTolerance;
  int index = -1;

  for (int i = first + 1; i < last; i++) {
    auto sqDist = getSqSegDist(points[i], points[first], points[last]);
    if (sqDist > maxSqDist) {
      index = i;
      maxSqDist = sqDist;
    }
  }

  if (maxSqDist > sqTolerance) {
    if (index - first > 1) {
      simplifyDPStep(points, first, index, sqTolerance, simplified);
    }
    simplified.push_back(points[index]);
    if (last - index > 1) {
      simplifyDPStep(points, index, last, sqTolerance, simplified);
    }
  }
}

std::vector<p2t::Point*> simplifyDouglasPeucker(const std::vector<p2t::Point*>& points, const double sqTolerance) {
  auto last = points.size() - 1;

  std::vector<p2t::Point*> simplified = {points[0]};
  simplifyDPStep(points, 0, last, sqTolerance, simplified);
  simplified.push_back(points[last]);

  if (simplified.size() == 2) {
    double maxSqDist(0.0);
    int index = -1;
    for (int i = 1; i < static_cast<int>(points.size()) - 1; i++) {
      auto sqDist = lineDist(simplified[0], simplified[1], points[i]);
      if (sqDist > maxSqDist) {
        index = i;
        maxSqDist = sqDist;
      }
    }
    if (index < 0) {
      index = 1;
    }
    simplified.insert(simplified.begin() + 1, points[index]);
  }

  return simplified;
}

// Returns a simplified set of points from an input list of connected points.
std::vector<p2t::Point*> simplify(const std::vector<p2t::Point*>& points,
                                  const double tolerance = 1.0,
                                  const bool highestQuality = false) {
  if (points.size() <= 3) {
    return points;
  }

  auto sqTolerance = tolerance * tolerance;
  return simplifyDouglasPeucker((highestQuality ? points : simplifyRadialDist(points, sqTolerance)), sqTolerance);
}
}  // namespace GeomSimplify

std::tuple<bool, double, double> buildRenderablePolyOutline(
    PolyData2d& poly,
    const std::vector<p2t::Point*>& vertexPtrs,
    std::unordered_map<p2t::Point*, int>& pointIndices,
    const std::unordered_map<p2t::Point*, int>& origPointIndices,
    const ssize_t featureIdx,
    const ssize_t multipolyIdx) {
  if (vertexPtrs.size() < 2) {
    throw std::runtime_error("invalid geometry: polygon" +
                             (multipolyIdx < 0 ? "" : (" at multipolygon index " + std::to_string(multipolyIdx + 1))) +
                             " in feature " + std::to_string(featureIdx + 1) + " - Cannot import a polygon with " +
                             std::to_string(vertexPtrs.size()) + " point" + (vertexPtrs.size() == 1 ? "." : "s."));
  } else if (vertexPtrs.size() == 2) {
    LOG(WARNING)
        << "invalid geometry: polygon"
        << (multipolyIdx < 0 ? "" : (" at multipolygon index " + std::to_string(multipolyIdx + 1))) << " in feature "
        << (featureIdx + 1) << " has only " << std::to_string(vertexPtrs.size()) << " point"
        << (vertexPtrs.size() == 1
                ? "."
                : "s. The geom will be imported, but it might not be visible unless you have stroking enabled.");

    poly.beginLine();
    poly.addLinePoint(vertexPtrs[0]);
    poly.addLinePoint(vertexPtrs[0]);
    poly.addLinePoint(vertexPtrs[1]);
    poly.addLinePoint(vertexPtrs[1]);
    poly.endLine(false);
    poly.beginPoly();
    poly.addTriangle(0, 2, 1);
    poly.endPoly();
    return std::make_tuple(true, 0.0, 0.0);
  }

  pointIndices.clear();
  std::set<std::pair<double, double>> dedupe;
  poly.beginLine();
  ScopeGuard guard = [&poly]() { poly.endLine(); };
  double mindist = std::numeric_limits<double>::max(), avgdist(0.0), sumdist(0.0);
  for (size_t k = 0; k < vertexPtrs.size(); ++k) {
    auto vert = vertexPtrs[k];
    auto origitr = origPointIndices.find(vert);
    CHECK(origitr != origPointIndices.end());
    auto origxypair = std::make_pair(vert->x, vert->y);
    auto a = dedupe.insert(origxypair);
    if (!a.second) {
      const auto& startpt = *vertexPtrs[k - 1];
      const auto next = vertexPtrs[(k + 1) % vertexPtrs.size()];
      p2t::Point nextpt(next->x, next->y);
      std::array<double, 2> v({startpt.x - nextpt.x, startpt.y - nextpt.y});

      v[0] *= 0.5;
      v[1] *= 0.5;
      nextpt.x += v[0];
      nextpt.y += v[1];
      v[0] = nextpt.x - vert->x;
      v[1] = nextpt.y - vert->y;
      if (!(v[0] * v[0] + v[1] * v[1])) {
        v[0] = startpt.x - vert->x;
        v[1] = startpt.y - vert->y;
        CHECK_NE(v[0] * v[0] + v[1] * v[1], 0);
      }
      const auto stepsz = 0.01;
      v[0] *= stepsz;
      v[1] *= stepsz;
      auto currstep = 0;
      while (!a.second && currstep < 1.0) {
        currstep += stepsz;
        vert->x += v[0];
        vert->y += v[1];
        a = dedupe.insert(std::make_pair(vert->x, vert->y));
      }

      if (currstep >= 1.0) {
        throw std::runtime_error(
            "invalid geometry: polygon" +
            (multipolyIdx < 0 ? "" : (" at multipolygon index " + std::to_string(multipolyIdx + 1))) + " in feature " +
            std::to_string(featureIdx + 1) + " - duplicate vertex found at index " + std::to_string(origitr->second) +
            " which cannot be properly transformed to avoid the duplication.");
      }

      LOG(WARNING) << "duplicate vertex at index " << origitr->second << " in polygon"
                   << (multipolyIdx < 0 ? "" : (" at multipolygon index " + std::to_string(multipolyIdx + 1)))
                   << " in feature " << (featureIdx + 1) << ". It has been transformed from [" << origxypair.first
                   << ", " << origxypair.second << "] to " << std::string(*vert)
                   << " in order to avoid overlap issues.";
    }
    if (k > 0) {
      const auto lastpt = vertexPtrs[k - 1];
      const auto dx = vert->x - lastpt->x;
      const auto dy = vert->y - lastpt->y;
      const auto dist = std::sqrt(dx * dx + dy * dy);
      sumdist += dist;
      if (dist < mindist) {
        mindist = dist;
      }
    }
    vert->edge_list.clear();  // make sure to clear out the containers build up from a previous triangulation run
    poly.addLinePoint(vert);
    pointIndices.insert({vert, k});
  }
  // already guaranteed to have a poly with at least 3 verts by the time we
  // reach this point
  const auto lastpt = vertexPtrs.back();
  const auto firstpt = vertexPtrs.front();
  const auto dx = firstpt->x - lastpt->x;
  const auto dy = firstpt->y - lastpt->y;
  const auto dist = std::sqrt(dx * dx + dy * dy);
  sumdist += dist;
  if (dist < mindist) {
    mindist = dist;
  }
  avgdist = sumdist / vertexPtrs.size();
  return std::make_tuple(false, mindist, avgdist);
}

p2t::CDT triangulate(bool& triangulated, const std::vector<p2t::Point*>& vertexPtrs) {
  triangulated = false;
  p2t::CDT triangulator(vertexPtrs);
  triangulator.Triangulate();
  triangulated = true;
  return triangulator;
}

void buildRenderablePolyAfterTriangulation(PolyData2d& poly,
                                           p2t::CDT& triangulator,
                                           const std::unordered_map<p2t::Point*, int>& pointIndices,
                                           const std::unordered_map<p2t::Point*, int>& origPointIndices) {
  int idx0, idx1, idx2;
  std::unordered_map<p2t::Point*, int>::const_iterator itr;
  poly.beginPoly();
  ScopeGuard guard = [&poly]() { poly.endPoly(); };
  for (p2t::Triangle* tri : triangulator.GetTriangles()) {
    itr = pointIndices.find(tri->GetPoint(0));
    if (itr == pointIndices.end()) {
      throw std::runtime_error("failed to triangulate polygon due to invalid triangle - triidx: " +
                               std::to_string(poly.numTris()) + ", pointidx: 0");
    }
    idx0 = itr->second;

    itr = pointIndices.find(tri->GetPoint(1));
    if (itr == pointIndices.end()) {
      throw std::runtime_error("failed to triangulate polygon due to invalid triangle - triidx: " +
                               std::to_string(poly.numTris()) + ", pointidx: 1");
    }
    idx1 = itr->second;

    itr = pointIndices.find(tri->GetPoint(2));
    if (itr == pointIndices.end()) {
      throw std::runtime_error("failed to triangulate polygon due to invalid triangle - triidx: " +
                               std::to_string(poly.numTris()) + ", pointidx: 2");
    }
    idx2 = itr->second;

    poly.addTriangle(idx0, idx1, idx2);
  }
}

void Importer::readVerticesFromGDALGeometryZ(const std::string& fileName,
                                             OGRPolygon* poPolygon,
                                             PolyData2d& poly,
                                             const bool,
                                             const ssize_t featureIdx,
                                             const ssize_t multipolyIdx = -1) {
  std::vector<std::shared_ptr<p2t::Point>> vertexShPtrs;
  std::vector<p2t::Point*> vertexPtrs;
  std::vector<int> tris;
  std::unordered_map<p2t::Point*, int> origPointIndices;
  std::unordered_map<p2t::Point*, int> pointIndices;
  OGRPoint ptTemp;

  OGRLinearRing* poExteriorRing = poPolygon->getExteriorRing();
  if (!poExteriorRing->isClockwise()) {
    poExteriorRing->reverseWindingOrder();
  }
  if (!poExteriorRing->isClockwise()) {
    return;
  }
  poExteriorRing->closeRings();
  int nExtVerts = poExteriorRing->getNumPoints();
  if (nExtVerts < 3) {
    throw std::runtime_error("invalid geometry: polygon" +
                             (multipolyIdx < 0 ? "" : (" at multipolygon index " + std::to_string(multipolyIdx + 1))) +
                             " in feature " + std::to_string(featureIdx + 1) + " needs at least 3 points. It has " +
                             std::to_string(nExtVerts) + ".");
  }

  for (int k = 0; k < nExtVerts - 1; k++) {
    poExteriorRing->getPoint(k, &ptTemp);
    if (k > 0 && vertexPtrs.back()->x == ptTemp.getX() && vertexPtrs.back()->y == ptTemp.getY()) {
      continue;
    }
    vertexShPtrs.emplace_back(new p2t::Point(ptTemp.getX(), ptTemp.getY()));
    auto vertPtr = vertexShPtrs.back().get();
    vertexPtrs.push_back(vertPtr);
    origPointIndices.insert({vertPtr, k});
  }

  bool finished;
  double mindist, avgdist;
  std::tie(finished, mindist, avgdist) =
      buildRenderablePolyOutline(poly, vertexPtrs, pointIndices, origPointIndices, featureIdx, multipolyIdx);
  if (finished) {
    return;
  }

  bool triangulated = false;
  try {
    auto triangulator = triangulate(triangulated, vertexPtrs);
    buildRenderablePolyAfterTriangulation(poly, triangulator, pointIndices, origPointIndices);
  } catch (std::exception& err) {
    if (triangulated) {
      poly.popPoly();
    }
    poly.popLine();

    // try a simplify with a modest tolerance to see if we can get the geom
    // to be more poly2tri friendly
    auto simplifyVertexPtrs = GeomSimplify::simplify(vertexPtrs, mindist + 0.1 * (avgdist - mindist), true);
    std::tie(finished, mindist, avgdist) =
        buildRenderablePolyOutline(poly, simplifyVertexPtrs, pointIndices, origPointIndices, featureIdx, multipolyIdx);
    try {
      if (!finished) {
        auto triangulator = triangulate(triangulated, simplifyVertexPtrs);
        buildRenderablePolyAfterTriangulation(poly, triangulator, pointIndices, origPointIndices);
      }
      LOG(WARNING) << "Failed to triangulate original polygon "
                   << (multipolyIdx < 0 ? "" : ("at multipolygon index " + std::to_string(multipolyIdx + 1)))
                   << " in feature " << (featureIdx + 1)
                   << ". A simplified geometry will be used as a proxy in its place. The simplified geometry has "
                   << simplifyVertexPtrs.size() << " vertices. The original polygon had " << vertexPtrs.size() << ".";
    } catch (std::exception& err) {
      if (triangulated) {
        poly.popPoly();
      }
      poly.popLine();
      auto convexHullVertexPtrs = GeomConvexHull::buildConvexHull(vertexPtrs);
      std::tie(finished, mindist, avgdist) = buildRenderablePolyOutline(
          poly, convexHullVertexPtrs, pointIndices, origPointIndices, featureIdx, multipolyIdx);
      try {
        if (!finished) {
          auto triangulator = triangulate(triangulated, convexHullVertexPtrs);
          buildRenderablePolyAfterTriangulation(poly, triangulator, pointIndices, origPointIndices);
        }
        LOG(WARNING)
            << "Failed to triangulate original polygon"
            << (multipolyIdx < 0 ? "" : (" at multipolygon index " + std::to_string(multipolyIdx + 1)))
            << " in feature " << (featureIdx + 1)
            << ". A convex hull of the original geometry will be used as a proxy in its place. The convex hull has "
            << convexHullVertexPtrs.size() << " vertices. The original polygon had " << vertexPtrs.size() << ".";
      } catch (std::exception& err) {
        throw std::runtime_error(
            "invalid geometry: polygon" +
            (multipolyIdx < 0 ? "" : (" at multipolygon index " + std::to_string(multipolyIdx + 1))) + " in feature " +
            std::to_string(featureIdx + 1) + "- " + err.what() +
            ". Cannot build proxy geometry after trying a simplification and convex hull.");
      }
    }
  }
}

void initGDAL() {
  static bool gdal_initialized = false;
  if (!gdal_initialized) {
    // FIXME(andrewseidl): investigate if CPLPushFinderLocation can be public
    setenv("GDAL_DATA", std::string(mapd_root_abs_path() + "/ThirdParty/gdal-data").c_str(), true);
    GDALAllRegister();
    OGRRegisterAll();
    gdal_initialized = true;
  }
}

static OGRDataSource* openGDALDataset(const std::string& fileName) {
  initGDAL();
  CPLSetErrorHandler(*GDALErrorHandler);
  OGRDataSource* poDS;
#if GDAL_VERSION_MAJOR == 1
  poDS = (OGRDataSource*)OGRSFDriverRegistrar::Open(fileName.c_str(), false);
#else
  poDS = (OGRDataSource*)GDALOpenEx(fileName.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
#endif
  if (poDS == nullptr) {
    LOG(INFO) << "ogr error: " << CPLGetLastErrorMsg();
  }
  return poDS;
}

void Importer::readMetadataSampleGDAL(const std::string& fileName,
                                      std::map<std::string, std::vector<std::string>>& metadata,
                                      int rowLimit) {
  auto poDS = openGDALDataset(fileName);
  if (poDS == nullptr) {
    throw std::runtime_error("Unable to open geo file " + fileName);
  }
  OGRLayer* poLayer;
  poLayer = poDS->GetLayer(0);
  if (poLayer == nullptr) {
    throw std::runtime_error("No layers found in " + fileName);
  }
  OGRFeatureDefn* poFDefn = poLayer->GetLayerDefn();

  // typeof GetFeatureCount() is different between GDAL 1.x (int32_t) and 2.x (int64_t)
  auto nFeats = poLayer->GetFeatureCount();
  size_t numFeatures =
      std::max(static_cast<decltype(nFeats)>(0), std::min(static_cast<decltype(nFeats)>(rowLimit), nFeats));
  for (auto iField = 0; iField < poFDefn->GetFieldCount(); iField++) {
    OGRFieldDefn* poFieldDefn = poFDefn->GetFieldDefn(iField);
    // FIXME(andrewseidl): change this to the faster one used by readVerticesFromGDAL
    metadata.emplace(poFieldDefn->GetNameRef(), std::vector<std::string>(numFeatures));
  }
  OGRFeature* poFeature;
  poLayer->ResetReading();
  size_t iFeature = 0;
  while ((poFeature = poLayer->GetNextFeature()) != nullptr && iFeature < numFeatures) {
    OGRGeometry* poGeometry;
    poGeometry = poFeature->GetGeometryRef();
    if (poGeometry != nullptr) {
      switch (wkbFlatten(poGeometry->getGeometryType())) {
        case wkbPolygon:
        case wkbMultiPolygon:
          break;
        default:
          throw std::runtime_error("Unsupported geometry type: " + std::string(poGeometry->getGeometryName()));
      }
      for (auto i : metadata) {
        auto iField = poFeature->GetFieldIndex(i.first.c_str());
        metadata[i.first].at(iFeature) = std::string(poFeature->GetFieldAsString(iField));
      }
      OGRFeature::DestroyFeature(poFeature);
    }
    iFeature++;
  }
  GDALClose(poDS);
}

void Importer::readVerticesFromGDAL(
    const std::string& fileName,
    std::vector<PolyData2d>& polys,
    std::pair<std::map<std::string, size_t>, std::vector<std::vector<std::string>>>& metadata) {
  auto poDS = openGDALDataset(fileName);
  if (poDS == nullptr) {
    throw std::runtime_error("Unable to open geo file " + fileName);
  }
  OGRLayer* poLayer;
  poLayer = poDS->GetLayer(0);
  if (poLayer == nullptr) {
    throw std::runtime_error("No layers found in " + fileName);
  }
  OGRFeatureDefn* poFDefn = poLayer->GetLayerDefn();
  size_t numFeatures = poLayer->GetFeatureCount();
  size_t nFields = poFDefn->GetFieldCount();
  for (size_t iField = 0; iField < nFields; iField++) {
    OGRFieldDefn* poFieldDefn = poFDefn->GetFieldDefn(iField);
    metadata.first[poFieldDefn->GetNameRef()] = iField;
    metadata.second.push_back(std::vector<std::string>(numFeatures));
  }
  OGRFeature* poFeature;
  poLayer->ResetReading();
  auto poSR = new OGRSpatialReference();
  poSR->importFromEPSG(3857);

  // typeof GetFeatureCount() is different between GDAL 1.x (int32_t) and 2.x (int64_t)
  auto nFeats = poLayer->GetFeatureCount();
  decltype(nFeats) iFeature = 0;
  for (iFeature = 0; iFeature < nFeats; iFeature++) {
    poFeature = poLayer->GetNextFeature();
    if (poFeature == nullptr) {
      break;
    }
    try {
      OGRGeometry* poGeometry;
      poGeometry = poFeature->GetGeometryRef();
      if (poGeometry != nullptr) {
        poGeometry->transformTo(poSR);
        if (polys.size()) {
          polys.emplace_back(polys.back().startVert() + polys.back().numVerts(),
                             polys.back().startIdx() + polys.back().numIndices());
        } else {
          polys.emplace_back();
        }
        switch (wkbFlatten(poGeometry->getGeometryType())) {
          case wkbPolygon: {
            OGRPolygon* poPolygon = (OGRPolygon*)poGeometry;
            readVerticesFromGDALGeometryZ(fileName, poPolygon, polys.back(), false, iFeature);
            break;
          }
          case wkbMultiPolygon: {
            OGRMultiPolygon* poMultiPolygon = (OGRMultiPolygon*)poGeometry;
            int NumberOfGeometries = poMultiPolygon->getNumGeometries();
            for (auto j = 0; j < NumberOfGeometries; j++) {
              OGRGeometry* poPolygonGeometry = poMultiPolygon->getGeometryRef(j);
              OGRPolygon* poPolygon = (OGRPolygon*)poPolygonGeometry;
              readVerticesFromGDALGeometryZ(fileName, poPolygon, polys.back(), false, iFeature, j);
            }
            break;
          }
          case wkbPoint:
          default:
            throw std::runtime_error("Unsupported geometry type: " + std::string(poGeometry->getGeometryName()));
        }
        for (size_t iField = 0; iField < nFields; iField++) {
          metadata.second[iField][iFeature] = std::string(poFeature->GetFieldAsString(iField));
        }
        OGRFeature::DestroyFeature(poFeature);
      }
    } catch (const std::exception& e) {
      throw std::runtime_error(e.what() + std::string(" Feature: ") + std::to_string(iFeature + 1));
    }
  }
  GDALClose(poDS);
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

ColumnDescriptor create_array_column(const SQLTypes& type, const std::string& name) {
  ColumnDescriptor cd;
  cd.columnName = name;
  SQLTypeInfo ti;
  ti.set_type(kARRAY);
  ti.set_subtype(type);
  ti.set_fixed_size();
  cd.columnType = ti;
  return cd;
}

const std::list<ColumnDescriptor> Importer::gdalToColumnDescriptors(const std::string& fileName) {
  std::list<ColumnDescriptor> cds;
  auto poDS = openGDALDataset(fileName);
  try {
    if (poDS == nullptr) {
      throw std::runtime_error("Unable to open geo file " + fileName + " : " + CPLGetLastErrorMsg());
    }
    OGRLayer* poLayer;
    poLayer = poDS->GetLayer(0);
    if (poLayer == nullptr) {
      throw std::runtime_error("No layers found in " + fileName);
    }
    OGRFeature* poFeature;
    poLayer->ResetReading();
    // TODO(andrewseidl): support multiple features
    if ((poFeature = poLayer->GetNextFeature()) != nullptr) {
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
    }
  } catch (const std::exception& e) {
    GDALClose(poDS);
    throw;
  }
  GDALClose(poDS);

  return cds;
}

ImportStatus Importer::importGDAL(std::map<std::string, std::string> colname_to_src) {
  set_import_status(import_id, import_status);
  std::vector<PolyData2d> polys;
  std::pair<std::map<std::string, size_t>, std::vector<std::vector<std::string>>> metadata;

  readVerticesFromGDAL(file_path.c_str(), polys, metadata);

  std::vector<std::unique_ptr<TypedImportBuffer>> import_buffers_vec;
  for (const auto cd : loader->get_column_descs())
    import_buffers_vec.push_back(
        std::unique_ptr<TypedImportBuffer>(new TypedImportBuffer(cd, loader->get_string_dict(cd))));

  for (size_t ipoly = 0; ipoly < polys.size(); ++ipoly) {
    auto poly = polys[ipoly];
    try {
      auto icol = 0;
      for (auto cd : loader->get_column_descs()) {
        if (cd->columnName == MAPD_GEO_PREFIX + "coords") {
          std::vector<TDatum> coords;
          for (auto coord : poly.coords) {
            TDatum td;
            td.val.real_val = coord;
            coords.push_back(td);
          }
          TDatum tdd;
          tdd.val.arr_val = coords;
          tdd.is_null = false;
          import_buffers_vec[icol++]->add_value(cd, tdd, false);
        } else if (cd->columnName == MAPD_GEO_PREFIX + "indices") {
          std::vector<TDatum> indices;
          for (auto tris : poly.triangulation_indices) {
            TDatum td;
            td.val.int_val = tris;
            indices.push_back(td);
          }
          TDatum tdd;
          tdd.val.arr_val = indices;
          tdd.is_null = false;
          import_buffers_vec[icol++]->add_value(cd, tdd, false);
        } else if (cd->columnName == MAPD_GEO_PREFIX + "linedrawinfo") {
          std::vector<TDatum> ldis;
          ldis.resize(4 * poly.lineDrawInfo.size());
          size_t ildi = 0;
          for (auto ldi : poly.lineDrawInfo) {
            ldis[ildi++].val.int_val = ldi.count;
            ldis[ildi++].val.int_val = ldi.instanceCount;
            ldis[ildi++].val.int_val = ldi.firstIndex;
            ldis[ildi++].val.int_val = ldi.baseInstance;
          }
          TDatum tdd;
          tdd.val.arr_val = ldis;
          tdd.is_null = false;
          import_buffers_vec[icol++]->add_value(cd, tdd, false);
        } else if (cd->columnName == MAPD_GEO_PREFIX + "polydrawinfo") {
          std::vector<TDatum> pdis;
          pdis.resize(5 * poly.polyDrawInfo.size());
          size_t ipdi = 0;
          for (auto pdi : poly.polyDrawInfo) {
            pdis[ipdi++].val.int_val = pdi.count;
            pdis[ipdi++].val.int_val = pdi.instanceCount;
            pdis[ipdi++].val.int_val = pdi.firstIndex;
            pdis[ipdi++].val.int_val = pdi.baseVertex;
            pdis[ipdi++].val.int_val = pdi.baseInstance;
          }
          TDatum tdd;
          tdd.val.arr_val = pdis;
          tdd.is_null = false;
          import_buffers_vec[icol++]->add_value(cd, tdd, false);
        } else {
          auto ifield = metadata.first.at(colname_to_src[cd->columnName]);
          auto str = metadata.second.at(ifield).at(ipoly);
          import_buffers_vec[icol++]->add_value(cd, str, false, copy_params);
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "importGDAL exception thrown: " << e.what() << ". Row discarded.";
    }
  }

  try {
    loader->load(import_buffers_vec, polys.size());
    return import_status;
  } catch (const std::exception& e) {
    LOG(WARNING) << e.what();
  }

  return import_status;
}

}  // Namespace Importer
