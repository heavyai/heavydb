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

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <stdexcept>
#include <list>
#include <vector>
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
#include "Importer.h"
#include "gen-cpp/MapD.h"
#include <vector>
#include <iostream>
using std::ostream;

namespace Importer_NS {

bool debug_timing = false;

static mapd_shared_mutex status_mutex;
static std::map<std::string, ImportStatus> import_status_map;

Importer::Importer(const Catalog_Namespace::Catalog& c,
                   const TableDescriptor* t,
                   const std::string& f,
                   const CopyParams& p)
    : Importer(new Loader(c, t), f, p) {}

Importer::Importer(Loader* providedLoader, const std::string& f, const CopyParams& p)
    : file_path(f), copy_params(p), loader(providedLoader), load_failed(false) {
  import_id = boost::filesystem::path(file_path).filename().string();
  file_size = 0;
  max_threads = 0;
  p_file = nullptr;
  buffer[0] = nullptr;
  buffer[1] = nullptr;
  which_buf = 0;
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

void parseStringArray(const std::string& s, const CopyParams& copy_params, std::vector<std::string>& string_vec) {
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
  const auto type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType) : cd->columnType.get_type();
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
        parseStringArray(val, copy_params, string_vec);
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
                                  const char* buffer,
                                  size_t begin_pos,
                                  size_t end_pos,
                                  size_t total_size) {
  ImportStatus import_status;
  int64_t total_get_row_time_us = 0;
  int64_t total_str_to_val_time_us = 0;
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

void Detector::read_file() {
  if (!boost::filesystem::exists(file_path)) {
    LOG(ERROR) << "File does not exist: " << file_path;
    return;
  }
  std::ifstream infile(file_path.string());
  std::string line;
  auto end_time = std::chrono::steady_clock::now() + timeout;
  try {
    while (std::getline(infile, line, copy_params.line_delim)) {
      raw_data += line;
      raw_data += copy_params.line_delim;
      if (std::chrono::steady_clock::now() > end_time) {
        break;
      }
    }
  } catch (std::exception& e) {
  }
  infile.close();
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

ImportStatus Importer::import() {
  // TODO(andrew): add file type detection
  return importDelimited();
}

#define IMPORT_FILE_BUFFER_SIZE 100000000  // 100M file size
#define MIN_FILE_BUFFER_SIZE 50000         // 50K min buffer

ImportStatus Importer::importDelimited() {
  bool load_truncated = false;
  set_import_status(import_id, import_status);
  p_file = fopen(file_path.c_str(), "rb");
  if (!p_file) {
    throw std::runtime_error("fopen failure for '" + file_path + "': " + strerror(errno));
  }
  (void)fseek(p_file, 0, SEEK_END);
  file_size = ftell(p_file);
  if (copy_params.threads == 0)
    max_threads = sysconf(_SC_NPROCESSORS_CONF);
  else
    max_threads = copy_params.threads;
  // deal with small files
  size_t alloc_size = IMPORT_FILE_BUFFER_SIZE;
  if (file_size < alloc_size) {
    alloc_size = file_size;
  }
  buffer[0] = (char*)checked_malloc(alloc_size);
  if (max_threads > 1)
    buffer[1] = (char*)checked_malloc(alloc_size);
  for (int i = 0; i < max_threads; i++) {
    import_buffers_vec.push_back(std::vector<std::unique_ptr<TypedImportBuffer>>());
    for (const auto cd : loader->get_column_descs())
      import_buffers_vec[i].push_back(
          std::unique_ptr<TypedImportBuffer>(new TypedImportBuffer(cd, loader->get_string_dict(cd))));
  }
  size_t current_pos = 0;
  size_t end_pos;
  (void)fseek(p_file, current_pos, SEEK_SET);
  size_t size = fread((void*)buffer[which_buf], 1, alloc_size, p_file);
  bool eof_reached = false;
  size_t begin_pos = 0;
  if (copy_params.has_header) {
    size_t i;
    for (i = 0; i < size && buffer[which_buf][i] != copy_params.line_delim; i++)
      ;
    if (i == size)
      LOG(WARNING) << "No line delimiter in block." << std::endl;
    begin_pos = i + 1;
  }
  ChunkKey chunkKey = {loader->get_catalog().get_currentDB().dbId, loader->get_table_desc()->tableId};
  while (size > 0) {
    // for each process through a buffer take a table lock
    mapd_unique_lock<mapd_shared_mutex> tableLevelWriteLock(*loader->get_catalog().get_dataMgr().getMutexForChunkPrefix(
        chunkKey));  // prevent two threads from trying to insert into the same table simultaneously
    if (eof_reached)
      end_pos = size;
    else
      end_pos = find_end(buffer[which_buf], size, copy_params);
    if (size <= alloc_size) {
      max_threads = std::min(max_threads, (int)std::ceil((double)(end_pos - begin_pos) / MIN_FILE_BUFFER_SIZE));
    }
    if (max_threads == 1) {
      import_status += import_thread(0, this, buffer[which_buf], begin_pos, end_pos, end_pos);
      current_pos += end_pos;
      (void)fseek(p_file, current_pos, SEEK_SET);
      size = fread((void*)buffer[which_buf], 1, IMPORT_FILE_BUFFER_SIZE, p_file);
      if (size < IMPORT_FILE_BUFFER_SIZE && feof(p_file))
        eof_reached = true;
    } else {
      std::vector<std::future<ImportStatus>> threads;
      for (int i = 0; i < max_threads; i++) {
        size_t begin = begin_pos + i * ((end_pos - begin_pos) / max_threads);
        size_t end = (i < max_threads - 1) ? begin_pos + (i + 1) * ((end_pos - begin_pos) / max_threads) : end_pos;
        threads.push_back(
            std::async(std::launch::async, import_thread, i, this, buffer[which_buf], begin, end, end_pos));
      }
      current_pos += end_pos;
      which_buf = (which_buf + 1) % 2;
      (void)fseek(p_file, current_pos, SEEK_SET);
      size = fread((void*)buffer[which_buf], 1, IMPORT_FILE_BUFFER_SIZE, p_file);
      if (size < IMPORT_FILE_BUFFER_SIZE && feof(p_file))
        eof_reached = true;
      for (auto& p : threads)
        p.wait();
      for (auto& p : threads)
        import_status += p.get();
    }
    import_status.rows_estimated = ((float)file_size / current_pos) * import_status.rows_completed;
    set_import_status(import_id, import_status);
    begin_pos = 0;
    if (import_status.rows_rejected > copy_params.max_reject) {
      load_truncated = true;
      LOG(ERROR) << "Maximum rows rejected exceeded. Halting load";
      break;
    }
    if (loader->get_table_desc()->persistenceLevel ==
        Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident tables
      // checkpoint before going again
      const auto shard_tables = loader->get_catalog().getPhysicalTablesDescriptors(loader->get_table_desc());
      for (const auto shard_table : shard_tables) {
        loader->get_catalog().get_dataMgr().checkpoint(loader->get_catalog().get_currentDB().dbId,
                                                       shard_table->tableId);
      }
    }
  }
  // checkpoint before going again
  if (loader->get_table_desc()->persistenceLevel ==
      Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident tables
    // todo MAT we need to review whether this checkpoint process makes sense
    auto ms = measure<>::execution([&]() {
      if (!load_failed) {
        for (auto& p : import_buffers_vec[0]) {
          if (!p->stringDictCheckpoint()) {
            LOG(ERROR) << "Checkpointing Dictionary for Column " << p->getColumnDesc()->columnName << " failed.";
            load_failed = true;
            break;
          }
        }
        loader->get_catalog().get_dataMgr().checkpoint(loader->get_catalog().get_currentDB().dbId,
                                                       loader->get_table_desc()->tableId);
      }
    });
    if (debug_timing)
      LOG(INFO) << "Checkpointing took " << (double)ms / 1000.0 << " Seconds." << std::endl;
  }
  free(buffer[0]);
  buffer[0] = nullptr;
  free(buffer[1]);
  buffer[1] = nullptr;
  fclose(p_file);
  p_file = nullptr;

  import_status.load_truncated = load_truncated;
  return import_status;
}

void GDALErrorHandler(CPLErr eErrClass, int err_no, const char* msg) {
  throw std::runtime_error("GDAL error: " + std::string(msg));
}

void Importer::readVerticesFromGDALGeometryZ(const std::string& fileName,
                                             OGRPolygon* poPolygon,
                                             PolyData2d& poly,
                                             bool) {
  std::vector<std::shared_ptr<p2t::Point>> vertexShPtrs;
  std::vector<p2t::Point*> vertexPtrs;
  std::vector<int> tris;
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
  std::set<std::pair<double, double>> dedupe;

  poly.beginLine();
  for (int k = 0; k < nExtVerts - 1; k++) {
    poExteriorRing->getPoint(k, &ptTemp);
    auto xy = std::make_pair(ptTemp.getX(), ptTemp.getY());
    if (k > 0 && vertexPtrs.back()->x == xy.first && vertexPtrs.back()->y == xy.second) {
      continue;
    }
    auto a = dedupe.insert(std::make_pair(xy.first, xy.second));
    if (!a.second) {
      throw std::runtime_error("invalid geometry: duplicate vertex found");
    }
    vertexShPtrs.emplace_back(new p2t::Point(xy.first, xy.second));
    poly.addLinePoint(vertexShPtrs.back());
    vertexPtrs.push_back(vertexShPtrs.back().get());
    pointIndices.insert({vertexShPtrs.back().get(), vertexPtrs.size() - 1});
  }
  poly.endLine();

  p2t::CDT triangulator(vertexPtrs);

  triangulator.Triangulate();

  int idx0, idx1, idx2;

  std::unordered_map<p2t::Point*, int>::iterator itr;

  poly.beginPoly();
  for (p2t::Triangle* tri : triangulator.GetTriangles()) {
    itr = pointIndices.find(tri->GetPoint(0));
    if (itr == pointIndices.end()) {
      throw std::runtime_error("failed to triangulate polygon");
    }
    idx0 = itr->second;

    itr = pointIndices.find(tri->GetPoint(1));
    if (itr == pointIndices.end()) {
      throw std::runtime_error("failed to triangulate polygon");
    }
    idx1 = itr->second;

    itr = pointIndices.find(tri->GetPoint(2));
    if (itr == pointIndices.end()) {
      throw std::runtime_error("failed to triangulate polygon");
    }
    idx2 = itr->second;

    poly.addTriangle(idx0, idx1, idx2);
  }
  poly.endPoly();
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
            readVerticesFromGDALGeometryZ(fileName, poPolygon, polys.back(), false);
            break;
          }
          case wkbMultiPolygon: {
            OGRMultiPolygon* poMultiPolygon = (OGRMultiPolygon*)poGeometry;
            int NumberOfGeometries = poMultiPolygon->getNumGeometries();
            for (auto j = 0; j < NumberOfGeometries; j++) {
              OGRGeometry* poPolygonGeometry = poMultiPolygon->getGeometryRef(j);
              OGRPolygon* poPolygon = (OGRPolygon*)poPolygonGeometry;
              readVerticesFromGDALGeometryZ(fileName, poPolygon, polys.back(), false);
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
