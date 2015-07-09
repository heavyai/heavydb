/*
 * @file Importer.cpp
 * @author Wei Hong <wei@mapd.com>
 * @brief Functions for Importer class
 */

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <unistd.h>
#include <stdexcept>
#include <list>
#include <vector>
#include <thread>
#include <mutex>
#include <glog/logging.h>
#include "../Shared/measure.h"
#include "Importer.h"
#include "gen-cpp/MapD.h"

namespace Importer_NS {

bool debug_timing = false;

static std::mutex insert_mutex;

Importer::Importer(const Catalog_Namespace::Catalog &c, const TableDescriptor *t, const std::string &f, const CopyParams &p) : file_path(f), copy_params(p), loader(c,t), load_failed(false)
{
  file_size = 0;
  max_threads = 0;
  p_file = nullptr;
  buffer[0] = nullptr;
  buffer[1] = nullptr;
  which_buf = 0;
  std::unique_ptr<bool> is_array = std::unique_ptr<bool>((bool*)malloc(loader.get_column_descs().size() * sizeof(bool)));
  int i = 0;
  bool has_array = false;
  for (auto &p : loader.get_column_descs()) {
    if (p->columnType.get_type() == kARRAY) {
      is_array.get()[i] = true;
      has_array = true;
    } else
      is_array.get()[i] = false;
    ++i;
  }
  if (has_array)
    is_array_a = std::unique_ptr<bool>(is_array.release());
  else
    is_array_a = std::unique_ptr<bool>(nullptr);
}

Importer::~Importer()
{
  if (p_file != nullptr)
    fclose(p_file);
  if (buffer[0] != nullptr)
    free(buffer[0]);
  if (buffer[1] != nullptr)
    free(buffer[1]);
}

static const char *
get_row(const char *buf, const char *buf_end, const char *entire_buf_end, const CopyParams &copy_params, bool is_begin, const bool *is_array, std::vector<std::string> &row)
{
  const char *field = buf;
  const char *p;
  bool in_quote = false;
  bool in_array = false;
  bool has_escape = false;
  bool strip_quotes = false;
  for (p = buf; p < buf_end && *p != copy_params.line_delim; p++) {
    if (*p == copy_params.escape && p < buf_end - 1 && *(p+1) == copy_params.quote) {
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
    } else if (*p == copy_params.delimiter) {
      if (!in_quote && !in_array) {
        if (!has_escape && !strip_quotes) {
          std::string s(field, p - field);
          row.push_back(s);
        } else {
          char field_buf[p - field + 1];
          int j = 0, i = 0;
          if (strip_quotes)
            i++;
          for (; i < p - field; i++, j++) {
            if (has_escape && field[i] == copy_params.escape && field[i+1] == copy_params.quote) {
              field_buf[j] = copy_params.quote;
              i++;
          } else
            field_buf[j] = field[i];
          }
          if (strip_quotes)
            field_buf[j - 1] = '\0';
          else
            field_buf[j] = '\0';
          row.push_back(std::string(field_buf));
        }
        field = p + 1;
        has_escape = false;
        strip_quotes = false;
      } 
    }
  }
  if (*p == copy_params.line_delim) {
    row.push_back(std::string(field, p - field));
    return p;
  }
  for (; p < entire_buf_end && (in_quote || *p != copy_params.line_delim); p++) {
    if (*p == copy_params.escape && p < buf_end - 1 && *(p+1) == copy_params.quote) {
      p++;
      has_escape = true;
    } else if (copy_params.quoted && *p == copy_params.quote) {
      in_quote = !in_quote;
      if (in_quote)
        strip_quotes = true;
    } else if (is_array != nullptr && *p == copy_params.array_begin && is_array[row.size()]) {
      in_array = true;
    } else if (is_array != nullptr && *p == copy_params.array_end && is_array[row.size()]) {
      in_array = false;
    } else if (*p == copy_params.delimiter) {
      if (!in_quote && !in_array) {
        if (!has_escape) {
          std::string s(field, p - field);
          row.push_back(s);
        } else {
          char field_buf[p - field + 1];
          int j = 0, i = 0;
          if (strip_quotes)
            i++;
          for (; i < p - field; i++, j++) {
            if (has_escape && field[i] == copy_params.escape && field[i+1] == copy_params.quote) {
              field_buf[j] = copy_params.quote;
              i++;
          } else
            field_buf[j] = field[i];
          }
          if (strip_quotes)
            field_buf[j - 1] = '\0';
          else
            field_buf[j] = '\0';
          row.push_back(std::string(field_buf));
        }
        field = p + 1;
        has_escape = false;
        strip_quotes = false;
      }
    } 
  }
  if (*p == copy_params.line_delim) {
    row.push_back(std::string(field, p - field));
    return p;
  }
  /*
  @TODO(wei) do error handling
  */
  if (in_quote)
    LOG(ERROR) << "unmatched quote.";
  return p;
}

int8_t *
appendDatum(int8_t *buf, Datum d, const SQLTypeInfo &ti)
{
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

ArrayDatum
StringToArray(const std::string &s, const SQLTypeInfo &ti, const CopyParams &copy_params)
{
  SQLTypeInfo elem_ti = ti.get_elem_type();
  if (s[0] != copy_params.array_begin || s[s.size() - 1] != copy_params.array_end) {
    LOG(WARNING) << "Malformed array: " << s;
    return ArrayDatum(0, NULL, true);
  }
  std::vector<std::string> elem_strs;
  size_t last = 1;
  for (size_t i = s.find(copy_params.delimiter, 1); i != std::string::npos; i = s.find(copy_params.delimiter, last)) {
    elem_strs.push_back(s.substr(last, i - last));
    last = i + 1;
  }
  if (last + 1 < s.size()) {
    elem_strs.push_back(s.substr(last, s.size() - 1 - last));
  }
  if (!elem_ti.is_string()) {
    size_t len = elem_strs.size() * elem_ti.get_size();
    int8_t *buf = (int8_t*)malloc(len);
    int8_t *p = buf;
    for (auto &e : elem_strs) {
      Datum d = StringToDatum(e, elem_ti);
      p = appendDatum(p, d, elem_ti);
    }
    return ArrayDatum(len, buf, len == 0);
  }
  // must not be called for array of strings
  CHECK(false);
  return ArrayDatum(0, NULL, true);
}

void
parseStringArray(const std::string &s, const CopyParams &copy_params, std::vector<std::string> &string_vec)
{
  if (s[0] != copy_params.array_begin || s[s.size() - 1] != copy_params.array_end) {
    LOG(WARNING) << "Malformed array: " << s;
    return;
  }
  size_t last = 1;
  for (size_t i = s.find(copy_params.delimiter, 1); i != std::string::npos; i = s.find(copy_params.delimiter, last)) {
    if (i > last) // if not empty string - disallow empty strings for now
      string_vec.push_back(s.substr(last, i - last));
    last = i + 1;
  }
  if (s.size()-1 > last)  // if not empty string - disallow empty strings for now 
    string_vec.push_back(s.substr(last, s.size() - 1 - last));
}

void
addBinaryStringArray(const TDatum &datum, std::vector<std::string> &string_vec)
{
  const auto& arr = datum.val.arr_val;
  for (const auto& elem_datum : arr) {
    string_vec.push_back(elem_datum.val.str_val);
  }
}

Datum
TDatumToDatum(const TDatum &datum, SQLTypeInfo &ti)
{
	Datum d;
  const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
	switch (type) {
		case kBOOLEAN:
      d.boolval = datum.is_null ? NULL_BOOLEAN : datum.val.int_val;
			break;
		case kBIGINT:
			d.bigintval = datum.is_null ? NULL_BIGINT : datum.val.int_val;
			break;
		case kINT:
			d.intval = datum.is_null ? NULL_INT : datum.val.int_val;
			break;
		case kSMALLINT:
			d.smallintval = datum.is_null ? NULL_SMALLINT : datum.val.int_val;
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
      d.timeval = datum.is_null ? (sizeof(time_t) == 8 ? NULL_BIGINT : NULL_INT) : datum.val.int_val;
      break;
		default:
			throw std::runtime_error("Internal error: invalid type in StringToDatum.");
	}
	return d;
}

ArrayDatum
TDatumToArrayDatum(const TDatum &datum, const SQLTypeInfo &ti)
{
  SQLTypeInfo elem_ti = ti.get_elem_type();
  CHECK(!elem_ti.is_string());
  size_t len = datum.val.arr_val.size() * elem_ti.get_size();
  int8_t *buf = (int8_t*)malloc(len);
  int8_t *p = buf;
  for (auto &e : datum.val.arr_val) {
    p = appendDatum(p, TDatumToDatum(e, elem_ti), elem_ti);
  }
  return ArrayDatum(len, buf, len == 0);
}

static size_t
find_beginning(const char *buffer, size_t begin, size_t end, const CopyParams &copy_params)
{
  // @TODO(wei) line_delim is in quotes note supported
  if (begin == 0 || (begin > 0 && buffer[begin - 1] == copy_params.line_delim))
    return 0;
  size_t i;
  const char *buf = buffer + begin;
  for (i = 0; i < end - begin; i++)
    if (buf[i] == copy_params.line_delim)
      return i + 1;
  return i;
}

void
TypedImportBuffer::add_value(const ColumnDescriptor *cd, const std::string &val, const bool is_null, const CopyParams &copy_params)
{
  const auto type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType) : cd->columnType.get_type();
  switch (type) {
  case kBOOLEAN: {
    if (is_null) {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addBoolean(NULL_BOOLEAN);
    } else {
      SQLTypeInfo ti = cd->columnType;
      Datum d = StringToDatum(val, ti);
      addBoolean((int8_t)d.boolval);
    }
    break;
  }
  case kSMALLINT: {
    if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
      addSmallint(cd->columnType.is_decimal()
        ? std::atof(val.c_str()) * pow(10, cd->columnType.get_scale())
        : std::atoi(val.c_str()));
    } else {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addSmallint(NULL_SMALLINT);
    }
    break;
  }
  case kINT: {
    if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
      addInt(cd->columnType.is_decimal()
        ? std::atof(val.c_str()) * pow(10, cd->columnType.get_scale())
        : std::atoi(val.c_str()));
    } else {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addInt(NULL_INT);
    }
    break;
  }
  case kBIGINT: {
    if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
      addBigint(cd->columnType.is_decimal()
        ? std::strtod(val.c_str(), NULL) * pow(10, cd->columnType.get_scale())
        : std::atoll(val.c_str()));
    } else {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addBigint(NULL_BIGINT);
    }
    break;
  }
  case kFLOAT:
    if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
      addFloat((float)std::atof(val.c_str()));
    } else {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addFloat(NULL_FLOAT);
    }
    break;
  case kDOUBLE:
    if (!is_null && (isdigit(val[0]) || val[0] == '-')) {
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
    } else
      addString(val);
    break;
  }
  case kTIME:
  case kTIMESTAMP:
  case kDATE:
    if (!is_null && isdigit(val[0])) {
      SQLTypeInfo ti = cd->columnType;
      Datum d = StringToDatum(val, ti);
      addTime(d.timeval);
    } else {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addTime(cd->columnType.get_size() == 4 ? NULL_INT : NULL_BIGINT);
    }
    break;
  case kARRAY:
    if (!is_null) {
      if (IS_STRING(cd->columnType.get_subtype())) {
        std::vector<std::string> &string_vec = addStringArray();
        parseStringArray(val, copy_params, string_vec);
      } else {
        ArrayDatum d = StringToArray(val, cd->columnType, copy_params);
        addArray(d);
      }
    } else {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addArray(ArrayDatum(0, NULL, true));
    }
    break;
  default:
    CHECK(false);
  }
}

void
TypedImportBuffer::add_value(const ColumnDescriptor *cd, const TDatum &datum, const bool is_null)
{
  const auto type = cd->columnType.is_decimal() ? decimal_to_int_type(cd->columnType) : cd->columnType.get_type();
  switch (type) {
  case kBOOLEAN: {
    if (is_null) {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addBoolean(NULL_BOOLEAN);
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
      addSmallint(NULL_SMALLINT);
    }
    break;
  case kINT:
    if (!is_null) {
      addInt((int32_t)datum.val.int_val);
    } else {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addInt(NULL_INT);
    }
    break;
  case kBIGINT:
    if (!is_null) {
      addBigint(datum.val.int_val);
    } else {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addBigint(NULL_BIGINT);
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
      addTime(sizeof(time_t) == 4 ? NULL_INT : NULL_BIGINT);
    }
    break;
  case kARRAY:
    if (!is_null) {
      if (IS_STRING(cd->columnType.get_subtype())) {
        std::vector<std::string> &string_vec = addStringArray();
        addBinaryStringArray(datum, string_vec);
      } else {
        addArray(TDatumToArrayDatum(datum, cd->columnType));
      }
    } else {
      if (cd->columnType.get_notnull())
        throw std::runtime_error("NULL for column " + cd->columnName);
      addArray(ArrayDatum(0, NULL, true));
    }
    break;
  default:
    CHECK(false);
  }
}

static void
import_thread(int thread_id, Importer *importer, const char *buffer, size_t begin_pos, size_t end_pos, size_t total_size)
{
  size_t row_count = 0;
  int64_t total_get_row_time_us = 0;
  int64_t total_str_to_val_time_us = 0;
  auto load_ms = measure<>::execution([]() {});
  auto ms = measure<>::execution([&]() {
  const CopyParams &copy_params = importer->get_copy_params();
  const std::list<const ColumnDescriptor*> &col_descs = importer->get_column_descs();
  size_t begin = find_beginning(buffer, begin_pos, end_pos, copy_params);
  const char *thread_buf = buffer + begin_pos + begin;
  const char *thread_buf_end = buffer + end_pos;
  const char *buf_end = buffer + total_size;
  std::vector<std::unique_ptr<TypedImportBuffer>> &import_buffers = importer->get_import_buffers(thread_id);
  auto us = measure<std::chrono::microseconds>::execution([&]() {});
  for (const auto& p : import_buffers)
    p->clear();
  std::vector<std::string> row;
  for (const char *p = thread_buf; p < thread_buf_end; p++) {
    {
      decltype(row) empty;
      row.swap(empty);
    }
    if (debug_timing) {
      us = measure<std::chrono::microseconds>::execution([&]() {
        p = get_row(p, thread_buf_end, buf_end, copy_params, p == thread_buf, importer->get_is_array(), row);
      });
      total_get_row_time_us += us;
    } else 
      p = get_row(p, thread_buf_end, buf_end, copy_params, p == thread_buf, importer->get_is_array(), row);
    /*
    std::cout << "Row " << row_count << " : ";
    for (auto p : row)
      std::cout << p << ", ";
    std::cout << std::endl;
    */
    if (row.size() != col_descs.size()) {
    LOG(ERROR) << "Incorrect Row (expected " << col_descs.size() << " columns, has " << row.size() << "): ";
    for (auto p : row)
        std::cerr << p << ", ";
    std::cerr << std::endl;
      continue;
    }
    us = measure<std::chrono::microseconds>::execution([&]() {
    try {
      int col_idx = 0;
      for (const auto cd : col_descs) {
        bool is_null = (row[col_idx] == copy_params.null_str);
        if (!cd->columnType.is_string() && row[col_idx].empty())
          is_null = true;
        import_buffers[col_idx]->add_value(cd, row[col_idx], is_null, copy_params);
        ++col_idx;
      }
      row_count++;
    } catch (const std::exception &e) {
      LOG(WARNING) << "Input exception thrown: " << e.what() << ". Row discarded.";
    }
    });
    total_str_to_val_time_us += us;
  }
  if (row_count > 0) {
    load_ms = measure<>::execution([&]() {
    importer->load(import_buffers, row_count);
    });
  }
  });
  if (debug_timing && row_count > 0) {
    LOG(INFO) << "Thread" << std::this_thread::get_id() << ":" << row_count << " rows inserted in " << (double)ms/1000.0 << "sec, Insert Time: " << (double)load_ms/1000.0 << "sec, get_row: " << (double)total_get_row_time_us/1000000.0 << "sec, str_to_val: " << (double)total_str_to_val_time_us/1000000.0 << "sec" << std::endl;
  }
}

static size_t
find_end(const char *buffer, size_t size, const CopyParams &copy_params)
{
  int i;
  // @TODO(wei) line_delim is in quotes note supported
  for (i = size - 1; i >= 0 && buffer[i] != copy_params.line_delim; i--)
  ;

  if (i < 0)
    LOG(ERROR) << "No line delimiter in block.";
  return i + 1;
}

bool
Loader::load(const std::vector<std::unique_ptr<TypedImportBuffer>> &import_buffers, size_t row_count)
{
  Fragmenter_Namespace::InsertData ins_data(insert_data);
  ins_data.numRows = row_count;
  bool success = true;
  for (const auto& import_buff : import_buffers) {
    DataBlockPtr p;
    if (import_buff->getTypeInfo().is_number() ||
        import_buff->getTypeInfo().is_time() ||
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
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(insert_mutex));
    try {
      table_desc->fragmenter->insertData(ins_data);
    } catch (std::exception &e) {
      LOG(ERROR) << "Fragmenter Insert Exception: " << e.what();
      success = false;
    }
  }
  return success;
}

void
Loader::init()
{
  insert_data.databaseId = catalog.get_currentDB().dbId;
  insert_data.tableId = table_desc->tableId;
  for (auto cd : column_descs) {
    insert_data.columnIds.push_back(cd->columnId);
    if (cd->columnType.get_compression() == kENCODING_DICT) {
      CHECK(cd->columnType.is_string() || cd->columnType.is_string_array());
      const auto dd = catalog.getMetadataForDict(cd->columnType.get_comp_param());
      CHECK(dd);
      dict_map[cd->columnId] = dd->stringDict;
    }
  }
  insert_data.numRows = 0;
}

#define IMPORT_FILE_BUFFER_SIZE   100000000
#define MIN_FILE_BUFFER_SIZE      1000000

void
Importer::import()
{
  p_file = fopen(file_path.c_str(), "rb");
  (void)fseek(p_file,0,SEEK_END);
  file_size = ftell(p_file);
  if (copy_params.threads == 0)
    max_threads = sysconf(_SC_NPROCESSORS_CONF);
  else
    max_threads = copy_params.threads;
  buffer[0] = (char*)malloc(IMPORT_FILE_BUFFER_SIZE);
  if (max_threads > 1)
    buffer[1] = (char*)malloc(IMPORT_FILE_BUFFER_SIZE);
  for (int i = 0; i < max_threads; i++) {
    import_buffers_vec.push_back(std::vector<std::unique_ptr<TypedImportBuffer>>());
    for (const auto cd : loader.get_column_descs())
      import_buffers_vec[i].push_back(std::unique_ptr<TypedImportBuffer>(new TypedImportBuffer(cd, loader.get_string_dict(cd))));
  }
  size_t current_pos = 0;
  size_t end_pos;
  (void)fseek(p_file, current_pos, SEEK_SET);
  size_t size = fread((void*)buffer[which_buf], 1, IMPORT_FILE_BUFFER_SIZE, p_file);
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
  while (size > 0) {
    if (eof_reached)
      end_pos = size;
    else
      end_pos = find_end(buffer[which_buf], size, copy_params);
    if (size < IMPORT_FILE_BUFFER_SIZE) {
      max_threads = std::min(max_threads, (int)std::ceil((double)(end_pos - begin_pos)/MIN_FILE_BUFFER_SIZE));
    }
    if (max_threads == 1) {
      import_thread(0, this, buffer[which_buf], begin_pos, end_pos, end_pos);
      current_pos += end_pos;
      (void)fseek(p_file, current_pos, SEEK_SET);
      size = fread((void*)buffer[which_buf], 1, IMPORT_FILE_BUFFER_SIZE, p_file);
      if (size < IMPORT_FILE_BUFFER_SIZE && feof(p_file))
        eof_reached = true;
    } else {
      std::vector<std::thread> threads;
      for (int i = 0; i < max_threads; i++) {
        size_t begin = begin_pos + i * ((end_pos - begin_pos) / max_threads);
        size_t end = (i < max_threads - 1) ? begin_pos + (i + 1) * ((end_pos - begin_pos) / max_threads) : end_pos;
        threads.push_back(std::thread(import_thread, i, this, buffer[which_buf], begin, end, end_pos));
      }
      current_pos += end_pos;
      which_buf = (which_buf + 1) % 2;
      (void)fseek(p_file, current_pos, SEEK_SET);
      size = fread((void*)buffer[which_buf], 1, IMPORT_FILE_BUFFER_SIZE, p_file);
      if (size < IMPORT_FILE_BUFFER_SIZE && feof(p_file))
        eof_reached = true;
      for (auto& p : threads)
        p.join();
    }
    begin_pos = 0;
  }
  auto ms = measure<>::execution([&] () {
    if (!load_failed) {
      for (auto &p : import_buffers_vec[0]) {
        if (!p->stringDictCheckpoint()) {
          LOG(ERROR) << "Checkpointing Dictionary for Column " << p->getColumnDesc()->columnName << " failed.";
          load_failed = true;
          break;
        }
      }
      if (!load_failed)
        loader.checkpoint();
    }
  });
  if (debug_timing)
    LOG(INFO) << "Checkpointing took " << (double)ms/1000.0 << " Seconds." << std::endl;

  free(buffer[0]);
  buffer[0] = nullptr;
  free(buffer[1]);
  buffer[1] = nullptr;
  fclose(p_file); 
  p_file = nullptr;
}

} // Namespace Importer
