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

namespace Importer_NS {

bool debug_timing = false;

static std::mutex insert_mutex;

Importer::Importer(const Catalog_Namespace::Catalog &c, const TableDescriptor *t, const std::string &f, const CopyParams &p) : catalog(c), table_desc(t), file_path(f), copy_params(p), load_failed(false)
{
  file_size = 0;
  max_threads = 0;
  p_file = nullptr;
  buffer[0] = nullptr;
  buffer[1] = nullptr;
  which_buf = 0;
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
get_row(const char *buf, const char *buf_end, const char *entire_buf_end, const CopyParams &copy_params, bool is_begin, std::vector<std::string> &row)
{
  const char *field = buf;
  const char *p;
  bool in_quote = false;
  bool has_escape = false;
  bool strip_quotes = false;
  for (p = buf; p < buf_end && *p != copy_params.line_delim; p++) {
    if (*p == copy_params.escape && p < buf_end - 1 && *(p+1) == copy_params.quote) {
      p++;
      has_escape = true;
    } else if (*p == copy_params.quote) {
      in_quote = !in_quote;
      if (in_quote)
        strip_quotes = true;
    } else if (*p == copy_params.delimiter) {
      if (!in_quote) {
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
    } else if (*p == copy_params.quote) {
      in_quote = !in_quote;
      if (in_quote)
        strip_quotes = true;
    } else if (*p == copy_params.delimiter) {
      if (!in_quote) {
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
  for (const auto& p : import_buffers)
    p->flush();
  std::vector<std::string> row;
  for (const char *p = thread_buf; p < thread_buf_end; p++) {
    {
      decltype(row) empty;
      row.swap(empty);
    }
    auto us = measure<std::chrono::microseconds>::execution([&]() {
    p = get_row(p, thread_buf_end, buf_end, copy_params, p == thread_buf, row);
    });
    total_get_row_time_us += us;
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
        switch (cd->columnType.get_type()) {
        case kBOOLEAN: {
          if (is_null) {
            if (cd->columnType.get_notnull())
              throw std::runtime_error("NULL for column " + cd->columnName);
            import_buffers[col_idx]->addBoolean(NULL_BOOLEAN);
          } else {
            SQLTypeInfo ti = cd->columnType;
            Datum d = StringToDatum(row[col_idx], ti);
            import_buffers[col_idx]->addBoolean((int8_t)d.boolval);
          }
          break;
        }
        case kSMALLINT:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addSmallint((int16_t)std::atoi(row[col_idx].c_str()));
          } else {
            if (cd->columnType.get_notnull())
              throw std::runtime_error("NULL for column " + cd->columnName);
            import_buffers[col_idx]->addSmallint(NULL_SMALLINT);
          }
          break;
        case kINT:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addInt(std::atoi(row[col_idx].c_str()));
          } else {
            if (cd->columnType.get_notnull())
              throw std::runtime_error("NULL for column " + cd->columnName);
            import_buffers[col_idx]->addInt(NULL_INT);
          }
          break;
        case kBIGINT:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addBigint(std::atoll(row[col_idx].c_str()));
          } else {
            if (cd->columnType.get_notnull())
              throw std::runtime_error("NULL for column " + cd->columnName);
            import_buffers[col_idx]->addBigint(NULL_BIGINT);
          }
          break;
        case kFLOAT:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addFloat((float)std::atof(row[col_idx].c_str()));
          } else {
            if (cd->columnType.get_notnull())
              throw std::runtime_error("NULL for column " + cd->columnName);
            import_buffers[col_idx]->addFloat(NULL_FLOAT);
          }
          break;
        case kDOUBLE:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addDouble(std::atof(row[col_idx].c_str()));
          } else {
            if (cd->columnType.get_notnull())
              throw std::runtime_error("NULL for column " + cd->columnName);
            import_buffers[col_idx]->addDouble(NULL_DOUBLE);
          }
          break;
        case kTEXT:
        case kVARCHAR:
        case kCHAR: {
          // @TODO(wei) for now, use empty string for nulls
          if (is_null) {
            if (cd->columnType.get_notnull())
              throw std::runtime_error("NULL for column " + cd->columnName);
            import_buffers[col_idx]->addString(std::string());
          } else
            import_buffers[col_idx]->addString(row[col_idx]);
          break;
        }
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          if (!is_null && isdigit(row[col_idx][0])) {
            SQLTypeInfo ti = cd->columnType;
            Datum d = StringToDatum(row[col_idx], ti);
            import_buffers[col_idx]->addTime(d.timeval);
          } else {
            if (cd->columnType.get_notnull())
              throw std::runtime_error("NULL for column " + cd->columnName);
            import_buffers[col_idx]->addTime(sizeof(time_t) == 4 ? NULL_INT : NULL_BIGINT);
          }
          break;
        default:
          CHECK(false);
        }
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

void
Importer::load(const std::vector<std::unique_ptr<TypedImportBuffer>> &import_buffers, size_t row_count)
{
  Fragmenter_Namespace::InsertData ins_data(insert_data);
  ins_data.numRows = row_count;
  for (const auto& import_buff : import_buffers) {
    DataBlockPtr p;
    if (import_buff->getTypeInfo().is_number() ||
        import_buff->getTypeInfo().is_time() ||
        import_buff->getTypeInfo().get_type() == kBOOLEAN) {
      p.numbersPtr = import_buff->getAsBytes();
    } else {
      CHECK(import_buff->getTypeInfo().is_string());
      auto string_payload_ptr = import_buff->getStringBuffer();
      if (import_buff->getTypeInfo().get_compression() == kENCODING_NONE) {
        p.stringsPtr = string_payload_ptr;
      } else {
        CHECK_EQ(kENCODING_DICT, import_buff->getTypeInfo().get_compression());
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(insert_mutex));
        for (const auto& str : *string_payload_ptr) {
          import_buff->addDictEncodedString(str);
        }
        p.numbersPtr = import_buff->getStringDictBuffer();
      }
    }
    ins_data.data.push_back(p);
  }
  {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(insert_mutex));
    try {
      table_desc->fragmenter->insertData(ins_data);
    } catch (std::exception &e) {
      LOG(ERROR) << "Fragmenter Insert Exception: " << e.what();
      set_load_failed(true);
    }
  }
}

#define IMPORT_FILE_BUFFER_SIZE   100000000
#define MIN_FILE_BUFFER_SIZE      1000000

void
Importer::import()
{
  column_descs = catalog.getAllColumnMetadataForTable(table_desc->tableId);
  insert_data.databaseId = catalog.get_currentDB().dbId;
  insert_data.tableId = table_desc->tableId;
  for (auto cd : column_descs) {
    insert_data.columnIds.push_back(cd->columnId);
    if (cd->columnType.is_string() && cd->columnType.get_compression() == kENCODING_DICT) {
      const auto dd = catalog.getMetadataForDict(cd->columnType.get_comp_param());
      CHECK(dd);
      dict_map[cd->columnId] = std::unique_ptr<StringDictionary>(new StringDictionary(dd->dictFolderPath));
    }
  }
  insert_data.numRows = 0;
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
    for (const auto cd : column_descs)
      import_buffers_vec[i].push_back(std::unique_ptr<TypedImportBuffer>(new TypedImportBuffer(cd, get_string_dict(cd))));
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
    if (!load_failed)
      catalog.get_dataMgr().checkpoint();
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
