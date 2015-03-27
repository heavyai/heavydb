/*
 * @file Importer.cpp
 * @author Wei Hong <wei@mapd.com>
 * @brief Functions for Importer class
 */

#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <stdexcept>
#include <list>
#include <vector>
#include <thread>
#include <mutex>
#include "Importer.h"

namespace Importer_NS {

static std::mutex insert_mutex;

Importer::Importer(const Catalog_Namespace::Catalog &c, const TableDescriptor *t, const std::string &f, const CopyParams &p) : catalog(c), table_desc(t), file_path(f), copy_params(p) 
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
get_row(const char *buf, const char *buf_end, const CopyParams &copy_params, bool is_begin, std::vector<std::string> &row)
{
  const char *field = buf;
  const char *p;
  bool in_quote = false;
  for (p = buf; p < buf_end && !in_quote && *p != copy_params.line_delim; p++) {
    if (*p == copy_params.quote) {
      if (is_begin || *(p - 1) != copy_params.escape)
        in_quote = !in_quote;
    } else if (*p == copy_params.delimiter) {
      if (!in_quote) {
        std::string s(field, p - field);
        row.push_back(s);
        field = p + 1;
      }
    } 
  }
  row.push_back(std::string(field, p - field));
  /*
  std::cout << "Row: ";
  for (auto p : row)
    std::cout << p << ", ";
  std::cout << std::endl;
  */
  /*
  @TODO(wei) do error handling
  */
  if (in_quote)
    std::cerr << "unmatched quote." << std::endl;
  return p + 1;
}

static size_t
find_beginning(const char *buffer, size_t begin, size_t end, const CopyParams &copy_params)
{
  // @TODO(wei) what if line_delim is in quotes?
  if (begin > 0 && buffer[begin - 1] == copy_params.line_delim)
    return begin;
  if (begin == 0 && !copy_params.has_header)
    return 0;
  size_t i;
  const char *buf = buffer + begin;
  for (i = 0; i < end - begin; i++)
    if (buf[i] == copy_params.line_delim)
      return i + 1;
  return i;
}

static void
import_thread(const Importer &importer, const char *buffer, size_t begin_pos, size_t end_pos)
{
  const CopyParams &copy_params = importer.get_copy_params();
  const std::list<const ColumnDescriptor*> &col_descs = importer.get_column_descs();
  size_t begin = find_beginning(buffer, begin_pos, end_pos, copy_params);
  const char *thread_buf = buffer + begin_pos + begin;
  const char *thread_buf_end = buffer + end_pos;
  std::vector<std::unique_ptr<TypedImportBuffer>> import_buffers;
  for (const auto cd : importer.get_column_descs()) {
    import_buffers.push_back(std::unique_ptr<TypedImportBuffer>(new TypedImportBuffer(cd, importer.get_string_dict(cd))));
  }
  size_t row_count = 0;
  std::vector<std::string> row;
  for (const char *p = thread_buf; p < thread_buf_end; p++) {
    {
      decltype(row) empty;
      row.swap(empty);
    }
    p = get_row(p, thread_buf_end, copy_params, p == thread_buf, row);
    if (row.size() != col_descs.size()) {
      std::cerr << "incorrect rowsize." << std::endl;
      continue;
    }
    try {
      int col_idx = 0;
      for (const auto cd : col_descs) {
        bool is_null = (row[col_idx] == copy_params.null_str);
        if (!cd->columnType.is_string() && row[col_idx].empty())
          is_null = true;
        switch (cd->columnType.get_type()) {
        case kBOOLEAN: {
          if (is_null)
            import_buffers[col_idx]->addBoolean(NULL_BOOLEAN);
          else {
            SQLTypeInfo ti = cd->columnType;
            Datum d = StringToDatum(row[col_idx], ti);
            import_buffers[col_idx]->addBoolean((int8_t)d.boolval);
          }
          break;
        }
        case kSMALLINT:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addSmallint((int16_t)std::atoi(row[col_idx].c_str()));
          } else
            import_buffers[col_idx]->addSmallint(NULL_SMALLINT);
          break;
        case kINT:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addInt(std::atoi(row[col_idx].c_str()));
          } else
            import_buffers[col_idx]->addInt(NULL_INT);
          break;
        case kBIGINT:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addBigint(std::atoll(row[col_idx].c_str()));
          } else
            import_buffers[col_idx]->addBigint(NULL_BIGINT);
          break;
        case kFLOAT:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addFloat((float)std::atof(row[col_idx].c_str()));
          } else
            import_buffers[col_idx]->addFloat(NULL_FLOAT);
          break;
        case kDOUBLE:
          if (!is_null && (isdigit(row[col_idx][0]) || row[col_idx][0] == '-')) {
            import_buffers[col_idx]->addDouble(std::atof(row[col_idx].c_str()));
          } else
            import_buffers[col_idx]->addDouble(NULL_DOUBLE);
          break;
        case kTEXT:
        case kVARCHAR:
        case kCHAR: {
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
          } else
            import_buffers[col_idx]->addTime(sizeof(time_t) == 4 ? NULL_INT : NULL_BIGINT);
          break;
        default:
          CHECK(false);
        }
        ++col_idx;
      }
      row_count++;
    } catch (const std::exception&) {
      std::cerr << "input exception throw." << std::endl;
    }
  }
  if (row_count > 0) {
    importer.load(import_buffers, row_count);
    std::cout << row_count << " rows inserted." << std::endl;
  }
}

static size_t
find_end(const char *buffer, size_t size, const CopyParams &copy_params)
{
  bool in_quote = false;
  int i;
  for (i = size - 1; i >= 0 && (buffer[i] != copy_params.line_delim || in_quote); i--) {
    if (!in_quote && buffer[i] == copy_params.quote && (i == 0 || buffer[i - 1] != copy_params.escape))
      in_quote = true;
    else if (in_quote && buffer[i] == copy_params.quote && i > 0 && buffer[i - 1] != copy_params.escape)
      in_quote = false;
  }
  // @TODO(wei) check for error of unmatched quote
  return i + 1;
}

void
Importer::load(const std::vector<std::unique_ptr<TypedImportBuffer>> &import_buffers, size_t row_count) const
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
    table_desc->fragmenter->insertData(ins_data);
    catalog.get_dataMgr().checkpoint();
  }
  for (const auto& import_buff : import_buffers) {
    import_buff->flush();
  }
}

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
  p_file = fopen(file_path.c_str(), "r");
  (void)fseek(p_file,0,SEEK_END);
  file_size = ftell(p_file);
  max_threads = sysconf(_SC_NPROCESSORS_CONF);
  max_threads = 1;
  buffer[0] = (char*)malloc(IMPORT_FILE_BUFFER_SIZE);
  buffer[1] = (char*)malloc(IMPORT_FILE_BUFFER_SIZE);
  size_t current_pos = 0;
  size_t end_pos;
  (void)fseek(p_file, current_pos, SEEK_SET);
  size_t size = fread((void*)buffer[which_buf], 1, IMPORT_FILE_BUFFER_SIZE, p_file);
  bool eof_reached = false;
  while (size > 0) {
    if (eof_reached)
      end_pos = size;
    else
      end_pos = find_end(buffer[which_buf], size, copy_params);
    {
      /*
      std::vector<std::unique_ptr<std::thread>> gc_threads;
      std::vector<std::thread*> threads;
      for (int i = 0; i < max_threads; i++) {
        size_t begin = i * (end_pos / max_threads);
        size_t end = (i < max_threads - 1) ? (i + 1) * (end_pos / max_threads) : end_pos;
        std::thread *thd = new std::thread(import_thread, *this, buffer[which_buf], begin, end);
        threads.push_back(thd);
        gc_threads.push_back(std::unique_ptr<std::thread>(thd));
      }
      */
      import_thread(*this, buffer[which_buf], 0, end_pos);
      current_pos += end_pos;
      which_buf = (which_buf + 1) % 2;
      (void)fseek(p_file, current_pos, SEEK_SET);
      size = fread((void*)buffer[which_buf], 1, IMPORT_FILE_BUFFER_SIZE, p_file);
      if (size < IMPORT_FILE_BUFFER_SIZE && feof(p_file))
        eof_reached = true;
      /*
      for (auto p : threads)
        p->join();
      */
    }
  }

  free(buffer[0]);
  buffer[0] = nullptr;
  free(buffer[1]);
  buffer[1] = nullptr;
  fclose(p_file); 
  p_file = nullptr;
}

} // Namespace Importer
