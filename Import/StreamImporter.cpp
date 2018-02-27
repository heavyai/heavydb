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

/**
 * @file    StreamImporter.cpp
 * @author  Michael <michael@mapd.com>
 * @brief   Based on StreamInsert code but using binary columnar format for inserting a stream of rows
 * with optional transformations from stdin to a MapD table.
 *
 * Copyright (c) 2017 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cstring>
#include <string>
#include <iostream>
#include <iterator>
#include <boost/regex.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>

#include "../Shared/sqltypes.h"
#include "RowToColumnLoader.h"

#include <thread>
#include <chrono>

#include <boost/program_options.hpp>

#define MAX_FIELD_LEN 20000

bool print_error_data = false;
bool print_transformation = false;

// reads copy_params.delimiter delimited rows from std::cin and load them to
// table_name in batches of size copy_params.batch_size until EOF
void stream_insert(RowToColumnLoader& row_loader,
                   const std::map<std::string, std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>>&
                       transformations,
                   const Importer_NS::CopyParams& copy_params,
                   const bool remove_quotes) {
  std::ios_base::sync_with_stdio(false);
  std::istream_iterator<char> eos;
  std::cin >> std::noskipws;
  std::istream_iterator<char> iit(std::cin);

  char field[MAX_FIELD_LEN];
  size_t field_i = 0;

  int nrows = 0;
  int nskipped = 0;
  bool backEscape = false;

  auto row_desc = row_loader.get_row_descriptor();

  const std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>* xforms[row_desc.size()];
  for (size_t i = 0; i < row_desc.size(); i++) {
    auto it = transformations.find(row_desc[i].col_name);
    if (it != transformations.end())
      xforms[i] = &(it->second);
    else
      xforms[i] = nullptr;
  }

  std::vector<TStringValue> row;  // used to store each row as we move through the stream

  int read_rows = 0;
  while (iit != eos) {
    // construct a row
    while (iit != eos) {
      if (*iit == copy_params.delimiter || *iit == copy_params.line_delim) {
        bool end_of_field = (*iit == copy_params.delimiter);
        bool end_of_row;
        if (end_of_field)
          end_of_row = false;
        else {
          end_of_row = (row_desc[row.size()].col_type.type != TDatumType::STR) || (row.size() == row_desc.size() - 1);
          if (!end_of_row) {
            size_t l = copy_params.null_str.size();
            if (field_i >= l && strncmp(field + field_i - l, copy_params.null_str.c_str(), l) == 0) {
              end_of_row = true;
            }
          }
        }
        if (!end_of_field && !end_of_row) {
          // not enough columns yet and it is a string column
          // treat the line delimiter as part of the string
          field[field_i++] = *iit;
        } else {
          field[field_i] = '\0';
          field_i = 0;
          TStringValue ts;
          ts.str_val = std::string(field);
          ts.is_null = (ts.str_val.empty() || ts.str_val == copy_params.null_str);
          auto xform = row.size() < row_desc.size() ? xforms[row.size()] : nullptr;
          if (!ts.is_null && xform != nullptr) {
            if (print_transformation)
              std::cout << "\ntransforming\n" << ts.str_val << "\nto\n";
            ts.str_val = boost::regex_replace(ts.str_val, *xform->first, *xform->second);
            if (ts.str_val.empty())
              ts.is_null = true;
            if (print_transformation)
              std::cout << ts.str_val << std::endl;
          }

          row.push_back(ts);  // add column value to row
          if (end_of_row || (row.size() > row_desc.size())) {
            break;  // found row
          }
        }
      } else {
        if (*iit == '\\') {
          backEscape = true;
        } else if (backEscape || !remove_quotes || *iit != '\"') {
          field[field_i++] = *iit;
          backEscape = false;
        }
        // else if unescaped double-quote, continue without adding the
        // character to the field string.
      }
      if (field_i >= MAX_FIELD_LEN) {
        field[MAX_FIELD_LEN - 1] = '\0';
        std::cerr << "String too long for buffer." << std::endl;
        if (print_error_data)
          std::cerr << field << std::endl;
        field_i = 0;
        break;
      }
      ++iit;
    }
    if (row.size() == row_desc.size()) {
      // add the new data in the column format
      bool record_loaded = row_loader.convert_string_to_column(row, copy_params);
      read_rows++;
      if (!record_loaded) {
        // record could not be parsed correctly conside rit skipped
        ++nskipped;
      }
      row.clear();
      if (read_rows % copy_params.batch_size == 0) {
        row_loader.do_load(nrows, nskipped, copy_params);
      }
    } else {
      ++nskipped;
      if (print_error_data) {
        std::cerr << "Incorrect number of columns for row: ";
        std::cerr << row_loader.print_row_with_delim(row, copy_params) << std::endl;
      }
      if (row.size() > row_desc.size()) {
        // skip to the next line delimiter
        while (*iit != copy_params.line_delim)
          ++iit;
      }
      row.clear();
    }
    ++iit;
  }
  // load remaining rows if any
  if (read_rows % copy_params.batch_size != 0) {
    LOG(INFO) << " read_rows " << read_rows;
    row_loader.do_load(nrows, nskipped, copy_params);
  }
}

int main(int argc, char** argv) {
  std::string server_host("localhost");  // default to localhost
  int port = 9091;                       // default port number
  std::string table_name;
  std::string db_name;
  std::string user_name;
  std::string passwd;
  std::string delim_str(","), nulls("\\N"), line_delim_str("\n"), quoted("false");
  size_t batch_size = 10000;
  size_t retry_count = 10;
  size_t retry_wait = 5;
  bool remove_quotes = false;
  std::vector<std::string> xforms;
  std::map<std::string, std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>> transformations;

  google::InitGoogleLogging(argv[0]);

  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ");
  desc.add_options()("table", po::value<std::string>(&table_name)->required(), "Table Name");
  desc.add_options()("database", po::value<std::string>(&db_name)->required(), "Database Name");
  desc.add_options()("user,u", po::value<std::string>(&user_name)->required(), "User Name");
  desc.add_options()("passwd,p", po::value<std::string>(&passwd)->required(), "User Password");
  desc.add_options()("host", po::value<std::string>(&server_host)->default_value(server_host), "MapD Server Hostname");
  desc.add_options()("port", po::value<int>(&port)->default_value(port), "MapD Server Port Number");
  desc.add_options()("delim", po::value<std::string>(&delim_str)->default_value(delim_str), "Field delimiter");
  desc.add_options()("null", po::value<std::string>(&nulls), "NULL string");
  desc.add_options()("line", po::value<std::string>(&line_delim_str), "Line delimiter");
  desc.add_options()("quoted",
                     po::value<std::string>(&quoted),
                     "Whether the source contains quoted fields (true/false, default false)");
  desc.add_options()("batch", po::value<size_t>(&batch_size)->default_value(batch_size), "Insert batch size");
  desc.add_options()(
      "retry_count", po::value<size_t>(&retry_count)->default_value(retry_count), "Number of time to retry an insert");
  desc.add_options()(
      "retry_wait", po::value<size_t>(&retry_wait)->default_value(retry_wait), "wait in secs between retries");
  desc.add_options()(
      "transform,t", po::value<std::vector<std::string>>(&xforms)->multitoken(), "Column Transformations");
  desc.add_options()("print_error", "Print Error Rows");
  desc.add_options()("print_transform", "Print Transformations");

  po::positional_options_description positionalOptions;
  positionalOptions.add("table", 1);
  positionalOptions.add("database", 1);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
    if (vm.count("help")) {
      std::cout
          << "Usage: <table name> <database name> {-u|--user} <user> {-p|--passwd} <password> [{--host} "
             "<hostname>][--port <port number>][--delim <delimiter>][--null <null string>][--line <line "
             "delimiter>][--batch <batch size>][{-t|--transform} transformation [--quoted <true|false>] "
             "...][--retry_count <num_of_retries>] [--retry_wait <wait in secs>][--print_error][--print_transform]\n\n";
      std::cout << desc << std::endl;
      return 0;
    }
    if (vm.count("print_error"))
      print_error_data = true;
    if (vm.count("print_transform"))
      print_transformation = true;

    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  char delim = delim_str[0];
  if (delim == '\\') {
    if (delim_str.size() < 2 || (delim_str[1] != 'x' && delim_str[1] != 't' && delim_str[1] != 'n')) {
      std::cerr << "Incorrect delimiter string: " << delim_str << std::endl;
      return 1;
    }
    if (delim_str[1] == 't')
      delim = '\t';
    else if (delim_str[1] == 'n')
      delim = '\n';
    else {
      std::string d(delim_str);
      d[0] = '0';
      delim = (char)std::stoi(d, nullptr, 16);
    }
  }
  if (isprint(delim))
    std::cout << "Field Delimiter: " << delim << std::endl;
  else if (delim == '\t')
    std::cout << "Field Delimiter: "
              << "\\t" << std::endl;
  else if (delim == '\n')
    std::cout << "Field Delimiter: "
              << "\\n"
              << std::endl;
  else
    std::cout << "Field Delimiter: \\x" << std::hex << (int)delim << std::endl;
  char line_delim = line_delim_str[0];
  if (line_delim == '\\') {
    if (line_delim_str.size() < 2 ||
        (line_delim_str[1] != 'x' && line_delim_str[1] != 't' && line_delim_str[1] != 'n')) {
      std::cerr << "Incorrect delimiter string: " << line_delim_str << std::endl;
      return 1;
    }
    if (line_delim_str[1] == 't')
      line_delim = '\t';
    else if (line_delim_str[1] == 'n')
      line_delim = '\n';
    else {
      std::string d(line_delim_str);
      d[0] = '0';
      line_delim = (char)std::stoi(d, nullptr, 16);
    }
  }
  if (isprint(line_delim))
    std::cout << "Line Delimiter: " << line_delim << std::endl;
  else if (line_delim == '\t')
    std::cout << "Line Delimiter: "
              << "\\t" << std::endl;
  else if (line_delim == '\n')
    std::cout << "Line Delimiter: "
              << "\\n"
              << std::endl;
  else
    std::cout << "Line Delimiter: \\x" << std::hex << (int)line_delim << std::endl;
  std::cout << "Null String: " << nulls << std::endl;
  std::cout << "Insert Batch Size: " << std::dec << batch_size << std::endl;

  if (quoted == "true")
    remove_quotes = true;

  for (auto& t : xforms) {
    auto n = t.find_first_of(':');
    if (n == std::string::npos) {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    std::string col_name = t.substr(0, n);
    if (t.size() < n + 3 || t[n + 1] != 's' || t[n + 2] != '/') {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    auto n1 = n + 3;
    auto n2 = t.find_first_of('/', n1);
    if (n2 == std::string::npos) {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    std::string regex_str = t.substr(n1, n2 - n1);
    n1 = n2 + 1;
    n2 = t.find_first_of('/', n1);
    if (n2 == std::string::npos) {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    std::string fmt_str = t.substr(n1, n2 - n1);
    std::cout << "transform " << col_name << ": s/" << regex_str << "/" << fmt_str << "/" << std::endl;
    transformations[col_name] = std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>(
        std::unique_ptr<boost::regex>(new boost::regex(regex_str)),
        std::unique_ptr<std::string>(new std::string(fmt_str)));
  }

  Importer_NS::CopyParams copy_params(delim, nulls, line_delim, batch_size, retry_count, retry_wait);
  RowToColumnLoader row_loader(server_host, port, db_name, user_name, passwd, table_name);
  stream_insert(row_loader, transformations, copy_params, remove_quotes);
  return 0;
}
