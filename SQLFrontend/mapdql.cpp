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
 * @file    mapdql.cpp
 * @author  Wei Hong <wei@map-d.com>
 * @brief   MapD SQL Client Tool
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <termios.h>
#include <signal.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <rapidjson/document.h>

#include "../Fragmenter/InsertOrderFragmenter.h"
#include "Shared/checked_alloc.h"
#include "gen-cpp/MapD.h"
#include "MapDServer.h"
#include "MapDRelease.h"
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/THttpClient.h>

#include "linenoise.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

const std::string MapDQLRelease(MAPD_RELEASE);

using boost::shared_ptr;

void completion(const char* buf, linenoiseCompletions* lc) {
  if (toupper(buf[0]) == 'S') {
    linenoiseAddCompletion(lc, "SELECT ");
  }
}

#define INVALID_SESSION_ID ""

struct ClientContext {
  std::string user_name;
  std::string passwd;
  std::string db_name;
  std::string server_host;
  int port;
  bool http;
  TTransport& transport;
  MapDClient& client;
  TSessionId session;
  TQueryResult query_return;
  std::vector<std::string> names_return;
  std::vector<TDBInfo> dbinfos_return;
  TExecuteMode::type execution_mode;
  std::string version;
  std::string memory_usage;
  TMemorySummary memory_summary;
  TTableDetails table_details;
  std::string table_name;
  std::string file_name;
  TCopyParams copy_params;

  ClientContext(TTransport& t, MapDClient& c)
      : transport(t), client(c), session(INVALID_SESSION_ID), execution_mode(TExecuteMode::GPU) {}
};

enum ThriftService {
  kCONNECT,
  kDISCONNECT,
  kSQL,
  kGET_TABLES,
  kGET_DATABASES,
  kGET_USERS,
  kSET_EXECUTION_MODE,
  kGET_VERSION,
  kGET_MEMORY_GPU,
  kGET_MEMORY_CPU,
  kGET_MEMORY_SUMMARY,
  kGET_TABLE_DETAILS,
  kCLEAR_MEMORY_GPU,
  kCLEAR_MEMORY_CPU,
  kIMPORT_GEO_TABLE,
  kINTERRUPT
};

namespace {

bool thrift_with_retry(ThriftService which_service, ClientContext& context, const char* arg) {
  try {
    switch (which_service) {
      case kCONNECT:
        context.client.connect(context.session, context.user_name, context.passwd, context.db_name);
        break;
      case kDISCONNECT:
        context.client.disconnect(context.session);
        break;
      case kINTERRUPT:
        context.client.interrupt(context.session);
        break;
      case kSQL:
        context.client.sql_execute(context.query_return, context.session, arg, true, "", -1);
        break;
      case kGET_TABLES:
        context.client.get_tables(context.names_return, context.session);
        break;
      case kGET_DATABASES:
        context.client.get_databases(context.dbinfos_return, context.session);
        break;
      case kGET_USERS:
        context.client.get_users(context.names_return, context.session);
        break;
      case kSET_EXECUTION_MODE:
        context.client.set_execution_mode(context.session, context.execution_mode);
        break;
      case kGET_VERSION:
        context.client.get_version(context.version);
        break;
      case kGET_MEMORY_GPU:
        context.client.get_memory_gpu(context.memory_usage, context.session);
        break;
      case kGET_MEMORY_CPU:
        context.client.get_memory_cpu(context.memory_usage, context.session);
        break;
      case kGET_MEMORY_SUMMARY:
        context.client.get_memory_summary(context.memory_summary, context.session);
        break;
      case kGET_TABLE_DETAILS:
        context.client.get_table_details(context.table_details, context.session, arg);
        break;
      case kCLEAR_MEMORY_GPU:
        context.client.clear_gpu_memory(context.session);
        break;
      case kCLEAR_MEMORY_CPU:
        context.client.clear_cpu_memory(context.session);
        break;
      case kIMPORT_GEO_TABLE:
        context.client.import_geo_table(
            context.session, context.table_name, context.file_name, context.copy_params, TRowDescriptor());
        break;
    }
  } catch (TMapDException& e) {
    std::cerr << e.error_msg << std::endl;
    return false;
  } catch (TException& te) {
    try {
      context.transport.open();
      if (which_service == kDISCONNECT)
        return false;
      if (which_service != kCONNECT) {
        if (!thrift_with_retry(kCONNECT, context, nullptr))
          return false;
      }
      return thrift_with_retry(which_service, context, arg);
    } catch (TException& te1) {
      std::cerr << "Thrift error: " << te1.what() << std::endl;
      return false;
    }
  }
  return true;
}

#define LOAD_PATCH_SIZE 10000

void copy_table(char* filepath, char* table, ClientContext& context) {
  if (context.session == INVALID_SESSION_ID) {
    std::cerr << "Not connected to any databases." << std::endl;
    return;
  }
  if (!boost::filesystem::exists(filepath)) {
    std::cerr << "File does not exist." << std::endl;
    return;
  }
  if (!thrift_with_retry(kGET_TABLE_DETAILS, context, table)) {
    std::cerr << "Cannot connect to table." << std::endl;
    return;
  }
  const TRowDescriptor& row_desc = context.table_details.row_desc;
  std::ifstream infile(filepath);
  std::string line;
  const char* delim = ",";
  int l = strlen(filepath);
  if (l >= 4 && strcmp(filepath + l - 4, ".tsv") == 0) {
    delim = "\t";
  }
  std::vector<TStringRow> input_rows;
  TStringRow row;
  boost::char_separator<char> sep{delim, "", boost::keep_empty_tokens};
  try {
    while (std::getline(infile, line)) {
      row.cols.clear();
      boost::tokenizer<boost::char_separator<char>> tok{line, sep};
      for (const auto& s : tok) {
        TStringValue ts;
        ts.str_val = s;
        ts.is_null = s.empty();
        row.cols.push_back(ts);
      }
      /*
      std::cout << "Row: ";
      for (const auto &p : row.cols) {
        std::cout << p.str_val << ", ";
      }
      std::cout << std::endl;
       */
      if (row.cols.size() != row_desc.size()) {
        std::cerr << "Incorrect number of columns: (" << row.cols.size() << " vs " << row_desc.size() << ") " << line
                  << std::endl;
        continue;
      }
      input_rows.push_back(row);
      if (input_rows.size() >= LOAD_PATCH_SIZE) {
        try {
          context.client.load_table(context.session, table, input_rows);
        } catch (TMapDException& e) {
          std::cerr << e.error_msg << std::endl;
        }
        input_rows.clear();
      }
    }
    if (input_rows.size() > 0)
      context.client.load_table(context.session, table, input_rows);
  } catch (TMapDException& e) {
    std::cerr << e.error_msg << std::endl;
  } catch (TException& te) {
    std::cerr << "Thrift error: " << te.what() << std::endl;
  }
}

void detect_table(char* file_name, TCopyParams& copy_params, ClientContext& context) {
  if (context.session == INVALID_SESSION_ID) {
    std::cerr << "Not connected to any databases." << std::endl;
    return;
  }

  TDetectResult _return;

  try {
    context.client.detect_column_types(_return, context.session, file_name, copy_params);
  } catch (TMapDException& e) {
    std::cerr << e.error_msg << std::endl;
  } catch (TException& te) {
    std::cerr << "Thrift error in detect_table: " << te.what() << std::endl;
  }
}

size_t get_row_count(const TQueryResult& query_result) {
  CHECK(!query_result.row_set.row_desc.empty());
  if (query_result.row_set.columns.empty()) {
    return 0;
  }
  CHECK_EQ(query_result.row_set.columns.size(), query_result.row_set.row_desc.size());
  return query_result.row_set.columns.front().nulls.size();
}

// runs a simple single integer value query and returns that single int value returned
int run_query(ClientContext& context, std::string query) {
  thrift_with_retry(kSQL, context, query.c_str());
  CHECK(get_row_count(context.query_return));
  // std::cerr << "return value is " <<  context.query_return.row_set.columns[0].data.int_col[0];
  return context.query_return.row_set.columns[0].data.int_col[0];
  // col.data.int_col[row_idx];
  // return 4;
}

int get_optimal_size(ClientContext& context, std::string table_name, std::string col_name, int col_type) {
  switch (col_type) {
    case TDatumType::STR: {
      int strings = run_query(context, "select count(distinct " + col_name + ") from " + table_name + ";");
      if (strings < pow(2, 8)) {
        return 8;
      } else {
        if (strings < pow(2, 16)) {
          return 16;
        } else {
          return 32;
        }
      }
    }
    case TDatumType::TIME: {
      return 32;
    }
    case TDatumType::DATE:
    case TDatumType::TIMESTAMP: {
      return run_query(context,
                       "select case when (extract( epoch from mn)  > -2147483648 and extract (epoch from mx) < "
                       "2147483647) then 32 else 0 end from (select min(" +
                           col_name + ") mn, max(" + col_name + ") mx from " + table_name + " );");
    }
    case TDatumType::BIGINT: {
      return run_query(context,
                       "select  case when (mn > -128 and mx < 127) then 8 else case when (mn > -32768 and mx < 32767) "
                       "then 16 else case when (mn  > -2147483648 and mx < 2147483647) then 32 else 0 end end end from "
                       "(select min(" +
                           col_name + ") mn, max(" + col_name + ") mx from " + table_name + " );");
    }
    case TDatumType::INT: {
      return run_query(context,
                       "select  case when (mn > -128 and mx < 127) then 8 else case when (mn > -32768 and mx < 32767) "
                       "then 16 else 0 end end from "
                       "(select min(" +
                           col_name + ") mn, max(" + col_name + ") mx from " + table_name + " );");
    }
  }
  return 0;
}

namespace {

std::vector<std::string> unserialize_key_metainfo(const std::string key_metainfo) {
  std::vector<std::string> keys_with_spec;
  rapidjson::Document document;
  document.Parse(key_metainfo.c_str());
  CHECK(!document.HasParseError());
  CHECK(document.IsArray());
  for (auto it = document.Begin(); it != document.End(); ++it) {
    const auto& key_with_spec_json = *it;
    CHECK(key_with_spec_json.IsObject());
    const std::string type = key_with_spec_json["type"].GetString();
    const std::string name = key_with_spec_json["name"].GetString();
    auto key_with_spec = type + " (" + name + ")";
    if (type == "SHARED DICTIONARY") {
      key_with_spec += " REFERENCES ";
      const std::string foreign_table = key_with_spec_json["foreign_table"].GetString();
      const std::string foreign_column = key_with_spec_json["foreign_column"].GetString();
      key_with_spec += foreign_table + "(" + foreign_column + ")";
    } else {
      CHECK(type == "SHARD KEY");
    }
    keys_with_spec.push_back(key_with_spec);
  }
  return keys_with_spec;
}

}  // namespace

void process_backslash_commands(char* command, ClientContext& context) {
  switch (command[1]) {
    case 'h':
      std::cout << "\\u List all users.\n";
      std::cout << "\\l List all databases.\n";
      std::cout << "\\t List all tables.\n";
      std::cout << "\\d <table> List all columns of table.\n";
      std::cout << "\\c <database> <user> <password>.\n";
      std::cout << "\\gpu Execute in GPU mode's.\n";
      std::cout << "\\cpu Execute in CPU mode's.\n";
      std::cout << "\\multiline Set multi-line command line mode.\n";
      std::cout << "\\singleline Set single-line command line mode.\n";
      std::cout << "\\historylen <number> Set history buffer size (default 100).\n";
      std::cout << "\\timing Print timing information.\n";
      std::cout << "\\notiming Do not print timing information.\n";
      std::cout << "\\memory_summary Print memory usage summary.\n";
      std::cout << "\\version Print MapD Server version.\n";
      std::cout << "\\copy <file path> <table> Copy data from file to table.\n";
      std::cout << "\\q Quit.\n";
      return;
    case 'd': {
      if (command[2] != ' ') {
        std::cerr << "Invalid \\d command usage.  Do \\d <table name>" << std::endl;
        return;
      }
      std::string table_name(command + 3);
      if (!thrift_with_retry(kGET_TABLE_DETAILS, context, command + 3)) {
        return;
      }
      const auto table_details = context.table_details;
      if (table_details.view_sql.empty()) {
        std::string temp_holder(" ");
        if (table_details.is_temporary) {
          temp_holder = " TEMPORARY ";
        }
        std::cout << "CREATE" + temp_holder + "TABLE " + table_name + " (\n";
      } else {
        std::cout << "CREATE VIEW " + table_name + " AS " + table_details.view_sql << "\n";
        std::cout << "\n"
                  << "View columns:"
                  << "\n\n";
      }
      std::string comma_or_blank("");
      for (TColumnType p : table_details.row_desc) {
        if (p.is_system) {
          continue;
        }
        std::string encoding;
        if (p.col_type.type == TDatumType::STR) {
          encoding =
              (p.col_type.encoding == 0 ? " ENCODING NONE" : " ENCODING " + thrift_to_encoding_name(p.col_type) + "(" +
                                                                 std::to_string(p.col_type.comp_param) + ")");

        } else {
          encoding = (p.col_type.encoding == 0 ? "" : " ENCODING " + thrift_to_encoding_name(p.col_type) + "(" +
                                                          std::to_string(p.col_type.comp_param) + ")");
        }
        std::cout << comma_or_blank << p.col_name << " " << thrift_to_name(p.col_type)
                  << (p.col_type.nullable ? "" : " NOT NULL") << encoding;
        comma_or_blank = ",\n";
      }
      if (table_details.view_sql.empty()) {
        const auto keys_with_spec = unserialize_key_metainfo(table_details.key_metainfo);
        for (const auto& key_with_spec : keys_with_spec) {
          std::cout << ",\n" << key_with_spec;
        }
        // push final ")\n";
        std::cout << ")\n";
        comma_or_blank = "";
        std::string frag = "";
        std::string page = "";
        std::string row = "";
        if (DEFAULT_FRAGMENT_ROWS != table_details.fragment_size) {
          frag = "FRAGMENT_SIZE = " + std::to_string(table_details.fragment_size);
          comma_or_blank = ", ";
        }
        if (table_details.shard_count) {
          frag += comma_or_blank + "SHARD_COUNT = " + std::to_string(table_details.shard_count);
          comma_or_blank = ", ";
        }
        if (DEFAULT_PAGE_SIZE != table_details.page_size) {
          page = comma_or_blank + "PAGE_SIZE = " + std::to_string(table_details.page_size);
          comma_or_blank = ", ";
        }
        if (DEFAULT_MAX_ROWS != table_details.max_rows) {
          row = comma_or_blank + "MAX_ROWS = " + std::to_string(table_details.max_rows);
        }
        std::string with = frag + page + row;
        if (with.length() > 0) {
          std::cout << "WITH (" << with << ")\n";
        }
      } else {
        std::cout << "\n";
      }
      return;
    }
    case 'o': {
      if (command[2] != ' ') {
        std::cerr << "Invalid \\o command usage.  Do \\o <table name>" << std::endl;
        return;
      }
      std::string table_name(command + 3);
      if (!thrift_with_retry(kGET_TABLE_DETAILS, context, command + 3)) {
        return;
      }
      const auto table_details = context.table_details;
      if (table_details.view_sql.empty()) {
        std::cout << "CREATE TABLE " + table_name + " (\n";
      } else {
        std::cerr << "Can't optimize a view, only the underlying tables\n";
        return;
      }
      std::string comma_or_blank("");
      for (TColumnType p : table_details.row_desc) {
        std::string encoding;
        if (p.col_type.type == TDatumType::STR) {
          encoding =
              (p.col_type.encoding == 0
                   ? " ENCODING NONE"
                   : " ENCODING " + thrift_to_encoding_name(p.col_type) + "(" +
                         std::to_string(get_optimal_size(context, table_name, p.col_name, p.col_type.type)) + ")");

        } else {
          int opt_size = get_optimal_size(context, table_name, p.col_name, p.col_type.type);
          encoding = (opt_size == 0 ? "" : " ENCODING FIXED(" + std::to_string(opt_size) + ")");
        }
        std::cout << comma_or_blank << p.col_name << " " << thrift_to_name(p.col_type)
                  << (p.col_type.nullable ? "" : " NOT NULL") << encoding;
        comma_or_blank = ",\n";
      }
      // push final "\n";
      if (table_details.view_sql.empty()) {
        std::cout << ")\n";
        comma_or_blank = "";
        std::string frag = "";
        std::string page = "";
        std::string row = "";
        if (DEFAULT_FRAGMENT_ROWS != table_details.fragment_size) {
          frag = " FRAGMENT_SIZE = " + std::to_string(table_details.fragment_size);
          comma_or_blank = ",";
        }
        if (DEFAULT_PAGE_SIZE != table_details.page_size) {
          page = comma_or_blank + " PAGE_SIZE = " + std::to_string(table_details.page_size);
          comma_or_blank = ",";
        }
        if (DEFAULT_MAX_ROWS != table_details.max_rows) {
          row = comma_or_blank + " MAX_ROWS = " + std::to_string(table_details.max_rows);
        }
        std::string with = frag + page + row;
        if (with.length() > 0) {
          std::cout << "WITH (" << with << ")\n";
        }
      } else {
        std::cout << "\n";
      }
      return;
    }
    case 't': {
      if (thrift_with_retry(kGET_TABLES, context, nullptr))
        for (auto p : context.names_return)
          if (thrift_with_retry(kGET_TABLE_DETAILS, context, p.c_str()))
            if (context.table_details.view_sql.empty()) {
              std::cout << p << std::endl;
            }

      return;
    }
    case 'v': {
      if (thrift_with_retry(kGET_TABLES, context, nullptr))
        for (auto p : context.names_return)
          if (thrift_with_retry(kGET_TABLE_DETAILS, context, p.c_str()))
            if (!context.table_details.view_sql.empty()) {
              std::cout << p << std::endl;
            }

      return;
    }
    case 'c': {
      if (command[2] != ' ') {
        std::cerr << "Invalid \\c command usage.  Do \\c <database> <user> <password>" << std::endl;
        return;
      }
      char* db = strtok(command + 3, " ");
      char* user = strtok(NULL, " ");
      char* passwd = strtok(NULL, " ");
      if (db == NULL || user == NULL || passwd == NULL) {
        std::cerr << "Invalid \\c command usage.  Do \\c <database> <user> <password>" << std::endl;
        return;
      }
      if (context.session != INVALID_SESSION_ID) {
        if (thrift_with_retry(kDISCONNECT, context, nullptr))
          std::cout << "Disconnected from database " << context.db_name << std::endl;
      }
      context.db_name = db;
      context.user_name = user;
      context.passwd = passwd;
      if (thrift_with_retry(kCONNECT, context, nullptr)) {
        std::cout << "User " << context.user_name << " connected to database " << context.db_name << std::endl;
      }
    } break;
    case 'u': {
      if (thrift_with_retry(kGET_USERS, context, nullptr))
        for (auto p : context.names_return)
          std::cout << p << std::endl;
      return;
    }
    case 'l': {
      if (thrift_with_retry(kGET_DATABASES, context, nullptr)) {
        std::cout << "Database | Owner" << std::endl;
        for (auto p : context.dbinfos_return)
          std::cout << p.db_name << " | " << p.db_owner << std::endl;
      }
      return;
    }
    default:
      std::cerr << "Invalid backslash command: " << command << std::endl;
  }
}

std::string scalar_datum_to_string(const TDatum& datum, const TTypeInfo& type_info) {
  if (datum.is_null) {
    return "NULL";
  }
  switch (type_info.type) {
    case TDatumType::SMALLINT:
    case TDatumType::INT:
    case TDatumType::BIGINT:
      return std::to_string(datum.val.int_val);
    case TDatumType::DECIMAL:
    case TDatumType::FLOAT:
    case TDatumType::DOUBLE:
      return std::to_string(datum.val.real_val);
    case TDatumType::STR:
      return datum.val.str_val;
    case TDatumType::TIME: {
      time_t t = datum.val.int_val;
      std::tm tm_struct;
      gmtime_r(&t, &tm_struct);
      char buf[9];
      strftime(buf, 9, "%T", &tm_struct);
      return buf;
    }
    case TDatumType::TIMESTAMP: {
      time_t t = datum.val.int_val;
      std::tm tm_struct;
      gmtime_r(&t, &tm_struct);
      char buf[20];
      strftime(buf, 20, "%F %T", &tm_struct);
      return buf;
    }
    case TDatumType::DATE: {
      time_t t = datum.val.int_val;
      std::tm tm_struct;
      gmtime_r(&t, &tm_struct);
      char buf[11];
      strftime(buf, 11, "%F", &tm_struct);
      return buf;
    }
    case TDatumType::BOOL:
      return (datum.val.int_val ? "true" : "false");
    case TDatumType::INTERVAL_DAY_TIME:
      return std::to_string(datum.val.int_val) + " ms (day-time interval)";
    case TDatumType::INTERVAL_YEAR_MONTH:
      return std::to_string(datum.val.int_val) + " month(s) (year-month interval)";
    default:
      return "Unknown column type.\n";
  }
}

std::string datum_to_string(const TDatum& datum, const TTypeInfo& type_info) {
  if (datum.is_null) {
    return "NULL";
  }
  if (type_info.is_array) {
    std::vector<std::string> elem_strs;
    elem_strs.reserve(datum.val.arr_val.size());
    for (const auto& elem_datum : datum.val.arr_val) {
      TTypeInfo elem_type_info{type_info};
      elem_type_info.is_array = false;
      elem_strs.push_back(scalar_datum_to_string(elem_datum, elem_type_info));
    }
    return "{" + boost::algorithm::join(elem_strs, ", ") + "}";
  }
  return scalar_datum_to_string(datum, type_info);
}

TDatum columnar_val_to_datum(const TColumn& col, const size_t row_idx, const TTypeInfo& col_type) {
  TDatum datum;
  if (col_type.is_array) {
    auto elem_type = col_type;
    elem_type.is_array = false;
    datum.is_null = false;
    CHECK_LT(row_idx, col.data.arr_col.size());
    const auto& arr_col = col.data.arr_col[row_idx];
    for (size_t elem_idx = 0; elem_idx < arr_col.nulls.size(); ++elem_idx) {
      TColumn elem_col;
      elem_col.data = arr_col.data;
      elem_col.nulls = arr_col.nulls;
      datum.val.arr_val.push_back(columnar_val_to_datum(elem_col, elem_idx, elem_type));
    }
    return datum;
  }
  datum.is_null = col.nulls[row_idx];
  switch (col_type.type) {
    case TDatumType::SMALLINT:
    case TDatumType::INT:
    case TDatumType::BIGINT:
    case TDatumType::TIME:
    case TDatumType::TIMESTAMP:
    case TDatumType::DATE:
    case TDatumType::BOOL:
    case TDatumType::INTERVAL_DAY_TIME:
    case TDatumType::INTERVAL_YEAR_MONTH: {
      datum.val.int_val = col.data.int_col[row_idx];
      break;
    }
    case TDatumType::DECIMAL:
    case TDatumType::FLOAT:
    case TDatumType::DOUBLE: {
      datum.val.real_val = col.data.real_col[row_idx];
      break;
    }
    case TDatumType::STR: {
      datum.val.str_val = col.data.str_col[row_idx];
      break;
    }
    default:
      CHECK(false);
  }
  return datum;
}

// based on http://www.gnu.org/software/libc/manual/html_node/getpass.html
std::string mapd_getpass() {
  struct termios origterm, tmpterm;
  int nread;

  size_t MAX_PASSWORD_LENGTH{256};
  char* password = (char*)checked_malloc(MAX_PASSWORD_LENGTH);

  tcgetattr(STDIN_FILENO, &origterm);
  tmpterm = origterm;
  tmpterm.c_lflag &= ~ECHO;
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &tmpterm);

  std::cout << "Password: ";
  nread = getline(&password, &MAX_PASSWORD_LENGTH, stdin);
  std::cout << std::endl;

  tcsetattr(STDIN_FILENO, TCSAFLUSH, &origterm);

  return std::string(password, nread - 1);
}

enum Action { INITIALIZE, TURN_ON, TURN_OFF, INTERRUPT };

bool backchannel(int action, ClientContext* cc) {
  enum State { UNINITIALIZED, INITIALIZED, INTERRUPTIBLE };
  static int state = UNINITIALIZED;
  static ClientContext* context{nullptr};

  if (action == INITIALIZE) {
    CHECK(cc);
    context = cc;
    state = INITIALIZED;
    return false;
  }
  if (state == INITIALIZED && action == TURN_ON) {
    state = INTERRUPTIBLE;
    return false;
  }
  if (state == INTERRUPTIBLE && action == TURN_OFF) {
    state = INITIALIZED;
    return false;
  }
  if (state == INTERRUPTIBLE && action == INTERRUPT) {
    CHECK(context);
    shared_ptr<TTransport> transport2;
    shared_ptr<TProtocol> protocol2;
    shared_ptr<TTransport> socket2;
    if (context->http) {
      transport2 = shared_ptr<TTransport>(new THttpClient(context->server_host, context->port, "/"));
      protocol2 = shared_ptr<TProtocol>(new TJSONProtocol(transport2));
    } else {
      socket2 = shared_ptr<TTransport>(new TSocket(context->server_host, context->port));
      transport2 = shared_ptr<TTransport>(new TBufferedTransport(socket2));
      protocol2 = shared_ptr<TProtocol>(new TBinaryProtocol(transport2));
    }
    MapDClient c2(protocol2);
    ClientContext context2(*transport2, c2);

    context2.db_name = context->db_name;
    context2.user_name = context->user_name;
    context2.passwd = context->passwd;

    context2.session = INVALID_SESSION_ID;

    transport2->open();
    if (!transport2->isOpen()) {
      std::cout << "Unable to send interrupt to the server.\n" << std::flush;
      return false;
    }

    (void)thrift_with_retry(kCONNECT, context2, nullptr);

    std::cout << "Asking server to interrupt query.\n" << std::flush;
    (void)thrift_with_retry(kINTERRUPT, context2, nullptr);

    if (context2.session != INVALID_SESSION_ID) {
      (void)thrift_with_retry(kDISCONNECT, context2, nullptr);
    }
    transport2->close();
    state = INITIALIZED;
    return true;
  }
  return false;
}

void mapdql_signal_handler(int signal_number) {
  std::cout << "\nInterrupt signal (" << signal_number << ") received.\n" << std::flush;

  if (backchannel(INTERRUPT, nullptr)) {
    return;
  }

  // terminate program
  exit(signal_number);
}

void register_signal_handler() {
  signal(SIGTERM, mapdql_signal_handler);
  signal(SIGKILL, mapdql_signal_handler);
  signal(SIGINT, mapdql_signal_handler);
}

}  // namespace

int main(int argc, char** argv) {
  std::string server_host{"localhost"};
  int port = 9091;
  std::string delimiter("|");
  bool print_header = true;
  bool print_connection = true;
  bool print_timing = false;
  bool http = false;
  char* line;
  TQueryResult _return;
  std::string db_name{"mapd"};
  std::string user_name{"mapd"};
  std::string passwd;

  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ");
  desc.add_options()("version,v", "Print mapdql version number");
  desc.add_options()("no-header,n", "Do not print query result header");
  desc.add_options()("timing,t",
                     po::bool_switch(&print_timing)->default_value(print_timing)->implicit_value(true),
                     "Print timing information");
  desc.add_options()(
      "delimiter,d", po::value<std::string>(&delimiter)->default_value(delimiter), "Field delimiter in row output");
  desc.add_options()("db", po::value<std::string>(&db_name)->default_value(db_name), "Database name");
  desc.add_options()("user,u", po::value<std::string>(&user_name)->default_value(user_name), "User name");
  desc.add_options()("passwd,p", po::value<std::string>(&passwd), "Password");
  desc.add_options()("server,s", po::value<std::string>(&server_host)->default_value(server_host), "Server hostname");
  desc.add_options()("port", po::value<int>(&port)->default_value(port), "Port number");
  desc.add_options()("http", po::bool_switch(&http)->default_value(http)->implicit_value(true), "Use HTTP transport");
  desc.add_options()("quiet,q", "Do not print result headers or connection strings ");

  po::variables_map vm;
  po::positional_options_description positionalOptions;
  positionalOptions.add("db", 1);

  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
    if (vm.count("help")) {
      std::cout << "Usage: mapdql [<database>] [{--user|-u} <user>] [{--passwd|-p} <password>] [--port <port number>] "
                   "[{-s|--server} <server host>] [--http] [{--no-header|-n}] [{--quiet|-q}] [{--delimiter|-d}]\n\n";
      std::cout << desc << std::endl;
      return 0;
    }
    if (vm.count("version")) {
      std::cout << "MapDQL Version: " << MapDQLRelease << std::endl;
      return 0;
    }
    if (vm.count("quiet")) {
      print_header = false;
      print_connection = false;
    }
    if (vm.count("no-header"))
      print_header = false;
    if (vm.count("db") && !vm.count("user")) {
      std::cerr << "Must specify a user name to access database " << db_name << std::endl;
      return 1;
    }

    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  if (!vm.count("passwd")) {
    passwd = mapd_getpass();
  }

  shared_ptr<TTransport> transport;
  shared_ptr<TProtocol> protocol;
  shared_ptr<TTransport> socket;
  if (http) {
    transport = shared_ptr<TTransport>(new THttpClient(server_host, port, "/"));
    protocol = shared_ptr<TProtocol>(new TJSONProtocol(transport));
  } else {
    socket = shared_ptr<TTransport>(new TSocket(server_host, port));
    transport = shared_ptr<TTransport>(new TBufferedTransport(socket));
    protocol = shared_ptr<TProtocol>(new TBinaryProtocol(transport));
  }
  MapDClient c(protocol);
  ClientContext context(*transport, c);

  context.db_name = db_name;
  context.user_name = user_name;
  context.passwd = passwd;
  context.server_host = server_host;
  context.port = port;
  context.http = http;

  context.session = INVALID_SESSION_ID;

  transport->open();

  if (context.db_name.empty()) {
    std::cout
        << "Not connected to any database.  Only \\u and \\l commands are allowed in this state.  See \\h for help."
        << std::endl;
  } else {
    if (thrift_with_retry(kCONNECT, context, nullptr))
      if (print_connection) {
        std::cout << "User " << context.user_name << " connected to database " << context.db_name << std::endl;
      }
  }

  register_signal_handler();
  (void)backchannel(INITIALIZE, &context);

  /* Set the completion callback. This will be called every time the
   * user uses the <tab> key. */
  linenoiseSetCompletionCallback(completion);

  /* Load history from file. The history file is just a plain text file
   * where entries are separated by newlines. */
  linenoiseHistoryLoad("mapdql_history.txt"); /* Load the history at startup */
  /* default to multi-line mode */
  linenoiseSetMultiLine(1);

  std::string current_line;
  std::string prompt("mapdql> ");

  /* Now this is the main loop of the typical linenoise-based application.
   * The call to linenoise() will block as long as the user types something
   * and presses enter.
   *
   * The typed string is returned as a malloc() allocated string by
   * linenoise, so the user needs to free() it. */
  while ((line = linenoise(prompt.c_str())) != NULL) {
    {
      TQueryResult empty;
      swap(_return, empty);
    }
    /* Do something with the string. */
    if (line[0] != '\0' && line[0] != '\\') {
      // printf("echo: '%s'\n", line);
      if (context.session == INVALID_SESSION_ID) {
        std::cerr << "Not connected to any MapD databases." << std::endl;
        continue;
      }
      current_line.append(" ").append(std::string(line));
      boost::algorithm::trim(current_line);
      if (current_line.back() == ';') {
        linenoiseHistoryAdd(current_line.c_str());  /* Add to the history. */
        linenoiseHistorySave("mapdql_history.txt"); /* Save the history on disk. */
        std::string query(current_line);
        current_line.clear();
        prompt.assign("mapdql> ");
        (void)backchannel(TURN_ON, nullptr);
        if (thrift_with_retry(kSQL, context, query.c_str())) {
          (void)backchannel(TURN_OFF, nullptr);
          if (context.query_return.row_set.row_desc.empty()) {
            continue;
          }
          const size_t row_count{get_row_count(context.query_return)};
          if (!row_count) {
            static const std::string insert{"INSERT"};
            std::string verb(query, 0, insert.size());
            if (!boost::iequals(verb, insert)) {
              std::cout << "No rows returned." << std::endl;
            }
            if (print_timing) {
              std::cout << "Execution time: " << context.query_return.execution_time_ms << " ms,"
                        << " Total time: " << context.query_return.total_time_ms << " ms" << std::endl;
            }
            continue;
          }
          bool not_first = false;
          if (print_header) {
            for (auto p : context.query_return.row_set.row_desc) {
              if (not_first)
                std::cout << delimiter;
              else
                not_first = true;
              std::cout << p.col_name;
            }
            std::cout << std::endl;
          }
          for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
            const auto& col_desc = context.query_return.row_set.row_desc;
            for (size_t col_idx = 0; col_idx < col_desc.size(); ++col_idx) {
              if (col_idx) {
                std::cout << delimiter;
              }
              const auto& col_type = col_desc[col_idx].col_type;
              std::cout << datum_to_string(
                  columnar_val_to_datum(context.query_return.row_set.columns[col_idx], row_idx, col_type), col_type);
            }
            std::cout << std::endl;
          }
          if (print_timing) {
            std::cout << row_count << " rows returned." << std::endl;
            std::cout << "Execution time: " << context.query_return.execution_time_ms << " ms,"
                      << " Total time: " << context.query_return.total_time_ms << " ms" << std::endl;
          }
        } else {
          (void)backchannel(TURN_OFF, nullptr);
        }
      } else {
        // change the prommpt
        prompt.assign("..> ");
      }
    } else if (!strncmp(line, "\\interrupt", 10)) {
      (void)thrift_with_retry(kINTERRUPT, context, nullptr);
    } else if (!strncmp(line, "\\cpu", 4)) {
      context.execution_mode = TExecuteMode::CPU;
      (void)thrift_with_retry(kSET_EXECUTION_MODE, context, nullptr);
    } else if (!strncmp(line, "\\gpu", 4)) {
      context.execution_mode = TExecuteMode::GPU;
      (void)thrift_with_retry(kSET_EXECUTION_MODE, context, nullptr);
    } else if (!strncmp(line, "\\hybrid", 5)) {
      context.execution_mode = TExecuteMode::HYBRID;
      (void)thrift_with_retry(kSET_EXECUTION_MODE, context, nullptr);
    } else if (!strncmp(line, "\\version", 8)) {
      if (thrift_with_retry(kGET_VERSION, context, nullptr)) {
        std::cout << "MapD Server Version: " << context.version << std::endl;
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    } else if (!strncmp(line, "\\memory_gpu", 11)) {
      if (thrift_with_retry(kGET_MEMORY_GPU, context, nullptr)) {
        std::cout << "MapD Server GPU Detailed Memory Usage " << context.memory_usage << std::endl;
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    } else if (!strncmp(line, "\\memory_cpu", 11)) {
      if (thrift_with_retry(kGET_MEMORY_CPU, context, nullptr)) {
        std::cout << "MapD Server CPU Detailed Memory Usage " << context.memory_usage << std::endl;
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    } else if (!strncmp(line, "\\clear_gpu", 11)) {
      if (thrift_with_retry(kCLEAR_MEMORY_GPU, context, nullptr)) {
        std::cout << "MapD Server GPU memory Cleared " << std::endl;
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    } else if (!strncmp(line, "\\clear_cpu", 11)) {
      if (thrift_with_retry(kCLEAR_MEMORY_CPU, context, nullptr)) {
        std::cout << "MapD Server CPU memory Cleared " << std::endl;
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    } else if (!strncmp(line, "\\memory_summary", 11)) {
      if (thrift_with_retry(kGET_MEMORY_SUMMARY, context, nullptr)) {
        std::ostringstream tss;
        size_t mb = 1024 * 1024;
        tss << std::endl;
        tss << "CPU RAM IN BUFFER USE : " << std::fixed << std::setw(9) << std::setprecision(2)
            << ((float)context.memory_summary.cpu_memory_in_use / mb) << " MB" << std::endl;
        int gpuNum = 0;
        tss << "GPU VRAM USAGE (in MB's)" << std::endl;
        tss << "GPU     MAX    ALLOC    IN-USE     FREE" << std::endl;
        for (auto gpu : context.memory_summary.gpu_summary) {
          int64_t real_max = gpu.is_allocation_capped ? gpu.allocated : gpu.max;
          tss << std::setfill(' ') << std::setw(2) << gpuNum << std::setw(9) << std::setprecision(2)
              << ((float)gpu.max / mb) << std::setw(9) << std::setprecision(2) << ((float)gpu.allocated / mb)
              << (gpu.is_allocation_capped ? "*" : " ") << std::setw(9) << std::setprecision(2)
              << ((float)gpu.in_use / mb) << std::setw(9) << std::setprecision(2)
              << ((float)(real_max - gpu.in_use) / mb) << std::endl;
          gpuNum++;
        }
        std::cout << "MapD Server Memory Usage " << tss.str() << std::endl;
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    } else if (!strncmp(line, "\\copygeo", 8)) {
      context.file_name = strtok(line + 9, " ");
      context.table_name = strtok(nullptr, " ");
      (void)thrift_with_retry(kIMPORT_GEO_TABLE, context, nullptr);
    } else if (!strncmp(line, "\\copy", 5)) {
      char* filepath = strtok(line + 6, " ");
      char* table = strtok(NULL, " ");
      copy_table(filepath, table, context);
    } else if (!strncmp(line, "\\detect", 7)) {
      char* filepath = strtok(line + 8, " ");
      TCopyParams copy_params;
      copy_params.delimiter = delimiter;
      detect_table(filepath, copy_params, context);
    } else if (!strncmp(line, "\\historylen", 11)) {
      /* The "/historylen" command will change the history len. */
      int len = atoi(line + 11);
      linenoiseHistorySetMaxLen(len);
    } else if (!strncmp(line, "\\multiline", 10)) {
      linenoiseSetMultiLine(1);
    } else if (!strncmp(line, "\\singleline", 11)) {
      linenoiseSetMultiLine(0);
    } else if (!strncmp(line, "\\keycodes", 9)) {
      linenoisePrintKeyCodes();
    } else if (!strncmp(line, "\\timing", 7)) {
      print_timing = true;
    } else if (!strncmp(line, "\\notiming", 9)) {
      print_timing = false;
    } else if (line[0] == '\\' && line[1] == 'q')
      break;
    else if (line[0] == '\\') {
      process_backslash_commands(line, context);
    }
    linenoiseHistoryAdd(line);                  /* Add to the history. */
    linenoiseHistorySave("mapdql_history.txt"); /* Save the history on disk. */
    free(line);
  }

  if (context.session != INVALID_SESSION_ID) {
    if (thrift_with_retry(kDISCONNECT, context, nullptr)) {
      if (print_connection) {
        std::cout << "User " << context.user_name << " disconnected from database " << context.db_name << std::endl;
      }
    }
  }
  transport->close();

  return 0;
}
