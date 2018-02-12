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

#include <glog/logging.h>
#include <rapidjson/document.h>
#include <signal.h>
#include <termios.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/THttpClient.h>
#include <thrift/transport/TSocket.h>
#include "../Fragmenter/InsertOrderFragmenter.h"
#include "MapDRelease.h"
#include "MapDServer.h"
#include "Shared/checked_alloc.h"
#include "gen-cpp/MapD.h"

#include "linenoise.h"

#include "ClientContext.h"
#include "CommandFunctors.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

const std::string MapDQLRelease(MAPD_RELEASE);

using boost::shared_ptr;

namespace {

ClientContext* g_client_context_ptr{nullptr};

void completion(const char* buf, linenoiseCompletions* lc) {
  CHECK(g_client_context_ptr);
  thrift_with_retry(kGET_COMPLETION_HINTS, *g_client_context_ptr, buf);
  for (const auto& completion_hint : g_client_context_ptr->completion_hints) {
    for (const auto& hint_str : completion_hint.hints) {
      CHECK_LE(completion_hint.replaced.size(), strlen(buf));
      std::string partial_query(buf, buf + strlen(buf) - completion_hint.replaced.size());
      partial_query += hint_str;
      linenoiseAddCompletion(lc, partial_query.c_str());
    }
  }
}

// code from https://stackoverflow.com/questions/7053538/how-do-i-encode-a-string-to-base64-using-only-boost

std::string decode64(const std::string& val) {
  using namespace boost::archive::iterators;
  using It = transform_width<binary_from_base64<std::string::const_iterator>, 8, 6>;
  return boost::algorithm::trim_right_copy_if(std::string(It(std::begin(val)), It(std::end(val))),
                                              [](char c) { return c == '\0'; });
}

std::string encode64(const std::string& val) {
  using namespace boost::archive::iterators;
  using It = base64_from_binary<transform_width<std::string::const_iterator, 6, 8>>;
  auto tmp = std::string(It(std::begin(val)), It(std::end(val)));
  return tmp.append((3 - val.size() % 3) % 3, '=');
}

#define LOAD_PATCH_SIZE 10000

void copy_table(char const* filepath, char const* table, ClientContext& context) {
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


void get_table_epoch(ClientContext& context, const std::string& table_specifier) {
  if (table_specifier.size() == 0) {
    std::cerr << "get table epoch requires table to be specified by name or by db_id:table_id" << std::endl;
    return;
  }

  if (isdigit(table_specifier.at(0))) {
    std::vector<std::string> split_result;

    boost::split(split_result,
                 table_specifier,
                 boost::is_any_of(":"),
                 boost::token_compress_on);  // SplitVec == { "hello abc","ABC","aBc goodbye" }

    if (split_result.size() != 2) {
      std::cerr << "get table epoch does not contain db_id:table_id " << table_specifier << std::endl;
      return;
    }

    // validate db identifier is a number
    try {
      context.db_id = std::stoi(split_result[0]);
    } catch (std::exception& e) {
      std::cerr << "non numeric db number: " << table_specifier << std::endl;
      return;
    }
    // validate table identifier is a number
    try {
      context.table_id = std::stoi(split_result[1]);
    } catch (std::exception& e) {
      std::cerr << "non-numeric table number: " << table_specifier << std::endl;
      return;
    }

    if (thrift_with_retry(kGET_TABLE_EPOCH, context, nullptr)) {
      std::cout << "table epoch is " << context.epoch_value << std::endl;
    } else {
      std::cerr << "Cannot connect to MapD Server." << std::endl;
    }
  } else {
    // presume we have a table name
    // get the db_id and table_id from the table metadata
    if (thrift_with_retry(kGET_PHYSICAL_TABLES, context, nullptr)) {
      if (std::find(context.names_return.begin(), context.names_return.end(), table_specifier) ==
          context.names_return.end()) {
        std::cerr << "table " << table_specifier << " not found" << std::endl;
        return;
      }
    } else {
      std::cerr << "Cannot connect to MapD Server." << std::endl;
      return;
    }
    context.table_name = table_specifier;

    if (thrift_with_retry(kGET_TABLE_EPOCH_BY_NAME, context, nullptr)) {
      std::cout << "table epoch is " << context.epoch_value << std::endl;
    } else {
      std::cerr << "Cannot connect to MapD Server." << std::endl;
    }
  }
}

bool replace(std::string& str, const std::string& from, const std::string& to) {
  size_t start_pos = str.find(from);
  if (start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

void import_dashboard(ClientContext& context, std::string dash_details) {
  std::vector<std::string> split_result;

  boost::split(split_result, dash_details, boost::is_any_of(","), boost::token_compress_on);

  if (split_result.size() != 2) {
    std::cerr << "import_dashboard requires a dashboard name and a filename eg:dash1,/tmp/dash.out " << dash_details
              << std::endl;
    return;
  }

  context.view_name = split_result[0];
  std::string filename = split_result[1];
  // context.view_metadata = std::string("{\"table\":\"") + table_name + std::string("\",\"version\":\"v2\"}");
  context.view_state = "";
  context.view_metadata = "";
  // read file to get view state
  std::ifstream dashfile;
  std::string state;
  std::string old_name;
  dashfile.open(filename);
  if (dashfile.is_open()) {
    std::getline(dashfile, old_name);
    std::getline(dashfile, context.view_metadata);
    std::getline(dashfile, state);
    dashfile.close();
  } else {
    std::cout << "Could not open file " << filename << std::endl;
    return;
  }

  if (!replace(state,
               std::string("\"title\":\"" + old_name + "\","),
               std::string("\"title\":\"" + context.view_name + "\","))) {
    std::cout << "Failed to update title." << std::endl;
    return;
  }

  context.view_state = encode64(state);

  std::cout << "Importing dashboard " << context.view_name << " from file " << filename << std::endl;
  if (!thrift_with_retry(kIMPORT_DASHBOARD, context, nullptr)) {
    std::cout << "Failed to import dashboard." << std::endl;
  }
}

void export_dashboard(ClientContext& context, std::string dash_details) {
  std::vector<std::string> split_result;

  boost::split(split_result, dash_details, boost::is_any_of(","), boost::token_compress_on);

  if (split_result.size() != 2) {
    std::cerr << "export_dashboard requires a dashboard name and a filename eg:dash1,/tmp/dash.out " << dash_details
              << std::endl;
    return;
  }
  context.view_name = split_result[0];
  std::string filename = split_result[1];

  if (thrift_with_retry(kEXPORT_DASHBOARD, context, nullptr)) {
    std::cout << "Exporting dashboard " << context.view_name << " to file " << filename
              << std::endl;  // create file and dump string to it
    std::ofstream dashfile;
    dashfile.open(filename);
    if (dashfile.is_open()) {
      dashfile << context.view_name << std::endl;
      dashfile << context.view_return.view_metadata << std::endl;
      dashfile << decode64(context.view_return.view_state);
      dashfile.close();
    } else {
      std::cout << "Could not open file " << filename << std::endl;
    }
  } else {
    std::cout << "Failed to export dashboard." << std::endl;
  }
}

void set_table_epoch(ClientContext& context, const std::string& table_details) {
  if (table_details.size() == 0) {
    std::cerr << "set table epoch requires table and epoch to be specified by name epoch or by db_id:table_id:epoch"
              << std::endl;
    return;
  }
  if (isdigit(table_details.at(0))) {
    std::vector<std::string> split_result;

    boost::split(split_result,
                 table_details,
                 boost::is_any_of(":"),
                 boost::token_compress_on);  // SplitVec == { "hello abc","ABC","aBc goodbye" }

    if (split_result.size() != 3) {
      std::cerr << "Set table epoch does not contain db_id:table_id:epoch " << table_details << std::endl;
      return;
    }

    // validate db identifier is a number
    try {
      context.db_id = std::stoi(split_result[0]);
    } catch (std::exception& e) {
      std::cerr << "non numeric db number: " << table_details << std::endl;
      return;
    }

    // validate table identifier is a number
    try {
      context.table_id = std::stoi(split_result[1]);
    } catch (std::exception& e) {
      std::cerr << "non-numeric table number: " << table_details << std::endl;
      return;
    }

    // validate epoch value
    try {
      context.epoch_value = std::stoi(split_result[2]);
    } catch (std::exception& e) {
      std::cerr << "non-numeric epoch number: " << table_details << std::endl;
      return;
    }
    if (context.epoch_value < 0) {
      std::cerr << "Epoch value can not be negative: " << table_details << std::endl;
      return;
    }

    if (thrift_with_retry(kSET_TABLE_EPOCH, context, nullptr)) {
      std::cout << "table epoch set" << std::endl;
    } else {
      std::cout << "Cannot connect to MapD Server." << std::endl;
    }
  } else {
    std::vector<std::string> split_result;
    boost::split(split_result, table_details, boost::is_any_of(" "), boost::token_compress_on);

    if (split_result.size() < 2) {
      std::cerr << "Set table epoch does not contain table_name epoch " << std::endl;
      return;
    }

    if (thrift_with_retry(kGET_PHYSICAL_TABLES, context, nullptr)) {
      if (std::find(context.names_return.begin(), context.names_return.end(), split_result[0]) ==
          context.names_return.end()) {
        std::cerr << "table " << split_result[0] << " not found" << std::endl;
        return;
      }
    } else {
      std::cerr << "Cannot connect to MapD Server." << std::endl;
      return;
    }
    context.table_name = split_result[0];
    // validate epoch value
    try {
      context.epoch_value = std::stoi(split_result[1]);
    } catch (std::exception& e) {
      std::cerr << "non-numeric epoch number: " << table_details << std::endl;
      return;
    }
    if (context.epoch_value < 0) {
      std::cerr << "Epoch value can not be negative: " << table_details << std::endl;
      return;
    }
    if (thrift_with_retry(kSET_TABLE_EPOCH_BY_NAME, context, nullptr)) {
      std::cout << "table epoch set" << std::endl;
    } else {
      std::cout << "Cannot connect to MapD Server." << std::endl;
    }
  }
}

void process_backslash_commands(char* command, ClientContext& context) {
  // clang-format off
  auto resolution_status = CommandResolutionChain<>( command, "\\h", 1, HelpCmd<>( context ) )
  ( "\\d", 2, ListColumnsCmd<>( context ), "Usage: \\d <table>" )
  ( "\\o", 2, GetOptimizedSchemaCmd<>( context ), "Usage: \\o <table>" )
  ( "\\t", 1, ListTablesCmd<>( context ) )
  ( "\\v", 1, ListViewsCmd<>( context ) )
  ( "\\c", 4, ConnectToDBCmd<>( context ), "Usage: \\c <database> <user> <password>." )
  ( "\\u", 1, ListUsersCmd<>( context ) )
  ( "\\l", 1, ListDatabasesCmd<>( context ) )
  .is_resolved();

  if( !resolution_status ) {
      std::cerr << "Invalid backslash command: " << command << std::endl;
  }
  // clang-format on
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
    case TDatumType::DECIMAL: {
      std::ostringstream dout;
      dout << std::setprecision(type_info.precision) << datum.val.real_val;
      return dout.str();
    }
    case TDatumType::DOUBLE: {
      std::ostringstream dout;
      dout << std::setprecision(std::numeric_limits<double>::digits10 + 1) << datum.val.real_val;
      return dout.str();
    }
    case TDatumType::FLOAT: {
      std::ostringstream out;
      out << std::setprecision(std::numeric_limits<float>::digits10 + 1) << datum.val.real_val;
      return out.str();
    }
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

void print_memory_summary(ClientContext context, std::string memory_level) {
  std::ostringstream tss;
  std::vector<TNodeMemoryInfo> memory_info;
  std::string sub_system;
  std::string cur_host = "^";
  bool hasGPU = false;
  bool multiNode = context.cpu_memory.size() > 1;
  if (!memory_level.compare("gpu")) {
    memory_info = context.gpu_memory;
    sub_system = "GPU";
    hasGPU = true;
  } else {
    memory_info = context.cpu_memory;
    sub_system = "CPU";
  }

  tss << "MapD Server " << sub_system << " Memory Summary:" << std::endl;

  if (multiNode) {
    if (hasGPU) {
      tss << "        NODE[GPU]            MAX            USE      ALLOCATED           FREE" << std::endl;
    } else {
      tss << "        NODE            MAX            USE      ALLOCATED           FREE" << std::endl;
    }
  } else {
    if (hasGPU) {
      tss << "[GPU]            MAX            USE      ALLOCATED           FREE" << std::endl;
    } else {
      tss << "            MAX            USE      ALLOCATED           FREE" << std::endl;
    }
  }
  u_int16_t gpu_num = 0;
  for (auto& nodeIt : memory_info) {
    int MB = 1024 * 1024;
    u_int64_t page_count = 0;
    u_int64_t free_page_count = 0;
    u_int64_t used_page_count = 0;
    for (auto& segIt : nodeIt.node_memory_data) {
      page_count += segIt.num_pages;
      if (segIt.is_free) {
        free_page_count += segIt.num_pages;
      }
    }
    if (context.cpu_memory.size() > 1) {
    }
    used_page_count = page_count - free_page_count;
    if (cur_host.compare(nodeIt.host_name)) {
      gpu_num = 0;
      cur_host = nodeIt.host_name;
    } else {
      ++gpu_num;
    }
    if (multiNode) {
      tss << std::setfill(' ') << std::setw(12) << nodeIt.host_name;
      if (hasGPU) {
        tss << std::setfill(' ') << std::setw(3);
        tss << "[" << gpu_num << "]";
      } else {
      }
    } else {
      if (hasGPU) {
        tss << std::setfill(' ') << std::setw(3);
        tss << "[" << gpu_num << "]";
      } else {
      }
    }
    tss << std::fixed;
    tss << std::setprecision(2);
    tss << std::setfill(' ') << std::setw(12) << ((float)nodeIt.page_size * nodeIt.max_num_pages) / MB << " MB";
    tss << std::setfill(' ') << std::setw(12) << ((float)nodeIt.page_size * used_page_count) / MB << " MB";
    tss << std::setfill(' ') << std::setw(12) << ((float)nodeIt.page_size * page_count) / MB << " MB";
    tss << std::setfill(' ') << std::setw(12) << ((float)nodeIt.page_size * free_page_count) / MB << " MB";
    if (nodeIt.is_allocation_capped) {
      tss << " The allocation is capped!";
    }
    tss << std::endl;
  }
  std::cout << tss.str() << std::endl;
}

void print_memory_info(ClientContext context, std::string memory_level) {
  int MB = 1024 * 1024;
  std::ostringstream tss;
  std::vector<TNodeMemoryInfo> memory_info;
  std::string sub_system;
  std::string cur_host = "^";
  bool multiNode;
  int mgr_num = 0;
  if (!memory_level.compare("gpu")) {
    memory_info = context.gpu_memory;
    sub_system = "GPU";
    multiNode = context.gpu_memory.size() > 1;
  } else {
    memory_info = context.cpu_memory;
    sub_system = "CPU";
    multiNode = context.cpu_memory.size() > 1;
  }

  tss << "MapD Server Detailed " << sub_system << " Memory Usage:" << std::endl;
  for (auto& nodeIt : memory_info) {
    if (cur_host.compare(nodeIt.host_name)) {
      mgr_num = 0;
      cur_host = nodeIt.host_name;
      if (multiNode) {
        tss << "Node: " << nodeIt.host_name << std::endl;
      }
      tss << "Maximum Bytes for one page: " << nodeIt.page_size << " Bytes" << std::endl;
      tss << "Maximum Bytes for node: " << (nodeIt.max_num_pages * nodeIt.page_size) / MB << " MB" << std::endl;
      tss << "Memory allocated: " << (nodeIt.num_pages_allocated * nodeIt.page_size) / MB << " MB" << std::endl;
    } else {
      ++mgr_num;
    }
    cur_host = nodeIt.host_name;

    tss << sub_system << "[" << mgr_num << "]"
        << " Slab Information:" << std::endl;
    if (nodeIt.is_allocation_capped) {
      tss << "The allocation is capped!";
    }
    tss << "SLAB     ST_PAGE NUM_PAGE  TOUCH         CHUNK_KEY" << std::endl;
    for (auto segIt = nodeIt.node_memory_data.begin(); segIt != nodeIt.node_memory_data.end(); ++segIt) {
      tss << std::setfill(' ') << std::setw(4) << segIt->slab;
      tss << std::setfill(' ') << std::setw(12) << segIt->start_page;
      tss << std::setfill(' ') << std::setw(9) << segIt->num_pages;
      tss << std::setfill(' ') << std::setw(7) << segIt->touch;
      tss << std::setfill(' ') << std::setw(5);

      if (segIt->is_free) {
        tss << "FREE";
      } else {
        tss << "USED";
        tss << std::setfill(' ') << std::setw(5);
        for (auto& vecIt : segIt->chunk_key) {
          tss << vecIt << ",";
        }
      }

      tss << std::endl;
    }
    tss << "---------------------------------------------------------------" << std::endl;
  }
  std::cout << tss.str() << std::endl;
}

std::string print_gpu_specification(TGpuSpecification gpu_spec) {
  int Giga = 1024 * 1024 * 1024;
  int Kilo = 1024;
  std::ostringstream tss;
  tss << "Number of SM             :" << gpu_spec.num_sm << std::endl;
  tss << "Clock frequency          :" << gpu_spec.clock_frequency_kHz / Kilo << " MHz" << std::endl;
  tss << "Physical GPU Memory      :" << gpu_spec.memory / Giga << " GB" << std::endl;
  tss << "Compute capability       :" << gpu_spec.compute_capability_major << "." << gpu_spec.compute_capability_minor
      << std::endl;
  return tss.str();
}

std::string print_hardware_specification(THardwareInfo hw_spec) {
  std::ostringstream tss;
  if (hw_spec.host_name != "") {
    tss << "Host name                :" << hw_spec.host_name << std::endl;
  }
  tss << "Number of Physical GPUs  :" << hw_spec.num_gpu_hw << std::endl;
  tss << "Number of CPU core       :" << hw_spec.num_cpu_hw << std::endl;
  tss << "Number of GPUs allocated :" << hw_spec.num_gpu_allocated << std::endl;
  tss << "Start GPU                :" << hw_spec.start_gpu << std::endl;
  for (auto gpu_spec : hw_spec.gpu_info) {
    tss << "-------------------------------------------" << std::endl;
    tss << print_gpu_specification(gpu_spec);
  }
  tss << "-------------------------------------------" << std::endl;
  return tss.str();
}

void print_all_hardware_info(ClientContext context) {
  std::ostringstream tss;
  for (auto hw_info : context.cluster_hardware_info.hardware_info) {
    tss << "===========================================" << std::endl;
    tss << print_hardware_specification(hw_info) << std::endl;
  }
  tss << "===========================================" << std::endl;
  std::cout << tss.str();
}

void get_role(ClientContext context) {
  context.role_names.clear();
  context.userPrivateRole = false;
  if (thrift_with_retry(kGET_ROLE, context, context.privs_role_name.c_str())) {
    if (context.role_names.size() > 0) {
      std::cout << "Role " << context.privs_role_name << " exists." << std::endl;
    } else {
      std::cout << "Role " << context.privs_role_name << " does not exist." << std::endl;
    }
  } else {
    std::cout << "Cannot connect to MapD Server." << std::endl;
  }
}

void get_db_objects_for_role(ClientContext context) {
  context.role_names.clear();
  context.userPrivateRole = true;
  if (thrift_with_retry(kGET_ROLE, context, context.privs_role_name.c_str())) {
    if (context.role_names.size() == 0) {
      std::cout << "Role or user " << context.privs_role_name << " does not exist." << std::endl;
    } else {
      context.db_objects.clear();
      if (thrift_with_retry(kGET_OBJECTS_FOR_ROLE, context, context.privs_role_name.c_str())) {
        for (size_t i = 0; i < context.db_objects.size(); i++) {
          bool any_granted_privs = false;
          for (size_t j = 0; j < context.db_objects[i].privs.size(); j++) {
            if (context.db_objects[i].privs[j]) {
              any_granted_privs = true;
              break;
            }
          }
          if (!any_granted_privs) {
            continue;
          }
          std::cout << context.db_objects[i].objectName.c_str();
          switch (context.db_objects[i].objectType) {
            case (TDBObjectType::DatabaseDBObjectType): {
              std::cout << " (database)";
              break;
            }
            case (TDBObjectType::TableDBObjectType): {
              std::cout << " (table)";
              break;
            }
            default: { CHECK(false); }
          }
          std::cout << " privileges:";
          for (size_t j = 0; j < context.db_objects[i].privs.size(); j++) {
            if (context.db_objects[i].privs[j]) {
              switch (j) {
                case (0): {
                  std::cout << " select";
                  break;
                }
                case (1): {
                  std::cout << " insert";
                  break;
                }
                case (2): {
                  std::cout << " create";
                  break;
                }
                case (3): {
                  std::cout << " truncate";
                  break;
                }
                default: { CHECK(false); }
              }
            }
          }
          std::cout << std::endl;
        }
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    }
  } else {
    std::cout << "Cannot connect to MapD Server." << std::endl;
  }
}

void get_db_object_privs(ClientContext context) {
  context.role_names.clear();
  context.userPrivateRole = true;
  if (thrift_with_retry(kGET_ALL_ROLES, context, nullptr)) {
    for (size_t i = 0; i < context.role_names.size(); i++) {
      context.db_objects.clear();
      context.privs_role_name = context.role_names[i];
      if (thrift_with_retry(kGET_OBJECTS_FOR_ROLE, context, context.privs_role_name.c_str())) {
        bool print_role(true);
        for (size_t i = 0; i < context.db_objects.size(); i++) {
          if (boost::to_upper_copy<std::string>(context.privs_object_name)
                  .compare(boost::to_upper_copy<std::string>(context.db_objects[i].objectName))) {
            continue;
          }
          bool any_granted_privs = false;
          for (size_t j = 0; j < context.db_objects[i].privs.size(); j++) {
            if (context.db_objects[i].privs[j]) {
              any_granted_privs = true;
              break;
            }
          }
          if (!any_granted_privs) {
            continue;
          }
          if (print_role) {
            std::cout << "Role/User: " << context.privs_role_name.c_str() << std::endl;
            print_role = false;
          }
          std::cout << "           DB Object: " << context.db_objects[i].objectName.c_str();
          switch (context.db_objects[i].objectType) {
            case (TDBObjectType::DatabaseDBObjectType): {
              std::cout << " (database)";
              break;
            }
            case (TDBObjectType::TableDBObjectType): {
              std::cout << " (table)   ";
              break;
            }
            default: { CHECK(false); }
          }
          std::cout << " privileges:";
          for (size_t j = 0; j < context.db_objects[i].privs.size(); j++) {
            if (context.db_objects[i].privs[j]) {
              switch (j) {
                case (0): {
                  std::cout << " select";
                  break;
                }
                case (1): {
                  std::cout << " insert";
                  break;
                }
                case (2): {
                  std::cout << " create";
                  break;
                }
                case (3): {
                  std::cout << " truncate";
                  break;
                }
                default: { CHECK(false); }
              }
            }
          }
          std::cout << std::endl;
        }
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    }
  } else {
    std::cout << "Cannot connect to MapD Server." << std::endl;
  }
}

void set_license_key(ClientContext context, const std::string& token) {
  context.license_key = token;
  if (thrift_with_retry(kSET_LICENSE_KEY, context, nullptr)) {
    for (auto claims : context.license_info.claims) {
      std::vector<std::string> jwt;
      boost::split(jwt, claims, boost::is_any_of("."));
      if (jwt.size() > 1) {
        std::cout << decode64(jwt[1]) << std::endl;
      }
    }
  }
}

void get_license_claims(ClientContext context) {
  if (thrift_with_retry(kGET_LICENSE_CLAIMS, context, nullptr)) {
    for (auto claims : context.license_info.claims) {
      std::vector<std::string> jwt;
      boost::split(jwt, claims, boost::is_any_of("."));
      if (jwt.size() > 1) {
        std::cout << decode64(jwt[1]) << std::endl;
      }
    }
  } else {
    std::cout << "Cannot connect to MapD Server." << std::endl;
  }
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
  g_client_context_ptr = &context;

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
   * The typed string that is malloc() allocated by linenoise() is managed by a smart pointer
   * with a custom free() deleter; no need to free this memory explicitly. */

  while (true) {
    using LineType = std::remove_pointer<__decltype(linenoise(prompt.c_str()))>::type;
    using LineTypePtr = LineType*;

    std::unique_ptr<LineType, std::function<void(LineTypePtr)>> smart_line(linenoise(prompt.c_str()),
                                                                           [](LineTypePtr l) { free((l)); });
    if (smart_line == nullptr)
      break;
    LineType* line = smart_line.get();  // Alias to make the C stuff work

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
      std::string trimmed_line = std::string(line);
      boost::algorithm::trim(trimmed_line);
      current_line.append(" ").append(trimmed_line);
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
      continue;
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
        print_memory_info(context, "gpu");
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    } else if (!strncmp(line, "\\memory_cpu", 11)) {
      if (thrift_with_retry(kGET_MEMORY_CPU, context, nullptr)) {
        print_memory_info(context, "cpu");
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
        print_memory_summary(context, "cpu");
        print_memory_summary(context, "gpu");
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
    } else if (!strncmp(line, "\\hardware_info", 13)) {
      if (context.cluster_hardware_info.hardware_info.size() > 0 ||
          thrift_with_retry(kGET_HARDWARE_INFO, context, nullptr)) {
        // TODO(vraj): try not to abuse using short circuit here
        print_all_hardware_info(context);
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }

    } else if (!strncmp(line, "\\status", 8)) {
      if (thrift_with_retry(kGET_SERVER_STATUS, context, nullptr)) {
        time_t t = (time_t)context.cluster_status[0].start_time;
        std::tm* tm_ptr = gmtime(&t);
        char buf[12] = {0};
        strftime(buf, 11, "%F", tm_ptr);
        std::string server_version = context.cluster_status[0].version;

        std::cout << "The Server Version Number  : " << context.cluster_status[0].version << std::endl;
        std::cout << "The Server Start Time      : " << buf << " : " << tm_ptr->tm_hour << ":" << tm_ptr->tm_min << ":"
                  << tm_ptr->tm_sec << std::endl;
        std::cout << "The Server edition         : " << server_version << std::endl;

        if (context.cluster_status.size() > 1) {
          std::cout << "The Number of Leaves       : " << context.cluster_status.size() - 1 << std::endl;
          for (auto leaf = context.cluster_status.begin() + 1; leaf != context.cluster_status.end(); ++leaf) {
            t = (time_t)leaf->start_time;
            buf[11] = 0;
            std::tm* tm_ptr = gmtime(&t);
            strftime(buf, 11, "%F", tm_ptr);
            std::cout << "--------------------------------------------------" << std::endl;
            std::cout << "Name of Leaf               : " << leaf->host_name << std::endl;
            if (server_version.compare(leaf->version) != 0) {
              std::cout << "The Leaf Version Number   : " << leaf->version << std::endl;
              std::cerr << "Version number mismatch!" << std::endl;
            }
            std::cout << "The Leaf Start Time        : " << buf << " : " << tm_ptr->tm_hour << ":" << tm_ptr->tm_min
                      << ":" << tm_ptr->tm_sec << std::endl;
          }
        }
      } else {
        std::cout << "Cannot connect to MapD Server." << std::endl;
      }
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
    } else if (!strncmp(line, "\\role_check", 11)) {
      std::string temp_line(line);
      boost::algorithm::trim(temp_line);
      if (temp_line.size() > 11) {
        context.privs_role_name.clear();
        context.privs_role_name = strtok(line + 12, " ");
        get_role(context);
      } else {
        std::cout << "Command role_check failed because parameter role name is missing." << std::endl;
      }
    } else if (!strncmp(line, "\\privileges", 11)) {
      std::string temp_line(line);
      boost::algorithm::trim(temp_line);
      if (temp_line.size() > 11) {
        context.privs_role_name.clear();
        context.privs_role_name = strtok(line + 12, " ");
        if (!context.privs_role_name.compare(MAPD_ROOT_USER)) {
          std::cout << "Command privileges failed because " << MAPD_ROOT_USER
                    << " root user has all privileges by default." << std::endl;
        } else {
          get_db_objects_for_role(context);
        }
      } else {
        std::cout << "Command privileges failed because parameter role name or user name is missing." << std::endl;
      }
    } else if (!strncmp(line, "\\object_privileges", 18)) {
      std::string temp_line(line);
      boost::algorithm::trim(temp_line);
      if (temp_line.size() > 18) {
        context.privs_object_name.clear();
        context.privs_object_name = strtok(line + 19, " ");
        get_db_object_privs(context);
      } else {
        std::cout << "Command object_privileges failed because parameter object name is missing." << std::endl;
      }
    } else if (line[0] == '\\' && line[1] == 'q') {
      break;
    } else {  // Experimental Cleanup
      using Params = CommandResolutionChain<>::CommandTokenList;

      auto resolution_status =
          CommandResolutionChain<>(
              line, "\\copygeo", 3, CopyGeoCmd<>(context), "Usage: \\copygeo <serverGeoFileName> <tableName>")(
              "\\copy",
              3,
              [&](Params const& p) { copy_table(p[1].c_str() /* filepath */, p[2].c_str() /* table */, context); })(
              "\\ste", 2, [&](Params const& p) { set_table_epoch(context, p[1] /* table_details */); })(
              "\\gte", 2, [&](Params const& p) { get_table_epoch(context, p[1] /* table_details */); })(
              "\\export_dashboard", 2, [&](Params const& p) { export_dashboard(context, p[1] /* dash_details */); })(
              "\\import_dashboard", 2, [&](Params const& p) { import_dashboard(context, p[1] /*dash_details */); })(
              "\\role_list", 2, RoleListCmd<>(context), "Usage: \\role_list <userName>")(
              "\\roles", 1, RolesCmd<>(context))("\\set_license", 2, [&](Params const& p) {
            set_license_key(context, p[1]);
          })("\\get_license", 1, [&](Params const&) {
            get_license_claims(context);
          }).is_resolved();

      if (resolution_status == false) {
        if (line[0] == '\\' && line[1] == 'q') {
          break;
        } else if (line[0] == '\\') {
          process_backslash_commands(line, context);
        }
      }
    }

    linenoiseHistoryAdd(line);                  /* Add to the history. */
    linenoiseHistorySave("mapdql_history.txt"); /* Save the history on disk. */
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
