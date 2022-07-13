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

/**
 * @file    RowToColumnLoader.h
 * @brief   Utility Function to convert rows to input columns for loading via
 * load_table_binary_columnar
 *
 */

#ifndef _ROWTOCOLUMNLOADER_H_
#define _ROWTOCOLUMNLOADER_H_

#include "Logger/Logger.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <cstring>
#include <iostream>
#include <iterator>
#include <string>

#include "Shared/ThriftClient.h"
#include "Shared/sqltypes.h"

#include <chrono>
#include <thread>

#include <boost/program_options.hpp>

// include files for Thrift and MapD Thrift Services
#include "CopyParams.h"
#include "gen-cpp/Heavy.h"
#include "gen-cpp/heavy_types.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

class RowToColumnLoader {
 public:
  RowToColumnLoader(const ThriftClientConnection& conn_details,
                    const std::string& user_name,
                    const std::string& passwd,
                    const std::string& db_name,
                    const std::string& table_name);
  ~RowToColumnLoader();
  void do_load(int& nrows, int& nskipped, import_export::CopyParams copy_params);
  bool convert_string_to_column(std::vector<TStringValue> row,
                                const import_export::CopyParams& copy_params);
  TRowDescriptor get_row_descriptor();
  std::string print_row_with_delim(std::vector<TStringValue> row,
                                   const import_export::CopyParams& copy_params);

 private:
  std::string user_name_;
  std::string passwd_;
  std::string db_name_;
  std::string table_name_;
  ThriftClientConnection conn_details_;

  std::vector<TColumn> input_columns_;
  std::vector<SQLTypeInfo> column_type_info_;
  std::vector<SQLTypeInfo> array_column_type_info_;

  TRowDescriptor row_desc_;

  std::shared_ptr<HeavyClient> client_;
  TSessionId session_;

  void createConnection(const ThriftClientConnection& con);
  void closeConnection();
  void wait_disconnect_reconnect_retry(size_t tries,
                                       import_export::CopyParams copy_params);
};

#endif  // _ROWTOCOLUMNLOADER_H_
