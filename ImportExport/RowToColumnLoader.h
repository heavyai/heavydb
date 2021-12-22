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

#ifndef _ROWTOCOLUMNLOADER_H_
#define _ROWTOCOLUMNLOADER_H_

/**
 * @file    RowToColumnLoader.h
 * @author  Michael <michael@mapd.com>
 * @brief   Utility Function to convert rows to input columns for loading via
 *load_table_binary_columnar
 *
 * Copyright (c) 2017 MapD Technologies, Inc.  All rights reserved.
 **/

#include "Logger/Logger.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>

#if defined(_WIN32) && !defined(WIN32_LEAN_AND_MEAN)
// boost/regex.hpp on win32 includes Windows.h
// and we need to clean up macros such as ERROR and GetObject
#define WIN32_LEAN_AND_MEAN
#endif

#include <boost/regex.hpp>

#if defined(_WIN32) && defined(WIN32_LEAN_AND_MEAN)
#include "Shared/cleanup_global_namespace.h"
#undef WIN32_LEAN_AND_MEAN
#endif

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
#include "gen-cpp/OmniSci.h"
#include "gen-cpp/omnisci_types.h"

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

  std::shared_ptr<OmniSciClient> client_;
  TSessionId session_;

  void createConnection(const ThriftClientConnection& con);
  void closeConnection();
  void wait_disconnect_reconnect_retry(size_t tries,
                                       import_export::CopyParams copy_params);
};

#endif  // _ROWTOCOLUMNLOADER_H_
