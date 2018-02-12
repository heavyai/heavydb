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
 * @brief   Utility Function to convert rows to input columns for loading via load_table_binary_columnar
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

#include <thread>
#include <chrono>

#include <boost/program_options.hpp>

// include files for Thrift and MapD Thrift Services
#include "gen-cpp/MapD.h"
#include "gen-cpp/mapd_types.h"
#include "Importer.h"
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

// Thrift uses boost::shared_ptr instead of std::shared_ptr
using boost::shared_ptr;

struct ConnectionDetails {
  std::string server_host;
  int port;
  std::string db_name;
  std::string user_name;
  std::string passwd;

  ConnectionDetails(const std::string in_server_host,
                    const int in_port,
                    const std::string in_db_name,
                    const std::string in_user_name,
                    const std::string in_passwd)
      : server_host(in_server_host), port(in_port), db_name(in_db_name), user_name(in_user_name), passwd(in_passwd) {}
  ConnectionDetails(){};
};

class RowToColumnLoader {
 public:
  RowToColumnLoader(const std::string server_host,
                    const int port,
                    const std::string db_name,
                    const std::string user_name,
                    const std::string passwd,
                    const std::string table_name);
  ~RowToColumnLoader();
  void do_load(int& nrows, int& nskipped, Importer_NS::CopyParams copy_params);
  bool convert_string_to_column(std::vector<TStringValue> row, const Importer_NS::CopyParams& copy_params);
  TRowDescriptor get_row_descriptor();
  std::string print_row_with_delim(std::vector<TStringValue> row, const Importer_NS::CopyParams& copy_params);

 private:
  std::string table_name_;

  std::vector<TColumn> input_columns_;
  std::vector<SQLTypeInfo> column_type_info_;
  std::vector<SQLTypeInfo> array_column_type_info_;

  TRowDescriptor row_desc_;

  ConnectionDetails conn_details_;

  shared_ptr<MapDClient> client_;
  TSessionId session_;
  shared_ptr<apache::thrift::transport::TTransport> mytransport_;

  void createConnection(ConnectionDetails con);
  void closeConnection();
  void wait_disconnet_reconnnect_retry(size_t tries,
                                       Importer_NS::CopyParams copy_params,
                                       ConnectionDetails conn_details);
};

#endif  // _ROWTOCOLUMNLOADER_H_
