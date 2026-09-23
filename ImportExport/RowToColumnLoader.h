/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
