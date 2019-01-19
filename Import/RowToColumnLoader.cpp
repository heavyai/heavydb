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
 * @file    RowToColumnLoader.cpp
 * @author  Michael <michael@mapd.com>
 * @brief   Based on StreamInsert code but using binary columnar format for inserting a
 *stream of rows with optional transformations from stdin to a MapD table.
 *
 * Copyright (c) 2017 MapD Technologies, Inc.  All rights reserved.
 **/

#include <glog/logging.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/regex.hpp>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>

#include "Shared/mapd_shared_ptr.h"
#include "Shared/sqltypes.h"

#include <chrono>
#include <thread>

#include <boost/program_options.hpp>

// include files for Thrift and MapD Thrift Services
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TSocket.h>
#include "Importer.h"
#include "RowToColumnLoader.h"
#include "gen-cpp/MapD.h"
#include "gen-cpp/mapd_types.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

SQLTypes get_sql_types(const TColumnType& ct) {
  switch (ct.col_type.type) {
    case TDatumType::BIGINT:
      return SQLTypes::kBIGINT;
    case TDatumType::BOOL:
      return SQLTypes::kBOOLEAN;
    case TDatumType::DATE:
      return SQLTypes::kDATE;
    case TDatumType::DECIMAL:
      return SQLTypes::kDECIMAL;
    case TDatumType::DOUBLE:
      return SQLTypes::kDOUBLE;
    case TDatumType::FLOAT:
      return SQLTypes::kFLOAT;
    case TDatumType::INT:
      return SQLTypes::kINT;
    case TDatumType::STR:
      // Tdataum is lossy here so need to look at precision to see if it was defined
      if (ct.col_type.precision == 0) {
        return SQLTypes::kTEXT;
      } else {
        return SQLTypes::kVARCHAR;
      }
    case TDatumType::TIME:
      return SQLTypes::kTIME;
    case TDatumType::TIMESTAMP:
      return SQLTypes::kTIMESTAMP;
    case TDatumType::SMALLINT:
      return SQLTypes::kSMALLINT;
    default:
      LOG(FATAL) << "Unsupported TColumnType found, should not be possible";
  }
}

SQLTypeInfo create_sql_type_info_from_col_type(const TColumnType& ct) {
  if (ct.col_type.is_array) {
    return SQLTypeInfo(SQLTypes::kARRAY,
                       ct.col_type.precision,
                       ct.col_type.scale,
                       ct.col_type.nullable,
                       kENCODING_NONE,
                       0,
                       get_sql_types(ct));
  } else {
    // normal column
    return SQLTypeInfo(get_sql_types(ct),
                       ct.col_type.precision,
                       ct.col_type.scale,
                       ct.col_type.nullable,
                       kENCODING_NONE,
                       0,
                       SQLTypes::kNULLT);
  }
}

// this function allows us to treat array columns natively in the rest of the code
// by creating  fact column description
SQLTypeInfo create_array_sql_type_info_from_col_type(const TColumnType& ct) {
  return SQLTypeInfo(get_sql_types(ct),
                     ct.col_type.precision,
                     ct.col_type.scale,
                     ct.col_type.nullable,
                     kENCODING_NONE,
                     0,
                     SQLTypes::kNULLT);
}

std::string RowToColumnLoader::print_row_with_delim(
    std::vector<TStringValue> row,
    const Importer_NS::CopyParams& copy_params) {
  std::ostringstream out;
  bool first = true;
  for (TStringValue ts : row) {
    if (first) {
      first = false;
    } else {
      out << copy_params.delimiter;
    }
    out << ts.str_val;
  }
  return out.str();
}

// remove the entries from a row that has a failure during processing
// we must remove the entries that have been pushed onto the input_col so far
void remove_partial_row(size_t failed_column,
                        std::vector<SQLTypeInfo> column_type_info_vector,
                        std::vector<TColumn>& input_col_vec) {
  for (size_t idx = 0; idx < failed_column; idx++) {
    switch (column_type_info_vector[idx].get_type()) {
      case SQLTypes::kARRAY:
        input_col_vec[idx].nulls.pop_back();
        input_col_vec[idx].data.arr_col.pop_back();
        break;
      case SQLTypes::kTEXT:
      case SQLTypes::kCHAR:
      case SQLTypes::kVARCHAR:
        input_col_vec[idx].nulls.pop_back();
        input_col_vec[idx].data.str_col.pop_back();
        break;
      case SQLTypes::kINT:
      case SQLTypes::kBIGINT:
      case SQLTypes::kSMALLINT:
      case SQLTypes::kDATE:
      case SQLTypes::kTIME:
      case SQLTypes::kTIMESTAMP:
      case SQLTypes::kNUMERIC:
      case SQLTypes::kDECIMAL:
      case SQLTypes::kBOOLEAN:
        input_col_vec[idx].nulls.pop_back();
        input_col_vec[idx].data.int_col.pop_back();
        break;
      case SQLTypes::kFLOAT:
      case SQLTypes::kDOUBLE:
        input_col_vec[idx].nulls.pop_back();
        input_col_vec[idx].data.real_col.pop_back();
        break;
      default:
        LOG(FATAL) << "Trying to process an unsupported datatype, should be impossible";
    }
  }
}

void populate_TColumn(TStringValue ts,
                      SQLTypeInfo column_type_info,
                      TColumn& input_col,
                      const Importer_NS::CopyParams& copy_params) {
  // create datum and push data to column structure from row data

  switch (column_type_info.get_type()) {
    case SQLTypes::kARRAY:
      LOG(FATAL) << "Trying to process ARRAY at item level something is wrong";
      break;
    case SQLTypes::kTEXT:
    case SQLTypes::kCHAR:
    case SQLTypes::kVARCHAR:
      if (ts.is_null) {
        input_col.nulls.push_back(true);
        input_col.data.str_col.emplace_back("");

      } else {
        input_col.nulls.push_back(false);
        switch (column_type_info.get_type()) {
          case SQLTypes::kCHAR:
          case SQLTypes::kVARCHAR:
            input_col.data.str_col.push_back(
                ts.str_val.substr(0, column_type_info.get_precision()));
            break;
          case SQLTypes::kTEXT:

            input_col.data.str_col.push_back(ts.str_val);
            break;
          default:
            LOG(FATAL) << " trying to process a STRING transport type not handled "
                       << column_type_info.get_type();
        }
      }
      break;
    case SQLTypes::kINT:
    case SQLTypes::kBIGINT:
    case SQLTypes::kSMALLINT:
    case SQLTypes::kDATE:
    case SQLTypes::kTIME:
    case SQLTypes::kTIMESTAMP:
    case SQLTypes::kNUMERIC:
    case SQLTypes::kDECIMAL:
    case SQLTypes::kBOOLEAN:
      if (ts.is_null) {
        input_col.nulls.push_back(true);
        input_col.data.int_col.push_back(0);
      } else {
        input_col.nulls.push_back(false);
        Datum d = StringToDatum(ts.str_val, column_type_info);
        switch (column_type_info.get_type()) {
          case SQLTypes::kINT:
          case SQLTypes::kBOOLEAN:
            input_col.data.int_col.push_back(d.intval);
            break;
          case SQLTypes::kBIGINT:
          case SQLTypes::kNUMERIC:
          case SQLTypes::kDECIMAL:
            input_col.data.int_col.push_back(d.bigintval);
            break;
          case SQLTypes::kSMALLINT:
            input_col.data.int_col.push_back(d.smallintval);
            break;
          case SQLTypes::kDATE:
          case SQLTypes::kTIME:
          case SQLTypes::kTIMESTAMP:
            input_col.data.int_col.push_back(d.timeval);
            break;
          default:
            LOG(FATAL) << " trying to process an INT transport type not handled "
                       << column_type_info.get_type();
        }
      }
      break;
    case SQLTypes::kFLOAT:
    case SQLTypes::kDOUBLE:
      if (ts.is_null) {
        input_col.nulls.push_back(true);
        input_col.data.real_col.push_back(0);

      } else {
        input_col.nulls.push_back(false);
        Datum d = StringToDatum(ts.str_val, column_type_info);
        switch (column_type_info.get_type()) {
          case SQLTypes::kFLOAT:
            input_col.data.real_col.push_back(d.floatval);
            break;
          case SQLTypes::kDOUBLE:
            input_col.data.real_col.push_back(d.doubleval);
            break;
          default:
            LOG(FATAL) << " trying to process a REAL transport type not handled "
                       << column_type_info.get_type();
        }
      }
      break;
    default:
      LOG(FATAL) << "Trying to process an unsupported datatype, should be impossible";
  }
}

TRowDescriptor RowToColumnLoader::get_row_descriptor() {
  return row_desc_;
};

bool RowToColumnLoader::convert_string_to_column(
    std::vector<TStringValue> row,
    const Importer_NS::CopyParams& copy_params) {
  // create datum and push data to column structure from row data
  uint curr_col = 0;
  for (TStringValue ts : row) {
    try {
      switch (column_type_info_[curr_col].get_type()) {
        case SQLTypes::kARRAY: {
          std::vector<std::string> arr_ele;
          Importer_NS::ImporterUtils::parseStringArray(ts.str_val, copy_params, arr_ele);
          TColumn array_tcol;
          for (std::string item : arr_ele) {
            boost::algorithm::trim(item);
            TStringValue tsa;
            tsa.str_val = item;
            tsa.is_null = (tsa.str_val.empty() || tsa.str_val == copy_params.null_str);
            // now put into TColumn
            populate_TColumn(
                tsa, array_column_type_info_[curr_col], array_tcol, copy_params);
          }
          input_columns_[curr_col].nulls.push_back(false);
          input_columns_[curr_col].data.arr_col.push_back(array_tcol);

        } break;
        default:
          populate_TColumn(
              ts, column_type_info_[curr_col], input_columns_[curr_col], copy_params);
      }
    } catch (const std::exception& e) {
      remove_partial_row(curr_col, column_type_info_, input_columns_);
      // import_status.rows_rejected++;
      LOG(ERROR) << "Input exception thrown: " << e.what()
                 << ". Row discarded, issue at column : " << (curr_col + 1)
                 << " data :" << print_row_with_delim(row, copy_params);
      return false;
    }
    curr_col++;
  }
  return true;
}

RowToColumnLoader::RowToColumnLoader(const ThriftClientConnection& conn_details,
                                     const std::string& user_name,
                                     const std::string& passwd,
                                     const std::string& db_name,
                                     const std::string& table_name)
    : user_name_(user_name)
    , passwd_(passwd)
    , db_name_(db_name)
    , table_name_(table_name)
    , conn_details_(conn_details) {
  createConnection(conn_details_);

  TTableDetails table_details;
  client_->get_table_details(table_details, session_, table_name_);

  row_desc_ = table_details.row_desc;

  // create vector with column details
  for (TColumnType ct : row_desc_) {
    column_type_info_.push_back(create_sql_type_info_from_col_type(ct));
  }

  // create vector with array column details presented as real column for easier resue
  // of othe code
  for (TColumnType ct : row_desc_) {
    array_column_type_info_.push_back(create_array_sql_type_info_from_col_type(ct));
  }

  // create vector for storage of the actual column data
  for (TColumnType column : row_desc_) {
    TColumn t;
    input_columns_.push_back(t);
  }
}
RowToColumnLoader::~RowToColumnLoader() {
  closeConnection();
}

void RowToColumnLoader::createConnection(const ThriftClientConnection& con) {
  mapd::shared_ptr<TProtocol> protocol;
  mapd::shared_ptr<TTransport> socket;
  if (con.conn_type_ == ThriftConnectionType::HTTP ||
      con.conn_type_ == ThriftConnectionType::HTTPS) {
    mytransport_ = openHttpClientTransport(con.server_host_,
                                           con.port_,
                                           con.ca_cert_name_,
                                           con.conn_type_ == ThriftConnectionType::HTTPS,
                                           con.skip_host_verify_);
    protocol = mapd::shared_ptr<TProtocol>(new TJSONProtocol(mytransport_));
  } else {
    mytransport_ =
        openBufferedClientTransport(con.server_host_, con.port_, con.ca_cert_name_);
    protocol = mapd::shared_ptr<TProtocol>(new TBinaryProtocol(mytransport_));
  }
  client_.reset(new MapDClient(protocol));

  try {
    mytransport_->open();
    client_->connect(session_, user_name_, passwd_, db_name_);
  } catch (TMapDException& e) {
    std::cerr << e.error_msg << std::endl;
  } catch (TException& te) {
    std::cerr << "Thrift error on connect: " << te.what() << std::endl;
  }
}

void RowToColumnLoader::closeConnection() {
  try {
    client_->disconnect(session_);  // disconnect from omnisci_server
    mytransport_->close();          // close transport
  } catch (TMapDException& e) {
    std::cerr << e.error_msg << std::endl;
  } catch (TException& te) {
    std::cerr << "Thrift error on close: " << te.what() << std::endl;
  }
}

void RowToColumnLoader::wait_disconnet_reconnnect_retry(
    size_t tries,
    Importer_NS::CopyParams copy_params) {
  std::cout << "  Waiting  " << copy_params.retry_wait
            << " secs to retry Inserts , will try " << (copy_params.retry_count - tries)
            << " times more " << std::endl;
  sleep(copy_params.retry_wait);

  closeConnection();
  createConnection(conn_details_);
}

void RowToColumnLoader::do_load(int& nrows,
                                int& nskipped,
                                Importer_NS::CopyParams copy_params) {
  for (size_t tries = 0; tries < copy_params.retry_count;
       tries++) {  // allow for retries in case of insert failure
    try {
      client_->load_table_binary_columnar(session_, table_name_, input_columns_);
      //      client->load_table(session, table_name, input_rows);
      nrows += input_columns_[0].nulls.size();
      std::cout << nrows << " Rows Inserted, " << nskipped << " rows skipped."
                << std::endl;
      // we successfully loaded the data, lets move on
      input_columns_.clear();
      // create vector for storage of the actual column data
      for (TColumnType column : row_desc_) {
        TColumn t;
        input_columns_.push_back(t);
      }
      return;
    } catch (TMapDException& e) {
      std::cerr << "Exception trying to insert data " << e.error_msg << std::endl;
      wait_disconnet_reconnnect_retry(tries, copy_params);
    } catch (TException& te) {
      std::cerr << "Exception trying to insert data " << te.what() << std::endl;
      wait_disconnet_reconnnect_retry(tries, copy_params);
    }
  }
  std::cerr << "Retries exhausted program terminated" << std::endl;
  exit(1);
}
