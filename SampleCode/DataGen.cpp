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
 * @file    DataGen.cpp
 * @brief   Sample MapD Client code for generating random data that can be
 * inserted into a given MapD table.
 *
 * Usage: <table> <database> <user> <password> [<num rows>] [hostname[:port]]
 * The program executes the following:
 * 1. connect to mapd_server at hostname:port (default: localhost:9091)
 *    with <database> <user> <password>
 * 2. get the table descriptor of <table>
 * 3. randomly generate tab-delimited data that can be imported to <table>
 * 4. disconnect from mapd_server
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>
#include <cstdint>
#include <cfloat>
#include <random>
#include <ctime>

// include files for Thrift and MapD Thrift Services
#include "gen-cpp/MapD.h"
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

// Thrift uses boost::shared_ptr instead of std::shared_ptr
using boost::shared_ptr;

namespace {
// anonymous namespace for private functions
std::default_random_engine random_gen(std::random_device{}());

// returns a random int as string
std::string gen_int() {
  std::uniform_int_distribution<int> dist(INT_MIN, INT_MAX);
  return std::to_string(dist(random_gen));
}

// returns a random float as string
std::string gen_real() {
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  return std::to_string(dist(random_gen));
}

const int max_str_len = 100;

// returns a random string of length up to max_str_len
std::string gen_string() {
  std::string chars("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");
  std::uniform_int_distribution<> char_dist(0, chars.size() - 1);
  std::uniform_int_distribution<> len_dist(0, max_str_len);
  int len = len_dist(random_gen);
  std::string s(len, ' ');
  for (int i = 0; i < len; i++)
    s[i] = chars[char_dist(random_gen)];
  return s;
}

// returns a random boolean as string
std::string gen_bool() {
  std::uniform_int_distribution<int> dist(0, 1);
  if (dist(random_gen) == 1)
    return "t";
  return "f";
}

// returns a random time as string
std::string gen_time() {
  std::uniform_int_distribution<int> dist(0, INT32_MAX);
  time_t t = dist(random_gen);
  std::tm* tm_ptr = gmtime(&t);
  char buf[9];
  strftime(buf, 9, "%T", tm_ptr);
  return buf;
}

// returns a random timestamp as string
std::string gen_timestamp() {
  std::uniform_int_distribution<int> dist(0, INT32_MAX);
  time_t t = dist(random_gen);
  std::tm* tm_ptr = gmtime(&t);
  char buf[20];
  strftime(buf, 20, "%F %T", tm_ptr);
  return buf;
}

// returns a random date as string
std::string gen_date() {
  std::uniform_int_distribution<int> dist(0, INT32_MAX);
  time_t t = dist(random_gen);
  std::tm* tm_ptr = gmtime(&t);
  char buf[11];
  strftime(buf, 11, "%F", tm_ptr);
  return buf;
}

// output to std::cout num_rows number of rows conforming to row_desc.
// each column value is separated by delimiter.
void data_gen(const TRowDescriptor& row_desc, const char* delimiter, int num_rows) {
  for (int i = 0; i < num_rows; i++) {
    bool not_first = false;
    for (auto p = row_desc.begin(); p != row_desc.end(); ++p) {
      if (not_first)
        std::cout << delimiter;
      else
        not_first = true;
      switch (p->col_type.type) {
        case TDatumType::SMALLINT:
        case TDatumType::INT:
        case TDatumType::BIGINT:
          std::cout << gen_int();
          break;
        case TDatumType::FLOAT:
        case TDatumType::DOUBLE:
        case TDatumType::DECIMAL:
          std::cout << gen_real();
          break;
        case TDatumType::STR:
          std::cout << gen_string();
          break;
        case TDatumType::TIME:
          std::cout << gen_time();
          break;
        case TDatumType::TIMESTAMP:
        case TDatumType::INTERVAL_DAY_TIME:
        case TDatumType::INTERVAL_YEAR_MONTH:
          std::cout << gen_timestamp();
          break;
        case TDatumType::DATE:
          std::cout << gen_date();
          break;
        case TDatumType::BOOL:
          std::cout << gen_bool();
          break;
      }
    }
    std::cout << std::endl;
  }
}
}

int main(int argc, char** argv) {
  std::string server_host("localhost");  // default to localhost
  int port = 9091;                       // default port number
  int num_rows = 1000000;                // default number of rows to generate
  const char* delimiter = "\t";          // only support tab delimiter for now

  if (argc < 5) {
    std::cout << "Usage: <table> <database> <user> <password> [<num rows>] [hostname[:port]]" << std::endl;
    return 1;
  }
  std::string table_name(argv[1]);
  std::string db_name(argv[2]);
  std::string user_name(argv[3]);
  std::string passwd(argv[4]);

  if (argc >= 6) {
    num_rows = atoi(argv[5]);
    if (argc >= 7) {
      char* host = strtok(argv[6], ":");
      char* portno = strtok(NULL, ":");
      server_host = host;
      if (portno != NULL)
        port = atoi(portno);
    }
  }

  shared_ptr<TTransport> socket(new TSocket(server_host, port));
  shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  MapDClient client(protocol);
  TSessionId session;
  try {
    transport->open();                                     // open transport
    client.connect(session, user_name, passwd, db_name);   // connect to mapd_server
    TTableDetails table_details;
    client.get_table_details(table_details, session, table_name);
    data_gen(table_details.row_desc, delimiter, num_rows);
    client.disconnect(session);  // disconnect from mapd_server
    transport->close();          // close transport
  } catch (TMapDException& e) {
    std::cerr << e.error_msg << std::endl;
    return 1;
  } catch (TException& te) {
    std::cerr << "Thrift error: " << te.what() << std::endl;
    return 1;
  }

  return 0;
}
