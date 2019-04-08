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
 * @file    StreamInsert.cpp
 * @author  Wei Hong <wei@mapd.com>
 * @brief   Sample MapD Client code for inserting a stream of rows from stdin
 * to a MapD table.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <boost/tokenizer.hpp>
#include <cstring>
#include <iostream>
#include <string>

// include files for Thrift and MapD Thrift Services
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TSocket.h>
#include "gen-cpp/MapD.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

#ifdef HAVE_THRIFT_STD_SHAREDPTR
#include <memory>
namespace mapd {
using std::make_shared;
using std::shared_ptr;
}  // namespace mapd
#else
#include <boost/make_shared.hpp>
namespace mapd {
using boost::make_shared;
using boost::shared_ptr;
}  // namespace mapd
#endif  // HAVE_THRIFT_STD_SHAREDPTR

namespace {
// anonymous namespace for private functions
const size_t INSERT_BATCH_SIZE = 10000;

// reads tab-delimited rows from std::cin and load them to
// table_name in batches of size INSERT_BATCH_SIZE until done
void stream_insert(MapDClient& client,
                   const TSessionId session,
                   const std::string& table_name,
                   const TRowDescriptor& row_desc,
                   const char* delimiter) {
  std::string line;
  std::vector<TStringRow> input_rows;
  TStringRow row;
  boost::char_separator<char> sep{delimiter, "", boost::keep_empty_tokens};
  while (std::getline(std::cin, line)) {
    row.cols.clear();
    boost::tokenizer<boost::char_separator<char>> tok{line, sep};
    for (const auto& s : tok) {
      TStringValue ts;
      ts.str_val = s;
      ts.is_null = s.empty();
      row.cols.push_back(ts);
    }
    if (row.cols.size() != row_desc.size()) {
      std::cerr << "Incorrect number of columns: (" << row.cols.size() << " vs "
                << row_desc.size() << ") " << line << std::endl;
      continue;
    }
    input_rows.push_back(row);
    if (input_rows.size() >= INSERT_BATCH_SIZE) {
      try {
        client.load_table(session, table_name, input_rows);
      } catch (TMapDException& e) {
        std::cerr << e.error_msg << std::endl;
      }
      input_rows.clear();
    }
  }
  // load remaining rowset if any
  if (input_rows.size() > 0) {
    client.load_table(session, table_name, input_rows);
  }
}
}  // namespace

int main(int argc, char** argv) {
  std::string server_host("localhost");  // default to localohost
  int port = 6274;                       // default port number
  const char* delimiter = "\t";          // only support tab delimiter for now

  if (argc < 5) {
    std::cout << "Usage: <table> <database> <user> <password> [hostname[:port]]"
              << std::endl;
    return 1;
  }
  std::string table_name(argv[1]);
  std::string db_name(argv[2]);
  std::string user_name(argv[3]);
  std::string passwd(argv[4]);

  if (argc >= 6) {
    char* host = strtok(argv[5], ":");
    char* portno = strtok(NULL, ":");
    server_host = host;
    if (portno != NULL) {
      port = atoi(portno);
    }
  }

  mapd::shared_ptr<TTransport> socket(new TSocket(server_host, port));
  mapd::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  mapd::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  MapDClient client(protocol);
  TSessionId session;
  try {
    transport->open();                                    // open transport
    client.connect(session, user_name, passwd, db_name);  // connect to omnisci_server
    TTableDetails table_details;
    client.get_table_details(table_details, session, table_name);
    stream_insert(client, session, table_name, table_details.row_desc, delimiter);
    client.disconnect(session);  // disconnect from omnisci_server
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
