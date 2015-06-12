/**
 * @file    StreamInsert.cpp
 * @author  Wei Hong <wei@mapd.com>
 * @brief   Sample MapD Client code for inserting a stream of rows 
 * with optional transformations from stdin to a MapD table.
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cstring>
#include <string>
#include <iostream>
#include <iterator>
#include <regex>

#include <thread>
#include <chrono>

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
  struct CopyParams {
    char delimiter;
    std::string null_str;
    char line_delim;
    size_t batch_size;
    CopyParams() : delimiter('\x01'), null_str("\\N"), line_delim('\n'), batch_size(10000) {}
  } copy_params;

#define MAX_FIELD_LEN   20000
  const bool print_error_data = false;
  const bool print_transformation = false;

  // reads copy_params.delimiter delimited rows from std::cin and load them to
  // table_name in batches of size copy_params.batch_size until EOF
  void
  stream_insert(MapDClient &client, const TSessionId session, const std::string &table_name, const TRowDescriptor &row_desc, const std::map<std::string, std::pair<std::unique_ptr<std::regex>, std::unique_ptr<std::string>>> &transformations)
  {
    std::vector<TStringRow> input_rows;
    TStringRow row;

    std::istream_iterator<char> eos;
    std::cin >> std::noskipws;
    std::istream_iterator<char> iit(std::cin);

    char field[MAX_FIELD_LEN];
    size_t field_i = 0;

    int nrows = 0;
    int nskipped = 0;
    
    const std::pair<std::unique_ptr<std::regex>, std::unique_ptr<std::string>> *xforms[row_desc.size()];
    for (size_t i = 0; i < row_desc.size(); i++) {
      auto it = transformations.find(row_desc[i].col_name);
      if (it != transformations.end())
        xforms[i] = &(it->second);
      else
        xforms[i] = nullptr;
    }

    while (iit != eos) {
      {
        // free previous row's memory
        std::vector<TStringValue> empty;
        row.cols.swap(empty);
      }
      // construct a row
      while (iit != eos) {
        if (*iit == copy_params.delimiter || *iit == copy_params.line_delim) {
          bool end_of_field = (*iit == copy_params.delimiter);
          bool end_of_row;
          if (end_of_field)
            end_of_row = false;
          else {
            end_of_row = (row_desc[row.cols.size()].col_type.type  != TDatumType::STR) || (row.cols.size() == row_desc.size() - 1);
            if (!end_of_row) {
              size_t l = copy_params.null_str.size();
              if (field_i >= l && strncmp(field + field_i - l, copy_params.null_str.c_str(), l) == 0) {
                end_of_row = true;
                // std::cout << "new line after null.\n";
              }
            }
          }
          if (!end_of_field && !end_of_row)
          {
            // not enough columns yet and it is a string column
            // treat the line delimiter as part of the string
            field[field_i++] = *iit;
          } else {
            field[field_i] = '\0';
            field_i = 0;
            TStringValue ts;
            ts.str_val = std::string(field);
            ts.is_null = (ts.str_val.empty() || ts.str_val == copy_params.null_str);
            auto xform = xforms[row.cols.size()];
            if (!ts.is_null && xform != nullptr) {
              if (print_transformation)
                std::cout << "\ntransforming\n" << ts.str_val << "\nto\n";
              ts.str_val = std::regex_replace(ts.str_val, *xform->first, *xform->second);
              if (ts.str_val.empty())
                ts.is_null = true;
              if (print_transformation)
                std::cout << ts.str_val << std::endl;
            }
            row.cols.push_back(ts); // add column value to row
            if (end_of_row || (row.cols.size() > row_desc.size()))
              break; // found row
          }
        } else {
          field[field_i++] = *iit;
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
      if (row.cols.size() == row_desc.size()) {
        input_rows.push_back(row);
        if (input_rows.size() >= copy_params.batch_size) {
          try {
            client.load_table(session, table_name, input_rows);
            nrows += input_rows.size();
            std::cout << nrows << " rows inserted, " << nskipped << " rows skipped." << std::endl;
          }
          catch (TMapDException &e) {
            std::cerr << e.error_msg << std::endl;
          }
          {
            // free rowset that has already been loaded
            std::vector<TStringRow> empty;
            input_rows.swap(empty);
          }
        }
      } else {
        ++nskipped;
        if (print_error_data) {
          std::cerr << "Incorrect number of columns for row at: ";
          bool not_first = false;
          for (const auto &p : row.cols) {
            if (not_first)
              std::cerr << copy_params.delimiter;
            else
              not_first = true;
            std::cerr << p;
          }
          std::cerr << std::endl;
        }
        if (row.cols.size() > row_desc.size()) {
          // skip to the next line delimiter
          while (*iit != copy_params.line_delim)
            ++iit;
        }
      }
      ++iit;
    }
    // load remaining rowset if any
    if (input_rows.size() > 0) {
      try {
        client.load_table(session, table_name, input_rows);
        nrows += input_rows.size();
        std::cout << nrows << " rows inserted, " << nskipped << " rows skipped." << std::endl;
      }
      catch (TMapDException &e) {
        std::cerr << e.error_msg << std::endl;
      }
    }
  }
}

int main(int argc, char **argv) {
  std::string server_host("localhost"); // default to localohost
  int port = 9091; // default port number
  std::map<std::string, std::pair<std::unique_ptr<std::regex>, std::unique_ptr<std::string>>> transformations;

  if (argc < 5) {
    std::cout << "Usage: <table> <database> <user> <password> [hostname[:port]] [transformation ...]" << std::endl;
    return 1;
  }
  std::string table_name(argv[1]);
  std::string db_name(argv[2]);
  std::string user_name(argv[3]);
  std::string passwd(argv[4]);

  if (argc >= 6) {
    char *host = strtok(argv[5], ":");
    char *portno = strtok(NULL, ":");
    server_host = host;
    if (portno != NULL)
      port = atoi(portno);
  }
  for (int i = 6; i < argc; i++) {
    char *col_name = argv[i];
    char *p;
    for (p = col_name; *p != ':' && *p != '\0'; ++p);
    if (*p == '\0') {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    *(p++) = '\0';
    if (*(p++) != 's' || *(p++) != '/') {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    char *regex_str = p;
    for (; (*p != '/' || *(p - 1) == '\\') && *p != '\0'; ++p);
    if (*p == '\0') {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    *(p++) = '\0';
    char *fmt_str = p;
    for (; (*p != '/' || *(p - 1) == '\\') && *p != '\0'; ++p);
    if (*p == '\0') {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    *p = '\0';
    std::cout << "transform " << col_name << ": s/" << regex_str << "/" << fmt_str << "/" << std::endl;
    transformations[col_name] = std::pair<std::unique_ptr<std::regex>, std::unique_ptr<std::string>>(std::unique_ptr<std::regex>(new std::regex(regex_str)), std::unique_ptr<std::string>(new std::string(fmt_str)));
  }

  //for attaching debugger std::this_thread::sleep_for (std::chrono::seconds(20));

  shared_ptr<TTransport> socket(new TSocket(server_host, port));
  shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  MapDClient client(protocol);
  TSessionId session;
  try {
    transport->open(); // open transport
    session = client.connect(user_name, passwd, db_name); // connect to mapd_server
    TRowDescriptor row_desc;
    client.get_row_descriptor(row_desc, session, table_name);
    stream_insert(client, session, table_name, row_desc, transformations);
    client.disconnect(session); // disconnect from mapd_server
    transport->close(); // close transport
  }
  catch (TMapDException &e) {
    std::cerr << e.error_msg << std::endl;
    return 1;
  }
  catch (TException &te) {
    std::cerr << "Thrift error: " << te.what() << std::endl;
    return 1;
  }

  return 0;
}
