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
#include <boost/regex.hpp>

#include <thread>
#include <chrono>

#include <boost/program_options.hpp>

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

struct CopyParams {
  char delimiter;
  std::string null_str;
  char line_delim;
  size_t batch_size;
  CopyParams(char d, const std::string &n, char l, size_t b) : delimiter(d), null_str(n), line_delim(l), batch_size(b) {}
};

bool print_error_data = false;
bool print_transformation = false;


namespace {
  // anonymous namespace for private functions

#define MAX_FIELD_LEN   20000
  // reads copy_params.delimiter delimited rows from std::cin and load them to
  // table_name in batches of size copy_params.batch_size until EOF
  void
  stream_insert(MapDClient &client, const TSessionId session, const std::string &table_name, const TRowDescriptor &row_desc, const std::map<std::string, std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>> &transformations, const CopyParams &copy_params)
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
    
    const std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>> *xforms[row_desc.size()];
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
            auto xform = row.cols.size() < row_desc.size() ? xforms[row.cols.size()] : nullptr;
            if (!ts.is_null && xform != nullptr) {
              if (print_transformation)
                std::cout << "\ntransforming\n" << ts.str_val << "\nto\n";
              ts.str_val = boost::regex_replace(ts.str_val, *xform->first, *xform->second);
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
            std::cerr << &p;
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
  std::string table_name;
  std::string db_name;
  std::string user_name;
  std::string passwd;
  std::string delim_str(","), nulls("\\N"), line_delim_str("\n");
  size_t batch_size = 10000;
  std::vector<std::string> xforms;
  std::map<std::string, std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>> transformations;

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
		("table", po::value<std::string>(&table_name)->required(), "Table Name")
		("database", po::value<std::string>(&db_name)->required(), "Database Name")
		("user,u", po::value<std::string>(&user_name)->required(), "User Name")
		("passwd,p", po::value<std::string>(&passwd)->required(), "User Password")
    ("host", po::value<std::string>(&server_host), "MapD Server Hostname")
    ("port", po::value<int>(&port), "MapD Server Port Number")
    ("delim", po::value<std::string>(&delim_str), "Field delimiter")
    ("null", po::value<std::string>(&nulls), "NULL string")
    ("line", po::value<std::string>(&line_delim_str), "Line delimiter")
    ("batch", po::value<size_t>(&batch_size), "Insert batch size")
    ("transform,t", po::value<std::vector<std::string>>(&xforms)->multitoken(), "Column Transformations")
    ("print_error", "Print Error Rows")
    ("print_transform", "Print Transformations");

	po::positional_options_description positionalOptions;
	positionalOptions.add("table", 1);
	positionalOptions.add("database", 1);

	po::variables_map vm;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
		if (vm.count("help")) {
			std::cout << "Usage: <table name> <database name> {-u|--user} <user> {-p|--passwd} <password> [{--host} <hostname>][--port <port number>][--delim <delimiter>][--null <null string>][--line <line delimiter>][--batch <batch size>][{-t|--transform} transformation ...][--print_error][--print_transform]\n";
			return 0;
		}
    if (vm.count("print_error"))
      print_error_data = true;
    if (vm.count("print_transform"))
      print_transformation = true;

		po::notify(vm);
	}
	catch (boost::program_options::error &e)
	{
		std::cerr << "Usage Error: " << e.what() << std::endl;
		return 1;
	}

  char delim = delim_str[0];
  if (delim == '\\') {
    if (delim_str.size() < 2 || (delim_str[1] != 'x' && delim_str[1] != 't' && delim_str[1] != 'n')) {
      std::cerr << "Incorrect delimiter string: " << delim_str << std::endl;
      return 1;
    }
    if (delim_str[1] == 't')
      delim = '\t';
    else if (delim_str[1] == 'n')
      delim = '\n';
    else {
      std::string d(delim_str);
      d[0] = '0';
      delim = (char)std::stoi(d, nullptr, 16);
    }
  }
  if (isprint(delim))
    std::cout << "Field Delimiter: " << delim << std::endl;
  else if (delim == '\t')
    std::cout << "Line Delimiter: " << "\\t" << std::endl;
  else if (delim == '\n')
    std::cout << "Line Delimiter: " << "\\n" << std::endl;
  else
    std::cout << "Field Delimiter: \\x" << std::hex << (int)delim << std::endl;
  char line_delim = line_delim_str[0];
  if (line_delim == '\\') {
    if (line_delim_str.size() < 2 || (line_delim_str[1] != 'x' && line_delim_str[1] != 't' && line_delim_str[1] != 'n')) {
      std::cerr << "Incorrect delimiter string: " << line_delim_str << std::endl;
      return 1;
    }
    if (line_delim_str[1] == 't')
      line_delim = '\t';
    else if (delim_str[1] == 'n')
      line_delim = '\n';
    else {
      std::string d(delim_str);
      d[0] = '0';
      line_delim = (char)std::stoi(d, nullptr, 16);
    }
  }
  if (isprint(line_delim))
    std::cout << "Line Delimiter: " << line_delim << std::endl;
  else if (line_delim == '\t')
    std::cout << "Line Delimiter: " << "\\t" << std::endl;
  else if (line_delim == '\n')
    std::cout << "Line Delimiter: " << "\\n" << std::endl;
  else
    std::cout << "Line Delimiter: \\x" << std::hex << (int)line_delim << std::endl;
  std::cout << "Null String: " << nulls << std::endl;
  std::cout << "Insert Batch Size: " << std::dec << batch_size << std::endl;

  for (auto &t : xforms) {
    auto n = t.find_first_of(':');
    if (n == std::string::npos) {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    std::string col_name = t.substr(0, n);
    if (t.size() < n + 3 || t[n+1] != 's' || t[n+2] != '/') {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    auto n1 = n + 3;
    auto n2 = t.find_first_of('/', n1);
    if (n2 == std::string::npos) {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    std::string regex_str = t.substr(n1, n2-n1);
    n1 = n2 + 1;
    n2 = t.find_first_of('/', n1);
    if (n2 == std::string::npos) {
      std::cerr << "Transformation format: <column name>:s/<regex pattern>/<fmt string>/" << std::endl;
      return 1;
    }
    std::string fmt_str = t.substr(n1, n2-n1);
    std::cout << "transform " << col_name << ": s/" << regex_str << "/" << fmt_str << "/" << std::endl;
    transformations[col_name] = std::pair<std::unique_ptr<boost::regex>, std::unique_ptr<std::string>>(std::unique_ptr<boost::regex>(new boost::regex(regex_str)), std::unique_ptr<std::string>(new std::string(fmt_str)));
  }

  CopyParams copy_params(delim, nulls, line_delim, batch_size);

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
    stream_insert(client, session, table_name, row_desc, transformations, copy_params);
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
