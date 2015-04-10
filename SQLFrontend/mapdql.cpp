/**
 * @file    mapdql.cpp
 * @author  Wei Hong <wei@map-d.com>
 * @brief   MapD SQL Client Tool
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <boost/program_options.hpp>

#include "gen-cpp/MapD.h"
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include "linenoise.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;


using boost::shared_ptr;

void completion(const char *buf, linenoiseCompletions *lc) {
    if (buf[0] == 'h') {
        linenoiseAddCompletion(lc,"hello");
        linenoiseAddCompletion(lc,"hello there");
    }
}

int main(int argc, char **argv) {
  std::string server_host("localhost");
  int port = 9091;
  char *line;
  QueryResult _return;

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
    ("server,s", po::value<std::string>(&server_host), "MapD Server Hostname (default localhost)")
    ("port,p", po::value<int>(&port), "Port number (default 9091)");

	po::variables_map vm;
	po::positional_options_description positionalOptions;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
		if (vm.count("help")) {
			std::cout << "Usage: mapdql [{-p|--port} <port number>] [{-s|--server} <server host>]\n";
			return 0;
		}

		po::notify(vm);
	}
	catch (boost::program_options::error &e)
	{
		std::cerr << "Usage Error: " << e.what() << std::endl;
		return 1;
	}

  shared_ptr<TTransport> socket(new TSocket(server_host, port));
  shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  MapDClient client(protocol);
  transport->open();

  /* Set the completion callback. This will be called every time the
   * user uses the <tab> key. */
  linenoiseSetCompletionCallback(completion);

  /* Load history from file. The history file is just a plain text file
   * where entries are separated by newlines. */
  linenoiseHistoryLoad("mapdql_history.txt"); /* Load the history at startup */

  /* Now this is the main loop of the typical linenoise-based application.
   * The call to linenoise() will block as long as the user types something
   * and presses enter.
   *
   * The typed string is returned as a malloc() allocated string by
   * linenoise, so the user needs to free() it. */
  while((line = linenoise("mapd> ")) != NULL) {
      /* Do something with the string. */
      if (line[0] != '\0' && line[0] != '/') {
          // printf("echo: '%s'\n", line);
          linenoiseHistoryAdd(line); /* Add to the history. */
          linenoiseHistorySave("mapdql_history.txt"); /* Save the history on disk. */
        try {
          client.select(_return, line);
          for (auto row : _return.rows) {
            bool not_first = false;
            for (auto col_val : row.cols) {
              if (not_first)
                std::cout << "|";
              else
                not_first = true;
              if (col_val.is_null)
                std::cout << "NULL";
              else
                switch (col_val.type) {
                  case TDatumType::INT:
                    std::cout << std::to_string(col_val.datum.int_val);
                    break;
                  case TDatumType::REAL:
                    std::cout << std::to_string(col_val.datum.real_val);
                    break;
                  case TDatumType::STR:
                    std::cout << col_val.datum.str_val;
                    break;
                  case TDatumType::TIME:
                    {
                      time_t t = (time_t)col_val.datum.int_val;
                      std::tm tm_struct;
                      gmtime_r(&t, &tm_struct);
                      char buf[9];
                      strftime(buf, 9, "%T", &tm_struct);
                      std::cout << buf;
                      break;
                    }
                  case TDatumType::TIMESTAMP:
                    {
                      time_t t = (time_t)col_val.datum.int_val;
                      std::tm tm_struct;
                      gmtime_r(&t, &tm_struct);
                      char buf[20];
                      strftime(buf, 20, "%F %T", &tm_struct);
                      std::cout << buf;
                      break;
                    }
                  case TDatumType::DATE:
                    {
                      time_t t = (time_t)col_val.datum.int_val;
                      std::tm tm_struct;
                      gmtime_r(&t, &tm_struct);
                      char buf[11];
                      strftime(buf, 11, "%F", &tm_struct);
                      std::cout << buf;
                      break;
                    }
                  default:
                    std::cerr << "Unknown column type." << std::endl;
                }
            }
          std::cout << std::endl;
          }
        }
        catch (MapDException &e) {
          std::cerr << e.error_msg << std::endl;
        }
      } else if (!strncmp(line,"/historylen",11)) {
          /* The "/historylen" command will change the history len. */
          int len = atoi(line+11);
          linenoiseHistorySetMaxLen(len);
      } else if (line[0] == '/') {
          printf("Unreconized command: %s\n", line);
      }
      free(line);
  }

  transport->close();
  return 0;
}
