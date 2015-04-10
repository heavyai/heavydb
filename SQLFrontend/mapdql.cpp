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

static void
process_backslash_commands(const char *command, MapDClient &client)
{
  switch (command[1]) {
    case 'h':
      std::cout << "\\t List all tables.\n";
      std::cout << "\\d <table> List all columns of table.\n";
      std::cout << "\\q Quit.\n";
      return;
    case 'd':
      {
        if (command[2] != ' ') {
          std::cerr << "Invalid \\d command usage.  Do \\d <table name>" << std::endl;
          return;
        }
        ColumnTypes _return;
        std::string table_name(command+3);
        client.getColumnTypes(_return, table_name);
        for (auto p : _return) {
          std::cout << p.first << " ";
          switch (p.second.type) {
            case TDatumType::INT:
              std::cout << "INTEGER\n";
              break;
            case TDatumType::REAL:
              std::cout << "DOUBLE\n";
              break;
            case TDatumType::STR:
              std::cout << "STRING\n";
              break;
            case TDatumType::TIME:
              std::cout << "TIME\n";
              break;
            case TDatumType::TIMESTAMP:
              std::cout << "TIMESTAMP\n";
              break;
            case TDatumType::DATE:
              std::cout << "DATE\n";
              break;
            default:
              std::cerr << "Invalid Column Type.\n";
          }
        }
        return;
      }
    case 't':
      {
        std::vector<std::string> _return;
        client.getTables(_return);
        for (auto p : _return)
          std::cout << p << std::endl;
        return;
      }
    case 'q':
    default:
      std::cerr << "Invalid backslash command: " << command << std::endl;
  }
}

int main(int argc, char **argv) {
  std::string server_host("localhost");
  int port = 9091;
  std::string delimiter("|");
  bool print_header = true;
  char *line;
  QueryResult _return;

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
    ("no-header,n", "Do not print query result header")
    ("delimiter,d", po::value<std::string>(&delimiter), "Field delimiter in row output (default is |)")
    ("server,s", po::value<std::string>(&server_host), "MapD Server Hostname (default localhost)")
    ("port,p", po::value<int>(&port), "Port number (default 9091)");

	po::variables_map vm;
	po::positional_options_description positionalOptions;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
		if (vm.count("help")) {
			std::cout << "Usage: mapdql [{-p|--port} <port number>] [{-s|--server} <server host>] [{--no-header|-n}] [{--delimiter|-d}]\n";
			return 0;
		}
    if (vm.count("no-header"))
      print_header = false;

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
      {
        QueryResult empty;
        swap(_return, empty);
      }
      /* Do something with the string. */
      if (line[0] != '\0' && line[0] != '\\') {
          // printf("echo: '%s'\n", line);
          linenoiseHistoryAdd(line); /* Add to the history. */
          linenoiseHistorySave("mapdql_history.txt"); /* Save the history on disk. */
        try {
          client.select(_return, line);
          bool not_first = false;
          if (print_header) {
            for (auto p : _return.proj_info) {
              if (not_first)
                std::cout << delimiter;
              else
                not_first = true;
              std::cout << p.proj_name;
            }
            std::cout << std::endl;
          }
          for (auto row : _return.rows) {
            not_first = false;
            for (auto col_val : row.cols) {
              if (not_first)
                std::cout << delimiter;
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
      } else if (!strncmp(line,"\\historylen",11)) {
          /* The "/historylen" command will change the history len. */
          int len = atoi(line+11);
          linenoiseHistorySetMaxLen(len);
      } else if (line[0] == '\\' && line[1] == 'q')
        break;
      else if (line[0] == '\\') {
          process_backslash_commands(line, client);
      }
      free(line);
  }

  transport->close();
  return 0;
}
