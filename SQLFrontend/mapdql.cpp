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

#define INVALID_SESSION_ID -1

struct ClientContext {
  std::string user_name;
  std::string passwd;
  std::string db_name;
  SessionId session;
};

static void
process_backslash_commands(char *command, MapDClient &client, ClientContext &context)
{
  switch (command[1]) {
    case 'h':
      std::cout << "\\u List all users.\n";
      std::cout << "\\l List all databases.\n";
      std::cout << "\\t List all tables.\n";
      std::cout << "\\d <table> List all columns of table.\n";
      std::cout << "\\c <database> <user> <password>.\n";
      std::cout << "\\multiline Set multi-line command line mode.\n";
      std::cout << "\\singleline Set single-line command line mode.\n";
      std::cout << "\\historylen <number> Set history buffer size (default 100).\n";
      std::cout << "\\timing Print timing information.\n";
      std::cout << "\\notiming Do not print timing information.\n";
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
        try {
        client.getColumnTypes(_return, context.session, table_name);
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
        }
        catch (MapDException &e) {
          std::cerr << e.error_msg << std::endl;
        }
        catch (TException &te) {
          std::cerr << "Thrift error: " << te.what() << std::endl;
        }
        return;
      }
    case 't':
      {
        std::vector<std::string> _return;
        try {
          client.getTables(_return, context.session);
          for (auto p : _return)
            std::cout << p << std::endl;
        }
        catch (MapDException &e) {
          std::cerr << e.error_msg << std::endl;
        }
        catch (TException &te) {
          std::cerr << "Thrift error: " << te.what() << std::endl;
        }
        return;
      }
    case 'c':
      {
        if (command[2] != ' ') {
          std::cerr << "Invalid \\c command usage.  Do \\c <database> <user> <password>" << std::endl;
          return;
        }
        char *db = strtok(command+3, " ");
        char *user = strtok(NULL, " ");
        char *passwd = strtok(NULL, " ");
        if (db == NULL || user == NULL || passwd == NULL) {
          std::cerr << "Invalid \\c command usage.  Do \\c <database> <user> <password>" << std::endl;
          return;
        }
        if (context.session != INVALID_SESSION_ID) {
          try {
            client.disconnect(context.session);
            std::cout << "Disconnected from database " << context.db_name << std::endl;
          }
          catch (MapDException &e) {
            std::cerr << e.error_msg << std::endl;
          }
          catch (TException &te) {
            std::cerr << "Thrift error: " << te.what() << std::endl;
          }
        }
        context.db_name = db;
        context.user_name = user;
        context.passwd = passwd;
        try {
          context.session = client.connect(context.user_name, context.passwd, context.db_name);
          std::cout << "Connected to database " << context.db_name << std::endl;
        }
        catch (MapDException &e) {
          std::cerr << e.error_msg << std::endl;
          context.session = INVALID_SESSION_ID;
        }
        catch (TException &te) {
          std::cerr << "Thrift error: " << te.what() << std::endl;
        }
      }
      break;
    case 'u':
      {
        std::vector<std::string> _return;
        try {
          client.getUsers(_return);
          for (auto p : _return)
            std::cout << p << std::endl;
        }
        catch (TException &te) {
          std::cerr << "Thrift error: " << te.what() << std::endl;
        }
        return;
      }
    case 'l':
      {
        std::vector<DBInfo> _return;
        try {
          client.getDatabases(_return);
          std::cout << "Database | Owner" << std::endl;
          for (auto p : _return)
            std::cout << p.db_name << " | " << p.db_owner << std::endl;
        }
        catch (TException &te) {
          std::cerr << "Thrift error: " << te.what() << std::endl;
        }
        return;
      }
    default:
      std::cerr << "Invalid backslash command: " << command << std::endl;
  }
}

int main(int argc, char **argv) {
  std::string server_host("localhost");
  int port = 9091;
  std::string delimiter("|");
  bool print_header = true;
  bool print_timing = false;
  char *line;
  QueryResult _return;
  ClientContext context;
  std::string db_name;
  std::string user_name;
  std::string passwd;

  context.session = INVALID_SESSION_ID;

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
    ("no-header,n", "Do not print query result header")
    ("timing,t", "Print timing information")
    ("delimiter,d", po::value<std::string>(&delimiter), "Field delimiter in row output (default is |)")
    ("db", po::value<std::string>(&context.db_name), "Database name")
    ("user,u", po::value<std::string>(&context.user_name), "User name")
    ("passwd,p", po::value<std::string>(&context.passwd), "Password")
    ("server,s", po::value<std::string>(&server_host), "MapD Server Hostname (default localhost)")
    ("port", po::value<int>(&port), "Port number (default 9091)");

	po::variables_map vm;
	po::positional_options_description positionalOptions;
  positionalOptions.add("db", 1);

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
		if (vm.count("help")) {
			std::cout << "Usage: mapdql [<database>][{--user|-u} <user>][{--passwd|-p} <password>][--port <port number>] [{-s|--server} <server host>] [{--no-header|-n}] [{--delimiter|-d}]\n";
			return 0;
		}
    if (vm.count("no-header"))
      print_header = false;
    if (vm.count("timing"))
      print_timing = true;
    if (vm.count("db") && (!vm.count("user") || !vm.count("passwd"))) {
      std::cerr << "Must specify a user name and password to access database " << context.db_name << std::endl;
      return 1;
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

  if (context.db_name.empty()) {
    std::cout << "Not connected to any database.  Only \\u and \\l commands are allowed in this state.  See \\h for help." << std::endl;
  } else {
    try {
      context.session = client.connect(context.user_name, context.passwd, context.db_name);
      std::cout << "Connected to database " << context.db_name << std::endl;
    }
    catch (MapDException &e) {
      std::cerr << e.error_msg << std::endl;
    }
    catch (TException &te) {
      std::cerr << "Thrift error: " << te.what() << std::endl;
    }
  }

  /* Set the completion callback. This will be called every time the
   * user uses the <tab> key. */
  linenoiseSetCompletionCallback(completion);

  /* Load history from file. The history file is just a plain text file
   * where entries are separated by newlines. */
  linenoiseHistoryLoad("mapdql_history.txt"); /* Load the history at startup */
  /* default to multi-line mode */
  linenoiseSetMultiLine(1);

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
        if (context.session == INVALID_SESSION_ID) {
          std::cerr << "Not connected to any MapD databases." << std::endl;
          continue;
        }
        try {
          client.sql_execute(_return, context.session, line);
          if (_return.proj_info.empty() || _return.rows.empty())
            continue;
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
          if (print_timing)
            std::cout << "Execution time: " << _return.execution_time_ms << " miliseconds" << std::endl;
        }
        catch (MapDException &e) {
          std::cerr << e.error_msg << std::endl;
        }
        catch (TException &te) {
          std::cerr << "Thrift error: " << te.what() << std::endl;
        }
      } else if (!strncmp(line,"\\historylen",11)) {
          /* The "/historylen" command will change the history len. */
          int len = atoi(line+11);
          linenoiseHistorySetMaxLen(len);
      } else if (!strncmp(line,"\\multiline", 10)) {
        linenoiseSetMultiLine(1);
      } else if (!strncmp(line,"\\singleline", 11)) {
        linenoiseSetMultiLine(0);
      } else if (!strncmp(line,"\\keycodes", 9)) {
        linenoisePrintKeyCodes();
      } else if (!strncmp(line,"\\timing", 7)) {
        print_timing = true;
      } else if (!strncmp(line,"\\notiming", 9)) {
        print_timing = false;
      } else if (line[0] == '\\' && line[1] == 'q')
        break;
      else if (line[0] == '\\') {
          process_backslash_commands(line, client, context);
      }
      free(line);
  }

  if (context.session != INVALID_SESSION_ID) {
    try {
      client.disconnect(context.session);
    }
    catch (MapDException &e) {
      std::cerr << e.error_msg << std::endl;
    }
    catch (TException &te) {
      std::cerr << "Thrift error: " << te.what() << std::endl;
    }
  }
  transport->close();
  return 0;
}
