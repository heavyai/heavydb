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
#include <fstream>
#include <boost/algorithm/string/join.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>

#include "gen-cpp/MapD.h"
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include "linenoise.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

const std::string MapDQLRelease("0.1");

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
  TTransport &transport;
  MapDClient &client;
  TSessionId session;
  TQueryResult query_return;
  std::vector<std::string> names_return;
  std::vector<TDBInfo> dbinfos_return;
  TTableDescriptor columns_return;
  TRowDescriptor rowdesc_return;
  TExecuteMode::type execution_mode;
  std::string version;

  ClientContext(TTransport &t, MapDClient &c) : transport(t), client(c), session(INVALID_SESSION_ID), execution_mode(TExecuteMode::GPU) {}
};

enum ThriftService {
  kCONNECT,
  kDISCONNECT,
  kSQL,
  kGET_COLUMNS,
  kGET_TABLES,
  kGET_DATABASES,
  kGET_USERS,
  kSET_EXECUTION_MODE,
  kGET_VERSION,
  kGET_ROW_DESC
};

static bool
thrift_with_retry(ThriftService which_service, ClientContext &context, char *arg)
{
  try {
    switch (which_service) {
      case kCONNECT:
        context.session = context.client.connect(context.user_name, context.passwd, context.db_name);
        break;
      case kDISCONNECT:
        context.client.disconnect(context.session);
        break;
      case kSQL:
        context.client.sql_execute(context.query_return, context.session, arg);
        break;
      case kGET_COLUMNS:
        context.client.get_table_descriptor(context.columns_return, context.session, arg);
        break;
      case kGET_ROW_DESC:
        context.client.get_row_descriptor(context.rowdesc_return, context.session, arg);
        break;
      case kGET_TABLES:
        context.client.get_tables(context.names_return, context.session);
        break;
      case kGET_DATABASES:
        context.client.get_databases(context.dbinfos_return);
        break;
      case kGET_USERS:
        context.client.get_users(context.names_return);
        break;
      case kSET_EXECUTION_MODE:
        context.client.set_execution_mode(context.session, context.execution_mode);
        break;
      case kGET_VERSION:
        context.client.get_version(context.version);
        break;
    }
  }
  catch (TMapDException &e) {
    std::cerr << e.error_msg << std::endl;
    return false;
  }
  catch (TException &te) {
    try {
      context.transport.open();
      if (which_service == kDISCONNECT)
        return false;
      if (which_service != kCONNECT) {
        if (!thrift_with_retry(kCONNECT, context, nullptr))
          return false;
      }
      return thrift_with_retry(which_service, context, arg);
    }
    catch (TException &te1) {
      std::cerr << "Thrift error: " << te1.what() << std::endl;
      return false;
    }
  }
  return true;
}

#define LOAD_PATCH_SIZE  10000
static void
copy_table(char *filepath, char *table, ClientContext &context)
{
  if (context.session == INVALID_SESSION_ID) {
    std::cerr << "Not connected to any databases." << std::endl;
    return;
  }
  if (!boost::filesystem::exists(filepath)) {
    std::cerr << "File does not exist." << std::endl;
    return;
  }
  if (!thrift_with_retry(kGET_COLUMNS, context, table)) {
    std::cerr << "Cannot connect to table." << std::endl;
    return;
  }
  const TTableDescriptor &table_desc = context.columns_return;
  std::ifstream infile(filepath);
  std::string line;
  const char *delim = ",";
  int l = strlen(filepath);
  if (l >= 4 && strcmp(filepath+l-4, ".tsv") == 0) {
    delim = "\t";
  }
  std::vector<TStringRow> input_rows;
  TStringRow row;
  boost::char_separator<char> sep{delim, "", boost::keep_empty_tokens};
  try {
    while (std::getline(infile, line)) {
      {
        std::vector<TStringValue> empty;
        row.cols.swap(empty);
      }
      boost::tokenizer<boost::char_separator<char>> tok{line, sep};
      for (const auto &s : tok) {
        TStringValue ts;
        ts.str_val = s;
        ts.is_null = s.empty();
        row.cols.push_back(ts);
      }
      /*
      std::cout << "Row: ";
      for (const auto &p : row.cols) {
        std::cout << p.str_val << ", ";
      }
      std::cout << std::endl;
      */
      if (row.cols.size() != table_desc.size()) {
        std::cerr << "Incorrect number of columns: (" << row.cols.size() << " vs " << table_desc.size() << ") " << line << std::endl;
        continue;
      }
      input_rows.push_back(row);
      if (input_rows.size() >= LOAD_PATCH_SIZE) {
        try {
          context.client.load_table(context.session, table, input_rows);
        }
        catch (TMapDException &e) {
          std::cerr << e.error_msg << std::endl;
        }
        {
          std::vector<TStringRow> empty;
          input_rows.swap(empty);
        }
      }
    }
    if (input_rows.size() > 0)
      context.client.load_table(context.session, table, input_rows);
  }
  catch (TMapDException &e) {
    std::cerr << e.error_msg << std::endl;
  }
  catch (TException &te) {
    std::cerr << "Thrift error: " << te.what() << std::endl;
  }
}

static void
process_backslash_commands(char *command, ClientContext &context)
{
  switch (command[1]) {
    case 'h':
      std::cout << "\\u List all users.\n";
      std::cout << "\\l List all databases.\n";
      std::cout << "\\t List all tables.\n";
      std::cout << "\\d <table> List all columns of table.\n";
      std::cout << "\\c <database> <user> <password>.\n";
      std::cout << "\\gpu Execute in GPU mode's.\n";
      std::cout << "\\cpu Execute in CPU mode's.\n";
      std::cout << "\\hybrid Execute in Hybrid mode.\n";
      std::cout << "\\multiline Set multi-line command line mode.\n";
      std::cout << "\\singleline Set single-line command line mode.\n";
      std::cout << "\\historylen <number> Set history buffer size (default 100).\n";
      std::cout << "\\timing Print timing information.\n";
      std::cout << "\\notiming Do not print timing information.\n";
      std::cout << "\\version Print MapD Server version.\n";
      std::cout << "\\copy <file path> <table> Copy data from file to table.\n";
      std::cout << "\\q Quit.\n";
      return;
    case 'd':
      {
        if (command[2] != ' ') {
          std::cerr << "Invalid \\d command usage.  Do \\d <table name>" << std::endl;
          return;
        }
        std::string table_name(command+3);
        if (thrift_with_retry(kGET_ROW_DESC, context, command+3))
          for (auto p : context.rowdesc_return) {
            std::cout << p.col_name << " ";
            switch (p.col_type.type) {
              case TDatumType::INT:
                std::cout << "INTEGER";
                break;
              case TDatumType::REAL:
                std::cout << "DOUBLE";
                break;
              case TDatumType::STR:
                std::cout << "STRING";
                break;
              case TDatumType::TIME:
                std::cout << "TIME";
                break;
              case TDatumType::TIMESTAMP:
                std::cout << "TIMESTAMP";
                break;
              case TDatumType::DATE:
                std::cout << "DATE";
                break;
              case TDatumType::BOOL:
                std::cout << "BOOLEAN";
                break;
              default:
                std::cerr << "Invalid Column Type.";
            }
            if (p.col_type.is_array)
              std::cout << "[]";
            std::cout << "\n";
          }
        return;
      }
    case 't':
      {
        if (thrift_with_retry(kGET_TABLES, context, nullptr))
          for (auto p : context.names_return)
            std::cout << p << std::endl;
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
          if (thrift_with_retry(kDISCONNECT, context, nullptr))
            std::cout << "Disconnected from database " << context.db_name << std::endl;
        }
        context.db_name = db;
        context.user_name = user;
        context.passwd = passwd;
        if (thrift_with_retry(kCONNECT, context, nullptr)) {
          std::cout << "User " << context.user_name << " connected to database " << context.db_name << std::endl;
        }
      }
      break;
    case 'u':
      {
        if (thrift_with_retry(kGET_USERS, context, nullptr))
          for (auto p : context.names_return)
            std::cout << p << std::endl;
        return;
      }
    case 'l':
      {
        if (thrift_with_retry(kGET_DATABASES, context, nullptr)) {
          std::cout << "Database | Owner" << std::endl;
          for (auto p : context.dbinfos_return)
            std::cout << p.db_name << " | " << p.db_owner << std::endl;
        }
        return;
      }
    default:
      std::cerr << "Invalid backslash command: " << command << std::endl;
  }
}

std::string datum_to_string(const TDatum& datum, const TTypeInfo& type_info) {
  if (type_info.is_array) {
    std::vector<std::string> elem_strs;
    elem_strs.reserve(datum.arr_val.size());
    for (const auto& elem_datum : datum.arr_val) {
      TTypeInfo elem_type_info { type_info };
      elem_type_info.is_array = false;
      elem_strs.push_back(datum_to_string(elem_datum, elem_type_info));
    }
    return "{" + boost::algorithm::join(elem_strs, ", ") + "}";
  }
  switch (type_info.type) {
  case TDatumType::INT:
    return std::to_string(datum.int_val);
  case TDatumType::REAL:
    return std::to_string(datum.real_val);
  case TDatumType::STR:
    return datum.str_val;
  case TDatumType::TIME: {
    time_t t = datum.int_val;
    std::tm tm_struct;
    gmtime_r(&t, &tm_struct);
    char buf[9];
    strftime(buf, 9, "%T", &tm_struct);
    return buf;
  }
  case TDatumType::TIMESTAMP: {
    time_t t = datum.int_val;
    std::tm tm_struct;
    gmtime_r(&t, &tm_struct);
    char buf[20];
    strftime(buf, 20, "%F %T", &tm_struct);
    return buf;
  }
  case TDatumType::DATE: {
    time_t t = datum.int_val;
    std::tm tm_struct;
    gmtime_r(&t, &tm_struct);
    char buf[11];
    strftime(buf, 11, "%F", &tm_struct);
    return buf;
  }
  case TDatumType::BOOL:
    return (datum.int_val ? "true" : "false");
  default:
    return "Unknown column type.\n";
  }
}

int main(int argc, char **argv) {
  std::string server_host("localhost");
  int port = 9091;
  std::string delimiter("|");
  bool print_header = true;
  bool print_timing = false;
  char *line;
  TQueryResult _return;
  std::string db_name;
  std::string user_name;
  std::string passwd;
  shared_ptr<TTransport> socket(new TSocket(server_host, port));
  shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  MapDClient c(protocol);
  ClientContext context(*transport, c);

  context.session = INVALID_SESSION_ID;

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
    ("version,v", "Print mapdql version number")
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
		if (vm.count("version")) {
			std::cout << "MapDQL Version: " << MapDQLRelease << std::endl;
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

  transport->open();

  if (context.db_name.empty()) {
    std::cout << "Not connected to any database.  Only \\u and \\l commands are allowed in this state.  See \\h for help." << std::endl;
  } else {
    if (thrift_with_retry(kCONNECT, context, nullptr))
      std::cout << "User " << context.user_name << " connected to database " << context.db_name << std::endl;
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
        TQueryResult empty;
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
        if (thrift_with_retry(kSQL, context, line)) {
          if (context.query_return.row_set.row_desc.empty() || context.query_return.row_set.rows.empty())
            continue;
          bool not_first = false;
          if (print_header) {
            for (auto p : context.query_return.row_set.row_desc) {
              if (not_first)
                std::cout << delimiter;
              else
                not_first = true;
              std::cout << p.col_name;
            }
            std::cout << std::endl;
          }
          for (auto row : context.query_return.row_set.rows) {
            not_first = false;
            int i = 0;
            for (auto col_val : row.cols) {
              if (not_first)
                std::cout << delimiter;
              else
                not_first = true;
              if (col_val.is_null)
                std::cout << "NULL";
              else
                std::cout << datum_to_string(col_val.datum, context.query_return.row_set.row_desc[i].col_type);
              i++;
            }
          std::cout << std::endl;
          }
          if (print_timing) {
            std::cout << context.query_return.row_set.rows.size() << " rows returned." << std::endl;
            std::cout << "Execution time: " << context.query_return.execution_time_ms << " miliseconds" << std::endl;
          }
        }
      } else if (!strncmp(line,"\\cpu",4)) {
        context.execution_mode = TExecuteMode::CPU;
        (void)thrift_with_retry(kSET_EXECUTION_MODE, context, nullptr);
      } else if (!strncmp(line,"\\gpu",4)) {
        context.execution_mode = TExecuteMode::GPU;
        (void)thrift_with_retry(kSET_EXECUTION_MODE, context, nullptr);
      } else if (!strncmp(line,"\\hybrid",5)) {
        context.execution_mode = TExecuteMode::HYBRID;
        (void)thrift_with_retry(kSET_EXECUTION_MODE, context, nullptr);
      } else if (!strncmp(line,"\\version",8)) {
        if (thrift_with_retry(kGET_VERSION, context, nullptr)) {
          std::cout << "MapD Server Version: " << context.version << std::endl;
        } else {
          std::cout << "Cannot connect to MapD Server." << std::endl;
        }
      } else if (!strncmp(line,"\\copy",5)) {
        char *filepath = strtok(line+6, " ");
        char *table = strtok(NULL, " ");
        copy_table(filepath, table, context);
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
          process_backslash_commands(line, context);
      }
      free(line);
  }

  if (context.session != INVALID_SESSION_ID) {
    if (thrift_with_retry(kDISCONNECT, context, nullptr)) {
      std::cout << "User " << context.user_name << " disconnected from database " << context.db_name << std::endl;
    }
  }
  transport->close();
  return 0;
}
