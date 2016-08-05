#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include <unistd.h>
#include <signal.h>
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "DataMgr/DataMgr.h"
#include "Catalog/Catalog.h"
#include "Parser/parser.h"
#include "Analyzer/Analyzer.h"
#include "Parser/ParserNode.h"
#include "Planner/Planner.h"
#include "QueryEngine/Execute.h"
#include "MapDRelease.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Planner;
using namespace Fragmenter_Namespace;

void process_backslash_commands(const string& command, const Catalog& cat, SysCatalog& syscat) {
  switch (command[1]) {
    case 'h':
      cout << "\\d <table> List all columns of table.\n";
      cout << "\\d <table>.<column> dump all chunk stats for column.\n";
      cout << "\\t List all tables.\n";
      cout << "\\u List all users. \n";
      cout << "\\l List all databases.\n";
      cout << "\\q Quit.\n";
      return;
    case 'd': {
      if (command[2] != ' ')
        throw runtime_error("Correct use is \\d <table> or \\d <table>.<column>.");
      size_t dot = command.find_first_of('.');
      if (dot == string::npos) {
        string table_name = command.substr(3);
        const TableDescriptor* td = cat.getMetadataForTable(table_name);
        if (td == nullptr)
          throw runtime_error("Table " + table_name + " does not exist.");
        list<const ColumnDescriptor*> col_list = cat.getAllColumnMetadataForTable(td->tableId, false, true);
        cout << "TableId|ColumnId|ColumnName|Type|Dimension|Scale|NotNull|Compression|comp_param|size|chunks\n";

        for (auto cd : col_list) {
          cout << cd->tableId << "|";
          cout << cd->columnId << "|";
          cout << cd->columnName << "|";
          cout << cd->columnType.get_type_name() << "|";
          cout << cd->columnType.get_dimension() << "|";
          cout << cd->columnType.get_scale() << "|";
          if (cd->columnType.get_notnull())
            cout << "true|";
          else
            cout << "false|";
          cout << cd->columnType.get_compression_name() << "|";
          cout << cd->columnType.get_comp_param() << "|";
          cout << cd->columnType.get_size() << "|";
          cout << cd->chunks << "\n";
        }
      } else {
        string table_name = command.substr(3, dot - 3);
        string col_name = command.substr(dot + 1);
        const TableDescriptor* td = cat.getMetadataForTable(table_name);
        if (td == nullptr)
          throw runtime_error("Table " + table_name + " does not exist.");
        const ColumnDescriptor* cd = cat.getMetadataForColumn(td->tableId, col_name);
        if (cd == nullptr)
          throw runtime_error("Column " + col_name + " does not exist.");
        TableInfo query_info = td->fragmenter->getFragmentsForQuery();
        cout << "Chunk Stats for " + table_name + "." + col_name << ":" << std::endl;
        for (const auto& frag : query_info.fragments) {
          auto chunk_meta_it = frag.chunkMetadataMap.find(cd->columnId);
          const ChunkMetadata& chunkMetadata = chunk_meta_it->second;
          cout << "(" << cat.get_currentDB().dbId << "," << td->tableId << "," << cd->columnId << "," << frag.fragmentId
               << ") ";
          cout << "numBytes:" << chunkMetadata.numBytes;
          cout << " numElements:" << chunkMetadata.numElements;
          cout << " has_nulls:" << chunkMetadata.chunkStats.has_nulls;
          SQLTypes t = cd->columnType.get_type() == kARRAY ? cd->columnType.get_subtype() : cd->columnType.get_type();
          switch (t) {
            case kBOOLEAN:
              cout << " min:" << (short)chunkMetadata.chunkStats.min.tinyintval;
              cout << " max:" << (short)chunkMetadata.chunkStats.max.tinyintval;
              break;
            case kSMALLINT:
              cout << " min:" << chunkMetadata.chunkStats.min.smallintval;
              cout << " max:" << chunkMetadata.chunkStats.max.smallintval;
              break;
            case kINT:
            case kTEXT:
            case kVARCHAR:
            case kCHAR:
              if (cd->columnType.is_string() && cd->columnType.get_compression() != kENCODING_DICT)
                break;
              cout << " min:" << chunkMetadata.chunkStats.min.intval;
              cout << " max:" << chunkMetadata.chunkStats.max.intval;
              break;
            case kBIGINT:
            case kNUMERIC:
            case kDECIMAL:
              cout << " min:" << chunkMetadata.chunkStats.min.bigintval;
              cout << " max:" << chunkMetadata.chunkStats.max.bigintval;
              break;
            case kFLOAT:
              cout << " min:" << chunkMetadata.chunkStats.min.floatval;
              cout << " max:" << chunkMetadata.chunkStats.max.floatval;
              break;
            case kDOUBLE:
              cout << " min:" << chunkMetadata.chunkStats.min.doubleval;
              cout << " max:" << chunkMetadata.chunkStats.max.doubleval;
              break;
            case kTIME:
            case kTIMESTAMP:
            case kDATE:
              cout << " min:" << chunkMetadata.chunkStats.min.timeval;
              cout << " max:" << chunkMetadata.chunkStats.max.timeval;
              break;
            default:
              break;
          }
          cout << std::endl;
        }
      }
    } break;
    case 't': {
      cout << "TableId|TableName|NColumns|IsView|IsMaterialized|ViewSQL|Fragments|FragType|FragSize|PageSize|"
              "Partitions|Storage|Refresh|Ready\n";
      list<const TableDescriptor*> table_list = cat.getAllTableMetadata();
      std::string storage_name[] = {"DISK", "GPU", "CPU"};
      std::string refresh_name[] = {"MANUAL", "AUTO", "IMMEDIATE"};
      for (auto td : table_list) {
        cout << td->tableId << "|";
        cout << td->tableName << "|";
        cout << td->nColumns << "|";
        if (td->isView)
          cout << "true|";
        else
          cout << "false|";
        if (td->isMaterialized)
          cout << "true|";
        else
          cout << "false|";
        cout << td->viewSQL << "|";
        cout << td->fragments << "|";
        cout << td->fragType << "|";
        cout << td->maxFragRows << "|";
        cout << td->fragPageSize << "|";
        cout << td->partitions << "|";
        cout << storage_name[td->storageOption] << "|";
        cout << refresh_name[td->refreshOption] << "|";
        if (td->isReady)
          cout << "true\n";
        else
          cout << "false\n";
      }
    } break;
    case 'l': {
      cout << "DatabaseId|DatabaseName|OwnerId\n";
      list<DBMetadata> db_list = syscat.getAllDBMetadata();
      for (auto d : db_list) {
        cout << d.dbId << "|";
        cout << d.dbName << "|";
        cout << d.dbOwner << "\n";
      }
    } break;
    case 'u': {
      cout << "UserId|UserName|IsSuper\n";
      list<UserMetadata> user_list = syscat.getAllUserMetadata();
      for (auto u : user_list) {
        cout << u.userId << "|";
        cout << u.userName << "|";
        if (u.isSuper)
          cout << "true\n";
        else
          cout << "false\n";
      }
    } break;
    case 'q':
      exit(0);
      break;
    default:
      throw runtime_error("Invalid backslash command.  See \\h");
  }
}

int main(int argc, char* argv[]) {
  string base_path;
  string db_name;
  string user_name;
  string passwd;
  bool debug = false;
  bool execute = false;
  bool timer = false;
  bool jit_debug = false;
  bool allow_loop_joins = false;
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ")(
      "path", po::value<string>(&base_path)->required(), "Directory path to Mapd catalogs")(
      "db", po::value<string>(&db_name), "Database name")(
      "user,u", po::value<string>(&user_name)->required(), "User name")(
      "passwd,p", po::value<string>(&passwd)->required(), "Password")("debug,d", "Verbose debug mode")(
      "allow-loop-joins", "Enable loop joins")(
      "jit-debug", "Enable debugger support for the JIT. The generated code can be found at /tmp/mapdquery")(
      "execute,e", "Execute queries")("version,v", "Print MapD Version")("timer,t", "Show query time information");

  po::positional_options_description positionalOptions;
  positionalOptions.add("path", 1);
  positionalOptions.add("db", 1);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
    if (vm.count("help")) {
      cout << "Usage: mapd -u <user name> -p <password> <catalog path> [<database name>][--version|-v]\n";
      return 0;
    }
    if (vm.count("version")) {
      cout << "MapD Version: " << MapDRelease << std::endl;
      return 0;
    }
    if (vm.count("debug"))
      debug = true;
    if (vm.count("execute"))
      execute = true;
    if (vm.count("timer"))
      timer = true;
    if (vm.count("jit-debug"))
      jit_debug = true;
    if (vm.count("allow-loop-joins"))
      allow_loop_joins = true;

    po::notify(vm);
  } catch (boost::program_options::error& e) {
    cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  if (!vm.count("db"))
    db_name = MAPD_SYSTEM_DB;

  if (!boost::filesystem::exists(base_path)) {
    cerr << "Catalog path " + base_path + " does not exist.\n";
    return 1;
  }
  std::string system_db_file = base_path + "/mapd_catalogs/mapd";
  if (!boost::filesystem::exists(system_db_file)) {
    cerr << "MapD not initialized at " + base_path + "\nPlease run initdb first.\n";
    return 1;
  }
  const auto lock_file = boost::filesystem::path(base_path) / "mapd_server_pid.lck";
  if (boost::filesystem::exists(lock_file)) {
    std::ifstream lockf;
    lockf.open(lock_file.c_str());
    pid_t pid;
    lockf >> pid;
    lockf.close();
    if (kill(pid, 0) == 0) {
      std::cerr << "Another MapD Server is running on the same MapD directory." << std::endl;
      return 1;
    }
  }
  std::ofstream lockf;
  lockf.open(lock_file.c_str(), std::ios::out | std::ios::trunc);
  lockf << getpid();
  lockf.close();

#ifdef HAVE_CUDA
  const bool use_gpus{true};
#else
  const bool use_gpus{false};
#endif
  auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(base_path + "/mapd_data/", 0, use_gpus, -1);
#ifdef HAVE_CALCITE
  auto calcite = std::make_shared<Calcite>(-1, base_path + "/mapd_data/");
#endif  // HAVE_CALCITE
  SysCatalog sys_cat(base_path,
                     dataMgr
#ifdef HAVE_CALCITE
                     ,
                     calcite
#endif  // HAVE_CALCITE
                     );
  UserMetadata user;
  if (!sys_cat.getMetadataForUser(user_name, user)) {
    cerr << "User " << user_name << " does not exist." << std::endl;
    return 1;
  }
  if (user.passwd != passwd) {
    cerr << "Invalid password for User " << user_name << std::endl;
    return 1;
  }
  DBMetadata db;
  if (!sys_cat.getMetadataForDB(db_name, db)) {
    cerr << "Database " << db_name << " does not exist." << std::endl;
    return 1;
  }
  if (!user.isSuper && user.userId != db.dbOwner) {
    cerr << "User " << user_name << " is not authorized to access database " << db_name << std::endl;
    return 1;
  }
  auto cat = std::make_shared<Catalog>(base_path,
                                       db,
                                       dataMgr
#ifdef HAVE_CALCITE
                                       ,
                                       calcite
#endif  // HAVE_CALCITE
                                       );
  SessionInfo session(cat, user, ExecutorDeviceType::GPU, 0);
  while (true) {
    try {
      cout << "mapd> ";
      string input_str;
      getline(cin, input_str);
      if (cin.eof()) {
        cout << std::endl;
        break;
      }
      if (input_str[0] == '\\') {
        process_backslash_commands(input_str, *cat, sys_cat);
        continue;
      }
      SQLParser parser;
      list<std::unique_ptr<Parser::Stmt>> parse_trees;
      string last_parsed;
      int numErrors = parser.parse(input_str, parse_trees, last_parsed);
      if (numErrors > 0)
        throw runtime_error("Syntax error at: " + last_parsed);
      for (const auto& stmt : parse_trees) {
        Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
        Parser::ExplainStmt* explain_stmt = dynamic_cast<Parser::ExplainStmt*>(ddl);
        if (ddl != nullptr && !explain_stmt)
          ddl->execute(session);
        else {
          auto dml = explain_stmt ? explain_stmt->get_stmt() : static_cast<const Parser::DMLStmt*>(stmt.get());
          Query query;
          dml->analyze(*cat, query);
          Optimizer optimizer(query, *cat);
          RootPlan* plan = optimizer.optimize();
          unique_ptr<RootPlan> plan_ptr(plan);  // make sure it's deleted
          if (debug)
            plan->print();
          if (execute) {
            if (explain_stmt != nullptr) {
              plan->set_plan_dest(Planner::RootPlan::Dest::kEXPLAIN);
            }
            auto executor = Executor::getExecutor(
                plan->get_catalog().get_currentDB().dbId, jit_debug ? "/tmp" : "", jit_debug ? "mapdquery" : "");
            ResultRows results({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU);
            ResultRows results_cpu({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU);
            {
              auto ms = measure<>::execution([&]() {
                results_cpu = executor->execute(plan,
                                                session,
                                                -1,
                                                true,
                                                ExecutorDeviceType::CPU,
                                                ExecutorOptLevel::Default,
                                                true,
                                                allow_loop_joins);
              });
              if (timer) {
                cout << "Query took " << ms << " ms to execute." << endl;
              }
            }
            if (cat->get_dataMgr().gpusPresent() && plan->get_stmt_type() == kSELECT) {
              ResultRows results_gpu({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::GPU);
              {
                auto ms = measure<>::execution([&]() {
                  results_gpu = executor->execute(plan,
                                                  session,
                                                  -1,
                                                  true,
                                                  ExecutorDeviceType::GPU,
                                                  ExecutorOptLevel::Default,
                                                  true,
                                                  allow_loop_joins);
                });
                if (timer) {
                  cout << "Query took " << ms << " ms to execute." << endl;
                }
              }
              results = results_gpu;
            } else {
              results = results_cpu;
            }
            while (true) {
              const auto crt_row = results.getNextRow(true, true);
              if (crt_row.empty()) {
                break;
              }
              cout << fixed << setprecision(13) << row_col_to_string(crt_row, 0, results.getColType(0));
              for (size_t i = 1; i < results.colCount(); ++i) {
                cout << "|" << row_col_to_string(crt_row, i, results.getColType(i));
              }
              cout << endl;
            }
          }
        }
      }
    } catch (std::exception& e) {
      std::cerr << "Exception: " << e.what() << "\n";
    }
  }
  return 0;
}
