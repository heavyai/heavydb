#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "DataMgr/DataMgr.h"
#include "Catalog/Catalog.h"
#include "Parser/parser.h"
#include "Analyzer/Analyzer.h"
#include "Parser/ParserNode.h"
#include "Planner/Planner.h"
#include "QueryEngine/Execute.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Planner;
using namespace Fragmenter_Namespace;

void
process_backslash_commands(const string &command, const Catalog &cat, SysCatalog &syscat)
{
	switch (command[1]) {
		case 'h':
			cout << "\\d <table> List all columns of table.\n";
      cout << "\\d <table>.<column> dump all chunk stats for column.\n";
			cout << "\\t List all tables.\n";
			cout << "\\u List all users. \n";
			cout << "\\l List all databases.\n";
			cout << "\\q Quit.\n";
			return;
		case 'd':
			{
				if (command[2] != ' ')
					throw runtime_error("Correct use is \\d <table> or \\d <table>.<column>.");
        size_t dot = command.find_first_of('.');
        if (dot == string::npos) {
          string table_name = command.substr(3);
          const TableDescriptor *td = cat.getMetadataForTable(table_name);
          if (td == nullptr)
            throw runtime_error("Table " + table_name + " does not exist.");
          list <const ColumnDescriptor *> col_list = cat.getAllColumnMetadataForTable(td->tableId);
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
          string col_name = command.substr(dot+1);
          const TableDescriptor *td = cat.getMetadataForTable(table_name);
          if (td == nullptr)
            throw runtime_error("Table " + table_name + " does not exist.");
          const ColumnDescriptor *cd = cat.getMetadataForColumn(td->tableId, col_name);
          if (cd == nullptr)
            throw runtime_error("Column " + col_name + " does not exist.");
          QueryInfo query_info;
          td->fragmenter->getFragmentsForQuery(query_info);
          cout << "Chunk Stats for " + table_name + "." + col_name << ":" << std::endl;
          for (const auto& frag : query_info.fragments) {
            auto chunk_meta_it = frag.chunkMetadataMap.find(cd->columnId);
            const ChunkMetadata &chunkMetadata = chunk_meta_it->second;
            cout << "(" << cat.get_currentDB().dbId << "," << td->tableId << "," << cd->columnId << "," << frag.fragmentId << ") ";
            cout << "numBytes:" << chunkMetadata.numBytes;
            cout << " numElements:" << chunkMetadata.numElements;
            switch (cd->columnType.get_type()) {
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
			}
			break;
		case 't':
			{
				cout << "TableId|TableName|NColumns|IsView|IsMaterialized|ViewSQL|Fragments|FragType|FragSize|PageSize|Partitions|Storage|Refresh|Ready\n";
				list<const TableDescriptor *> table_list = cat.getAllTableMetadata();
				std::string storage_name[] = { "DISK", "GPU", "CPU" };
				std::string refresh_name[] = { "MANUAL", "AUTO", "IMMEDIATE" };
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
						cout  << "true\n";
					else
						cout << "false\n";
				}
			}
			break;
		case 'l':
			{
				cout << "DatabaseId|DatabaseName|OwnerId\n";
				list<DBMetadata> db_list = syscat.getAllDBMetadata();
				for (auto d : db_list) {
					cout << d.dbId << "|";
					cout << d.dbName << "|";
					cout << d.dbOwner << "\n";
				}
			}
			break;
		case 'u':
			{
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
			}
			break;
		case 'q':
			exit(0);
			break;
		default:
			throw runtime_error("Invalid backslash command.  See \\h");
	}
}

int
main(int argc, char* argv[])
{
	string base_path;
	string db_name;
	string user_name;
	string passwd;
	bool debug = false;
	bool execute = false;
	bool timer = false;
	bool jit_debug = false;
	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
		("path", po::value<string>(&base_path)->required(), "Directory path to Mapd catalogs")
		("db", po::value<string>(&db_name), "Database name")
		("user,u", po::value<string>(&user_name)->required(), "User name")
		("passwd,p", po::value<string>(&passwd)->required(), "Password")
		("debug,d", "Verbose debug mode")
		("jit-debug", "Enable debugger support for the JIT. The generated code can be found at /tmp/mapdquery")
		("execute,e", "Execute queries")
		("timer,t", "Show query time information");

	po::positional_options_description positionalOptions;
	positionalOptions.add("path", 1);
	positionalOptions.add("db", 1);

	po::variables_map vm;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: mapd -u <user name> -p <password> <catalog path> [<database name>]\n";
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

		po::notify(vm);
	}
	catch (boost::program_options::error &e)
	{
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

	Data_Namespace::DataMgr dataMgr (base_path + "/mapd_data/", true); // second param is to set up gpu buffer pool - yes in this case
	SysCatalog sys_cat(base_path, dataMgr);
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
	Catalog cat(base_path, db, dataMgr);
  SessionInfo session(std::shared_ptr<Catalog>(&cat), user);
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
				process_backslash_commands(input_str, cat, sys_cat);
				continue;
			}
			SQLParser parser;
			list<Parser::Stmt*> parse_trees;
			string last_parsed;
			int numErrors = parser.parse(input_str, parse_trees, last_parsed);
			if (numErrors > 0)
				throw runtime_error("Syntax error at: " + last_parsed);
			for (auto stmt : parse_trees) {
				unique_ptr<Stmt> stmt_ptr(stmt); // make sure it's deleted
				Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt *>(stmt);
				if ( ddl != nullptr)
					ddl->execute(session);
				else {
					Parser::DMLStmt *dml = dynamic_cast<Parser::DMLStmt*>(stmt);
					Query query;
					dml->analyze(cat, query);
					Optimizer optimizer(query, cat);
					RootPlan *plan = optimizer.optimize();
					unique_ptr<RootPlan> plan_ptr(plan); // make sure it's deleted
					if (debug) plan->print();
					if (execute) {
						auto executor = Executor::getExecutor(plan->get_catalog().get_currentDB().dbId, jit_debug ? "/tmp" : "", jit_debug ? "mapdquery" : "");
						ResultRows results({}, nullptr, nullptr);
						ResultRows results_cpu({}, nullptr, nullptr);
						{
							auto ms = measure<>::execution([&]() {
								results_cpu = executor->execute(plan, true, ExecutorDeviceType::CPU);
							});
							if (timer) {
								cout << "Query took " << ms << " ms to execute." << endl;
							}
						}
						if (cat.get_dataMgr().gpusPresent() && plan->get_stmt_type() == kSELECT) {
							ResultRows results_gpu({}, nullptr, nullptr);
							{
								auto ms = measure<>::execution([&]() {
									results_gpu = executor->execute(plan, true, ExecutorDeviceType::GPU);
								});
								if (timer) {
									cout << "Query took " << ms << " ms to execute." << endl;
								}
							}
							results = results_gpu;
							//CHECK(results_cpu == results_gpu);
						} else {
							results = results_cpu;
						}
						if (!results.empty()) {
							for (size_t row_idx = 0; row_idx < results.size(); ++row_idx) {
								cout << fixed << setprecision(13) << row_col_to_string(results, row_idx, 0);
								for (size_t i = 1; i < results.colCount(); ++i) {
									cout << "|" << row_col_to_string(results, row_idx, i);
								}
								cout << endl;
							}
						}
					}
				}
			}
		}
		catch (std::exception &e)
		{
			std::cerr << "Exception: " << e.what() << "\n";
		}
	}
	return 0;
}
