#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "DataMgr/DataMgr.h"
#include "Partitioner/TablePartitionerMgr.h"
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

void
process_backslash_commands(const string &command, const Catalog &cat, SysCatalog &syscat)
{
	switch (command[1]) {
		case 'h':
			cout << "\\d <table> List all columns of table.\n";
			cout << "\\t List all tables.\n";
			cout << "\\u List all users. \n";
			cout << "\\l List all databases.\n";
			cout << "\\q Quit.\n";
			return;
		case 'd':
			{
				if (command[2] != ' ')
					throw runtime_error("Correct use is \\d <table>.");
				string table_name = command.substr(3);
				const TableDescriptor *td = cat.getMetadataForTable(table_name);
				if (td == nullptr)
					throw runtime_error("Table " + table_name + " does not exist.");
				list <const ColumnDescriptor *> col_list = cat.getAllColumnMetadataForTable(td->tableId);
				cout << "TableId|ColumnId|ColumnName|Type|Dimension|Scale|NotNull|Compression|comp_param|chunks\n";
				std::string SQLTypeName[] = { "NULL", "BOOLEAN", "CHAR", "VARCHAR", "NUMERIC", "DECIMAL", "INTEGER", "SMALLINT", "FLOAT", "DOUBLE", "TIME", "TIMESTAMP", "BIGINT", "TEXT" };
				std::string compression[] = { "NONE", "FIXED", "RL", "DIFF", "DICT", "SPARSE" };

				for (auto cd : col_list) {
					cout << cd->tableId << "|";
					cout << cd->columnId << "|";
					cout << cd->columnName << "|";
					cout << SQLTypeName[cd->columnType.type] << "|";
					cout << cd->columnType.dimension << "|";
					cout << cd->columnType.scale << "|";
					if (cd->columnType.notnull)
						cout << "true|";
					else
						cout << "false|";
					cout << compression[cd->compression] << "|";
					cout << cd->comp_param << "|";
					cout << cd->chunks << "\n";
				}
			}
			break;
		case 't':
			{
				cout << "TableId|TableName|NColumns|IsView|IsMaterialized|ViewSQL|Fragments|Partitions|Storage|Refresh|Ready\n";
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
	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
		("path", po::value<string>(&base_path)->required(), "Directory path to Mapd catalogs")
		("db", po::value<string>(&db_name), "Database name")
		("user,u", po::value<string>(&user_name)->required(), "User name")
		("passwd,p", po::value<string>(&passwd)->required(), "Password")
		("debug,d", "Verbose debug mode")
		("execute,e", "Execute queries");

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

    Data_Namespace::DataMgr dataMgr (2, base_path + "/mapd_data/"); 
    Partitioner_Namespace::TablePartitionerMgr partitionerMgr (&dataMgr, base_path);
	SysCatalog sys_cat(base_path);
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
	Catalog cat(base_path, user, db);
	while (true) {
		try {
			cout << "MapD > ";
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
					ddl->execute(cat);
				else {
					Parser::DMLStmt *dml = dynamic_cast<Parser::DMLStmt*>(stmt);
					Query query;
					dml->analyze(cat, query);
					Optimizer optimizer(query, cat);
					RootPlan *plan = optimizer.optimize();
					unique_ptr<RootPlan> plan_ptr(plan); // make sure it's deleted
					if (debug) plan->print();
					if (execute) {
						Executor executor(plan);
						const auto results = executor.execute();
						CHECK(results.size());
						cout << results[0];
						for (size_t i = 1; i < results.size(); ++i) {
							cout << ", " << results[i];
						}
						cout << endl;
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
