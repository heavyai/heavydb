#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include <boost/functional/hash.hpp>
#include "../Catalog/Catalog.h"
#include "../Parser/parser.h"
#include "../Analyzer/Analyzer.h"
#include "../Parser/ParserNode.h"
#include "../DataMgr/DataMgr.h"
#include "../Fragmenter/Fragmenter.h"
#include "PopulateTableRandom.h"
#include "ScanTable.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Fragmenter_Namespace;

#define BASE_PATH "/tmp"

namespace {
	SysCatalog *gsys_cat = nullptr;
	Catalog *gcat = nullptr;
	Data_Namespace::DataMgr *dataMgr = nullptr;

	void run_ddl(const string &input_str)
	{
		SQLParser parser;
		list<Parser::Stmt*> parse_trees;
		string last_parsed;
		CHECK_EQ(parser.parse(input_str, parse_trees, last_parsed), 0);
		CHECK_EQ(parse_trees.size(), 1);
		auto stmt = parse_trees.front();
		unique_ptr<Stmt> stmt_ptr(stmt); // make sure it's deleted
		Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt *>(stmt);
		CHECK(ddl != nullptr);
		ddl->execute(*gcat);
	}

	class SQLTestEnv : public ::testing::Environment {
		public:
			virtual void SetUp()
			{
				boost::filesystem::path base_path { BASE_PATH };
				CHECK(boost::filesystem::exists(base_path));
				auto system_db_file = base_path / "mapd_catalogs" / MAPD_SYSTEM_DB ;
				auto data_dir = base_path / "mapd_data";
				dataMgr = new Data_Namespace::DataMgr(data_dir.string());
				if (!boost::filesystem::exists(system_db_file)) {
					SysCatalog syscat(base_path.string(), *dataMgr, true);
					syscat.initDB();
				}
				gsys_cat = new SysCatalog(base_path.string(), *dataMgr);
				UserMetadata user;
				CHECK(gsys_cat->getMetadataForUser(MAPD_ROOT_USER, user));
				gsys_cat->set_currentUser(user);
				if (!gsys_cat->getMetadataForUser("gtest", user)) {
					gsys_cat->createUser("gtest", "test!test!", false);
					CHECK(gsys_cat->getMetadataForUser("gtest", user));
				}
				DBMetadata db;
				if (!gsys_cat->getMetadataForDB("gtest_db", db)) {
					gsys_cat->createDatabase("gtest_db", user.userId);
					CHECK(gsys_cat->getMetadataForDB("gtest_db", db));
				}
				gcat = new Catalog(base_path.string(), user, db, *dataMgr);
			}
			virtual void TearDown()
			{
				if (gsys_cat != nullptr)
					delete gsys_cat;
				if (gcat != nullptr)
					delete gcat;
				if (dataMgr != nullptr)
					delete dataMgr;
			}
	};

	bool
	storage_test(string table_name, size_t num_rows)
	{
		vector<size_t> insert_col_hashs = populate_table_random(table_name, num_rows, *gcat);
		// cout << "insert hashs: ";
		// for (auto h : insert_col_hashs)
			// cout << h << " ";
		// cout << endl;
		vector<size_t> scan_col_hashs = scan_table_return_hash(table_name, *gcat);
		vector<size_t> scan_col_hashs2 = scan_table_return_hash_non_iter(table_name, *gcat);
		// cout << "scan hashs: ";
		// for (auto h : scan_col_hashs)
			// cout << h << " ";
		// cout << endl;
		return insert_col_hashs == scan_col_hashs && insert_col_hashs == scan_col_hashs2;
	}

}

#define SMALL		10000000
#define LARGE		100000000

TEST(StorageSmall,DISABLED_Numbers) {
	ASSERT_NO_THROW(run_ddl("drop table if exists numbers;"););
	ASSERT_NO_THROW(run_ddl("create table numbers (a smallint, b int, c bigint, d numeric(7,3), e double, f float) with (fragment_size = 1000, page_size = 1024);"););
	EXPECT_TRUE(storage_test("numbers", SMALL));
	ASSERT_NO_THROW(run_ddl("drop table numbers;"););
}

TEST(StorageLarge, Numbers) {
	ASSERT_NO_THROW(run_ddl("drop table if exists numbers;"););
	ASSERT_NO_THROW(run_ddl("create table numbers (a smallint, b int, c bigint, d numeric(7,3), e double, f float);"););
	EXPECT_TRUE(storage_test("numbers", LARGE));
	ASSERT_NO_THROW(run_ddl("drop table numbers;"););
}

TEST(StorageSmall, Strings) {
	ASSERT_NO_THROW(run_ddl("drop table if exists strings;"););
	ASSERT_NO_THROW(run_ddl("create table strings (x varchar(10), y text);"););
	EXPECT_TRUE(storage_test("strings", SMALL));
	ASSERT_NO_THROW(run_ddl("drop table strings;"););
}

TEST(StorageSmall, AllTypes) {
	ASSERT_NO_THROW(run_ddl("drop table if exists alltypes;"););
	ASSERT_NO_THROW(run_ddl("create table alltypes (a smallint, b int, c bigint, d numeric(7,3), e double, f float, g timestamp(0), h time(0), i date, x varchar(10), y text);"););
	EXPECT_TRUE(storage_test("alltypes", SMALL));
	ASSERT_NO_THROW(run_ddl("drop table alltypes;"););
}

int
main(int argc, char* argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	::testing::AddGlobalTestEnvironment(new SQLTestEnv);
	return RUN_ALL_TESTS();
}
