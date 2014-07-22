/**
 * @file 	parseTest.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "Parse.h"
#include "../../Shared/macros.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"

using namespace Testing;
using namespace Parse_Namespace;
using std::cout;
using std::endl;

// unit test function prototypes
bool test_SQLParse();



std::string selectTesting[60] = { 
"select * from table_A;",
"select ALL * from table_A;",
"select DISTINCT * from table_A;",
"select * from table_A, table_B;",
"select * from table_A, table_B, table_C;",
"select * from table_A.table_B;",
"select * from table_A as table_B;",
"select column_A from table_A;",
"select column_A, column_B from table_A, table_B;",
"select column_A, column_B, column_C from table_A;",
"select column_A.column_B from table_A;",
"select column_A as column_B from table_A;",
"select 1 from table_A;",
"select 1, 2+3 from table_A;",
"select avg(column_A), min(column_B), max(column_C), count(1+2), sum(sum(3)) from table_A;",
"select (avg(column_A)*min(column_C)) from table_A;",
"select * from table_A where 1 < 2;",
"select * from table_A where 1 < 2 AND 1 < 3;",
"select * from table_A where 1 < 2 OR 1 < 3;",
"select * from table_A where ((1 < 2));",
"select * from table_A where NOT 1 < 2;",
"select * from table_A where 1 < 1 + 1;",
"select * from table_A where ((1*3) + (6-5)) < (4/2);",
"select * from table_A where a = 1;",
"select * from table_A where a = 'twelve';",
"select * from table_A where a+b > c-d;",
"select * from table_A where a < b;",
"select * from table_A where a = b;",
"select * from table_A where a >= b;",
"select * from table_A where a <= b;",
"select * from table_A where a != b;",
"select * from table_A where a1.b2 < a3.b4;",
"select * from table_A where (a1.b2*(4-5)) LIKE 'twelve';",
"select * from table_A where (a1.b2*(4-5)) NOT LIKE 'twelve';",
"select * from table_A where (a1.b2*(4-5)) LIKE 'twelve' ESCAPE 'three';",
"select * from table_A where column_A.column_B IS NOT NULL;",
"select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen');",
"select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ((select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen')));",
"select * from table_A where (column_A.column_B) between a1 and a2;",
"select * from table_A where a+3 > ANY (select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen'));",
"select * from table_A where a+3 > SOME (select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen'));",
"select * from table_A where exists (select * from table_A where a+3 > SOME (select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen')));",
"select * from table_A group by column_A, column_B.column_C;",
"select * from table_A having 1 < 2;",
"select * from table_A having 1 < 2 AND 1 < 3;",
"select * from table_A having 1 < 2 OR 1 < 3;",
"select * from table_A order by 3, 5 asc, column_A desc, column_A.column_B asc;",
"select * from table_A limit 1;",
"select * from table_A limit 1 OFFSET 1;"
"select column_A.column_B, column_C.column_D, column_E as column_F from table_A, table_B.table_C, table_D as table_E where ((1*3) + (6-5)) < (4/2) AND (a1.b2*(4-5)) LIKE 'twelve' ESCAPE 'three' OR (NOT column_A.column_B IS NOT NULL) AND (NOT ((column_A.column_B*(a+b)/(3-4)) NOT IN ((select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen'))))) AND a+3 > ANY (select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen')) OR exists (select * from table_A where a+3 > SOME (select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen'))) group by column_A, column_B.column_C having 1 > 2 order by 3, 5 asc, column_A desc, column_A.column_B asc limit 10 OFFSET 10;"

};

std::string insertTesting[5] = {
"insert into table_A VALUES (NULL);",
"insert into table_A VALUES ('atom_1', 'atom_2', 'atom_3', 4);",
"insert into table_A (column_A, column_B, column_C) VALUES (NULL);",
"insert into table_A select * from table_A where exists (select * from table_A where a+3 > SOME (select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen')));",
"insert into table_A (column_A, column_B, column_C) VALUES ('atom_1', 'atom_2', 3);"
};

std::string updateTesting[5] = {
"update table_A set where current of cursor_a;",
"update table_A set a1 = 5 where current of cursor_a;",
"update table_A set a1 = 5, a2 = 6, a3 = 7 where current of cursor_a;",
"update table_A set a1 = 5 where a+3 > SOME (select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen'));"
"update table_A set a1 = 5, a2 = 6, a3 = 7 where a+3 > SOME (select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen'));"
};

std::string schemaTesting[10] = {
	"drop table table_A;",
	"create table table_A (column_A VARCHAR, column_B VARCHAR (1), column_C CHAR, column_D CHARACTER (1), column_E NUMERIC, column_F NUMERIC (1), column_G NUMERIC (1, 1), column_H DECIMAL, column_I DECIMAL (1), column_J DECIMAL (1,1), column_K INTEGER, column_L INT, column_F SMALLINT, column_M FLOAT, column_N FLOAT (1), column_O REAL, column_P DOUBLE PRECISION);",
	"create table table_A (column_A INT NOT NULL NOT NULL UNIQUE NOT NULL PRIMARY KEY DEFAULT 'literal1' DEFAULT NULL DEFAULT USER CHECK (sc = 'true') REFERENCES table_B REFERENCES table_C (column_A, column_B, column_C));",
	"create table table_A (UNIQUE (column_A, column_B, column_C));",
	"create table table_A (PRIMARY KEY (column_A, column_B, column_C));",
	"create table table_A (FOREIGN KEY (column_A, column_B, column_C) REFERENCES table_A);",
	"create table table_A (FOREIGN KEY (column_A, column_B, column_C) REFERENCES table_A (column_A, column_B, column_C));",
	"create table table_A (CHECK (a+3 > SOME (select * from table_A where (column_A.column_B*(a+b)/(3-4)) NOT IN ('five', 'twelve', 'thirteen'))));"
};

std::string amplab[4] = {
	"SELECT pageURL, pageRank FROM rankings WHERE pageRank > X;",
	"SELECT SUBSTR(sourceIP, 1, X), SUM(adRevenue) FROM uservisits GROUP BY SUBSTR(sourceIP, 1, X);",
	"SELECT sourceIP, totalRevenue, avgPageRank FROM (SELECT sourceIP, AVG(pageRank) as avgPageRank, SUM(adRevenue) as totalRevenue FROM Rankings AS R, UserVisits AS UV WHERE R.pageURL = UV.destURL AND UV.visitDate BETWEEN 'Date(`1980-01-01)' AND 'Date(`X)' GROUP BY UV.sourceIP) ORDER BY totalRevenue DESC LIMIT 1;",
	"CREATE TABLE url_counts_partial AS SELECT TRANSFORM (line) USING 'python /root/url_count.py' as (sourcePage, destPage, cnt) FROM documents; CREATE TABLE url_counts_total AS SELECT SUM(cnt) AS totalCount, destPage FROM url_counts_partial GROUP BY destPage;"
};

std::string tpcH[20] = {
	"create external table supplier (S_SUPPKEY bigint, S_NAME text, S_ADDRESS text, S_NATIONKEY bigint, S_PHONE text, S_ACCTBAL double, S_COMMENT text) using csv with ('csvfile.delimiter'='|') location 'hdfs://x/y';",
	"create external table lineitem (L_ORDERKEY bigint, L_PARTKEY bigint, L_SUPPKEY bigint, L_LINENUMBER bigint, L_QUANTITY double, L_EXTENDEDPRICE double, L_DISCOUNT double, L_TAX double, L_RETURNFLAG text, L_LINESTATUS text, L_SHIPDATE text, L_COMMITDATE text, L_RECEIPTDATE text, L_SHIPINSTRUCT text, L_SHIPMODE text, L_COMMENT text) using csv with ('csvfile.delimiter'='|') location 'hdfs://x/y';",
	"create external table part (P_PARTKEY bigint, P_NAME text, P_MFGR text, P_BRAND text, P_TYPE text, P_SIZE integer, P_CONTAINER text, P_RETAILPRICE double, P_COMMENT text) using csv with ('csvfile.delimiter'='|') location 'hdfs://x/y';",
	"create external table partsupp (PS_PARTKEY bigint, PS_SUPPKEY bigint, PS_AVAILQTY int, PS_SUPPLYCOST double, PS_COMMENT text) using csv with ('csvfile.delimiter'='|') location 'hdfs://x/y';",
	"create external table customer (C_CUSTKEY bigint, C_NAME text, C_ADDRESS text, C_NATIONKEY bigint, C_PHONE text, C_ACCTBAL double, C_MKTSEGMENT text, C_COMMENT text) using csv with ('csvfile.delimiter'='|') location 'hdfs://x/y';",
	"create external table orders (O_ORDERKEY bigint, O_CUSTKEY bigint, O_ORDERSTATUS text, O_TOTALPRICE double, O_ORDERDATE text, O_ORDERPRIORITY text, O_CLERK text, O_SHIPPRIORITY int, O_COMMENT text) using csv with ('csvfile.delimiter'='|') location 'hdfs://x/y';",	
	"create external table nation (N_NATIONKEY bigint, N_NAME text, N_REGIONKEY bigint, N_COMMENT text) using csv with ('csvfile.delimiter'='|') location 'hdfs://x/y';",
	"create external table region (R_REGIONKEY bigint, R_NAME text, R_COMMENT text) using csv with ('csvfile.delimiter'='|') location 'hdfs://x/y';",

	"select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice*(1-l_discount)) as sum_disc_price, sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order from lineitem where l_shipdate <= '1998-09-01' group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;",

	"create table nation_region as select n_regionkey, r_regionkey, n_nationkey, n_name, r_name from region join nation on n_regionkey = r_regionkey where r_name = 'EUROPE';",
	"create table r2_1 as select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment, ps_supplycost from nation_region join supplier on s_nationkey = n_nationkey join partsupp on s_suppkey = ps_suppkey join part on p_partkey = ps_partkey  where p_size = 15 and p_type like '%BRASS';",
	"create table r2_2 as select p_partkey, min(ps_supplycost) as min_ps_supplycost from r2_1 group by p_partkey;",
	"select s_acctbal, s_name, n_name, r2_1.p_partkey, p_mfgr, s_address, s_phone, s_comment from r2_1 join r2_2 on r2_1.p_partkey = r2_2.p_partkey where ps_supplycost = min_ps_supplycost order by s_acctbal, n_name, s_name, r2_1.p_partkey;",

	"select l_orderkey,  sum(l_extendedprice*(1-l_discount)) as revenue, o_orderdate, o_shippriority from customer as c join orders as o on c.c_mktsegment = 'BUILDING' and c.c_custkey = o.o_custkey join lineitem as l on l.l_orderkey = o.o_orderkey where o_orderdate < '1995-03-15' and l_shipdate > '1995-03-15' group by l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate;",

	"select sum(l_extendedprice*l_discount) as revenue from lineitem where l_shipdate >= '1994-01-01' and l_shipdate < '1995-01-01' and l_discount >= 0.05 and l_discount <= 0.07 and l_quantity < 24;",

	"select c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) as revenue, c_acctbal, n_name, c_address, c_phone, c_comment from customer_rc as c join nation_rc as n on c.c_nationkey = n.n_nationkey join orders as o on c.c_custkey = o.o_custkey and o.o_orderdate >= '1993-10-01' and o.o_orderdate < '1994-01-01' join lineitem_rc as l on l.l_orderkey = o.o_orderkey and l.l_returnflag = 'R' group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment order by revenue desc;",

	"select l_shipmode, sum(case when o_orderpriority ='1-URGENT' or o_orderpriority ='2-HIGH' then 1 else 0 end) as high_line_count, sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count from orders, lineitem where o_orderkey = l_orderkey and (l_shipmode = 'MAIL' or l_shipmode = 'SHIP') and l_commitdate < l_receiptdate and l_shipdate < l_commitdate and l_receiptdate >= '1994-01-01' and l_receiptdate < '1995-01-01' group by l_shipmode order by l_shipmode;",

	"select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice*(1-l_discount) else 0 end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue from lineitem, part where l_partkey = p_partkey and l_shipdate >= '1995-09-01' and l_shipdate < '1995-10-01';"
};

void testGrammar(SQLParse sql, std::string errMsg, std::pair<bool, ASTNode*> result) {

	PCLEAR("SQLParse()");

	cout << endl << "------------ Select statement testing -------------" << endl;

	// SELECT testing
	for (int i = 0; i < 50; i++) {

		// Test: select with predicate
		result = sql.parse(selectTesting[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			PFAIL("SQLParse()");
		}
		else PPASS("SQLParse()");
	}

	cout << endl << "------------ Insert statement testing -------------" << endl;

	// INSERT testing
	for (int i = 0; i < 5; i++) {

		// Test: insert
		result = sql.parse(insertTesting[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			PFAIL("SQLParse()");
		}
		else PPASS("SQLParse()");
	}

	cout << endl << "------------ Update statement testing -------------" << endl;

	// Update testing
	for (int i = 0; i < 5; i++) {

		// Test:update
		result = sql.parse(updateTesting[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			PFAIL("SQLParse()");
		}
		else PPASS("SQLParse()");
	}

	cout << endl << "------------ Schema statement testing -------------" << endl;

	// CREATE testing
	for (int i = 0; i < 10; i++) {

		// Test: create/drop
		result = sql.parse(schemaTesting[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			PFAIL("SQLParse()");
		}
		else PPASS("SQLParse()");
	}
}

void testBenchmarks(SQLParse sql, std::string errMsg, std::pair<bool, ASTNode*> result) {

	PCLEAR("SQLParse()");

	cout << endl << "------------ AMPLab Benchmark testing -------------" << endl;

	// AMPLAB testing
	for (int i = 0; i < 4; i++) {

		// Test: AMP lab benchmarks
		result = sql.parse(amplab[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			PFAIL("SQLParse()");
		}
		else PPASS("SQLParse()");
	}

	PCLEAR("SQLParse()");

	cout << endl << "------------ TPC-H Benchmark testing -------------" << endl;
	// TPC-H testing
	for (int i = 0; i < 20; i++) {

		// Test: TPC-H benchmarks
		result = sql.parse(tpcH[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			PFAIL("SQLParse()");
		}
		else PPASS("SQLParse()");

	}
}

int main() {
	PRINT_DLINE(80);
    
    // call unit tests
    test_SQLParse();

	PRINT_DLINE(80);
	printTestSummary();
	PRINT_DLINE(80);
    
    return 0;
}

bool test_SQLParse() {
	SQLParse sql;
	std::string errMsg;
	std::pair<bool, ASTNode*> result;

	std::string selectFromWhere = "select * from student where GradYear > 2004;";
	std::string selectFromWhereErr = "select * fromz student where GradYear > 2004;";

	testGrammar(sql, errMsg, result);
	testBenchmarks(sql, errMsg, result);
/*
	// Test: select with predicate, error
	result = sql.parse(selectFromWhereErr, errMsg);
	if (result.first) {
		fprintf(stderr, "[%s:%d] An invalid statement was parsed without producing an error.\n", __func__, __LINE__);
		PFAIL("SQLParse()");
	}
	else PPASS("SQLParse()");	*/

}