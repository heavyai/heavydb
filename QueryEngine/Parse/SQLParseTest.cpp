/**
 * @file 	parseTest.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include <cstdio>
#include <cstdlib>
#include "Parse.h"
#include "../../Shared/macros.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"

using namespace Testing;
using namespace Parse_Namespace;

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
	"SELECT SUM(adRevenue) FROM uservisits GROUP BY SUBSTR(sourceIP, 1, X)"
};

int main() {
	PRINT_DLINE(80);
    
    // call unit tests
    test_SQLParse() ? 
        PPASS("SQLParse()") : PFAIL("SQLParse()"); 

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

	// SELECT testing
	for (int i = 0; i < 50; i++) {

		// Test: select with predicate
		result = sql.parse(selectTesting[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			return false;
		}
	}

	// INSERT testing
	for (int i = 0; i < 5; i++) {

		// Test: select with predicate
		result = sql.parse(insertTesting[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			return false;
		}
	}

	// Update testing
	for (int i = 0; i < 5; i++) {

		// Test: select with predicate
		result = sql.parse(updateTesting[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			return false;
		}
	}

	// INSERT testing
	for (int i = 0; i < 10; i++) {

		// Test: select with predicate
		result = sql.parse(schemaTesting[i], errMsg);
		if (!result.first) {
			fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
			fprintf(stderr, "error was string number: %d\n", i);
			return false;
		}
	}
	// Test: select with predicate, error
	result = sql.parse(selectFromWhereErr, errMsg);
	if (result.first) {
		fprintf(stderr, "[%s:%d] An invalid statement was parsed without producing an error.\n", __func__, __LINE__);
		return false;
	}

	return true;

}