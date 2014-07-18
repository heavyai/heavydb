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

	// Test: select with predicate
	result = sql.parse(selectFromWhere, errMsg);
	if (!result.first) {
		fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
		return false;
	}

	// Test: select with predicate, error
	result = sql.parse(selectFromWhereErr, errMsg);
	if (result.first) {
		fprintf(stderr, "[%s:%d] An invalid statement was parsed without producing an error.\n", __func__, __LINE__);
		return false;
	}

	return true;
}
