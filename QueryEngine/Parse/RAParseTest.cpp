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
bool test_RAParse();

int main() {
	PRINT_DLINE(80);
    
    // call unit tests
    test_RAParse() ? 
        PPASS("RAParse()") : PFAIL("RAParse()"); 

	PRINT_DLINE(80);
	printTestSummary();
	PRINT_DLINE(80);
    
    return 0;
}

bool test_RAParse() {
	RAParse stmt;
	std::string errMsg;
	std::pair<bool, RelAlgNode*> result;

	std::string select = "select (student, GradYear > 2004);";
	std::string selectErr = "(student, GradYear > 2004);";

	// Test: select with predicate
	result = stmt.parse(select, errMsg);
	if (!result.first) {
		fprintf(stderr, "[%s:%d] %s\n", __func__, __LINE__, errMsg.c_str());
		return false;
	}

	// Test: select with predicate, error
	result = stmt.parse(selectErr, errMsg);
	if (result.first) {
		fprintf(stderr, "[%s:%d] An invalid statement was parsed without producing an error.\n", __func__, __LINE__);
		return false;
	}

	return true;
}
