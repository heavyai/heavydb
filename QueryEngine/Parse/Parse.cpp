/**
 * @file	Parse.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#include <string>
#include "Parse.h"
#include "RA/parser.h"
#include "SQL/parser.h"

namespace Parse_Namespace {

SQL_Namespace::ASTNode* parse_sql(const std::string &input, std::string &errMsg) {
	using namespace SQL_Namespace;
	SQLParser parser;
	ASTNode *parseRoot = nullptr;
    std::string lastParsed;
    int numErrors = parser.parse(input, parseRoot, lastParsed);
    if (numErrors > 0) {
        errMsg = "Error at: " + lastParsed;
		return nullptr;
	}
	return parseRoot;
}

RA_Namespace::RelAlgNode* parse_ra(const std::string &input, std::string &errMsg) {
	using namespace RA_Namespace;
	RAParser parser;
	RelAlgNode *parseRoot = nullptr;
    std::string lastParsed;
    int numErrors = parser.parse(input, parseRoot, lastParsed);
    if (numErrors > 0) {
        errMsg = "Error at: " + lastParsed;
		return nullptr;
	}
	return parseRoot;
}

} // Parse_Namespace