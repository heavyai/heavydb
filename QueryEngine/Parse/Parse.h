/**
 * @file	Parse.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#include <string>
#include "SQL/ast/ASTNode.h"
#include "RA/ast/RelAlgNode.h"

using namespace SQL_Namespace;
using namespace RA_Namespace;

namespace Parse_Namespace {

class SQLParse {
public:
	std::pair<bool, ASTNode*> parse(const std::string &input, std::string &errMsg);
};

class RAParse {
public:
	std::pair<bool, RelAlgNode*> parse(const std::string &input, std::string &errMsg);		
};

} // Parse_Namespace
