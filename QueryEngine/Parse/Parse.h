/**
 * @file	Parse.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#include <queue>
#include "SQL/ast/ASTNode.h"
#include "SQL/ast/Program.h"
#include "RA/ast/RelAlgNode.h"
#include "RA/ast/RA_Program.h"

namespace Parse_Namespace {

/**
 * @class Parse
 * @brief A Parse object handles the parsing of input strings.
 */
class Parse {

public:
	Parse();

	SQL_Namespace::ASTNode* parseSQL(const string &s);
	RA_Namespace::RelAlgNode* parseRA(const string &s);

	void printXML(SQL_Namespace::ASTNode *n);
	void printXML(RA_Namespace::RelAlgNode *n);

private:
	
};

} // Parse_Namespace
