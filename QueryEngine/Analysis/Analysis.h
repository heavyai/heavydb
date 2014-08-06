/**
 * @file	Analysis.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file contains functions belonging to the Analysis namespace (aptly called
 * Analysis_Namespace). These functions can be called to perform analysis, such as
 * type checking, by utilizing AST tree walkers. Please refer to the documentation
 * below for each respective function.
 */
#ifndef QUERYPLAN_ANALYSIS_ANALYSIS_H
#define QUERYPLAN_ANALYSIS_ANALYSIS_H

// includes
#include <string>
#include <utility>
#include "DdlWalker.h"
#include "InsertWalker.h"
#include "TypeChecker.h"

// forward declarations
namespace SQL_Namespace {
	class ASTNode;
}
using SQL_Namespace::ASTNode;

class Catalog;		// @todo Catalog should probably be wrapped in a namespace
class InsertData;	// @todo InsertData should probably be wrapped in a namespace

namespace Analysis_Namespace {

/**
 * @brief This function checks (validates) INSERT statements.
 *
 * This function makes use of an InsertWalker to check the insert statements
 * that are present in the "sql" string.
 *
 * @param tree
 * @param InsertData&							This will contain the insert data if the string passes
 * @return std::pair<bool, std::string>			Returns true if the string passes.
 */
std::pair<bool, std::string> checkInsert(ASTNode* tree, Catalog& c, InsertData& insert);

/**
 * @brief This function checks (validates) SQL statements (excluding INSERT).
 *
 * This function makes use of an TypeChecker (walker) to check the SQL statements
 * that are present in the "sql" string. Note that it does not check INSERT 
 * statements, which are handled separately via checkInsert.
 *
 * @see checkInsert
 *
 * @param tree
 * @return std::pair<bool, std::string>		Returns true if the string passes.
 */
std::pair<bool, std::string> checkSql(ASTNode *tree, Catalog &c);

/**
 * @brief 
 *
 * @see checkDdl
 *
 * @param tree
 * @return std::pair<bool, std::string>		Returns true if the string passes.
 */
std::pair<bool, std::string> checkDdl(ASTNode *tree, Catalog &c);

} // Analysis_Namespace

#endif // QUERYPLAN_ANALYSIS_ANALYSIS_H

