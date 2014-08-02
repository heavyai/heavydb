/**
 * @file	InsertWalker.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file contains the class specification for InsertWalker, which belongs
 * to the Analysis_Namespace.
 */
#ifndef QUERYENGINE_ANALYSIS_INSERTWALKER_H
#define QUERYENGINE_ANALYSIS_INSERTWALKER_H

#include <vector>
#include <string>
#include "../../Shared/types.h"
#include "../Parse/SQL/visitor/Visitor.h"
#include "../../DataMgr/Metadata/Catalog.h"

using namespace SQL_Namespace;

namespace Analysis_Namespace {

/**
 * @class InsertWalker
 * @brief Parses and type-checks INSERT statements.
 *
 * The InsertWalker will traverse the SQL AST in order to parse statements of
 * the following form:
 *
 * INSERT INTO table (column1 [, column2, column3 ... ]) VALUES (value1 [, value2, value3 ... ])
 *
 * It verifies the existence of the table and column names, and it verifies that
 * the specified values are of the correct type for the corresponding column. If
 * not, then a local member called "errFlag_" is set to true, and "errMsg_" will
 * contain an appropriate error message. These members are accessible to the
 * client via the isError() method, which are returned as a pair<bool, string>.
 *
 * Example usage given an SQL AST node "parseRoot" and a Catalog "c":
 *
 *		InsertWalker iw(c);
 *		if (parseRoot != 0) {
 *          parseRoot->accept(iw); 
 *          std::pair<bool, std::string> insertErr = iw.isError();
 *          if (insertErr.first == true) {
 *              cout << "Error: " << insertErr.second << std::endl;
 *          }
 *		}
 *
 */
class InsertWalker : public Visitor {

public:
	/// Constructor
	InsertWalker(Catalog &c) : c_(c), errFlag_(false) {}

	/// Returns an error message if an error was encountered
	inline std::pair<bool, std::string> isError() { return std::pair<bool, std::string>(errFlag_, errMsg_); }

	/// @brief Visit an SqlStmt node
	virtual void visit(SqlStmt *v);

	/// @brief Visit an DmlStmt node
	virtual void visit(DmlStmt *v);

	/// @brief Visit an InsertStmt node
	virtual void visit(InsertStmt *v);

	/// @brief Visit an ColumnList node
	virtual void visit(ColumnList *v);

	/// @brief Visit an Column node
	virtual void visit(Column *v);

	/// @brief Visit an LiteralList node
	virtual void visit(LiteralList *v);

	/// @brief Visit an Literal node
	virtual void visit(Literal *v);

private:
	Catalog &c_;							/// a reference to a Catalog, which holds table/column metadata
	std::vector<std::string> colNames_;		/// saves parsed column names from the INSERT statement
	std::vector<mapd_data_t> colTypes_;		/// saves parsed column types from the INSERT statement
	std::string errMsg_;					/// holds an error message, if applicable; otherwise, it is ""
	bool errFlag_ = false;					/// indicates the existence of an error when true
};

} // Analysis_Namespace

#endif // QUERYENGINE_ANALYSIS_INSERTWALKER_H
