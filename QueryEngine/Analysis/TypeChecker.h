/**
 * @file	TypeChecker.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef QUERYENGINE_ANALYSIS_TYPECHECKER_H
#define QUERYENGINE_ANALYSIS_TYPECHECKER_H

#include <iostream>
#include "../../Shared/types.h"
#include "../Parse/SQL/visitor/Visitor.h"
#include "../../DataMgr/Metadata/Catalog.h"

using namespace SQL_Namespace;

namespace Analysis_Namespace {

/**
 * @class 	TypeChecker
 * @brief	This class implements the Visitor interface for type checking of an SQL statement.
 */
class TypeChecker : public Visitor {

public:
	/// Constructor
	TypeChecker(Catalog &c) : c_(c), errFlag_(false) {}

	/// Returns an error message if an error was encountered
	inline std::pair<bool, std::string> isError() { return std::pair<bool, std::string>(errFlag_, errMsg_); }

	/// @brief Visit an Column node
	virtual void visit(Column *v);

	/// @brief Visit an DmlStmt node
	virtual void visit(DmlStmt *v);

	/// @brief Visit an Literal node
	virtual void visit(Literal *v);

	/// @brief Visit an ScalarExpr node
	virtual void visit(ScalarExpr *v);

	/// @brief Visit an ScalarExprList node
	virtual void visit(ScalarExprList *v);

	/// @brief Visit an SelectStmt node
	virtual void visit(SelectStmt *v);

	/// @brief Visit an Selection node
	virtual void visit(Selection *v);

	/// @brief Visit an FromClause node
	virtual void visit(FromClause *v);

	/// @brief Visit an SqlStmt node
	virtual void visit(SqlStmt *v);

	/// @brief Visit an FromClause node
	virtual void visit(TableList *v);

	/// @brief Visit an FromClause node
	virtual void visit(Table *v);

private:
	Catalog &c_;							/// a reference to a Catalog, which holds table/column metadata
	std::string errMsg_;					/// holds an error message, if applicable; otherwise, it is ""
	bool errFlag_ = false;					/// indicates the existence of an error when true

	std::vector<std::pair<std::string, std::string>> colNames_;	/// saves parsed column names from "Selection" node
	std::vector<std::string> tblNames_;		/// saves parsed table names from "FromClause" node

};

} // Analysis_Namespace

#endif // QUERYENGINE_ANALYSIS_TYPECHECKER_H
