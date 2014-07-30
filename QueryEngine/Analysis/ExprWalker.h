/**
 * @file	ExprWalker.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_EXPRWALKER_H
#define SQL_EXPRWALKER_H

#include <vector>
#include <string>
#include "../../Shared/types.h"
#include "../Parse/SQL/visitor/Visitor.h"
#include "../../DataMgr/Metadata/Catalog.h"

namespace Analysis_Namespace {

/**
 * @class ExprWalker
 * @brief The celery stalks at midnight.
 *
 */
class ExprWalker : public Visitor {

public:
	/// Constructor
	ExprWalker(Catalog &c) : c_(c), errFlag_(false) {}

	/// Returns an error message if an error was encountered
	inline std::pair<bool, std::string> isError() { return std::pair<bool, std::string>(errFlag_, errMsg_); }

	virtual void visit(SqlStmt *v);
	virtual void visit(DmlStmt *v);
	virtual void visit(SelectStmt *v);
	virtual void visit(Selection *v);
	virtual void visit(FromClause *v);
	virtual void visit(TableList *v);
	virtual void visit(Table *v);
	virtual void visit(ScalarExprList *v);
	virtual void visit(ScalarExpr *v);
	virtual void visit(Column *v);

private:
	Catalog &c_;
	std::vector<Table*> tblNodes_;
	std::vector<Column*> colNodes_;
	std::string errMsg_;
	bool errFlag_;
};

} // Analysis_Namespace

#endif // SQL_EXPRWALKER_H
