/**
 * @file	InsertWalker.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_INSERT_WALKER_H
#define SQL_INSERT_WALKER_H

#include <vector>
#include <string>
#include "../../Shared/types.h"
#include "../Parse/SQL/visitor/Visitor.h"
#include "../../DataMgr/Metadata/Catalog.h"

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
 * contain an appropriate error message. These members are accessible via the
 * isError() method.
 *
 */
class InsertWalker : public Visitor {

public:
	/// Constructor
	InsertWalker(Catalog &c) : c_(c), errFlag_(false) {}

	/// Returns an error message if an error was encountered
	inline std::pair<bool, std::string> isError() { return std::pair<bool, std::string>(errFlag_, errMsg_); }

	virtual void visit(SqlStmt *v);
	virtual void visit(DmlStmt *v);
	virtual void visit(InsertStmt *v);
	virtual void visit(ColumnList *v);
	virtual void visit(Column *v);
	virtual void visit(LiteralList *v);
	virtual void visit(Literal *v);

private:
	Catalog &c_;
	std::vector<std::string> colNames_;
	std::vector<mapd_data_t> colTypes_;
	std::string errMsg_;
	bool errFlag_;
};

} // Analysis_Namespace

#endif // SQL_INSERT_WALKER_H
