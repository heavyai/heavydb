/**
 * @file	SQL_RA_Translator.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This header file specifies the API for an SQL to RA translator.
 */

#include "../../RA/ast/RelAlgNode.h"

using RA_Namespace::RelAlgNode;

namespace SQL_Namespace {

class AggrExpr;
class AlterStmt;
class Column;
class ColumnDef;
class ColumnDefList;
class ColumnList;
class Comparison;
class CreateStmt;
class DdlStmt;
class DmlStmt;
class DropStmt;
class FromClause;
class InsertStmt;
class Literal;
class LiteralList;
class MapdDataT;
class MathExpr;
class OptAllDistinct;
class OptGroupby;
class OptHaving;
class OptLimit;
class OptOrderby;
class OptWhere;
class OrderbyColumn;
class OrderbyColumnList;
class Predicate;
class RenameStmt;
class ScalarExpr;
class ScalarExprList;
class SearchCondition;
class SelectStmt;
class Selection;
class SqlStmt;
class Table;
class TableList;
class sqlStmt;

class SQL_RA_Translator {

public:
	virtual RelAlgNode* visit(AggrExpr*) {}
	virtual RelAlgNode* visit(AlterStmt*) {}
	virtual RelAlgNode* visit(Column*) {}
	virtual RelAlgNode* visit(ColumnDef*) {}
	virtual RelAlgNode* visit(ColumnDefList*) {}
	virtual RelAlgNode* visit(ColumnList*) {}
	virtual RelAlgNode* visit(Comparison*) {}
	virtual RelAlgNode* visit(CreateStmt*) {}
	virtual RelAlgNode* visit(DdlStmt*) {}
	virtual RelAlgNode* visit(DmlStmt*);
	virtual RelAlgNode* visit(DropStmt*) {}
	virtual RelAlgNode* visit(FromClause*) {}
	virtual RelAlgNode* visit(InsertStmt*) {}
	virtual RelAlgNode* visit(Literal*) {}
	virtual RelAlgNode* visit(LiteralList*) {}
	virtual RelAlgNode* visit(MapdDataT*) {}
	virtual RelAlgNode* visit(MathExpr*) {}
	virtual RelAlgNode* visit(OptAllDistinct*) {}
	virtual RelAlgNode* visit(OptGroupby*) {}
	virtual RelAlgNode* visit(OptHaving*) {}
	virtual RelAlgNode* visit(OptLimit*) {}
	virtual RelAlgNode* visit(OptOrderby*) {}
	virtual RelAlgNode* visit(OptWhere*) {}
	virtual RelAlgNode* visit(OrderbyColumn*) {}
	virtual RelAlgNode* visit(OrderbyColumnList*) {}
	virtual RelAlgNode* visit(Predicate*) {}
	virtual RelAlgNode* visit(RenameStmt*) {}
	virtual RelAlgNode* visit(ScalarExpr*) {}
	virtual RelAlgNode* visit(ScalarExprList*) {}
	virtual RelAlgNode* visit(SearchCondition*) {}
	virtual RelAlgNode* visit(Selection*) {}
	virtual RelAlgNode* visit(SelectStmt*) {}
	virtual RelAlgNode* visit(SqlStmt*);
	virtual RelAlgNode* visit(Table*) {}
	virtual RelAlgNode* visit(TableList*) {}

 };

 } // SQL_Namespace
