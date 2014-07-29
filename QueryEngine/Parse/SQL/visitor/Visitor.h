#ifndef SQL_VISITOR_H
#define SQL_VISITOR_H

// forward declarations
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
class Predicate;
class RenameStmt;
class ScalarExpr;
class ScalarExprList;
class SearchCondition;
class SelectStmt;
class SqlStmt;
class Table;
class TableList;
class sqlStmt;

/**
 * @class Visitor
 * @brief This is the Visitor class.
 */
class Visitor {

public:
	Visitor();
	virtual ~Visitor();

	virtual void visit(const AggrExpr*);
	virtual void visit(const AlterStmt*);
	virtual void visit(const Column*);
	virtual void visit(const ColumnDef*);
	virtual void visit(const ColumnDefList*);
	virtual void visit(const ColumnList*);
	virtual void visit(const Comparison*);
	virtual void visit(const CreateStmt*);
	virtual void visit(const DdlStmt*);
	virtual void visit(const DmlStmt*);
	virtual void visit(const DropStmt*);
	virtual void visit(const FromClause*);
	virtual void visit(const InsertStmt*);
	virtual void visit(const Literal*);
	virtual void visit(const LiteralList*);
	virtual void visit(const MapdDataT*);
	virtual void visit(const MathExpr*);
	virtual void visit(const OptAllDistinct*);
	virtual void visit(const OptGroupby*);
	virtual void visit(const OptHaving*);
	virtual void visit(const OptLimit*);
	virtual void visit(const OptOrderby*);
	virtual void visit(const OptWhere*);
	virtual void visit(const Predicate*);
	virtual void visit(const RenameStmt*);
	virtual void visit(const ScalarExpr*);
	virtual void visit(const ScalarExprList*);
	virtual void visit(const SearchCondition*);
	virtual void visit(const SelectStmt*);
	virtual void visit(const SqlStmt*);
	virtual void visit(const Table*);
	virtual void visit(const TableList*);
	
};

#endif // SQL_VISITOR_H
