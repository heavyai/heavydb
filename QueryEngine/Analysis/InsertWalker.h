/**
 * @file	InsertWalker.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Semantic Analysis
 *
 */
#ifndef SQL_INSERT_WALKER_H
#define SQL_INSERT_WALKER_H

#include <vector>
#include <string>

class Catalog;

// SQL AST Nodes
#include "../Parse/SQL/visitor/Visitor.h"
 
using namespace SQL_Namespace;

namespace Analysis_Namespace {

class InsertWalker : public SQL_Namespace::Visitor {

public:
	InsertWalker(Catalog &c) : c_(c) {}

	virtual void visit(Program *v);
	virtual void visit(SQLList *v);
	virtual void visit(SQL *v);
	virtual void visit(ManipulativeStatement *v);
	virtual void visit(InsertStatement *v);
	virtual void visit(OptColumnCommalist *v);
	virtual void visit(ColumnCommalist *v);
	virtual void visit(Column *v);

	virtual void visit(AllOrAnyPredicate *v) {}
	virtual void visit(Ammsc *v) {}
	virtual void visit(AnyAllSome *v) {}
	virtual void visit(AssignmentCommalist *v) {}
	virtual void visit(Assignment *v) {}
	virtual void visit(AtomCommalist *v) {}
	virtual void visit(Atom *v) {}
	virtual void visit(BaseTableDef *v) {}
	virtual void visit(BaseTableElementCommalist *v) {}
	virtual void visit(BaseTableElement *v) {}
	virtual void visit(BetweenPredicate *v) {}
	virtual void visit(ColumnDef *v) {}
	virtual void visit(ColumnDefOpt *v) {}
	virtual void visit(ColumnDefOptList *v) {}
	virtual void visit(ColumnRefCommalist *v) {}
	virtual void visit(ColumnRef *v) {}
	virtual void visit(ComparisonPredicate *v) {}
	virtual void visit(Cursor *v) {}
	virtual void visit(DataType *v) {}
	virtual void visit(ExistenceTest *v) {}
	virtual void visit(FromClause *v) {}
	virtual void visit(FunctionRef *v) {}
	virtual void visit(GroupByList *v) {}
	virtual void visit(InPredicate *v) {}
	virtual void visit(InsertAtomCommalist *v) {}
	virtual void visit(InsertAtom *v) {}
	virtual void visit(LikePredicate *v) {}
	virtual void visit(Literal *v) {}
	virtual void visit(OptAllDistinct *v) {}
	virtual void visit(OptAscDesc *v) {}
	virtual void visit(OptEscape *v) {}
	virtual void visit(OptGroupByClause *v) {}
	virtual void visit(OptHavingClause *v) {}
	virtual void visit(OptLimitClause *v) {}
	virtual void visit(OptOrderByClause *v) {}
	virtual void visit(OptWhereClause *v) {}
	virtual void visit(OrderingSpecCommalist *v) {}
	virtual void visit(OrderingSpec *v) {}
	virtual void visit(Predicate *v) {}
	virtual void visit(QuerySpec *v) {}
	virtual void visit(ScalarExpCommalist *v) {}
	virtual void visit(ScalarExp *v) {}
	virtual void visit(Schema *v) {}
	virtual void visit(SearchCondition *v) {}
	virtual void visit(Selection *v) {}
	virtual void visit(SelectStatement *v) {}
	virtual void visit(Subquery *v) {}
	virtual void visit(TableConstraintDef *v) {}
	virtual void visit(TableExp *v) {}
	virtual void visit(Table *v) {}
	virtual void visit(TableRefCommalist *v) {}
	virtual void visit(TableRef *v) {}
	virtual void visit(TestForNull *v) {}
	virtual void visit(UpdateStatementPositioned *v) {}
	virtual void visit(UpdateStatementSearched *v) {}
	virtual void visit(ValuesOrQuerySpec *v) {}

private:
	Catalog &c_;
	std::vector<std::string> colNames_;
	std::string errMsg_ = "";
};

} // Analysis_Namespace

#endif // SQL_INSERT_WALKER_H
