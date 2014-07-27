/**
 * @file    SqlRaTranslator.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef AST_SIMPLE_TRANSLATOR_VISITOR_H
#define AST_SIMPLE_TRANSLATOR_VISITOR_H

// SQL AST Nodes
#include "../ast/SQL/visitor/Visitor.h"
#include "../ast/AbstractScalarExpr.h"
#include "../ast/AllOrAnyPredicate.h"
#include "../ast/Ammsc.h"
#include "../ast/AnyAllSome.h"
#include "../ast/AnyOrAllPredicate.h"
#include "../ast/AssignmentCommalist.h"
#include "../ast/Assignment.h"
#include "../ast/ASTNode.h"
#include "../ast/AtomCommalist.h"
#include "../ast/Atom.h"
#include "../ast/BaseTableDef.h"
#include "../ast/BaseTableElementCommalist.h"
#include "../ast/BaseTableElement.h"
#include "../ast/BetweenPredicate.h"
#include "../ast/ColumnCommalist.h"
#include "../ast/ColumnDef.h"
#include "../ast/ColumnDefOpt.h"
#include "../ast/ColumnDefOptList.h"
#include "../ast/Column.h"
#include "../ast/ColumnRefCommalist.h"
#include "../ast/ColumnRef.h"
#include "../ast/ComparisonPredicate.h"
#include "../ast/Cursor.h"
#include "../ast/DataType.h"
#include "../ast/ExistenceTest.h"
#include "../ast/FromClause.h"
#include "../ast/FunctionRef.h"
#include "../ast/GroupByList.h"
#include "../ast/InPredicate.h"
#include "../ast/InsertAtomCommalist.h"
#include "../ast/InsertAtom.h"
#include "../ast/InsertStatement.h"
#include "../ast/LikePredicate.h"
#include "../ast/Literal.h"
#include "../ast/ManipulativeStatement.h"
#include "../ast/OptAllDistinct.h"
#include "../ast/OptAscDesc.h"
#include "../ast/OptColumnCommalist.h"
#include "../ast/OptEscape.h"
#include "../ast/OptGroupByClause.h"
#include "../ast/OptHavingClause.h"
#include "../ast/OptLimitClause.h"
#include "../ast/OptOrderByClause.h"
#include "../ast/OptWhereClause.h"
#include "../ast/OrderingSpecCommalist.h"
#include "../ast/OrderingSpec.h"
#include "../ast/Predicate.h"
#include "../ast/Program.h"
#include "../ast/ProjectOp.h"
#include "../ast/QuerySpec.h"
#include "../ast/ScalarExpCommalist.h"
#include "../ast/ScalarExp.h"
#include "../ast/Schema.h"
#include "../ast/SearchCondition.h"
#include "../ast/Selection.h"
#include "../ast/SelectStatement.h"
#include "../ast/SQL.h"
#include "../ast/SQLList.h"
#include "../ast/Subquery.h"
#include "../ast/TableConstraintDef.h"
#include "../ast/TableExp.h"
#include "../ast/Table.h"
#include "../ast/TableRefCommalist.h"
#include "../ast/TableRef.h"
#include "../ast/TestForNull.h"
#include "../ast/UpdateStatementPositioned.h"
#include "../ast/UpdateStatementSearched.h"
#include "../ast/ValuesOrQuerySpec.h"

// Relational Algebra AST Nodes
#include "../../RA/ast/AggrExpr.h"
#include "../../RA/ast/AggrList.h"
#include "../../RA/ast/AntijoinOp.h"
#include "../../RA/ast/Attribute.h"
#include "../../RA/ast/AttrList.h"
#include "../../RA/ast/BinaryOp.h"
#include "../../RA/ast/Comparison.h"
#include "../../RA/ast/DiffOp.h"
#include "../../RA/ast/Expr.h"
#include "../../RA/ast/ExtendOp.h"
#include "../../RA/ast/GroupbyOp.h"
#include "../../RA/ast/JoinOp.h"
#include "../../RA/ast/MathExpr.h"
#include "../../RA/ast/OuterjoinOp.h"
#include "../../RA/ast/Predicate.h"
#include "../../RA/ast/ProductOp.h"
#include "../../RA/ast/Program.h"
#include "../../RA/ast/ProjectOp.h"
#include "../../RA/ast/RelAlgNode.h"
#include "../../RA/ast/Relation.h"
#include "../../RA/ast/RelExpr.h"
#include "../../RA/ast/RelExprList.h"
#include "../../RA/ast/RenameOp.h"
#include "../../RA/ast/SelectOp.h"
#include "../../RA/ast/SemijoinOp.h"
#include "../../RA/ast/SortOp.h"
#include "../../RA/ast/UnaryOp.h"
#include "../../RA/ast/UnionOp.h"

#include <iostream>
using std::cout;
using std::endl;

class QPTranslator : public SQL_Namespace::Visitor {

public:

void visit(AllOrAnyPredicate *v) {}

void visit(Ammsc *v) {}

void visit(AnyAllSome *v) {}

void visit(AnyOrAllPredicate *v) {}

void visit(AssignmentCommalist *v) {}

void visit(Assignment *v) {}

void visit(ASTNode *v) {}

void visit(AtomCommalist *v) {}

void visit(Atom *v) {}

void visit(BaseTableDef *v) {}

void visit(BaseTableElementCommalist *v) {}

void visit(BaseTableElement *v) {}

void visit(BetweenPredicate *v) {}

void visit(ColumnCommalist *v) {}

void visit(ColumnDef *v) {}

void visit(ColumnDefOpt *v) {}

void visit(ColumnDefOptList *v) {}

void visit(Column *v) {}

void visit(ColumnRefCommalist *v) {}

void visit(ColumnRef *v) {}

void visit(ComparisonPredicate *v) {}

void visit(Cursor *v) {}

void visit(DataType *v) {}

void visit(ExistenceTest *v) {}

void visit(FromClause *v) {}

void visit(FunctionRef *v) {}

void visit(GroupByList *v) {}

void visit(InPredicate *v) {}

void visit(InsertAtomCommalist *v) {}

void visit(InsertAtom *v) {}

void visit(InsertStatement *v) {}

void visit(LikePredicate *v) {}

void visit(Literal *v) {}

void visit(ManipulativeStatement *v) {
	// put something here!
}

void visit(OptAllDistinct *v) {}

void visit(OptAscDesc *v) {}

void visit(OptColumnCommalist *v) {}

void visit(OptEscape *v) {}

void visit(OptGroupByClause *v) {}

void visit(OptHavingClause *v) {}

void visit(OptLimitClause *v) {}

void visit(OptOrderByClause *v) {}

void visit(Opt *v) {}ereClause *v) {}

void visit(OrderingSpecCommalist *v) {}

void visit(OrderingSpec *v) {}

void visit(Predicate *v) {}

void visit(Program *v) {
	if (v->sqlList) v->sqlList->accept(*this);
}

void visit(ProjectOp *v) {}

void visit(QuerySpec *v) {}

void visit(ScalarExpCommalist *v) {}

void visit(ScalarExp *v) {}

void visit(Schema *v) {}

void visit(SearchCondition *v) {}

void visit(Selection *v) {}

void visit(SelectStatement *v) {
	// OptAllDistinct, Selection, TableExp
	
	// SELECT [list column_ref+] FROM [list table_ref+] ';
	RA_Namespace::Project *n0;
	RA_Namespace::RelExpr *n1;
	RA_Namespace::AttrList *n2;
	
}

void visit(SQL *v) {
	// if (v->sch) v->sch->accept(*this);
	if (v->manSta) v->manSta->accept(*this);
}

void visit(SQLList *v) {
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void visit(Subquery *v) {}

void visit(TableConstraintDef *v) {}

void visit(TableExp *v) {}

void visit(Table *v) {}

void visit(TableRefCommalist *v) {

}

void visit(TableRef *v) {}

void visit(TestForNull *v) {}

void visit(UpdateStatementPositioned *v) {}

void visit(UpdateStatementSear *v) {}ed *v) {}

void visit(ValuesOrQuerySpec *v) {}  

private:
    vector<Attribute*> attributeNames;
};

#endif // AST_SIMPLE_TRANSLATOR_VISITOR_H
