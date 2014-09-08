/**
 * @file	SQL_RA_Translator.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_RA_TRANSLATOR_H
#define SQL_RA_TRANSLATOR_H

#include <iostream>
#include <vector>
#include "../Parse/SQL/visitor/Visitor.h"
#include "../Parse/RA/ast/RelAlgNode.h"
 
using namespace SQL_Namespace;

namespace Translate_Namespace {

class SQL_RA_Translator : public SQL_Namespace::Visitor {

public:
	RA_Namespace::RelAlgNode *root = NULL;

	/// Constructor
	SQL_RA_Translator();

	virtual void visit(Column *v);
	virtual void visit(Comparison *v);
	virtual void visit(DdlStmt *v);
	virtual void visit(DmlStmt *v);
	virtual void visit(FromClause *v);
	virtual void visit(InsertStmt *v);
	virtual void visit(MathExpr *v);
	virtual void visit(OptWhere *v);
	virtual void visit(Predicate *v);
	virtual void visit(ScalarExpr *v);
	virtual void visit(ScalarExprList *v);
	virtual void visit(SearchCondition *v);
	virtual void visit(SelectStmt *v);
	virtual void visit(Selection *v);
	virtual void visit(SqlStmt *v);
	virtual void visit(Table *v);
	virtual void visit(TableList *v);

	/*
	virtual void visit(ColumnList *v);
	virtual void visit(Column *v);
	virtual void visit(LiteralList *v);
	virtual void visit(Literal *v);
	*/

private:
	//std::vector<int> tableIds_;
	//std::vector<int> columnIds_;
    std::vector<Table*> tableNodes_;
    std::vector<Column*> columnNodes_;
    
	RA_Namespace::RelExpr *nodeFromClause_ = NULL;
	RA_Namespace::AttrList *nodeSelection_ = NULL;
	RA_Namespace::ProjectOp *nodeSelectStmt_ = NULL;
	RA_Namespace::RelExpr *nodeSqlStmt_ = NULL;
    
    std::vector<RA_Namespace::Predicate*> nodePredicateVec_;
    std::vector<RA_Namespace::Comparison*> nodeComparisonVec_;
    std::vector<RA_Namespace::MathExpr*> nodeMathExprVec_;
};

} // Translate_Namespace

#endif // SQL_RA_TRANSLATOR_H
