#include "XMLTranslator.h"

#include "../ast/DdlStmt.h"
#include "../ast/DmlStmt.h"
#include "../ast/FromClause.h"
#include "../ast/InsertStmt.h"
#include "../ast/OptAllDistinct.h"
#include "../ast/OptGroupby.h"
#include "../ast/OptHaving.h"
#include "../ast/OptOrderby.h"
#include "../ast/OptLimit.h"
#include "../ast/OptWhere.h"
#include "../ast/Selection.h"
#include "../ast/SelectStmt.h"
#include "../ast/SqlStmt.h"


using namespace std;

namespace SQL_Namespace {

    void XMLTranslator::visit(AggrExpr *v) {
        cout << "<AggrExpr>" << endl;
        cout << "</AggrExpr>" << endl;
    }
    
    void XMLTranslator::visit(AlterStmt *v) {
        cout << "<AlterStmt>" << endl;
        cout << "</AlterStmt>" << endl;
    }
    
    void XMLTranslator::visit(Column *v) {
        cout << "<Column>" << endl;
        cout << "</Column>" << endl;
    }
    
    void XMLTranslator::visit(ColumnDef *v) {
        cout << "<ColumnDef>" << endl;
        cout << "</ColumnDef>" << endl;
    }
    
    void XMLTranslator::visit(ColumnDefList *v) {
        cout << "<ColumnDefList>" << endl;
        cout << "</ColumnDefList>" << endl;
    }
    
    void XMLTranslator::visit(ColumnList *v) {
        cout << "<ColumnList>" << endl;
        cout << "</ColumnList>" << endl;
    }
    
    void XMLTranslator::visit(Comparison *v) {
        cout << "<Comparison>" << endl;
        cout << "</Comparison>" << endl;
    }
    
    void XMLTranslator::visit(CreateStmt *v) {
        cout << "<CreateStmt>" << endl;
        cout << "</CreateStmt>" << endl;
    }
    
    void XMLTranslator::visit(DdlStmt *v) {
        cout << "<DdlStmt>" << endl;
        cout << "</DdlStmt>" << endl;
    }
    
    void XMLTranslator::visit(DmlStmt *v) {
        cout << "<DmlStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // InsertStmt
        if (v->n2) v->n2->accept(*this); // SelectStmt
        cout << "</DmlStmt>" << endl;
    }
    
    void XMLTranslator::visit(DropStmt *v) {
        cout << "<DropStmt>" << endl;
        cout << "</DropStmt>" << endl;
    }
    
    void XMLTranslator::visit(FromClause *v) {
        cout << "<FromClause>" << endl;
        cout << "</FromClause>" << endl;
    }
    
    void XMLTranslator::visit(InsertColumnList *v) {
        cout << "<InsertColumnList>" << endl;
        cout << "</InsertColumnList>" << endl;
    }
    
    void XMLTranslator::visit(InsertStmt *v) {
        cout << "<InsertStmt>" << endl;
        cout << "</InsertStmt>" << endl;
    }
    
    void XMLTranslator::visit(Literal *v) {
        cout << "<Literal>" << endl;
        cout << "</Literal>" << endl;
    }
    
    void XMLTranslator::visit(LiteralList *v) {
        cout << "<LiteralList>" << endl;
        cout << "</LiteralList>" << endl;
    }
    
    void XMLTranslator::visit(MapdDataT *v) {
        cout << "<MapdDataT>" << endl;
        cout << "</MapdDataT>" << endl;
    }
    
    void XMLTranslator::visit(MathExpr *v) {
        cout << "<MathExpr>" << endl;
        cout << "</MathExpr>" << endl;
    }
    
    void XMLTranslator::visit(OptAllDistinct *v) {
        cout << "<OptAllDistinct>" << endl;
        cout << "</OptAllDistinct>" << endl;
    }
    
    void XMLTranslator::visit(OptGroupby *v) {
        cout << "<OptGroupby>" << endl;
        cout << "</OptGroupby>" << endl;
    }
    
    void XMLTranslator::visit(OptHaving *v) {
        cout << "<OptHaving>" << endl;
        cout << "</OptHaving>" << endl;
    }
    
    void XMLTranslator::visit(OptLimit *v) {
        cout << "<OptLimit>" << endl;
        cout << "</OptLimit>" << endl;
    }
    
    void XMLTranslator::visit(OptOrderby *v) {
        cout << "<OptOrderby>" << endl;
        cout << "</OptOrderby>" << endl;
    }
    
    void XMLTranslator::visit(OptWhere *v) {
        cout << "<OptWhere>" << endl;
        cout << "</OptWhere>" << endl;
    }
    
    void XMLTranslator::visit(OrderbyColumn *v) {
        cout << "<OrderbyColumn>" << endl;
        cout << "</OrderbyColumn>" << endl;
    }
    
    void XMLTranslator::visit(OrderbyColumnList *v) {
        cout << "<OrderbyColumnList>" << endl;
        cout << "</OrderbyColumnList>" << endl;
    }
    
    void XMLTranslator::visit(Predicate *v) {
        cout << "<Predicate>" << endl;
        cout << "</Predicate>" << endl;
    }
    
    void XMLTranslator::visit(RenameStmt *v) {
        cout << "<RenameStmt>" << endl;
        cout << "</RenameStmt>" << endl;
    }
    
    void XMLTranslator::visit(ScalarExpr *v) {
        cout << "<ScalarExpr>" << endl;
        cout << "</ScalarExpr>" << endl;
    }
    
    void XMLTranslator::visit(ScalarExprList *v) {
        cout << "<ScalarExprList>" << endl;
        cout << "</ScalarExprList>" << endl;
    }
    
    void XMLTranslator::visit(SearchCondition *v) {
        cout << "<SearchCondition>" << endl;
        cout << "</SearchCondition>" << endl;
    }
    
    void XMLTranslator::visit(SelectStmt *v) {
        cout << "<SelectStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // OptAllDistinct
        if (v->n2) v->n2->accept(*this); // Selection
        if (v->n3) v->n3->accept(*this); // FromClause
        if (v->n4) v->n4->accept(*this); // OptWhere
        if (v->n5) v->n5->accept(*this); // OptGroupBy
        if (v->n6) v->n6->accept(*this); // OptHaving
        if (v->n7) v->n7->accept(*this); // OptOrderBy
        if (v->n8) v->n8->accept(*this); // OptLimit
        cout << "</SelectStmt>" << endl;
    }
    
    void XMLTranslator::visit(Selection *v) {
        cout << "<Selection>" << endl;
        cout << "</Selection>" << endl;
    }
    
    void XMLTranslator::visit(SqlStmt *v) {
        cout << "<SqlStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // DmlStmt
        if (v->n2) v->n2->accept(*this); // DdlStmt
        cout << "</SqlStmt>" << endl;
    }
    
    void XMLTranslator::visit(Table *v) {
        cout << "<Table>" << endl;
        cout << "</Table>" << endl;
    }
    
    void XMLTranslator::visit(TableList *v) {
        cout << "<TableList>" << endl;
        cout << "</TableList>" << endl;
    }
    
} // SQL_Namespace