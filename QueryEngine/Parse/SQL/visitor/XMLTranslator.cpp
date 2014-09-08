#include "XMLTranslator.h"

#include "../ast/AggrExpr.h"
#include "../ast/AlterStmt.h"
#include "../ast/Column.h"
#include "../ast/ColumnDef.h"
#include "../ast/ColumnDefList.h"
#include "../ast/ColumnList.h"
#include "../ast/Comparison.h"
#include "../ast/CreateStmt.h"
#include "../ast/DdlStmt.h"
#include "../ast/DmlStmt.h"
#include "../ast/DropStmt.h"
#include "../ast/FromClause.h"
#include "../ast/InsertColumnList.h"
#include "../ast/InsertStmt.h"
#include "../ast/Literal.h"
#include "../ast/LiteralList.h"
#include "../ast/MapdDataT.h"
#include "../ast/MathExpr.h"
#include "../ast/OptAllDistinct.h"
#include "../ast/OptGroupby.h"
#include "../ast/OptHaving.h"
#include "../ast/OptOrderby.h"
#include "../ast/OptLimit.h"
#include "../ast/OptWhere.h"
#include "../ast/OrderbyColumn.h"
#include "../ast/OrderByColumnList.h"
#include "../ast/Predicate.h"
#include "../ast/RenameStmt.h"
#include "../ast/ScalarExpr.h"
#include "../ast/ScalarExprList.h"
#include "../ast/SearchCondition.h"
#include "../ast/Selection.h"
#include "../ast/SelectStmt.h"
#include "../ast/SqlStmt.h"
#include "../ast/Table.h"
#include "../ast/TableList.h"

using namespace std;

namespace SQL_Namespace {

    void XMLTranslator::visit(AggrExpr *v) {
        cout << "<AggrExpr>" << endl;
        cout << v->func << " ";
        if (v->n1) v->n1->accept(*this); // Column
        cout << "</AggrExpr>" << endl;
    }
    
    void XMLTranslator::visit(AlterStmt *v) {
        cout << "<AlterStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // Column
        if (v->n3) v->n3->accept(*this); // MapdDataT
        cout << "</AlterStmt>" << endl;
    }
    
    void XMLTranslator::visit(Column *v) {
        cout << "<Column>";
        if (v->name.first != "")
            cout << v->name.first << ".";
        cout << v->name.second;
        cout << "</Column>" << endl;
    }
    
    void XMLTranslator::visit(ColumnDef *v) {
        cout << "<ColumnDef>" << endl;
        if (v->n1) v->n1->accept(*this); // Column
        if (v->n2) v->n2->accept(*this); // MapdDataT
        cout << "</ColumnDef>" << endl;
    }
    
    void XMLTranslator::visit(ColumnDefList *v) {
        cout << "<ColumnDefList>" << endl;
        if (v->n1) v->n1->accept(*this); // ColumnDefList
        if (v->n2) v->n2->accept(*this); // ColumnDef
        cout << "</ColumnDefList>" << endl;
    }
    
    void XMLTranslator::visit(ColumnList *v) {
        cout << "<ColumnList>" << endl;
        if (v->n1) v->n1->accept(*this); // ColumnList
        if (v->n2) v->n2->accept(*this); // Column
        cout << "</ColumnList>" << endl;
    }
    
    void XMLTranslator::visit(Comparison *v) {
        cout << "<Comparison op='";

        if (v->op == ">")
            cout << "GT";
        else if (v->op == "<")
            cout << "LT";
        else if (v->op == "<=")
            cout << "LTE";
        else if (v->op == ">=")
            cout << "GTE";
        else if (v->op == "<=")
            cout << "LTE";
        else
            cout << "UNKNOWN OP";

        cout << "'>" << endl;
        
        if (v->n1) v->n1->accept(*this); // MathExpr
        if (v->n2) v->n2->accept(*this); // MathExpr
        
        cout << "</Comparison>" << endl;
    }
    
    void XMLTranslator::visit(CreateStmt *v) {
        cout << "<CreateStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // ColumnDefList
        cout << "</CreateStmt>" << endl;
    }
    
    void XMLTranslator::visit(DdlStmt *v) {
        cout << "<DdlStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // CreateStmt
        if (v->n2) v->n2->accept(*this); // DropStmt
        if (v->n3) v->n3->accept(*this); // AlterStmt
        if (v->n4) v->n4->accept(*this); // RenameStmt
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
        if (v->n1) v->n1->accept(*this); // Table
        cout << "</DropStmt>" << endl;
    }
    
    void XMLTranslator::visit(FromClause *v) {
        cout << "<FromClause>" << endl;
        if (v->n1) v->n1->accept(*this); // TableList
        if (v->n2) v->n2->accept(*this); // SelectStmt
        cout << "</FromClause>" << endl;
    }
    
    void XMLTranslator::visit(InsertColumnList *v) {
        cout << "<InsertColumnList>" << endl;
        if (v->n1) v->n1->accept(*this); // InsertColumnList
        cout << v->name;
        cout << "</InsertColumnList>" << endl;
    }
    
    void XMLTranslator::visit(InsertStmt *v) {
        cout << "<InsertStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // InsertColumnList
        if (v->n3) v->n3->accept(*this); // LiteralList
        cout << "</InsertStmt>" << endl;
    }
    
    void XMLTranslator::visit(Literal *v) {
        cout << "<Literal>" << endl;

        if (v->type == INT_TYPE)
            cout << "int " << v->intData;
        else if (v->type == FLOAT_TYPE)
            cout << "float " << v->realData;
        else if (v->type == BOOLEAN_TYPE)
            cout << "boolean " << "?"; // @todo boolean type
        else
            cout << "UNKNOWN ";

        cout << "</Literal>" << endl;
    }
    
    void XMLTranslator::visit(LiteralList *v) {
        cout << "<LiteralList>" << endl;
        if (v->n1) v->n1->accept(*this); // LiteralList
        if (v->n2) v->n2->accept(*this); // Literal
        cout << "</LiteralList>" << endl;
    }
    
    void XMLTranslator::visit(MapdDataT *v) {
        cout << "<MapdDataT>" << endl;
        if (v->type == INT_TYPE)
            cout << "INT_TYPE";
        else if (v->type == FLOAT_TYPE)
            cout << "FLOAT_TYPE";
        else if (v->type == BOOLEAN_TYPE)
            cout << "BOOLEAN_TYPE";
        else
            cout << "UNKNOWN ";
        cout << "</MapdDataT>" << endl;
    }
    
    void XMLTranslator::visit(MathExpr *v) {
        cout << "<MathExpr>" << endl;
        if (v->n1) v->n1->accept(*this); // MathExpr
        if (v->n2) v->n2->accept(*this); // MathExpr
        if (v->n3) v->n3->accept(*this); // Column
        if (v->n4) v->n4->accept(*this); // AggrExpr
        // op
        cout << "</MathExpr>" << endl;
    }
    
    void XMLTranslator::visit(OptAllDistinct *v) {
        cout << "<OptAllDistinct>" << endl;
        cout << "</OptAllDistinct>" << endl;
    }
    
    void XMLTranslator::visit(OptGroupby *v) {
        cout << "<OptGroupby>" << endl;
        if (v->n1) v->n1->accept(*this); // ColumnList
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
        if (v->n1) v->n1->accept(*this); // OrderByColumnList
        cout << "</OptOrderby>" << endl;
    }
    
    void XMLTranslator::visit(OptWhere *v) {
        cout << "<OptWhere>" << endl;
        if (v->n1) v->n1->accept(*this); // SearchCondition
        cout << "</OptWhere>" << endl;
    }
    
    void XMLTranslator::visit(OrderbyColumn *v) {
        cout << "<OrderbyColumn>" << endl;
        cout << "</OrderbyColumn>" << endl;
    }
    
    void XMLTranslator::visit(OrderbyColumnList *v) {
        cout << "<OrderbyColumnList>" << endl;
        if (v->n1) v->n1->accept(*this); // OrderbyColumnList
        if (v->n2) v->n2->accept(*this); // OrderbyColumn
        cout << "</OrderbyColumnList>" << endl;
    }
    
    void XMLTranslator::visit(Predicate *v) {
        cout << "<Predicate>" << endl;
        if (v->n1) v->n1->accept(*this); // Predicate
        if (v->n2) v->n2->accept(*this); // Predicate
        if (v->n3) v->n3->accept(*this); // Comparison
        // op
        cout << "</Predicate>" << endl;
    }
    
    void XMLTranslator::visit(RenameStmt *v) {
        cout << "<RenameStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // Table
        cout << "<name>" << v->name << "</name>";
        cout << "</RenameStmt>" << endl;
    }
    
    void XMLTranslator::visit(ScalarExpr *v) {
        cout << "<ScalarExpr>" << endl;
        if (v->n1) v->n1->accept(*this); // ScalarExpr
        if (v->n2) v->n2->accept(*this); // ScalarExpr
        if (v->n3) v->n3->accept(*this); // Literal
        if (v->n4) v->n4->accept(*this); // Column
        // op
        cout << "</ScalarExpr>" << endl;
    }
    
    void XMLTranslator::visit(ScalarExprList *v) {
        cout << "<ScalarExprList>" << endl;
        if (v->n1) v->n1->accept(*this); // ScalarExprList
        if (v->n2) v->n2->accept(*this); // ScalarExpr
        cout << "</ScalarExprList>" << endl;
    }
    
    void XMLTranslator::visit(SearchCondition *v) {
        cout << "<SearchCondition>" << endl;
        if (v->n1) v->n1->accept(*this); // Predicate
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
        cout << "<Selection>";
        if (v->all)
            cout << "*";
        else if (v->n1) {
            cout << endl;
            v->n1->accept(*this); // ScalarExprList
        }
        cout << "</Selection>" << endl;
    }
    
    void XMLTranslator::visit(SqlStmt *v) {
        cout << "<SqlStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // DmlStmt
        if (v->n2) v->n2->accept(*this); // DdlStmt
        cout << "</SqlStmt>" << endl;
    }
    
    void XMLTranslator::visit(Table *v) {
        cout << "<Table>";
        if (v->name.first != "")
            cout << v->name.first << ".";
        cout << v->name.second;
        cout << "</Table>" << endl;
    }
    
    void XMLTranslator::visit(TableList *v) {
        cout << "<TableList>" << endl;
        if (v->n1) v->n1->accept(*this); // TableList
        if (v->n2) v->n2->accept(*this); // Table
        cout << "</TableList>" << endl;
    }
    
} // SQL_Namespace