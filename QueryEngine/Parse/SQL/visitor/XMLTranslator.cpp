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
#include "../ast/DeleteStmt.h"
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
        printTabs();
        tabCount_++;
        cout << "<AggrExpr \'" << v->func << "\'>" << endl;

        if (v->n1) v->n1->accept(*this); // Column
        
        tabCount_--;
        printTabs();
        cout << "</AggrExpr>" << endl;
    }
    
    void XMLTranslator::visit(AlterStmt *v) {
        printTabs();
        tabCount_++;
        cout << "<AlterStmt>" << endl;
        
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // Column
        if (v->n3) v->n3->accept(*this); // MapdDataT
        
        tabCount_--;
        printTabs();
        cout << "</AlterStmt>" << endl;
    }
    
    void XMLTranslator::visit(Column *v) {
        printTabs();
        tabCount_++;
        cout << "<Column>";
        
        if (v->name.first != "")
            cout << v->name.first << ".";
        cout << v->name.second;
        cout << "</Column>" << endl;
        
        tabCount_--;
    }
    
    void XMLTranslator::visit(ColumnDef *v) {
        printTabs();
        tabCount_++;
        
        tabCount_--;
        printTabs();
        cout << "<ColumnDef>" << endl;
        
        if (v->n1) v->n1->accept(*this); // Column
        if (v->n2) v->n2->accept(*this); // MapdDataT
        
        tabCount_--;
        printTabs();
        cout << "</ColumnDef>" << endl;
    }
    
    void XMLTranslator::visit(ColumnDefList *v) {
        printTabs();
        tabCount_++;
        cout << "<ColumnDefList>" << endl;
        
        if (v->n1) v->n1->accept(*this); // ColumnDefList
        if (v->n2) v->n2->accept(*this); // ColumnDef
        
        tabCount_--;
        printTabs();
        cout << "</ColumnDefList>" << endl;
    }
    
    void XMLTranslator::visit(ColumnList *v) {
        printTabs();
        tabCount_++;
        cout << "<ColumnList>" << endl;
        
        if (v->n1) v->n1->accept(*this); // ColumnList
        if (v->n2) v->n2->accept(*this); // Column
        
        tabCount_--;
        printTabs();
        cout << "</ColumnList>" << endl;
    }
    
    void XMLTranslator::visit(Comparison *v) {
        printTabs();
        tabCount_++;
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
        
        tabCount_--;
        printTabs();
        cout << "</Comparison>" << endl;
    }
    
    void XMLTranslator::visit(CreateStmt *v) {
        printTabs();
        tabCount_++;
        cout << "<CreateStmt>" << endl;
        
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // ColumnDefList
        
        tabCount_--;
        printTabs();
        cout << "</CreateStmt>" << endl;
    }
    
    void XMLTranslator::visit(DdlStmt *v) {
        printTabs();
        tabCount_++;
        cout << "<DdlStmt>" << endl;
        
        if (v->n1) v->n1->accept(*this); // CreateStmt
        if (v->n2) v->n2->accept(*this); // DropStmt
        if (v->n3) v->n3->accept(*this); // AlterStmt
        if (v->n4) v->n4->accept(*this); // RenameStmt
        
        tabCount_--;
        printTabs();
        cout << "</DdlStmt>" << endl;
    }
    
    void XMLTranslator::visit(DeleteStmt *v) {
        printTabs();
        tabCount_++;
        cout << "<DeleteStmt>" << endl;
        
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // Predicate
        
        tabCount_--;
        printTabs();
        cout << "</DeleteStmt>" << endl;
    }
    
    void XMLTranslator::visit(DmlStmt *v) {
        printTabs();
        tabCount_++;
        printTabs();
        tabCount_++;
        cout << "<DmlStmt>" << endl;
        
        if (v->n1) v->n1->accept(*this); // InsertStmt
        if (v->n2) v->n2->accept(*this); // SelectStmt
        if (v->n3) v->n3->accept(*this); // DeleteStmt
        
        tabCount_--;
        printTabs();
        cout << "</DmlStmt>" << endl;
    }
    
    void XMLTranslator::visit(DropStmt *v) {
        printTabs();
        tabCount_++;
        cout << "<DropStmt>" << endl;
        
        if (v->n1) v->n1->accept(*this); // Table
        
        tabCount_--;
        printTabs();
        cout << "</DropStmt>" << endl;
    }
    
    void XMLTranslator::visit(FromClause *v) {
        printTabs();
        tabCount_++;
        cout << "<FromClause>" << endl;
        
        if (v->n1) v->n1->accept(*this); // TableList
        if (v->n2) v->n2->accept(*this); // SelectStmt
        
        tabCount_--;
        printTabs();
        cout << "</FromClause>" << endl;
    }
    
    void XMLTranslator::visit(InsertColumnList *v) {
        printTabs();
        tabCount_++;
        cout << "<InsertColumnList>" << endl;
        
        if (v->n1) v->n1->accept(*this); // InsertColumnList
        cout << v->name;
        
        tabCount_--;
        printTabs();
        cout << "</InsertColumnList>" << endl;
    }
    
    void XMLTranslator::visit(InsertStmt *v) {
        printTabs();
        tabCount_++;
        cout << "<InsertStmt>" << endl;
        
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // InsertColumnList
        if (v->n3) v->n3->accept(*this); // LiteralList
        
        tabCount_--;
        printTabs();
        cout << "</InsertStmt>" << endl;
    }
    
    void XMLTranslator::visit(Literal *v) {
        printTabs();
        tabCount_++;
        cout << "<Literal>" << endl;

        if (v->type == INT_TYPE)
            cout << "int " << v->intData;
        else if (v->type == FLOAT_TYPE)
            cout << "float " << v->realData;
        else if (v->type == BOOLEAN_TYPE)
            cout << "boolean " << "?"; // @todo boolean type
        else
            cout << "UNKNOWN ";
        
        tabCount_--;
        printTabs();
        cout << "</Literal>" << endl;
    }
    
    void XMLTranslator::visit(LiteralList *v) {
        printTabs();
        tabCount_++;
        cout << "<LiteralList>" << endl;
        
        if (v->n1) v->n1->accept(*this); // LiteralList
        if (v->n2) v->n2->accept(*this); // Literal
        
        tabCount_--;
        printTabs();
        cout << "</LiteralList>" << endl;
    }
    
    void XMLTranslator::visit(MapdDataT *v) {
        printTabs();
        tabCount_++;
        cout << "<MapdDataT>" << endl;
        
        if (v->type == INT_TYPE)
            cout << "INT_TYPE";
        else if (v->type == FLOAT_TYPE)
            cout << "FLOAT_TYPE";
        else if (v->type == BOOLEAN_TYPE)
            cout << "BOOLEAN_TYPE";
        else
            cout << "UNKNOWN ";
        
        tabCount_--;
        printTabs();
        cout << "</MapdDataT>" << endl;
    }
    
    void XMLTranslator::visit(MathExpr *v) {
        printTabs();
        tabCount_++;
        cout << "<MathExpr>" << endl;
        if (v->numericFlag) {
            printTabs();
            
            if (v->intFlag)
                cout << v->intVal;
            else if (v->floatFlag)
                cout << v->floatFlag;
            cout << endl;
            tabCount_--;
            printTabs();
        }
        else {
            if (v->n1) v->n1->accept(*this); // MathExpr
            if (v->n2) v->n2->accept(*this); // MathExpr
            if (v->n3) v->n3->accept(*this); // Column
            if (v->n4) v->n4->accept(*this); // AggrExpr
        }
        // op
        
        tabCount_--;
        printTabs();
        cout << "</MathExpr>" << endl;
    }
    
    void XMLTranslator::visit(OptAllDistinct *v) {
        printTabs();
        tabCount_++;
        cout << "<OptAllDistinct>" << endl;
        
        tabCount_--;
        printTabs();
        cout << "</OptAllDistinct>" << endl;
    }
    
    void XMLTranslator::visit(OptGroupby *v) {
        printTabs();
        tabCount_++;
        cout << "<OptGroupby>" << endl;
        
        if (v->n1) v->n1->accept(*this); // ColumnList
        
        tabCount_--;
        printTabs();
        cout << "</OptGroupby>" << endl;
    }
    
    void XMLTranslator::visit(OptHaving *v) {
        printTabs();
        tabCount_++;
        cout << "<OptHaving>" << endl;
        
        tabCount_--;
        printTabs();
        cout << "</OptHaving>" << endl;
    }
    
    void XMLTranslator::visit(OptLimit *v) {
        printTabs();
        tabCount_++;
        cout << "<OptLimit>" << endl;
        
        tabCount_--;
        printTabs();
        cout << "</OptLimit>" << endl;
    }
    
    void XMLTranslator::visit(OptOrderby *v) {
        printTabs();
        tabCount_++;
        cout << "<OptOrderby>" << endl;
        
        if (v->n1) v->n1->accept(*this); // OrderByColumnList
        
        tabCount_--;
        printTabs();
        cout << "</OptOrderby>" << endl;
    }
    
    void XMLTranslator::visit(OptWhere *v) {
        printTabs();
        tabCount_++;
        cout << "<OptWhere>" << endl;
        
        if (v->n1) v->n1->accept(*this); // SearchCondition
        
        tabCount_--;
        printTabs();
        cout << "</OptWhere>" << endl;
    }
    
    void XMLTranslator::visit(OrderbyColumn *v) {
        printTabs();
        tabCount_++;
        cout << "<OrderbyColumn>" << endl;
        
        cout << "</OrderbyColumn>" << endl;
    }
    
    void XMLTranslator::visit(OrderbyColumnList *v) {
        printTabs();
        tabCount_++;
        cout << "<OrderbyColumnList>" << endl;
        
        if (v->n1) v->n1->accept(*this); // OrderbyColumnList
        if (v->n2) v->n2->accept(*this); // OrderbyColumn
        
        tabCount_--;
        printTabs();
        cout << "</OrderbyColumnList>" << endl;
    }
    
    void XMLTranslator::visit(Predicate *v) {
        printTabs();
        tabCount_++;
        cout << "<Predicate>" << endl;
        
        if (v->n1) v->n1->accept(*this); // Predicate
        if (v->n2) v->n2->accept(*this); // Predicate
        if (v->n3) v->n3->accept(*this); // Comparison
        // op
        
        tabCount_--;
        printTabs();
        cout << "</Predicate>" << endl;
    }
    
    void XMLTranslator::visit(RenameStmt *v) {
        printTabs();
        tabCount_++;
        cout << "<RenameStmt>" << endl;
        
        if (v->n1) v->n1->accept(*this); // Table
        cout << "<name>" << v->name << "</name>";
        
        tabCount_--;
        printTabs();
        cout << "</RenameStmt>" << endl;
    }
    
    void XMLTranslator::visit(ScalarExpr *v) {
        printTabs();
        tabCount_++;
        cout << "<ScalarExpr>" << endl;
        
        if (v->n1) v->n1->accept(*this); // ScalarExpr
        if (v->n2) v->n2->accept(*this); // ScalarExpr
        if (v->n3) v->n3->accept(*this); // Literal
        if (v->n4) v->n4->accept(*this); // Column
        if (v->n5) v->n5->accept(*this); // AggrExpr
        // op
        
        tabCount_--;
        printTabs();
        cout << "</ScalarExpr>" << endl;
    }
    
    void XMLTranslator::visit(ScalarExprList *v) {
        printTabs();
        tabCount_++;
        cout << "<ScalarExprList>" << endl;
        
        if (v->n1) v->n1->accept(*this); // ScalarExprList
        if (v->n2) v->n2->accept(*this); // ScalarExpr
        
        tabCount_--;
        printTabs();
        cout << "</ScalarExprList>" << endl;
    }
    
    void XMLTranslator::visit(SearchCondition *v) {
        printTabs();
        tabCount_++;
        cout << "<SearchCondition>" << endl;
        
        if (v->n1) v->n1->accept(*this); // Predicate
        
        tabCount_--;
        printTabs();
        cout << "</SearchCondition>" << endl;
    }
    
    void XMLTranslator::visit(SelectStmt *v) {
        printTabs();
        tabCount_++;
        cout << "<SelectStmt>" << endl;
        
        if (v->n1) v->n1->accept(*this); // OptAllDistinct
        if (v->n2) v->n2->accept(*this); // Selection
        if (v->n3) v->n3->accept(*this); // FromClause
        if (v->n4) v->n4->accept(*this); // OptWhere
        if (v->n5) v->n5->accept(*this); // OptGroupBy
        if (v->n6) v->n6->accept(*this); // OptHaving
        if (v->n7) v->n7->accept(*this); // OptOrderBy
        if (v->n8) v->n8->accept(*this); // OptLimit
        
        tabCount_--;
        printTabs();
        cout << "</SelectStmt>" << endl;
    }
    
    void XMLTranslator::visit(Selection *v) {
        printTabs();
        tabCount_++;
        cout << "<Selection>";
        
        if (v->all) {
            cout << "*";
            tabCount_--;
            cout << "</Selection>" << endl;
            return;
        }
        if (v->n1) {
            cout << endl;
            v->n1->accept(*this); // ScalarExprList
        }
        
        tabCount_--;
        printTabs();
        cout << "</Selection>" << endl;
    }
    
    void XMLTranslator::visit(SqlStmt *v) {
        cout << "<SqlStmt>" << endl;
        if (v->n1) v->n1->accept(*this); // DmlStmt
        if (v->n2) v->n2->accept(*this); // DdlStmt
        cout << "</SqlStmt>" << endl;
    }
    
    void XMLTranslator::visit(Table *v) {
        printTabs();
        tabCount_++;
        cout << "<Table>";
        
        if (v->name.first != "")
            cout << v->name.first << ".";
        cout << v->name.second;
        cout << "</Table>" << endl;
        
        tabCount_--;
    }
    
    void XMLTranslator::visit(TableList *v) {
        printTabs();
        tabCount_++;
        cout << "<TableList>" << endl;
        
        if (v->n1) v->n1->accept(*this); // TableList
        if (v->n2) v->n2->accept(*this); // Table
        
        tabCount_--;
        printTabs();
        cout << "</TableList>" << endl;
    }
    
} // SQL_Namespace