#ifndef AST_SIMPLE_PRINTER_VISITOR_H
#define AST_SIMPLE_PRINTER_VISITOR_H

#include "Visitor.h"

#include "../ast/ColumnCommalist.h"
#include "../ast/TableConstraintDef.h"
#include "../ast/BaseTableDef.h"
#include "../ast/BaseTableElementCommalist.h"
#include "../ast/BaseTableElement.h"
#include "../ast/Program.h"
#include "../ast/Table.h"
#include "../ast/Schema.h"
#include "../ast/SQL.h"
#include "../ast/ColumnDef.h"
#include "../ast/ColumnDefOpt.h"
#include "../ast/ColumnDefOptList.h"
#include "../ast/Column.h"
#include "../ast/Literal.h"
#include "../ast/DataType.h"
#include "../ast/SQLList.h"

#include "../ast/ManipulativeStatement.h"
#include "../ast/SelectStatement.h"
#include "../ast/Selection.h"
#include "../ast/OptAllDistinct.h"
#include "../ast/TableExp.h"
#include "../ast/FromClause.h"
#include "../ast/TableRefCommalist.h"
#include "../ast/TableRef.h"

#include <iostream>
using std::cout;
using std::endl;

#define TAB_SIZE 2 // number of spaces in a tab

enum tabFlag {INCR, DECR, NONE};
/**
 * @todo brief and detailed descriptions
 */
class XMLTranslator : public Visitor {

public:

    static void printTabs(tabFlag flag) {
        if (flag == INCR)
            tabLevel_++;
        //cout << tabLevel_;
        for (int i = 0; i < tabLevel_; ++i)
            for (int j = 0; j < TAB_SIZE; ++j)
                cout << " ";
        if (flag == DECR)
            tabLevel_--;
    }

    void visit(class BaseTableDef *v) {
        printTabs(INCR);
        cout << "<BaseTableDef ddlCmd='" << v->ddlCmd << "'>" << endl;

        if (v->ddlCmd == "CREATE") {
            if (!v->tbl || !v->btec) {
                printTabs(INCR);
                cout << "<ERROR msg='Invalid parameters to ddlCmd: " << v->ddlCmd << "'/>" << endl;
                tabLevel_--;
            }
            else {
                v->tbl->accept(*this);
                v->btec->accept(*this);
            }
        }
        else if (v->ddlCmd == "DROP") {
            if (!v->tbl) {
                printTabs(INCR);
                cout << "<ERROR msg='Invalid parameters to ddlCmd: " << v->ddlCmd << "'>" << endl;
                printTabs(DECR);
            }
            else
                v->tbl->accept(*this);
        }
        else {
            printTabs(INCR);
            cout << "<ERROR msg='Invalid ddlCmd: " << v->ddlCmd << "'>" << endl;
            printTabs(DECR);
        }

        printTabs(DECR);
        cout << "</BaseTableDef>" << endl;
    }

    void visit(class TableConstraintDef *v) {
        printTabs(INCR);
        cout << "<colDefOpt>" << endl;

         /* rule_Flags: 
        0 NOT NULL
        1 NOT NULL PRIMARY KEY
        2 NOT NULL UNIQUE
        3 DEFAULT [literal]
        4 DEFAULT NULL
        5 DEFAULT USER
        6 CHECK [search condition]
        7 REFERENCES [table]
        8 REFERENCES [table] [column_commalist]
        */

        int rf = v->rule_Flag;
        if (rf == 0) { // Handle UNIQUE column_commalist
            printTabs(INCR);
            cout << "<UNIQUE>" << endl;

            v->colCom1->accept(*this);
            
            cout << "</>" << endl;
            printTabs(DECR);
        }
        else if (rf == 1) { // Handle PRIMARY KEY column_commalist
            printTabs(INCR);
            cout << "<PRIMARY KEY>" << endl;

            v->colCom1->accept(*this);

            cout << "</>" << endl;
            printTabs(DECR);
        }
        else if (rf == 2) { // Handle FOREIGN KEY column_commalist REFERENCES table [possibly] (column_commalist)
            printTabs(INCR);
            cout << "<FOREIGN KEY>" << endl;

            v->colCom1->accept(*this);
            
            cout << "</>" << endl << "<REFERENCES>" << endl;
            
            v->tbl->accept(*this);

            cout << "</>" << endl;

            if(v->colCom2) {
                v->colCom2->accept(*this);
            }

            printTabs(DECR);
            cout << "</BaseTableDef>" << endl;
        }/*
        else if (rf == 3) { // Handle CHECK search_condition
            printTabs(INCR);
            cout << "<DEFAULT>" << endl;
            v->srchCon->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        } */
       cout << "</TableConstraintDef>" << endl;
    }

    void visit(class ColumnDefOpt *v) {
        printTabs(INCR);
        cout << "<colDefOpt>" << endl;

         /* rule_Flags: 
        0 NOT NULL
        1 NOT NULL PRIMARY KEY
        2 NOT NULL UNIQUE
        3 DEFAULT [literal]
        4 DEFAULT NULL
        5 DEFAULT USER
        6 CHECK [search condition]
        7 REFERENCES [table]
        8 REFERENCES [table] [column_commalist]
        */

        int rf = v->rule_Flag;
        if (rf == 0) { // Handle NOT NULL
            printTabs(INCR);
            cout << "<NOT NULL> </>" << endl;
            printTabs(DECR);
        }
        else if (rf == 1) { // Handle NOT NULL PRIMARY KEY
            printTabs(INCR);
            cout << "<NOT NULL PRIMARY KEY> </>" << endl;
            printTabs(DECR);
        }
        else if (rf == 2) { // Handle NOT NULL UNIQUE
            printTabs(INCR);
            cout << "<NOT NULL UNIQUE> </>" << endl;
            printTabs(DECR);
        }
        else if (rf == 3) { // Handle DEFAULT literal
            printTabs(INCR);
            cout << "<DEFAULT>" << endl;
            v->lit->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        } 
        else if (rf == 4) { // Handle DEFAULT NULL
            printTabs(INCR);
            cout << "<DEFAULT NULL> </>" << endl;
            printTabs(DECR);
        }
        else if (rf == 5) { // Handle DEFAULT USER
            printTabs(INCR);
            cout << "<DEFAULT USER> </>" << endl;
            printTabs(DECR);
        } 
       /*else if (rf == 6) { // Handle CHECK search_condition
            printTabs(INCR);
            cout << "<CHECK>" << endl;
            v->srchCon->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        } */
        else if (rf == 7) { // Handle REFERENCES table
            printTabs(INCR);
            cout << "<REFERENCES>" << endl;
            v->tbl->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        } 
        else if (rf == 8) { // Handle REFERENCES table commalist 
            printTabs(INCR);
            cout << "<REFERENCES>" << endl;
            v->tbl->accept(*this);
            v->colComList->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        }

        printTabs(DECR);
        cout << "</ColDefOpt>" << endl;
    }

    void visit(class ColumnDefOptList *v) {
        printTabs(INCR);
        cout << "<ColumnDefOptList>" << endl;

        if (v->colDefOptList) v->colDefOptList->accept(*this);
        v->colDefOpt->accept(*this);

        printTabs(DECR);
        cout << "</ColumnDefOptList>" << endl;
    }
    
    void visit(class ColumnDef *v) {
        printTabs(INCR);
        cout << "<ColumnDef>" << endl;
        
        v->col->accept(*this);
        v->dType->accept(*this);
        if (v->colDefOptList) v->colDefOptList->accept(*this);
        
        printTabs(DECR);
        cout << "</ColumnDef>" << endl;
    }

    void visit(class BaseTableElement *v) {
        printTabs(INCR);
        cout << "<BaseTableElement>" << endl;

        if (v->colDef) v->colDef->accept(*this);
        if (v->tblConDef) v->tblConDef->accept(*this);

        printTabs(DECR);
        cout << "</BaseTableElement>" << endl;
    }
    
    void visit(class BaseTableElementCommalist *v) {
        printTabs(INCR);
        cout << "<BaseTableElementCommalist>" << endl;

        if (v->btec) v->btec->accept(*this);
        v->bte->accept(*this);

        printTabs(DECR);
        cout << "</BaseTableElementCommalist>" << endl;
    }
    
    void visit(class ColumnCommalist *v) {
        printTabs(INCR);
        cout << "<ColumnCommalist>" << endl;

        if (v->colCom) v->colCom->accept(*this);
        v->col->accept(*this);

        printTabs(DECR);
        cout << "</ColumnCommalist>" << endl;
    }

    void visit(class TableRef *v) {
        printTabs(INCR);
        cout << "<TableRef>" << endl;

        if (v->tbl) v->tbl->accept(*this);

        printTabs(DECR);
        cout << "</TableRef>" << endl;
    }
    
    void visit(class TableRefCommalist *v) {
        printTabs(INCR);
        cout << "<TableRefCommalist>" << endl;

        if (v->trc) v->trc->accept(*this);
        v->tr->accept(*this);

        printTabs(DECR);
        cout << "</TableRefCommalist>" << endl;
    }

    void visit(class FromClause *v) {
        printTabs(INCR);
        cout << "<FromClause>" << endl;

        v->trc->accept(*this);

        printTabs(DECR);
        cout << "</FromClause>" << endl;
    }
    
    void visit(TableExp *v) {
        printTabs(INCR);
        cout << "<TableExp>" << endl;

        v->fc->accept(*this);

        printTabs(DECR);
        cout << "</TableExp>" << endl;
    }
    void visit(class Selection *v) {
        printTabs(INCR);
        cout << "<*>" << endl;
        // else: print the Scalar Expressions
    }

    void visit(class OptAllDistinct *v) {
        printTabs(INCR);
        cout << "<OptAllDistinct ddlCmd='" << v->ddlCmd << "'>" << endl;

        printTabs(DECR);
        cout << "</OptAllDistinct>" << endl;
    }
    
    void visit(class SelectStatement *v) {
        printTabs(INCR);
        cout << "<SelectStatement>" << endl;

        v->OAD->accept(*this);
        v->sel->accept(*this);
        v->tblExp->accept(*this);

        printTabs(DECR);
        cout << "</SelectStatement>" << endl;
    }
    
    void visit(class ManipulativeStatement *v) {
        printTabs(INCR);
        cout << "<ManipulativeStatement>" << endl;

        if (v->selSta) v->selSta->accept(*this);
/*        if (v->UPS) v->UPS->accept(*this);
        if (v->USS) v->USS->accept(*this);
        if (v->inSta) v->inSta->accept(*this); */

        printTabs(DECR);
        cout << "</ManipulativeStatement>" << endl;
    }

    void visit(class Schema *v) {
        printTabs(INCR);
        cout << "<Schema>" << endl;

        v->btd->accept(*this);

        printTabs(DECR);
        cout << "</Schema>" << endl;
    }
    
    void visit(class SQL *v) {
        printTabs(INCR);
        cout << "<SQL>" << endl;
        
        if (v->sch) v->sch->accept(*this);
        if (v->manSta) v->manSta->accept(*this);

        printTabs(DECR);
        cout << "</SQL>" << endl;
    }

    void visit(class SQLList *v) {
        printTabs(INCR);
        cout << "<SQLList>" << endl;
        

        if (v->sqlList) { v->sqlList->accept(*this);  }
        v->sql->accept(*this);

        printTabs(DECR);
        cout << "</SQLList>" << endl;
    }


    void visit(class Program *v) {
        printTabs(NONE);
        cout << "<Program>" << endl;

        v->sqlList->accept(*this);

        printTabs(DECR);
        cout << "</Program>" << endl;
    }

    void visit(class DataType *v) {
        printTabs(NONE);

        if (v->size1 != 0) 
        cout << "<DataType flag=" << v->dataType_Flag << "size1" << v->size1 << "/>" << endl;
        
        else 
        cout << "<DataType flag=" << v->dataType_Flag << "/>" << endl;
       
        //printTabs(DECR);
    }

    void visit(class Literal *v) {
        printTabs(INCR);

        if (v->name1 != "") cout << "<Literal string='" << v->name1 << "'/>" << endl;
        if (v->int1 != 0) cout << "<Literal int = " << v->int1 << "'/>" << endl;

        printTabs(DECR);
        cout << "</Literal>" << endl;
    }

    void visit(class Column *v) {
        printTabs(INCR);

        cout << "<Column name='" << v->name1 << "'/>" << endl;
        
        printTabs(DECR);
        cout << "</Column>" << endl;
    }

    void visit(class Table *v) {
        printTabs(INCR);

        if (v->name2 != "")
            cout << "<Table name1='" << v->name1 << "' name2='" << v->name2 << "' />" << endl;
        else
            cout << "<Table name='" << v->name1 << "'/>" << endl;
        
        printTabs(DECR);
        cout << "</Table>" << endl;
    }

private:
    static int tabLevel_;   /**< Keeps track of number of tabs to print out on a line. */
};

// Definition of static memble for SimplePrinterVisitor
int XMLTranslator::tabLevel_ = 0;

#endif // AST_SIMPLE_PRINTER_VISITOR_H
