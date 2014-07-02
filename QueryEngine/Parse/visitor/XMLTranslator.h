#ifndef AST_SIMPLE_PRINTER_VISITOR_H
#define AST_SIMPLE_PRINTER_VISITOR_H

#include "Visitor.h"

#include "../ast/BaseTableDef.h"
#include "../ast/BaseTableElementCommalist.h"
#include "../ast/Program.h"
#include "../ast/Table.h"
#include "../ast/Schema.h"
#include "../ast/SQL.h"
#include "../ast/SQLList.h"

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
    
    void visit(class BaseTableElement *v) {
        printTabs(INCR);
        cout << "<BaseTableElement>" << endl;

        //v->sqlList->accept(*this);

        printTabs(DECR);
        cout << "</BaseTableElement>" << endl;
    }
    
    void visit(class BaseTableElementCommalist *v) {
        printTabs(INCR);
        cout << "<BaseTableElementCommalist>" << endl;

        //v->sqlList->accept(*this);

        printTabs(DECR);
        cout << "</BaseTableElementCommalist>" << endl;
    }
    
    void visit(class Name *v) {
        printTabs(INCR);
        cout << "<Name>" << v->name;
        cout << "</Name>" << endl;
        tabLevel_--;
    }

    void visit(class Program *v) {
        printTabs(NONE);
        cout << "<Program>" << endl;

        v->sqlList->accept(*this);

        printTabs(DECR);
        cout << "</Program>" << endl;
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
        
        v->sch->accept(*this);
        
        printTabs(DECR);
        cout << "</SQL>" << endl;
    }

    void visit(class SQLList *v) {
        printTabs(INCR);
        cout << "<SQLList>" << endl;
        
        v->sql->accept(*this);
        
        printTabs(DECR);
        cout << "</SQLList>" << endl;
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