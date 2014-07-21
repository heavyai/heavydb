#ifndef REL_ALG_SIMPLE_PRINTER_VISITOR_H
#define REL_ALG_SIMPLE_PRINTER_VISITOR_H

#include "../visitor/Visitor.h"

#include "../ast/RelAlgNode.h"
#include "../ast/UnaryOp.h"
#include "../ast/BinaryOp.h"

#include "../ast/AggrExpr.h"
#include "../ast/AggrList.h"
#include "../ast/AntijoinOp.h"
#include "../ast/Attribute.h"
#include "../ast/AttrList.h"
#include "../ast/Comparison.h"
#include "../ast/DiffOp.h"
#include "../ast/Expr.h"
#include "../ast/ExtendOp.h"
#include "../ast/GroupbyOp.h"
#include "../ast/JoinOp.h"
#include "../ast/MathExpr.h"
#include "../ast/OuterjoinOp.h"
#include "../ast/Predicate.h"
#include "../ast/ProductOp.h"
#include "../ast/Program.h"
#include "../ast/ProjectOp.h"
#include "../ast/Relation.h"
#include "../ast/RelExpr.h"
#include "../ast/RelExprList.h"
#include "../ast/RenameOp.h"
#include "../ast/SelectOp.h"
#include "../ast/SemijoinOp.h"
#include "../ast/SortOp.h"
#include "../ast/UnionOp.h"

#include <iostream>
using std::cout;
using std::endl;

#define TAB_SIZE 2 // number of spaces in a tab

namespace RA_Namespace {

/**
 * @todo brief and detailed descriptions
 */
class XMLTranslator : public RA_Namespace::Visitor {

public:
    enum tabFlag {INCR, DECR, NONE};
    
    XMLTranslator() { tabLevel_ = -1; }
     
    void printTabs(tabFlag flag) {
        if (flag == INCR)
            tabLevel_++;

        for (int i = 0; i < tabLevel_; ++i)
            for (int j = 0; j < TAB_SIZE; ++j)
                cout << " ";
        if (flag == DECR)
            tabLevel_--;
    }

    void visit(class AggrExpr *v) {
        printTabs(INCR);
        cout << "<AggrExpr function=\"" << v->func << "\">" << endl;
        
        if (v->n1) v->n1->accept(*this);

        printTabs(DECR);
        cout << "</AggrExpr>" << endl;
    }

    void visit(class AggrList *v) {
        printTabs(INCR);
        cout << "<AggrList>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        printTabs(DECR);
        cout << "</AggrList>" << endl;
    }

    void visit(class AntijoinOp *v) {
        printTabs(INCR);
        cout << "<AntijoinOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        printTabs(DECR);
        cout << "</AntijoinOp>" << endl;
    }

    void visit(class Attribute *v) {
        printTabs(INCR);
        cout << "<Attribute>";
        cout << v->name1;
        if (v->name2 != "")
            cout << "." << v->name2;
        cout << "</Attribute>" << endl;
        tabLevel_--;
    }

    void visit(class AttrList *v) {
        printTabs(INCR);
        cout << "<AttrList>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        printTabs(DECR);
        cout << "</AttrList>" << endl;
    }

    void visit(class Comparison *v) {
        printTabs(INCR);
        if (v->op != "")
            cout << "<Comparison op=\"" << v->op << "\">" << endl;
        else
            cout << "<Comparison op=\"NOOP\">" << endl;

        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        printTabs(DECR);
        cout << "</Comparison>" << endl;
    }

    void visit(class DiffOp *v) {
        printTabs(INCR);
        cout << "<DiffOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        printTabs(DECR);
        cout << "</DiffOp>" << endl;
    }

    void visit(class Expr *v) {
        if (v->str == "") {
            printTabs(INCR);
            cout << "<Expr>" << endl;
            if (v->n1) v->n1->accept(*this);
            if (v->n2) v->n2->accept(*this);
            printTabs(DECR);
            cout << "</Expr>" << endl;
        }
        else {
            printTabs(INCR);
            cout << "<Expr>" << v->str << "</Expr>" << endl;
            tabLevel_--;
        }
    }

    void visit(class ExtendOp *v) {
        printTabs(INCR);
        cout << "<ExtendOp name=\"" << v->name << "\">" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        printTabs(DECR);
        cout << "</ExtendOp>" << endl;
    }

    void visit(class GroupbyOp *v) {
        printTabs(INCR);
        cout << "<GroupbyOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        printTabs(DECR);
        cout << "</GroupbyOp>" << endl;
    }

    void visit(class JoinOp *v) {
        printTabs(INCR);
        cout << "<JoinOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        printTabs(DECR);
        cout << "</JoinOp>" << endl;
    }

    void visit(class MathExpr *v) {
        printTabs(INCR);
        if (v->n1 || v->n2 || v->n3 || v->n4) { 
            if (v->op != "")
                cout << "<MathExpr op=\"" << v->op << "\">" << endl;
            else
                cout << "<MathExpr>" << endl;
            if (v->n1) v->n1->accept(*this);
            if (v->n2) v->n2->accept(*this);
            if (v->n3) v->n3->accept(*this);
            if (v->n4) v->n4->accept(*this);
            printTabs(DECR);
            cout << "</MathExpr>" << endl;
        }
        else {
            cout << "<MathExpr>";
            if (v->intFloatFlag)
                cout << v->intVal << "</MathExpr>" << endl;
            else
                cout << v->floatVal << "</MathExpr>" << endl;
            tabLevel_--;
        }
    }

    void visit(class OuterjoinOp *v) {
        printTabs(INCR);
        cout << "<OuterjoinOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);

        printTabs(DECR);
        cout << "</OuterjoinOp>" << endl;
    }

    void visit(class Predicate *v) {
        printTabs(INCR);
        if (v->op != "")
            cout << "<Predicate op=\"" << v->op << "\">" << endl;
        else
            cout << "<Predicate>" << endl;

        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);

        printTabs(DECR);
        cout << "</Predicate>" << endl;
    }

    void visit(class ProductOp *v) {
        printTabs(INCR);
        cout << "<ProductOp>" << endl;

        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);

        printTabs(DECR);
        cout << "</ProductOp>" << endl;
    }

    void visit(class Program *v) {
        printTabs(INCR);
        cout << "<Program>" << endl;
        
        if (v->n1) v->n1->accept(*this);

        printTabs(DECR);
        cout << "</Program>" << endl;
    }

    void visit(class ProjectOp *v) {
        printTabs(INCR);
        cout << "<ProjectOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        printTabs(DECR);
        cout << "</ProjectOp>" << endl;
    }

    void visit(class Relation *v) {
        printTabs(INCR);
        cout << "<Relation>";
        cout << v->name;
        cout << "</Relation>" << endl;
        tabLevel_--;
    }

    void visit(class RelExpr *v) {
        printTabs(INCR);
        cout << "<RelExpr>" << endl;
        
        if (v->n1)
            v->n1->accept(*this);
        else if (v->n2)
            v->n2->accept(*this);
        else if (v->n3)
            v->n3->accept(*this);
        else if (v->n4)
            v->n4->accept(*this);

        printTabs(DECR);
        cout << "</RelExpr>" << endl;
    }

    void visit(class RelExprList *v) {
        printTabs(INCR);
        cout << "<RelExprList>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);

        printTabs(DECR);
        cout << "</RelExprList>" << endl;
    }

    void visit(class RenameOp *v) {
        printTabs(INCR);
        cout << "<RenameOp old=\"" << v->name1 << "\" new=\"" << v->name2 << "\">" << endl;
        
        if (v->n1) v->n1->accept(*this);

        printTabs(DECR);
        cout << "</RenameOp>" << endl;
    }

    void visit(class SelectOp *v) {
        printTabs(INCR);
        cout << "<SelectOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);

        printTabs(DECR);
        cout << "</SelectOp>" << endl;
    }

    void visit(class SemijoinOp *v) {
        printTabs(INCR);
        cout << "<SemijoinOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);

        printTabs(DECR);
        cout << "</SemijoinOp>" << endl;
    }

    void visit(class SortOp *v) {
        printTabs(INCR);
        cout << "<SortOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);

        printTabs(DECR);
        cout << "</SortOp>" << endl;
    }

    void visit(class UnionOp *v) {
        printTabs(INCR);
        cout << "<UnionOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        printTabs(DECR);
        cout << "</UnionOp>" << endl;
    }


private:
    int tabLevel_ ;   /**< Keeps track of number of tabs to print out on a line. */
};

} // RA_Namespace

#endif // REL_ALG_SIMPLE_PRINTER_VISITOR_H
