#ifndef REL_ALG_SIMPLE_PRINTER_VISITOR_H
#define REL_ALG_SIMPLE_PRINTER_VISITOR_H

#include "Visitor.h"
#include "../relAlg/RelAlgNode.h"
#include "../relAlg/Program.h"
#include "../relAlg/RelExprList.h"
#include "../relAlg/RelExpr.h"
#include "../relAlg/UnaryOp.h"
#include "../relAlg/BinaryOp.h"
#include "../relAlg/MathExpr.h"
#include "../relAlg/SelectOp.h"
#include "../relAlg/ProjectOp.h"
#include "../relAlg/SortOp.h"
#include "../relAlg/ExtendOp.h"
#include "../relAlg/GroupByOp.h"
#include "../relAlg/RenameOp.h"

#include "../relAlg/JoinOp.h"
#include "../relAlg/SemijoinOp.h"
#include "../relAlg/ProductOp.h"
#include "../relAlg/OuterjoinOp.h"
#include "../relAlg/AntijoinOp.h"
#include "../relAlg/UnionOp.h"
#include "../relAlg/AggrExpr.h"
#include "../relAlg/AggrList.h"
#include "../relAlg/AttrList.h"
#include "../relAlg/Attribute.h"
#include "../relAlg/Relation.h"
#include "../relAlg/Data.h"

#include "../relAlg/RA_Predicate.h"
#include "../relAlg/Comparison.h"
#include "../relAlg/Compared.h"
#include "../relAlg/CompOp.h"
#include "../relAlg/Table.h"

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
    XMLTranslator() { tabLevel_ = 0; }
     
    void printTabs(tabFlag flag) {
        if (flag == INCR)
            tabLevel_++;
        //cout << tabLevel_;
        for (int i = 0; i < tabLevel_; ++i)
            for (int j = 0; j < TAB_SIZE; ++j)
                cout << " ";
        if (flag == DECR)
            tabLevel_--;
    }

    void visit(class Program *v) {
        printTabs(INCR);
        cout << "<Program>" << endl;
        
        if (v->rel) v->rel->accept(*this);

        printTabs(DECR);
        cout << "</Program>" << endl;
    }

    void visit(class RelExprList *v) {
        printTabs(INCR);
        cout << "<RelExprList>" << endl;

        if (v->relex) v->relex->accept(*this);
        if (v->relexlist) v->relexlist->accept(*this);

        printTabs(DECR);
        cout << "</RelExprList>" << endl;
    }

    void visit(class RelExpr *v) {
        printTabs(INCR);
        cout << "<RelExpr>" << endl;

        if (v->relex) v->relex->accept(*this);
        if (v->rel) v->rel->accept(*this);
        if (v->uno) v->uno->accept(*this);
        if (v->dos) v->dos->accept(*this);

        printTabs(DECR);
        cout << "</RelExpr>" << endl;
    }

    void visit(class UnaryOp *v) {
        printTabs(INCR);
        cout << "<UnaryOp>" << endl;
        
        if (v->relex) v->relex->accept(*this);
        
        printTabs(DECR);
        cout << "</UnaryOp>" << endl;
    }

    void visit(class BinaryOp *v) {
        printTabs(INCR);
        cout << "<BinaryOp>" << endl;
            
        if (v->relex1) v->relex1->accept(*this);
        if (v->relex2) v->relex2->accept(*this);
        
        printTabs(DECR);
        cout << "</BinaryOp>" << endl;
    }

    void visit(class MathExpr *v) {
        printTabs(INCR);
        cout << "<MathExpr>" << endl;
        
        if (v->attr) v->attr->accept(*this);
        if (v->data) v->data->accept(*this);
        if (v->agex) v->agex->accept(*this);
        if (v->me1) v->me1->accept(*this);
        if (v->me2) v->me2->accept(*this);

        printTabs(DECR);
        cout << "</MathExpr>" << endl;
    }

    void visit(class SelectOp *v) {
        printTabs(INCR);
        cout << "<SelectOp>" << endl;
        
        if (v->pred) v->pred->accept(*this);
        if (v->relex) v->relex->accept(*this);
        
        printTabs(DECR);
        cout << "</SelectOp>" << endl;
    }

    void visit(class ProjectOp *v) {
        printTabs(INCR);
        cout << "<ProjectOp>" << endl;
        
        if (v->atLi) v->atLi->accept(*this);
        if (v->relex) v->relex->accept(*this);

        printTabs(DECR);
        cout << "</ProjectOp>" << endl;
    }

    void visit(class SortOp *v) {
        printTabs(INCR);
        cout << "<SortOp>" << endl;
        
        if (v->relex) v->relex->accept(*this);
        if (v->atLi) v->atLi->accept(*this);

        printTabs(DECR);
        cout << "</SortOp>" << endl;
    }

    void visit(class ExtendOp *v) {
        printTabs(INCR);
        cout << "<ExtendOp>" << endl;
        
        if (v->me) v->me->accept(*this);
//        if (v->data) v->data->accept(*this);

        printTabs(DECR);
        cout << "</ExtendOp>" << endl;
    }

    void visit(class GroupByOp *v) {
        printTabs(INCR);
        cout << "<GroupByOp>" << endl;
        
        if (v->agLi) v->agLi->accept(*this);
        if (v->atLi) v->atLi->accept(*this);
        if (v->relex) v->relex->accept(*this);
        
        printTabs(DECR);
        cout << "</GroupByOp>" << endl;
    }

    void visit(class RenameOp *v) {
        printTabs(INCR);
        cout << "<RenameOp>" << endl;
    
        if (v->relex) v->relex->accept(*this);
        if (v->attr) v->attr->accept(*this);
        
        printTabs(DECR);
        cout << "</RenameOp>" << endl;
    }

    void visit(class JoinOp *v) {
        printTabs(INCR);
        cout << "<JoinOp>" << endl;
        
        if (v->relex1) v->relex1->accept(*this);
        if (v->relex2) v->relex2->accept(*this);
        if (v->pred) v->pred->accept(*this);

        printTabs(DECR);
        cout << "</JoinOp>" << endl;
    }

    void visit(class SemijoinOp *v) {
        printTabs(INCR);
        cout << "<SemijoinOp>" << endl;
        
        if (v->relex1) v->relex1->accept(*this);
        if (v->relex2) v->relex2->accept(*this);
        if (v->pred) v->pred->accept(*this);
        
        printTabs(DECR);
        cout << "</SemijoinOp>" << endl;
    }

    void visit(class ProductOp *v) {
        printTabs(INCR);
        cout << "<ProductOp>" << endl;
        
        if (v->relex1) v->relex1->accept(*this);
        if (v->relex2) v->relex2->accept(*this);

        printTabs(DECR);
        cout << "</ProductOp>" << endl;
    }

    void visit(class OuterjoinOp *v) {
        printTabs(INCR);
        cout << "<OuterjoinOp>" << endl;
        
        if (v->relex1) v->relex1->accept(*this);
        if (v->relex2) v->relex2->accept(*this);
        if (v->pred) v->pred->accept(*this);

        printTabs(DECR);
        cout << "</OuterjoinOp>" << endl;
    }

    void visit(class AntijoinOp *v) {
        printTabs(INCR);
        cout << "<AntijoinOp>" << endl;
        
        if (v->relex1) v->relex1->accept(*this);
        if (v->relex2) v->relex2->accept(*this);
        if (v->pred) v->pred->accept(*this);
        
        printTabs(DECR);
        cout << "</AntijoinOp>" << endl;
    }

    void visit(class UnionOp *v) {
        printTabs(INCR);
        cout << "<UnionOp>" << endl;
        
        if (v->relex1) v->relex1->accept(*this);
        if (v->relex2) v->relex2->accept(*this);
        
        printTabs(DECR);
        cout << "</UnionOp>" << endl;
    }

    void visit(class AggrExpr *v) {
        printTabs(INCR);
        cout << "<AggrExpr>" << endl;
    
        if (v->attr) v->attr->accept(*this);
        
        printTabs(DECR);
        cout << "</AggrExpr>" << endl;
    }

    void visit(class AggrList *v) {
        printTabs(INCR);
        cout << "<AggrList>" << endl;
        

        if (v->agex) v->agex->accept(*this);
        if (v->agLi) v->agLi->accept(*this);
        
        printTabs(DECR);
        cout << "</AggrList>" << endl;
    }

    void visit(class AttrList *v) {
        printTabs(INCR);
        cout << "<AttrList>" << endl;
        

        if (v->atLi) v->atLi->accept(*this);
        if (v->at) v->at->accept(*this);
    
        printTabs(DECR);
        cout << "</AttrList>" << endl;
    }

    void visit(class Attribute *v) {
        printTabs(INCR);
        cout << "<Attribute>" << endl;
        
        printTabs(DECR);
        cout << "</Attribute>" << endl;
    }

    void visit(class Relation *v) {
        printTabs(INCR);
        cout << "<Relation>" << endl;

        if (v->tbl) v->tbl->accept(*this);

        printTabs(DECR);
        cout << "</Relation>" << endl;
    }

    void visit(class Data *v) {
        printTabs(INCR);
        cout << "<Data>" << endl;
        
        printTabs(DECR);
        cout << "</Data>" << endl;
    }

    void visit(class RA_Predicate *v) {
        printTabs(INCR);
        cout << "<Predicate>" << endl;
        
        if (v->c) v->c->accept(*this);
        if (v->p1) v->p1->accept(*this);
        if (v->p2) v->p2->accept(*this);
        
        printTabs(DECR);
        cout << "</Predicate>" << endl;
    }

    void visit(class Comparison *v) {
        printTabs(INCR);
        cout << "<Comparison>" << endl;
        
        if (v->c1) v->c1->accept(*this);
        if (v->co) v->co->accept(*this);
        if (v->c2) v->c2->accept(*this);
        
        printTabs(DECR);
        cout << "</Comparison>" << endl;
    }

    void visit(class Compared *v) {
        printTabs(INCR);
        cout << "<Compared>" << endl;
        
//        if (v->a) v->a->accept(*this);
        if (v->me) v->me->accept(*this);
        
        printTabs(DECR);
        cout << "</Compared>" << endl;
    }

    void visit(class CompOp *v) {
        printTabs(INCR);
        cout << "<CompOp>" << endl;
        
        printTabs(DECR);
        cout << "</CompOp>" << endl;
    }

    void visit(class Table *v) {
        printTabs(INCR);
        cout << "<Table>" << endl;
        
        cout << "Name is " << v->name1 << endl;

        printTabs(DECR);
        cout << "</Table>" << endl;
    }

private:
    int tabLevel_ ;   /**< Keeps track of number of tabs to print out on a line. */
};

#endif // REL_ALG_SIMPLE_PRINTER_VISITOR_H
