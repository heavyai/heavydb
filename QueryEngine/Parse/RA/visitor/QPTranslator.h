#ifndef REL_ALG_QP_VISITOR_H
#define REL_ALG_QP_VISITOR_H

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

namespace RA_Namespace {

/**
 * @todo brief and detailed descriptions
 */
class QPTranslator : public RA_Namespace::Visitor {

public:
    
    QPTranslator() { }

    void visit(class AggrExpr *v) {
        cout << "<AggrExpr>";
    }

    void visit(class AggrList *v) {

    }

    void visit(class AntijoinOp *v) {
 
    }

    void visit(class Attribute *v) {
        cout << v->name1;
        if (v->name2 != "")
            cout << "." << v->name2;
    }

    void visit(class AttrList *v) {
        if (v->n1) v->n1->accept(*this);
        if (v->n2) {
            v->n2->accept(*this);
            cout << " ";
        }
    }

    void visit(class Comparison *v) {
        if (v->n1) v->n1->accept(*this);
        cout << v->op << " ";
        if (v->n2) v->n2->accept(*this);
    }

    void visit(class DiffOp *v) {

    }

    void visit(class Expr *v) {

    }

    void visit(class ExtendOp *v) {

    }

    void visit(class GroupbyOp *v) {

    }

    void visit(class JoinOp *v) {

    }

    void visit(class MathExpr *v) {
        if (v->op != "") {
            v->n1->accept(*this);
            cout << " " << v->op << " ";
            v->n2->accept(*this);
        }
        else if (v->n3) {
            v->n3->accept(*this);
            cout << " ";
        }
        else if (v->n4) {
            v->n4->accept(*this);
        }
        else {
            if (v->intFloatFlag)
                cout << v->intVal;
            else
                cout << v->floatVal;       
            cout << " ";
        }
    }

    void visit(class OuterjoinOp *v) {

    }

    void visit(class Predicate *v) {
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
    }

    void visit(class ProductOp *v) {
        bool x = false, y = false;
        int q = qCount++;
        cout << "Q" << q;
        cout << " = " << "product(";
        
        // Check if left child node is leaf (Relation)
        if (v->n1->n4) {
            v->n1->n4->accept(*this);
            x = true;
        }
        else
            cout << "Q" << ++q;
        cout << ", ";

        // Check for right child node is leaf (Relation)
        if (v->n1->n4) {
            v->n1->n4->accept(*this);
            y = true;
        }
        else
            cout << "Q" << ++q;

        cout << ");" << endl;

        // Otherwise, visit the left and/or right child
        if (!x) v->n1->accept(*this);
        if (!y) v->n2->accept(*this);
    }

    void visit(class Program *v) {
        if (v->n1) v->n1->accept(*this);
    }

    void visit(class ProjectOp *v) {
        int q = qCount;
        cout << "Q" << ++q;
        cout << " = " << "project( Q" << ++q << ", { ";
        if (v->n2) v->n2->accept(*this);
        cout << "});" << endl;
        if (v->n1) v->n1->accept(*this);
    }

    void visit(class Relation *v) {
        cout << v->name << " ";
    }

    void visit(class RelExpr *v) {
        if (v->n1)
            v->n1->accept(*this);
        else if (v->n2)
            v->n2->accept(*this);
        else if (v->n3)
            v->n3->accept(*this);
        else if (v->n4)
            v->n4->accept(*this);
    }

    void visit(class RelExprList *v) {
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
    }

    void visit(class RenameOp *v) {

    }

    void visit(class SelectOp *v) {
        int q = qCount;
        cout << "Q" << q << " = select( Q" << ++q << ", ";
        if (v->n2) v->n2->accept(*this);
        cout << ");" << endl;
        if (v->n1) v->n1->accept(*this);
    }

    void visit(class SemijoinOp *v) {

    }

    void visit(class SortOp *v) {

    }

    void visit(class UnionOp *v) {

    }

private:
    int qCount = 0;

};

} // RA_Namespace

#endif // REL_ALG_QP_VISITOR_H

