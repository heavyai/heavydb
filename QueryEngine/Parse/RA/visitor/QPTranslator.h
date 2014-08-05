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
        cout << v->func << "(";
        v->n1->accept(*this);
        cout << ")";
    }

    void visit(class AggrList *v) {
        if (v->n1) {
            v->n1->accept(*this);
            if (v->n1->n2)
                cout << ", ";
        }
        if (v->n2) {
            v->n2->accept(*this);
        }
    }

    void visit(class AntijoinOp *v) {
        int childCount = 0;
        if (!v->n1->n4) {
            v->n1->accept(*this);
            childCount++;
        }
        if (!v->n2->n4) {
            v->n2->accept(*this);
            childCount++;
        }

        cout << "Q" << qCount++ << " = antijoin(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        if (v->n2->n4)
            v->n2->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        v->n3->accept(*this);
        
        cout << ");" << endl;
    }

    void visit(class Attribute *v) {
        cout << v->name1;
        if (v->name2 != "")
            cout << "." << v->name2;
    }

    void visit(class AttrList *v) {
        if (v->n1) {
            v->n1->accept(*this);
            if (v->n1->n2)
                cout << ", ";
        }
        if (v->n2) {
            v->n2->accept(*this);
        }
    }

    void visit(class Comparison *v) {
        if (v->n1) v->n1->accept(*this);
        cout << " " << v->op << " ";
        if (v->n2) v->n2->accept(*this);
    }

    void visit(class DiffOp *v) {
        int childCount = 0;
        if (!v->n1->n4) {
            v->n1->accept(*this);
            childCount++;
        }
        if (!v->n2->n4) {
            v->n2->accept(*this);
            childCount++;
        }

        cout << "Q" << qCount++ << " = diff(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        if (v->n2->n4)
            v->n2->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        
        cout << ");" << endl;
    }

    void visit(class Expr *v) {
        if (v->str != "")
            cout << v->str;
        else {
            if (v->n1)
                v->n1->accept(*this);
            else if (v->n2)
                v->n2->accept(*this);
        }
    }

    void visit(class ExtendOp *v) {
        if (!v->n1->n4) {
            v->n1->accept(*this);
        }

        cout << "Q" << qCount++ << " = extend(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else
            cout << "Q" << (qCount-2);
        cout << ", ";

        v->n2->accept(*this);
        cout << ", " << v->name << ");" << endl;
    }

    void visit(class GroupbyOp *v) {
        if (!v->n1->n4)
            v->n1->accept(*this);

        cout << "Q" << qCount++ << " = groupby(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else
            cout << "Q" << qCount-2;
        cout << ", {";

        v->n2->accept(*this);
        cout << "}, {";
        v->n3->accept(*this);
        cout << "});" << endl;
    }

    void visit(class JoinOp *v) {
        int childCount = 0;
        if (!v->n1->n4) {
            v->n1->accept(*this);
            childCount++;
        }
        if (!v->n2->n4) {
            v->n2->accept(*this);
            childCount++;
        }

        cout << "Q" << qCount++ << " = join(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        if (v->n2->n4)
            v->n2->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        v->n3->accept(*this);
        
        cout << ");" << endl;
    }

    void visit(class MathExpr *v) {
        if (v->op != "") {
            v->n1->accept(*this);
            cout << " " << v->op << " ";
            v->n2->accept(*this);
        }
        else if (v->n3) {
            v->n3->accept(*this);
        }
        else if (v->n4) {
            v->n4->accept(*this);
        }
        else {
            if (v->intFloatFlag)
                cout << v->intVal;
            else
                cout << v->floatVal;       
        }
    }

    void visit(class OuterjoinOp *v) {
        int childCount = 0;
        if (!v->n1->n4) {
            v->n1->accept(*this);
            childCount++;
        }
        if (!v->n2->n4) {
            v->n2->accept(*this);
            childCount++;
        }

        cout << "Q" << qCount++ << " = outerjoin(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        if (v->n2->n4)
            v->n2->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        v->n3->accept(*this);
        
        cout << ");" << endl;
    }

    void visit(class Predicate *v) {
        if (v->n1 && v->n2) {
            v->n1->accept(*this);
            cout << " " << v->op << " ";
            v->n2->accept(*this);
        }
        else if (v->n1)
            v->n1->accept(*this);
        else if (v->n3)
            v->n3->accept(*this);
    }

    void visit(class ProductOp *v) {
        int childCount = 0;
        if (!v->n1->n4) {
            v->n1->accept(*this);
            childCount++;
        }
        if (!v->n2->n4) {
            v->n2->accept(*this);
            childCount++;
        }

        cout << "Q" << qCount++ << " = product(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        if (v->n2->n4)
            v->n2->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        
        cout << ");" << endl;
    }

    void visit(class Program *v) {
        if (v->n1) v->n1->accept(*this);
    }

    void visit(class ProjectOp *v) {
        if (!v->n1->n4)
            v->n1->accept(*this);

        cout << "Q" << qCount++ << " = project(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else
            cout << "Q" << qCount-2;
        cout << ", {";

        v->n2->accept(*this);
        cout << "});" << endl;
    }

    void visit(class Relation *v) {
        cout << v->name;
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
        if (!v->n1->n4) {
            v->n1->accept(*this);
        }

        cout << "Q" << qCount++ << " = rename(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else
            cout << "Q" << (qCount-2);
        cout << ", ";
        
        cout << v->name1 << ", ";
        cout << v->name2;
        
        cout << ");" << endl;
    }

    void visit(class SelectOp *v) {
        if (!v->n1->n4) {
            v->n1->accept(*this);
        }

        cout << "Q" << qCount++ << " = select(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else
            cout << "Q" << (qCount-2);
        cout << ", ";
        
        v->n2->accept(*this);
        
        cout << ");" << endl;
    }

    void visit(class SemijoinOp *v) {
        int childCount = 0;
        if (!v->n1->n4) {
            v->n1->accept(*this);
            childCount++;
        }
        if (!v->n2->n4) {
            v->n2->accept(*this);
            childCount++;
        }

        cout << "Q" << qCount++ << " = semijoin(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        if (v->n2->n4)
            v->n2->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        v->n3->accept(*this);
        
        cout << ");" << endl;
    }

    void visit(class SortOp *v) {
        if (!v->n1->n4)
            v->n1->accept(*this);

        cout << "Q" << qCount++ << " = sort(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else
            cout << "Q" << qCount-2;
        cout << ", [";

        v->n2->accept(*this);
        cout << "]);" << endl;
    }

    void visit(class UnionOp *v) {
        int childCount = 0;
        if (!v->n1->n4) {
            v->n1->accept(*this);
            childCount++;
        }
        if (!v->n2->n4) {
            v->n2->accept(*this);
            childCount++;
        }

        cout << "Q" << qCount++ << " = union(";
        if (v->n1->n4)
            v->n1->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }
        cout << ", ";

        if (v->n2->n4)
            v->n2->n4->accept(*this);
        else {
            cout << "Q" << (qCount-childCount-1);
            childCount--;
        }

        cout << ");" << endl;
    }

private:
    int qCount = 0;

};

} // RA_Namespace

#endif // REL_ALG_QP_VISITOR_H


