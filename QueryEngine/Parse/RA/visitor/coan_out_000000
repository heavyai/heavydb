
/**
 * @file    QPTranslator.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "QPTranslator.h"

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
#include <stdio.h>
using std::cout;
using std::endl;

namespace RA_Namespace {

void QPTranslator::visit(AggrExpr *v) {
    cout << v->func << "(";
    v->n1->accept(*this);
    cout << ")";
}

void QPTranslator::visit(AggrList *v) {
    if (v->n1) {
        v->n1->accept(*this);
        if (v->n1->n2)
            cout << ", ";
    }
    if (v->n2) {
        v->n2->accept(*this);
    }
}

void QPTranslator::visit(AntijoinOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);
    if (!v->n2->n4) v->n2->accept(*this);

    cout << "Q" << queryNum_ << " = antijoin(";

    int first;
    int second;
    if (!v->n1->n4) {
        second = qTracker_.back();
        qTracker_.pop_back();
    }
    if (!v->n2->n4) {
        first = qTracker_.back();
        qTracker_.pop_back();
    }

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else
        cout << "Q" << first;
     
    cout << ", ";

    if (v->n2->n4)
        v->n2->n4->accept(*this);
    else
        cout << "Q" << second;

    cout << ", ";
    if (v->n3) v->n3->accept(*this);
    
    cout << ")" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;

    
}

void QPTranslator::visit(Attribute *v) {
    cout << v->name1;
    if (v->name2 != "")
        cout << "." << v->name2;
}

void QPTranslator::visit(AttrList *v) {
    if (v->n1) {
        v->n1->accept(*this);
        if (v->n1->n2)
            cout << ", ";
    }
    if (v->n2) {
        v->n2->accept(*this);
    }
}

void QPTranslator::visit(Comparison *v) {
    if (v->n1) v->n1->accept(*this);
    cout << " " << v->op << " ";
    if (v->n2) v->n2->accept(*this);
}

void QPTranslator::visit(DiffOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);
    if (!v->n2->n4) v->n2->accept(*this);

    cout << "Q" << queryNum_ << " = diff(";

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else {
        cout << "Q" << qTracker_.front();
        qTracker_.pop_front();
    }
    cout << ", ";

    if (v->n2->n4)
        v->n2->n4->accept(*this);
    else {
        cout << "Q" << qTracker_.front();
        qTracker_.pop_front();
    }
    cout << ")" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;
}

void QPTranslator::visit(Expr *v) {
    if (v->str != "")
        cout << v->str;
    else {
        if (v->n1)
            v->n1->accept(*this);
        else if (v->n2)
            v->n2->accept(*this);
    }
}

void QPTranslator::visit(ExtendOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);

    cout << "Q" << queryNum_ << " = extend(";

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else {
        cout << "Q" << qTracker_.back();
        qTracker_.pop_back();
    }
    cout << ", ";

    v->n2->accept(*this);
    cout << ", " << v->name << ");" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;
}

void QPTranslator::visit(GroupbyOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);

    cout << "Q" << queryNum_ << " = group by(";

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else {
        cout << "Q" << qTracker_.back();
        qTracker_.pop_back();
    }
    cout << ", {";

    v->n2->accept(*this);
    cout << "}, {";
    v->n3->accept(*this);
    cout << "});" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;
}

void QPTranslator::visit(JoinOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);
    if (!v->n2->n4) v->n2->accept(*this);

    cout << "Q" << queryNum_ << " = join(";

    int first;
    int second;
    if (!v->n1->n4) {
        second = qTracker_.back();
        qTracker_.pop_back();
    }
    if (!v->n2->n4) {
        first = qTracker_.back();
        qTracker_.pop_back();
    }

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else
        cout << "Q" << first;
     
    cout << ", ";

    if (v->n2->n4)
        v->n2->n4->accept(*this);
    else
        cout << "Q" << second;
    
    cout << ", ";
    if (v->n3) v->n3->accept(*this);
     
    cout << ")" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;   
}

void QPTranslator::visit(MathExpr *v) {
    // case: MathExpr OP MathExpr
    if (v->op != "") {
        v->n1->accept(*this);
        cout << " " << v->op << " ";
        v->n2->accept(*this);
    }
    // case: (MathExpr)
    else if (v->n1 && !v->n2) {
        cout << "(";
        v->n1->accept(*this);
        cout << ")";
    }

    // case: Attribute
    else if (v->n3) {
        v->n3->accept(*this);
    }

    // case: AggrExpr
    else if (v->n4) {
        v->n4->accept(*this);
    }

    // case: INT or FLOAT
    else {
        if (v->intFloatFlag)
            cout << v->intVal;
        else
            cout << v->floatVal;       
    }
}

void QPTranslator::visit(OuterjoinOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);
    if (!v->n2->n4) v->n2->accept(*this);

    cout << "Q" << queryNum_ << " = outerjoin(";

    int first;
    int second;
    if (!v->n1->n4) {
        second = qTracker_.back();
        qTracker_.pop_back();
    }
    if (!v->n2->n4) {
        first = qTracker_.back();
        qTracker_.pop_back();
    }

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else
        cout << "Q" << first;
     
    cout << ", ";

    if (v->n2->n4)
        v->n2->n4->accept(*this);
    else
        cout << "Q" << second;

    cout << ", ";
    if (v->n3) v->n3->accept(*this);
     
    cout << ")" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;
}

void QPTranslator::visit(Predicate *v) {
    if (v->n1 && v->n2) {
        v->n1->accept(*this);
        cout << " " << v->op << " ";
        v->n2->accept(*this);
    }
    else if (v->n1) {
        if (v->op == "NOT")
            cout << v->op << " ";
        v->n1->accept(*this);
    }
    else if (v->n3)
        v->n3->accept(*this);
}

void QPTranslator::visit(ProductOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);
    if (!v->n2->n4) v->n2->accept(*this);

    cout << "Q" << queryNum_ << " = product(";

    int first;
    int second;
    if (!v->n1->n4) {
        second = qTracker_.back();
        qTracker_.pop_back();
    }
    if (!v->n2->n4) {
        first = qTracker_.back();
        qTracker_.pop_back();
    }

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else
        cout << "Q" << first;
     
    cout << ", ";

    if (v->n2->n4)
        v->n2->n4->accept(*this);
    else
        cout << "Q" << second;
     
    cout << ")" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;
}

void QPTranslator::visit(Program *v) {
    if (v->n1) v->n1->accept(*this);
}

void QPTranslator::visit(ProjectOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);

    cout << "Q" << queryNum_ << " = project(";

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else {
        cout << "Q" << qTracker_.back();
        qTracker_.pop_back();
    }
    cout << ", {";

    v->n2->accept(*this);
    cout << "});" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;  
}

void QPTranslator::visit(Relation *v) {
    cout << v->name;
}

void QPTranslator::visit(RelExpr *v) {
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
    if (v->n3) v->n3->accept(*this);
    //don't visit the Table, printing happens during query plan output
    //if (v->n4) v->n4->accept(*this);
}

void QPTranslator::visit(RelExprList *v) {
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
}

void QPTranslator::visit(RenameOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);

    cout << "Q" << queryNum_ << " = rename(";

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else {
        cout << "Q" << qTracker_.back();
        qTracker_.pop_back();
    }
    cout << ", ";
    
    cout << v->name1 << ", ";
    cout << v->name2;
    
    cout << ");" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;   
}

void QPTranslator::visit(SelectOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);

    cout << "Q" << queryNum_ << " = select(";

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else {
        cout << "Q" << qTracker_.back();
        qTracker_.pop_back();
    }
    cout << ", ";

    v->n2->accept(*this);
    cout << ");" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;
}

void QPTranslator::visit(SemijoinOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);
    if (!v->n2->n4) v->n2->accept(*this);

    cout << "Q" << queryNum_ << " = semijoin(";

    int first;
    int second;
    if (!v->n1->n4) {
        second = qTracker_.back();
        qTracker_.pop_back();
    }
    if (!v->n2->n4) {
        first = qTracker_.back();
        qTracker_.pop_back();
    }

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else
        cout << "Q" << first;
     
    cout << ", ";

    if (v->n2->n4)
        v->n2->n4->accept(*this);
    else
        cout << "Q" << second;

    cout << ", ";
    if (v->n3) v->n3->accept(*this);
    
    cout << ")" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;
    
}

void QPTranslator::visit(SortOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);

    cout << "Q" << queryNum_ << " = sort(";

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else {
        cout << "Q" << qTracker_.back();
        qTracker_.pop_back();
    }
    cout << ", [";

    v->n2->accept(*this);
    cout << "]);" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;
}

void QPTranslator::visit(UnionOp *v) {
    if (!v->n1->n4) v->n1->accept(*this);
    if (!v->n2->n4) v->n2->accept(*this);

    cout << "Q" << queryNum_ << " = union(";

    int first;
    int second;
    if (!v->n1->n4) {
        second = qTracker_.back();
        qTracker_.pop_back();
    }
    if (!v->n2->n4) {
        first = qTracker_.back();
        qTracker_.pop_back();
    }

    if (v->n1->n4)
        v->n1->n4->accept(*this);
    else
        cout << "Q" << first;
     
    cout << ", ";

    if (v->n2->n4)
        v->n2->n4->accept(*this);
    else
        cout << "Q" << second;
     
    cout << ")" << endl;

    qTracker_.push_back(queryNum_);
    queryNum_++;
}

} // RA_Namespace
