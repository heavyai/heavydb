/**
 * @file    ScalarExp.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef SQL_SCALAR_EXP_NODE_H
#define SQL_SCALAR_EXP_NODE_H

#include <cassert>
#include "ASTNode.h"
#include "AbstractScalarExpr.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class ScalarExp : public AbstractScalarExpr {
    
public:

	int rule_Flag = -1;
	/* rules are:
	0 (scalar_exp)
    1 addition
	2 subtraction
	3 multiplication
	4 division
    5 positive [scalar_exp]
    6 negative [scalar_exp] */

    ScalarExp* se1 = NULL;
    ScalarExp* se2 = NULL;
    Atom* a = NULL;
    ColumnRef *cr = NULL;
    FunctionRef *fr = NULL;

    /* constructor */
    ScalarExp(int rF, ScalarExp *n1, ScalarExp* n2) {
        assert(((rF >= 0) && (rF <= 6)) && n1 && n2);
        this->se1 = n1;
        this->se2 = n2;
        this->rule_Flag = rF;
    }

    ScalarExp(int rF, ScalarExp* n1) {
        assert((rF >= 0) && (rF <= 6) && n1);
        this->se1 = n1;
        this->rule_Flag = rF;
    }

    ScalarExp(Atom *n) { 
        assert(n);
        this->a = n; 
    }
    ScalarExp(ColumnRef* n) { 
        assert(n);
        this->cr = n; 
    }
    ScalarExp(FunctionRef* n) { 
        assert(n);
        this->fr = n; 
    }

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // SCALAR_EXP_NODE_H
