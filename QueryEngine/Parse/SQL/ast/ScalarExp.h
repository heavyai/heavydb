#ifndef SQL_SCALAR_EXP_NODE_H
#define SQL_SCALAR_EXP_NODE_H

#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class  ScalarExp : public ASTNode {
    
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

    ScalarExp* n1 = NULL;
    ScalarExp* n2 = NULL;
    Atom* n3 = NULL;
    ColumnRef *n4 = NULL;
    FunctionRef *n5 = NULL;

    /* constructor */
    ScalarExp(int rF, ScalarExp *n1, ScalarExp* n2) {
        assert(n1 && n2);
        this->n1 = n1;
        this->n2 = n2;
    }

    ScalarExp(int rF, ScalarExp* n) : rule_Flag(rF), se1(n), se2(NULL), a(NULL), cr(NULL), fr(NULL) {}
    ScalarExp(Atom *n) : rule_Flag(-1), se1(NULL), se2(NULL), a(n), cr(NULL), fr(NULL) {}
    ScalarExp(ColumnRef* n) : rule_Flag(-1), se1(NULL), se2(NULL), a(NULL), cr(n), fr(NULL) {}
    ScalarExp(FunctionRef* n) : rule_Flag(-1), se1(NULL), se2(NULL), a(NULL), cr(NULL), fr(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // SCALAR_EXP_NODE_H
