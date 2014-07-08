#ifndef SCALAR_EXP_NODE_H
#define SCALAR_EXP_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ScalarExp : public ASTNode {
    
public:

	int rule_Flag;
	/* rules are:
	0 (scalar_exp)
    1 addition
	2 subtraction
	3 multiplication
	4 division
    5 positive [scalar_exp]
    6 negative [scalar_exp] */

    ScalarExp* se1;
    ScalarExp* se2;
    Atom* a;
    ColumnRef* cr;
    FunctionRef* fr;

    /* constructor */
    explicit ScalarExp(int rF, ScalarExp *n1, ScalarExp* n2) : rule_Flag(rF), se1(n1), se2(n2), a(NULL), cr(NULL), fr(NULL) {}
    ScalarExp(int rF, ScalarExp* n) : rule_Flag(rF), se1(n), se2(NULL), a(NULL), cr(NULL), fr(NULL) {}
    ScalarExp(Atom *n) : rule_Flag(-1), se1(NULL), se2(NULL), a(n), cr(NULL), fr(NULL) {}
    ScalarExp(ColumnRef* n) : rule_Flag(-1), se1(NULL), se2(NULL), a(NULL), cr(n), fr(NULL) {}
    ScalarExp(FunctionRef* n) : rule_Flag(-1), se1(NULL), se2(NULL), a(NULL), cr(NULL), fr(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // SCALAR_EXP_NODE_H
