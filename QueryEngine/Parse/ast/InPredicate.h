#ifndef IN_PREDICATE_NODE_H
#define IN_PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class InPredicate : public ASTNode {
    
public:

	int rule_Flag;
	/* Rules:
	0 IN
	1 NOT IN */

    ScalarExp* se;
    Subquery* sq;
    AtomCommalist* ac;

    /* constructor */
    explicit InPredicate(int rF, ScalarExp* n1, Subquery* n2) : rule_Flag(rF), se(n1), sq(n2), ac(NULL) {}
    InPredicate(int rF, ScalarExp* n1, AtomCommalist* n2) : rule_Flag(rF), se(n1), sq(NULL), ac(n2) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // IN_PREDICATE_NODE_H
