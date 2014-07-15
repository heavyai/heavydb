#ifndef LIKE_PREDICATE_NODE_H
#define LIKE_PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class LikePredicate : public ASTNode {
    
public:

	int rule_Flag;
	/* Rules:
	0 LIKE
	1 NOT LIKE */

    ScalarExp* se;
    Atom* a;
    OptEscape* oe;

    /* constructor */
    explicit LikePredicate(int rF, ScalarExp* n1, Atom* n2, OptEscape* n3) : rule_Flag(rF), se(n1), a(n2), oe(n3) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // LIKE_PREDICATE_NODE_H
