#ifndef BETWEEN_PREDICATE_NODE_H
#define BETWEEN_PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  BetweenPredicate : public ASTNode {
    
public:

	int rule_Flag;
	/* Rules:
	0 BETWEEN
	1 NOT BETWEEN */

    ScalarExp* se1;
    ScalarExp* se2;
    ScalarExp* se3;

    /* constructor */
    explicit BetweenPredicate(int rF, ScalarExp* n1, ScalarExp* n2, ScalarExp* n3) : rule_Flag(rF), se1(n1), se2(n2), se3(n3) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // BETWEEN_PREDICATE_NODE_H
