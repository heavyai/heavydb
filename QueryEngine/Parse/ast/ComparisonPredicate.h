#ifndef COMPARISON_PREDICATE_NODE_H
#define COMPARISON_PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ComparisonPredicate : public ASTNode {
    
public:

    ScalarExp* se1;
    ScalarExp* se2;

    /* constructor */
    explicit ComparisonPredicate(ScalarExp* n1, ScalarExp* n2) : se1(n1), se2(n2) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // COMPARISON_PREDICATE_NODE_H
