#ifndef ALL_OR_ANY_PREDICATE_NODE_H
#define ALL_OR_ANY_PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class AllOrAnyPredicate : public ASTNode {
    
public:
    ScalarExp* se;
    AnyAllSome* aas;
    Subquery* sq;

    /* constructor */
    explicit AllOrAnyPredicate(ScalarExp *n1, AnyAllSome *n2, Subquery *n3) : se(n1), aas(n2), sq(n3) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // ALL_OR_ANY_PREDICATE_NODE_H
