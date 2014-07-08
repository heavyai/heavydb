#ifndef PREDICATE_NODE_H
#define PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Predicate : public ASTNode {
    
public:

    ComparisonPredicate* cp;
    BetweenPredicate* bp;
    LikePredicate* lp;

    /* constructor */
    explicit Predicate(ComparisonPredicate* n) : cp(n), bp(NULL), lp(NULL) {}
    Predicate(BetweenPredicate* n) : cp(NULL), bp(n), lp(NULL) {}
    Predicate(LikePredicate *n) : cp(NULL), bp(NULL), lp(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // PREDICATE_NODE_H
