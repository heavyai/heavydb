#ifndef PREDICATE_NODE_H
#define PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  Predicate : public ASTNode {
    
public:

    ComparisonPredicate* cp;
    BetweenPredicate* bp;
    LikePredicate* lp;
    ExistenceTest* et;
    TestForNull *tfn;
    AllOrAnyPredicate *aoap;
    InPredicate *ip;

    /* constructor */
    explicit Predicate(ComparisonPredicate* n) : cp(n), bp(NULL), lp(NULL), et(NULL), tfn(NULL), aoap(NULL), ip(NULL) {}
    Predicate(BetweenPredicate* n) : cp(NULL), bp(n), lp(NULL), et(NULL), tfn(NULL), aoap(NULL), ip(NULL) {}
    Predicate(LikePredicate *n) : cp(NULL), bp(NULL), lp(n), et(NULL), tfn(NULL), aoap(NULL), ip(NULL) {}
    Predicate(ExistenceTest* n) : cp(NULL), bp(NULL), lp(NULL), et(n), tfn(NULL), aoap(NULL), ip(NULL) {}
    Predicate(TestForNull *n) : cp(NULL), bp(NULL), lp(NULL), et(NULL), tfn(n), aoap(NULL), ip(NULL) {}
    Predicate(AllOrAnyPredicate* n) : cp(NULL), bp(NULL), lp(NULL), et(NULL), tfn(NULL), aoap(n), ip(NULL) {}
    Predicate(InPredicate *n) : cp(NULL), bp(NULL), lp(NULL), et(NULL), tfn(NULL), aoap(NULL), ip(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // PREDICATE_NODE_H
