#ifndef COMPARISON_PREDICATE_NODE_H
#define COMPARISON_PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  ComparisonPredicate : public ASTNode {
    
public:

    ScalarExp* se1;
    ScalarExp* se2;
    Subquery* s;

    /* constructor */
    explicit ComparisonPredicate(ScalarExp* n1, ScalarExp* n2) : se1(n1), se2(n2), s(NULL) {}
    ComparisonPredicate(ScalarExp* n1, Subquery* n2) : se1(n1), se2(NULL), s(n2) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // COMPARISON_PREDICATE_NODE_H
