#ifndef SQL_ANY_OR_ALL_PREDICATE_NODE_H
#define SQL_ANY_OR_ALL_PREDICATE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class AnyOrAllPredicate : public ASTNode {
    
public:
    ScalarExp* se;
    AnyAllSome* aas;
    Subquery* sq;

    /// Constructor
    AnyOrAllPredicate(ScalarExp *n1, AnyAllSome *n2, Subquery *n3) : se(n1), aas(n2), sq(n3) {}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // SQL_ANY_OR_ALL_PREDICATE_NODE_H
