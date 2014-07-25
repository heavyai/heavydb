#ifndef SQL_ALL_OR_ANY_PREDICATE_NODE_H
#define SQL_ALL_OR_ANY_PREDICATE_NODE_H

#include <cstddef>
#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class AllOrAnyPredicate : public ASTNode {
    
public:
    ScalarExp* se = NULL;
    AnyAllSome* aas = NULL;
    Subquery* sq = NULL;

    /// Constructor
    AllOrAnyPredicate(ScalarExp *n1, AnyAllSome *n2, Subquery *n3) {
    	assert(n1 && n2 && n3);
    	this->se = n1;
    	this->aas = n2;
    	this->sq = n3;
    }

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // SQL_ALL_OR_ANY_PREDICATE_NODE_H
