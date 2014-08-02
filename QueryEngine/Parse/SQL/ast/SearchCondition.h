#ifndef SQL_SEARCHCONDITION_H
#define SQL_SEARCHCONDITION_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class SearchCondition : public ASTNode {

public:

	Predicate *n1 = NULL;

	explicit SearchCondition(Predicate *n1) {
		assert(n1);
		this->n1 = n1;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

}

#endif // SQL_SEARCHCONDITION_H
