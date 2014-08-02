#ifndef SQL_LITERALLIST_H
#define SQL_LITERALLIST_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class LiteralList : public ASTNode {

public:

	LiteralList *n1 = NULL;
	Literal *n2 = NULL;

	explicit LiteralList(LiteralList *n1, Literal *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	explicit LiteralList(Literal *n2) {
		assert(n2);
		this->n2 = n2;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_LITERALIST_H
