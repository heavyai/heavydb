#ifndef SQL_OPTORDERBY_H
#define SQL_OPTORDERBY_H

#include <cassert>
#include "ASTNode.h"

class OptOrderby : public ASTNode {

public:
	OrderbyColumnList *n1 = NULL;

	explicit OptOrderby(OrderbyColumnList* n1) {
		assert(n1);
		this->n1 = n1;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_OPTORDERBY_H
