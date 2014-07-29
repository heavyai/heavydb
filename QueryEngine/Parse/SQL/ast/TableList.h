#ifndef SQL_TABLELIST_H
#define SQL_TABLELIST_H

#include <cassert>
#include "ASTNode.h"

class TableList : public ASTNode {

public:
	TableList *n1 = NULL;
	Table *n2 = NULL;

	explicit TableList(TableList *n1, Table *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}
	
	explicit TableList(Table *n2) {
		assert(n2);
		this->n2 = n2;
	}

	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_TABLELIST_H
