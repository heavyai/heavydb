#ifndef SQL_COLUMNDEFLIST_H
#define SQL_COLUMNDEFLIST_H

#include <cassert>
#include "ASTNode.h"

class ColumnDefList : public ASTNode {

public:

	ColumnDefList *n1 = NULL;
	ColumnDef *n2 = NULL;

	ColumnDefList(ColumnDefList *n1, ColumnDef *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	ColumnDefList(ColumnDef *n2) {
		assert(n2);
		this->n2 = n2;
	}

	~ColumnDefList() {

	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_COLUMNDEFLIST_H
