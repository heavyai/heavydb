#ifndef SQL_ORDERBYCOLUMN_H
#define SQL_ORDERBYCOLUMN_H

#include <cassert>
#include "ASTNode.h"

class OrderbyColumn : public ASTNode {

public:
	Column *n1 = NULL;
	bool sortFlag = false;
	bool asc_or_desc = false; // true => ASC, false => DESC

	explicit OrderbyColumn(Column *n1) {
		assert(n1);
		this->n1 = n1;
	}

	OrderbyColumn(Column *n1, bool asc_or_desc) {
		assert(n1);
		this->n1 = n1;
		this->sortFlag = true;
		this->asc_or_desc = asc_or_desc;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_ORDERBYCOLUMN_H
