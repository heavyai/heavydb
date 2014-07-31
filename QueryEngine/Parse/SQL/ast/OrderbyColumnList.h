#ifndef SQL_ORDERBYCOLUMNLIST_H
#define SQL_ORDERBYCOLUMNLIST_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class OrderbyColumnList : public ASTNode {

public:
	OrderbyColumnList *n1 = NULL;
	OrderbyColumn *n2 = NULL;

	explicit OrderbyColumnList(OrderbyColumnList* n1, OrderbyColumn* n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	explicit OrderbyColumnList(OrderbyColumn* n2) {
		assert(n2);
		this->n2 = n2;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

    virtual void accept(class SQL_RA_Translator &v) {
        v.visit(this);
    }

};

} // SQL_Namespace

#endif // SQL_ORDERBYCOLUMNLIST_H
