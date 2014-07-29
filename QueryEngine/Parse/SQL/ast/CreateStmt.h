#ifndef SQL_CREATESTMT_H
#define SQL_CREATESTMT_H

#include <cassert>
#include "Statement.h"

class CreateStmt : public Statement {

public:

	Table *n1 = NULL;
	ColumnDefList *n2 = NULL;
	
	CreateStmt(Table *n1, ColumnDefList *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_CREATESTMT_H
