#ifndef SQL_DROPSTMT_H
#define SQL_DROPSTMT_H

#include <cassert>
#include "Statement.h"

class DropStmt : public Statement {

public:

	Table *n1 = NULL;
	
	DropStmt(Table *n1) {
		assert(n1);
		this->n1 = n1;
	}

	~DropStmt() {
		if (n1) delete n1;
	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_DROPSTMT_H
