#ifndef SQL_DROPSTMT_H
#define SQL_DROPSTMT_H

#include <cassert>
#include "Statement.h"

namespace SQL_Namespace {

class DropStmt : public Statement {

public:

	Table *n1 = NULL;
	
	DropStmt(Table *n1) {
		assert(n1);
		this->n1 = n1;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_DROPSTMT_H
