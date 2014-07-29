#ifndef SQL_DMLSTMT_H
#define SQL_DMLSTMT_H

#include <cassert>
#include "Statement.h"

class DmlStmt : public Statement {

public:

	InsertStmt *n1 = NULL;
	SelectStmt *n2 = NULL;
	
	explicit DmlStmt(InsertStmt *n1) {
		assert(n1);
		this->n1 = n1;
	}

	explicit DmlStmt(SelectStmt *n2) {
		assert(n2);
		this->n2 = n2;
	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_DMLSTMT_H
