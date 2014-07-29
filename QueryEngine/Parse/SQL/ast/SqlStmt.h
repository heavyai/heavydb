#ifndef SQL_SQLSTMT_H
#define SQL_SQLSTMT_H

#include <cassert>
#include "Statement.h"

class SqlStmt : public Statement {

public:

	DdlStmt *n1 = NULL;
	DmlStmt *n2 = NULL;
	
	explicit SqlStmt(DdlStmt *n1) {
		assert(n1);
		this->n1 = n1;
	}

	explicit SqlStmt(DmlStmt *n2) {
		assert(n2);
		this->n2 = n2;
	}

	~SqlStmt() {
		if (n1) delete n1;
		if (n2) delete n2;
	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_SQLSTMT_H
