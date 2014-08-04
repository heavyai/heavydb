#ifndef SQL_DDLSTMT_H
#define SQL_DDLSTMT_H

#include <cassert>
#include "Statement.h"

namespace SQL_Namespace {

class DdlStmt : public Statement {

public:

	CreateStmt *n1 = NULL;
	DropStmt *n2 = NULL;
	AlterStmt *n3 = NULL;
	RenameStmt *n4 = NULL;
	
	explicit DdlStmt(CreateStmt *n1) {
		assert(n1);
		this->n1 = n1;
	}

	explicit DdlStmt(DropStmt *n2) {
		assert(n2);
		this->n2 = n2;
	}

	explicit DdlStmt(AlterStmt *n3) {
		assert(n3);
		this->n3 = n3;
	}

	explicit DdlStmt(RenameStmt *n4) {
		assert(n4);
		this->n4 = n4;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_DMLSTMT_H
