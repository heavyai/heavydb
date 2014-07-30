#ifndef SQL_ALTERSTMT_H
#define SQL_ALTERSTMT_H

#include <cassert>
#include "Statement.h"

namespace SQL_Namespace {

class AlterStmt : public Statement {

public:

	Table *n1 = NULL;
	Column *n2 = NULL;
	MapdDataT *n3 = NULL;
	
	AlterStmt(Table *n1, Column *n2, MapdDataT *n3) {
		assert(n1 && n2 && n3);
		this->n1 = n1;
		this->n2 = n2;
		this->n3 = n3;
	}

	AlterStmt(Table *n1, Column *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

} // SQL_Namespace

#endif // SQL_ALTERSTMT_H
