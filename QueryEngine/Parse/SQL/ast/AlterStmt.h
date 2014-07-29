#ifndef SQL_ALTERSTMT_H
#define SQL_ALTERSTMT_H

#include <cassert>
#include "Statement.h"

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

	~AlterStmt() {
		if (n1) delete n1;
		if (n2) delete n2;
		if (n3) delete n3;
	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_ALTERSTMT_H
