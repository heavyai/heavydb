#ifndef SQL_INSERTSTMT_H
#define SQL_INSERTSTMT_H

#include <cassert>
#include "Statement.h"

namespace SQL_Namespace {

class InsertStmt : public Statement {

public:

	Table *n1 = NULL;
	ColumnList *n2 = NULL;
	LiteralList *n3 = NULL;
	
	explicit InsertStmt(Table *n1, ColumnList *n2, LiteralList *n3) {
		assert(n1 && n2 && n3);
		this->n1 = n1;
		this->n2 = n2;
		this->n3 = n3;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

} // SQL_Namespace

#endif // SQL_INSERTSTMT_H
