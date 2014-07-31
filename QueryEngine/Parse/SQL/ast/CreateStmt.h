#ifndef SQL_CREATESTMT_H
#define SQL_CREATESTMT_H

#include <cassert>
#include "Statement.h"

namespace SQL_Namespace {

class CreateStmt : public Statement {

public:

	Table *n1 = NULL;
	ColumnDefList *n2 = NULL;
	
	CreateStmt(Table *n1, ColumnDefList *n2) {
		assert(n1 && n2);
		this->n1 = n1;
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

#endif // SQL_CREATESTMT_H
