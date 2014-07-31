#ifndef SQL_FROMCLAUSE_H
#define SQL_FROMCLAUSE_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class FromClause : public ASTNode {

public:
	TableList *n1 = NULL;
	SelectStmt *n2 = NULL;

	explicit FromClause(TableList *n1) {
		assert(n1);
		this->n1 = n1;
	}

	explicit FromClause(SelectStmt *n2) {
		assert(n2);
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

#endif // SQL_FROMCLAUSE_H
