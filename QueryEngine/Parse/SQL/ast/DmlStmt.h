#ifndef SQL_DMLSTMT_H
#define SQL_DMLSTMT_H

#include <cassert>
#include "Statement.h"

namespace SQL_Namespace {

class DmlStmt : public Statement {

public:

	InsertStmt *n1 = NULL;
	SelectStmt *n2 = NULL;
    DeleteStmt *n3 = NULL;
	
	explicit DmlStmt(InsertStmt *n1) {
		assert(n1);
		this->n1 = n1;
	}

	explicit DmlStmt(SelectStmt *n2) {
		assert(n2);
		this->n2 = n2;
	}
    
    explicit DmlStmt(DeleteStmt *n3) {
        assert(n3);
        this->n3 = n3;
    }
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_DMLSTMT_H
