#ifndef SQL_RENAMESTMT_H
#define SQL_RENAMESTMT_H

#include <cassert>
#include "Statement.h"

namespace SQL_Namespace {

class RenameStmt : public Statement {

public:

	Table *n1 = NULL;
	std::string name;
	
	RenameStmt(Table *n1, const std::string &name) {
		assert(n1 && name != "");
		this->n1 = n1;
		this->name = name;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_RENAMESTMT_H
