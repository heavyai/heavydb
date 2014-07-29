#ifndef SQL_RENAMESTMT_H
#define SQL_RENAMESTMT_H

#include <cassert>
#include "Statement.h"

class RenameStmt : public Statement {

public:

	Table *n1 = NULL;
	std::string name;
	
	RenameStmt(Table *n1, const std::string &name) {
		assert(n1 && name != "");
		this->n1 = n1;
		this->name = name;
	}

	~RenameStmt() {
		if (n1) delete n1;
	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_RENAMESTMT_H
