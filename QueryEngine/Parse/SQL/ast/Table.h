#ifndef SQL_TABLE_H
#define SQL_TABLE_H

#include <cassert>
#include "ASTNode.h"

class Table : public ASTNode {

public:

	explicit Table() {

	}

	~Table() {

	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_TABLE_H
