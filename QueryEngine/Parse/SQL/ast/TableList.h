#ifndef SQL_TABLELIST_H
#define SQL_TABLELIST_H

#include <cassert>
#include "ASTNode.h"

class TableList : public ASTNode {

public:

	explicit TableList() {

	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_TABLELIST_H
