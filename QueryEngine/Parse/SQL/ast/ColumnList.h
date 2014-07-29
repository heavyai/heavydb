#ifndef SQL_COLUMNLIST_H
#define SQL_COLUMNLIST_H

#include <cassert>
#include "ASTNode.h"

class ColumnList : public ASTNode {

public:

	explicit ColumnList() {

	}

	~ColumnList() {

	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_COLUMNLIST_H
