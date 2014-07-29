#ifndef SQL_FROMCLAUSE_H
#define SQL_FROMCLAUSE_H

#include <cassert>
#include "ASTNode.h"

class FromClause : public ASTNode {

public:

	explicit FromClause() {

	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_FROMCLAUSE_H
