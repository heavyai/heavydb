#ifndef SQL_OPTWHERE_H
#define SQL_OPTWHERE_H

#include <cassert>
#include "ASTNode.h"

class OptWhere : public ASTNode {

public:

	explicit OptWhere() {

	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_OPTWHERE_H
