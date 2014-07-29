#ifndef SQL_OPTHAVING_H
#define SQL_OPTHAVING_H

#include <cassert>
#include "ASTNode.h"

class OptHaving : public ASTNode {

public:

	explicit OptHaving() {

	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_OPTHAVING_H
