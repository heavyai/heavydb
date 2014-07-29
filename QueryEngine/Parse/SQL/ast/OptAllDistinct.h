#ifndef SQL_OPTALLDISTINCT_H
#define SQL_OPTALLDISTINCT_H

#include <cassert>
#include "ASTNode.h"

class OptAllDistinct : public ASTNode {

public:

	explicit OptAllDistinct() {

	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_OPTALLDISTINCT_H
