#ifndef SQL_OPTGROUPBY_H
#define SQL_OPTGROUPBY_H

#include <cassert>
#include "ASTNode.h"

class OptGroupby : public ASTNode {

public:

	explicit OptGroupby() {

	}

	~OptGroupby() {

	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_OPTGROUPBY_H
