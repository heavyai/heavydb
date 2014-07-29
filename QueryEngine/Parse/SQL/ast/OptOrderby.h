#ifndef SQL_OPTORDERBY_H
#define SQL_OPTORDERBY_H

#include <cassert>
#include "ASTNode.h"

class OptOrderby : public ASTNode {

public:

	explicit OptOrderby() {

	}

	~OptOrderby() {

	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_OPTORDERBY_H
