#ifndef SQL_OPTLIMIT_H
#define SQL_OPTLIMIT_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class OptLimit : public ASTNode {

public:

	explicit OptLimit() {

	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_OPTLIMIT_H
