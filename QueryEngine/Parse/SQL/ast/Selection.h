#ifndef SQL_SELECTION_H
#define SQL_SELECTION_H

#include <cassert>
#include "ASTNode.h"

class Selection : public ASTNode {

public:
	ScalarExprList *n1 = NULL;
	bool all = false;		// true when the MULTIPLY token is parsed (indicates a "SELECT *" statement)

	/// Constructor
	explicit Selection(ScalarExprList *n1) {
		assert(n1);
		this->n1 = n1;
	}

	/// Constructor
	explicit Selection(bool all) {
		this->all = all;
	}

	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_SELECTION_H
