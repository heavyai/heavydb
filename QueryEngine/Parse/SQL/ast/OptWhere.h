#ifndef SQL_OPTWHERE_H
#define SQL_OPTWHERE_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class OptWhere : public ASTNode {

public:
	SearchCondition *n1 = NULL;

	explicit OptWhere(SearchCondition *n1) {
		assert(n1);
		this->n1 = n1;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

    virtual void accept(class SQL_RA_Translator &v) {
        v.visit(this);
    }

};

} // SQL_Namespace

#endif // SQL_OPTWHERE_H
