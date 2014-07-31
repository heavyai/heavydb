#ifndef SQL_OPTHAVING_H
#define SQL_OPTHAVING_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class OptHaving : public ASTNode {

public:

	explicit OptHaving() {

	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

    virtual void accept(class SQL_RA_Translator &v) {
        v.visit(this);
    }

};

} // SQL_Namespace

#endif // SQL_OPTHAVING_H
