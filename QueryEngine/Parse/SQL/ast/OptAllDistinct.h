#ifndef SQL_OPTALLDISTINCT_H
#define SQL_OPTALLDISTINCT_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class OptAllDistinct : public ASTNode {

public:

	explicit OptAllDistinct() {

	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

    virtual void accept(class SQL_RA_Translator &v) {
        v.visit(this);
    }

};

} // SQL_Namespace

#endif // SQL_OPTALLDISTINCT_H
