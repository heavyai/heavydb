/**
 * @file    RelExprList.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef RA_RELEXPRLIST_NODE_H
#define RA_RELEXPRLIST_NODE_H

#include <cassert>
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class RelExprList : public RelAlgNode {
    
public:
    RelExprList *n1 = NULL;
    RelExpr *n2 = NULL;

    /// Constructor
    explicit RelExprList(RelExpr *n2) {
        assert(n2);
        printf("RelExprList(RelExpr*) n2=%p\n", n2);
    	this->n2 = n2;
    }

	RelExprList(RelExprList *n1, RelExpr *n2) {
        assert(n1 && n2);
    	this->n1 = n1;
    	this->n2 = n2;
    }

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // RA_RELEXPRLIST_NODE_H
