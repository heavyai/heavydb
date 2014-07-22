/**
 * @file    RelExpr.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef RA_RELEXPR_NODE_H
#define RA_RELEXPR_NODE_H

#include <cassert>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class RelExpr : public RelAlgNode {
    
public:
    UnaryOp *n1 = NULL;
    BinaryOp *n2 = NULL;
    RelExpr *n3 = NULL;
    Relation *n4 = NULL;

    /// Constructor
    explicit RelExpr(UnaryOp *n1) {
    	assert(n1);
        this->n1 = n1;
    }

    explicit RelExpr(BinaryOp *n2) {
        assert(n2);
        this->n2 = n2;
    }

    explicit RelExpr(RelExpr *n3) {
        assert(n3);
        this->n3 = n3;
    }

    explicit RelExpr(Relation *n4) {
        assert(n4);
        this->n4 = n4;
    }

	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

} // RA_Namespace

#endif // RA_RELEXPRLIST_NODE_H
