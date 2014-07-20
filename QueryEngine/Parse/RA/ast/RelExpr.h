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
    UnaryOp *n1;
    BinaryOp *n2;
    RelExpr *n3;
    Relation *n4;

    /// Constructor
    explicit RelExpr(UnaryOp *n1) {
    	assert(n1);
        this->n1 = n1;
        this->n2 = NULL;
        this->n3 = NULL;
        this->n4 = NULL;
    }

    explicit RelExpr(BinaryOp *n2) {
        assert(n2);
    	this->n1 = NULL;
        this->n2 = n2;
        this->n3 = NULL;
        this->n4 = NULL;
    }

    explicit RelExpr(RelExpr *n3) {
        assert(n3);
        this->n1 = NULL;
        this->n2 = NULL;
        this->n3 = n3;
        this->n4 = NULL;
    }

    explicit RelExpr(Relation *n4) {
        assert(n4);
        this->n1 = NULL;
        this->n2 = NULL;
        this->n3 = NULL;
        this->n4 = n4;
    }

	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

} // RA_Namespace

#endif // RA_RELEXPRLIST_NODE_H
