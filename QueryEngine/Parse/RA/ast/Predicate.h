/**
 * @file    Predicate.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef RA_PREDICATE_NODE_H
#define RA_PREDICATE_NODE_H

#include <cassert>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class Predicate : public RelAlgNode {
    
public:
	Predicate *n1 = NULL;
	Predicate *n2 = NULL;
	Comparison *n3 = NULL;
    OpType op;

    /// Constructor
    Predicate(const OpType opType, Predicate *n1, Predicate *n2): op(opType) {
    	assert(op == OP_AND || op == OP_OR);
        assert(n1 && n2);
    	this->n1 = n1;
    	this->n2 = n2;
        this->n3 = NULL;
    }

    Predicate(const OpType opType, Predicate *n1): op(opType) {
    	assert(op == OP_NOT);
        assert(n1);
    	this->n1 = n1;
        this->n2 = NULL;
        this->n3 = NULL;
    }

    explicit Predicate(Predicate *n1): op(OP_NOOP) {
        assert(n1);
    	this->n1 = n1;
        this->n2 = NULL;
        this->n3 = NULL;
    }

    explicit Predicate(Comparison *n3): op(OP_NOOP) {
        assert(n3);
        this->n1 = NULL;
        this->n2 = NULL;
    	this->n3 = n3;
    }

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // RA_PREDICATE_NODE_H
