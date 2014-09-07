/**
 * @file    Comparison.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef RA_COMPARISON_NODE_H
#define RA_COMPARISON_NODE_H

#include <cassert>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class Comparison : public RelAlgNode {
    
public:
    MathExpr *n1;
    MathExpr *n2;
    //std::string op;
    OpType op;
    bool isBinaryOp;
     

    /// Constructor
    Comparison(OpType opType, MathExpr *n1, MathExpr *n2): op(opType) {
    	assert(n1 && n2);
        this->n1 = n1;
    	this->n2 = n2;
        isBinaryOp = true;
    
        /*
        if (op == "GT")
            this->op = OP_GT;
        else if (op == "LT")
            this->op = OP_LT; 
        else if (op == "GTE")
            this->op = OP_GTE;
        else if (op == "LTE")
            this->op = OP_LTE;
        else if (op == "NEQ")
            this->op = OP_NEQ;
        else if (op == "EQ")
            this->op = OP_EQ;
        */
    }

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // RA_COMPARISON_NODE_H
