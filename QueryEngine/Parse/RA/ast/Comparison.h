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
    std::string op;

    /// Constructor
    Comparison(std::string op, MathExpr *n1, MathExpr *n2) {
    	assert(n1 && n2);
        this->n1 = n1;
    	this->n2 = n2;
    	
        if (op == "GT")
            this->op = ">";
        else if (op == "LT")
            this->op = "<";
        else if (op == "GTE")
            this->op = ">=";
        else if (op == "LTE")
            this->op = "<=";
        else if (op == "NEQ")
            this->op = "!=";
        else if (op == "EQ")
            this->op = "=";
        else
            this->op = op;
    }

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // RA_COMPARISON_NODE_H
