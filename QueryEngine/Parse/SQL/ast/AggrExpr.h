/**
 * @file    AggrExpr.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_AGGREXPRLIST_NODE_H
#define SQL_AGGREXPRLIST_NODE_H

#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

class AggrExpr : public ASTNode {
    
public:
    Column *n1 = NULL;
    std::string func = "";

    /// Constructor
    AggrExpr(std::string func, Column *n1) {
    	assert(n1);
        this->n1 = n1;
    	this->func = func;
    }

	virtual void accept(class Visitor &v) const {
		v.visit(this);
	}

};

#endif // SQL_AGGREXPRLIST_NODE_H