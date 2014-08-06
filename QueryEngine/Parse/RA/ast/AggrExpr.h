/**
 * @file    AggrExpr.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef RA_AGGREXPRLIST_NODE_H
#define RA_AGGREXPRLIST_NODE_H

#include <cassert>
#include <string>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class AggrExpr : public RelAlgNode {
    
public:
    Attribute *n1 = NULL;
    std::string func = "";

    /// Constructor
    AggrExpr(std::string func, Attribute *n1) {
    	assert(n1);
        this->n1 = n1;
    	this->func = func;
    }

	virtual void accept(class RA_Namespace::Visitor &v) {
		v.visit(this);
	}
};

}

#endif // RA_AGGREXPRLIST_NODE_H
