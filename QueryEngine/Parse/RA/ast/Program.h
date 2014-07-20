/**
 * @file	Program.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_PROGRAM_NODE_H
#define RA_PROGRAM_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class Program : public RelAlgNode {
    
public:
    RelExprList *n1;

    /// Constructor
    explicit Program(RelExprList *n1) {
    	this->n1 = n1;
    }

	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

}

#endif // RA_PROGRAM_NODE_H
