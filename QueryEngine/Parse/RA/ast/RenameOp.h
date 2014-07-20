/**
 * @file	RenameOp.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_RENAMEOP_NODE_H
#define RA_RENAMEOP_NODE_H

#include <cassert>
#include "UnaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class RenameOp : public UnaryOp {
    
public:
	RelExpr *n1 = NULL;
	std::string name1 = "";
	std::string name2 = "";

	/// Constructor
	RenameOp(RelExpr *n1, const std::string &name1, const std::string &name2) {
		assert(n1 && name1 != "" && name2 != "");
		this->n1 = n1;
		this->name1 = name1;
		this->name2 = name2;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // RA_Namespace

#endif // RA_RENAMEOP_NODE_H
