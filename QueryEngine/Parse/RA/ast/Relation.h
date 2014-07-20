/**
 * @file	Relation.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_RELATION_NODE_H
#define RA_RELATION_NODE_H

#include <string>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class Relation : public RelAlgNode {
    
public:
	std::string name;

	/// Constructor
	Relation(const std::string &name) {
		this->name = name;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

} // RA_Namespace

#endif // RA_RELATION_NODE_H
