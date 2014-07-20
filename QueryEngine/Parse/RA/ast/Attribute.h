/**
 * @file    Attribute.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef RA_ATTRIBUTE_NODE_H
#define RA_ATTRIBUTE_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class Attribute : public RelAlgNode {
    
public:
    std::string name1 = "";
    std::string name2 = "";

    /// Constructor
    explicit Attribute(const std::string &name1) {
    	this->name1 = name1;
    }

	Attribute(std::string name1, const std::string &name2) {
		this->name1 = name1;
		this->name2 = name2;
    }

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // RA_RELEXPRLIST_NODE_H
