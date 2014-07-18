#ifndef ATTRIBUTE_NODE_H
#define ATTRIBUTE_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class Attribute : public RelAlgNode {
    
public:

	std::string name1;
	std::string name2;
	
    Attribute(const std::string &n1) : name1(n1) {}
    Attribute(const std::string &n1, const std::string &n2) : name1(n1), name2(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // ATTRIBUTE_NODE_H