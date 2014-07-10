#ifndef ATTRIBUTE_NODE_H
#define ATTRIBUTE_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class Attribute : public RelAlgNode {
    
public:

	std::string name1;
	std::string name2;

    explicit Attribute(const std::string &n1) : name1(n1)
    Attribute(const std::string &n1, const std::string &n2) : name1(n1), name2(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // ATTRIBUTE_NODE_H