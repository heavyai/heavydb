#ifndef COMPOP_NODE_H
#define COMPOP_NODE_H


#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class CompOp : public RelAlgNode {
    
public:

	std::string *comparator;

	explicit CompOp(const std::string &n) : comparator(n) {}
};

#endif // COMPOP_NODE_H