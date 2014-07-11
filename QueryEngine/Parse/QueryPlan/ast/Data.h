#ifndef DATA_NODE_H
#define DATA_NODE_H


#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class Data : public RelAlgNode {
    
public:
	double d;
	std::string *s;

	explicit DATA(double n) : d(n), s("") {}
	explicit DATA(const std::string &n) : d(-1), s(n) {}
};

#endif // DATA_NODE_H