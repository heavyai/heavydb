#ifndef DATA_NODE_H
#define DATA_NODE_H


#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class Data : public RelAlgNode {
    
public:
	double d;
	std::string s;

	Data(double n) : d(n), s("") {}
	Data(const std::string &n) : d(-1), s(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
};

#endif // DATA_NODE_H