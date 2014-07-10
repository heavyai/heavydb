#ifndef COMPARED_NODE_H
#define COMPARED_NODE_H


#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class Compared : public RelAlgNode {
    
public:

	Attribute *a;
	Data *d;

	explicit Compared(Attribute *n1, Data* n2) : a(n1), d(n2) {}
};

#endif // COMPARED_NODE_H