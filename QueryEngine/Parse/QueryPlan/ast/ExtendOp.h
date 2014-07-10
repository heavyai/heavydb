#ifndef EXTEND_OP_NODE_H
#define EXTEND_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class ExtendOp : public UnaryOp {
    
public:

	MathExpr *me;
	Data *data;
	std::string name;

	explicit ExtendOp(RelExpr *n1, MathExpr* n2, std::string n3) : relex(n1), me(n2), data(NULL), name(n3) {}
	ExtendOp(RelExpr *n1, Data* n2, std::string n3) : relex(n1), me(NULL), data(n2), name(n3) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // EXTEND_OP_NODE_H