#ifndef RENAME_OP_NODE_H
#define RENAME_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class RenameOp : public UnaryOp {
    
public:

	Attribute attr;
	std::string name;

	explicit RenameOp(RelExpr *n1, Attribute* n2, std::string n3) : relex(n1), attr(n2), name(n3) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // RENAME_OP_NODE_H