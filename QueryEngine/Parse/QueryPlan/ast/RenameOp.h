#ifndef RENAME_OP_NODE_H
#define RENAME_OP_NODE_H

#include "RelAlgNode.h"
#include "UnaryOp.h"
#include "../visitor/Visitor.h"

class RenameOp : public UnaryOp {
    
public:

	Attribute* attr;
	std::string name;

	RenameOp(RelExpr *n1, Attribute* n2, std::string &n3) : attr(n2), name(n3) { relex = n1; }

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // RENAME_OP_NODE_H