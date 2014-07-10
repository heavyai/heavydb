#ifndef PROJECT_OP_NODE_H
#define PROJECT_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class ProjectOp : public UnaryOp {
    
public:

	AttrList* atLi;

	explicit ProjectOp(RelExpr *n1, AttrList* n2) : relex(n1), atLi(n2) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // PROJECT_OP_NODE_H