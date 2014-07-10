#ifndef GROUP_BY_OP_NODE_H
#define GROUP_BY_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class GroupByOp : public UnaryOp {
    
public:

	AttrList *atLi;
	AggrList *agLi;

	explicit GroupByOp(RelExpr *n1, MathExpr* n2, std::string n3) : relex(n1), atLi(n2), AggrList(n3) {}
	GroupByOp(RelExpr *n1, Data* n2, std::string n3) : relex(n1), atLi(NULL), agLi(n2) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // GROUP_BY_OP_NODE_H