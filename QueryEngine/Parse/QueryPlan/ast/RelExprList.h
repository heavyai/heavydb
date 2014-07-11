#ifndef REL_EXPR_LIST_NODE_H
#define REL_EXPR_LIST_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class RelExprList : public RelAlgNode {
    
public:

	RelExpr* relex;
	RelExprList* relexlist;

	explicit RelExprList(RelExprList *n1, RelExpr *n2) : relexlist(n1), relex(n2) {}
	RelExprList(RelExpr *n) : relex(n), relexlist(NULL) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // REL_EXPR_LIST_NODE_H