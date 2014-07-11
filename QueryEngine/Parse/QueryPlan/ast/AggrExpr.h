#ifndef AGGR_EXPR_NODE_H
#define AGGR_EXPR_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class AggrExpr : public RelAlgNode {

public:

	int ammsc;
	int distinct;
	/* ammsc:
	0 MAX
    1 MIN
	2 COUNT
	3 SUM
	4 AVG

	distinct:
	0 (not distinct)
	1 DISTINCT */

	Attribute* attr;

	explicit AggrExpr(int rf1, int rf2, Attribute* n) : ammsc(rf1), distinct(rf2), attr(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
};

#endif // AGGR_EXPR_NODE_H
