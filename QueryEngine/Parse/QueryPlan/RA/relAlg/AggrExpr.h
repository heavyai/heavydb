#ifndef AGGR_EXPR_NODE_H
#define AGGR_EXPR_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

using namespace RA_Namespace;

namespace RA_Namespace {
	class AggrExpr : public RelAlgNode {

public:

	std::string ammsc;			//AVG, MAX, etc
	std::string distinct;		//DISTINCT or not

	Attribute* attr;

	explicit AggrExpr(const std::string &n1, const std::string &n2, Attribute* n3) : ammsc(n1), distinct(n2), attr(n3) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // AGGR_EXPR_NODE_H
