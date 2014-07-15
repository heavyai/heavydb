#ifndef REL_EXPR_LIST_NODE_H
#define REL_EXPR_LIST_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class RelExpr;

namespace RA_Namespace {
	class RelExprList : public RelAlgNode {
	    
	public:

		RelExpr* relex;
		RelExprList* relexlist;

		explicit RelExprList(RelExprList *n1, RelExpr *n2) : relexlist(n1), relex(n2) {}
		RelExprList(RelExpr *n) : relex(n), relexlist(NULL) {}

	    virtual void accept(Visitor &v) {
	        v.visit(this);
	    }
	};
}

#endif // REL_EXPR_LIST_NODE_H