#ifndef PROJECT_OP_NODE_H
#define PROJECT_OP_NODE_H

#include "RelAlgNode.h"
#include "UnaryOp.h"
//#include "../visitor/Visitor.h"

namespace SQL_Namespace {
 ProjectOp : public UnaryOp {
    
public:

	AttrList* atLi;
	std::string selectAll;

	ProjectOp(RelExpr *n1, AttrList* n2) : atLi(n2) { relex = n1; }
	/* If the selection is "*"- this means, of course, the Attribute List spans the entirety of the table. */
	explicit ProjectOp(const std::string &n) : atLi(NULL), selectAll(n) { relex(NULL); }

/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // PROJECT_OP_NODE_H