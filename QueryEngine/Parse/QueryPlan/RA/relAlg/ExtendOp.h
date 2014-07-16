#ifndef EXTEND_OP_NODE_H
#define EXTEND_OP_NODE_H

#include "RelAlgNode.h"
#include "UnaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class ExtendOp : public UnaryOp {
    
public:

	MathExpr *me;
//	Data *data;
	std::string name;

	explicit ExtendOp(RelExpr *n1, MathExpr* n2, std::string n3) : me(n2), name(n3) { relex = n1; }
//	ExtendOp(RelExpr *n1, Data* n2, std::string n3) : me(NULL), data(n2), name(n3) { relex = n1; }

/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // EXTEND_OP_NODE_H