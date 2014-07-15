#ifndef ORDERING_SPEC_NODE_H
#define ORDERING_SPEC_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  OrderingSpec : public ASTNode {
    
public:
    int orderInt;
    ColumnRef *cr;
    OptAscDesc *oad;

    /**< Constructor */
    explicit OrderingSpec(int i, OptAscDesc *n) : orderInt(i), cr(NULL), oad(n) {}
    OrderingSpec(ColumnRef *n1, OptAscDesc *n2) 
        : cr(n1), oad(n2), orderInt(-1) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // ORDERING_SPEC_NODE_H
