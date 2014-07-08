#ifndef ORDERING_SPEC_NODE_H
#define ORDERING_SPEC_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class OrderingSpec : public ASTNode {
    
public:
    OrderingSpec *os;
    int OrderInt;

    /**< Constructor */
    explicit OrderingSpec(int i, OptAscDesc *n) : ia(n), iac(NULL) {}
    OrderingSpec(OrderingSpec *n1, InsertAtom *n2) 
        : iac(n1), ia(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // ORDERING_SPEC_NODE_H
