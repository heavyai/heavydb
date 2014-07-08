#ifndef ORDERING_SPEC_COMMALIST_NODE_H
#define ORDERING_SPEC_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class OrderingSpecCommalist : public ASTNode {
    
public:
    OrderingSpec *os;
    OrderingSpecCommalist *osc;
    
    /**< Constructor */
    explicit OrderingSpecCommalist(OrderingSpec *n) : os(n), osc(NULL) {}
    OrderingSpecCommalist(OrderingSpecCommalist *n1, OrderingSpec *n2) 
        : osc(n1), os(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // ORDERING_SPEC_COMMALIST_NODE_H
