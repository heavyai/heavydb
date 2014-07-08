#ifndef OPT_ORDER_BY_CLAUSE_NODE_H
#define OPT_ORDER_BY_CLAUSE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class OptOrderByClause : public ASTNode {
    
public:
    OrderingSpecCommalist* osc;
    
    /**< Constructor */
    explicit OptOrderByClause(OrderingSpecCommalist* n) : osc(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // OPT_ORDER_BY_CLAUSE_NODE_H
