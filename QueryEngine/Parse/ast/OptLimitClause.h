#ifndef OPT_LIMIT_CLAUSE_NODE_H
#define OPT_LIMIT_CLAUSE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class OptLimitClause : public ASTNode {
    
public:
    int lim1;
    // int lim2;
    
    /**< Constructor */
    explicit OptLimitClause(int Limit) : lim1(Limit) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // OPT_LIMIT_CLAUSE_NODE_H
