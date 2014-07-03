#ifndef FROM_CLAUSE_NODE_H
#define FROM_CLAUSE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class FromClause : public ASTNode {
    
public:
    TableRefCommalist* trc; 

    /**< Constructor */
    explicit FromClause(TableRefCommalist *n) : trc(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // FROM_CLAUSE_NODE_H
