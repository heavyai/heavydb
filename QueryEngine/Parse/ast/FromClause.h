#ifndef FROM_CLAUSE_NODE_H
#define FROM_CLAUSE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class FromClause : public ASTNode {
    
public:
    TableRefCommalist* trc; 
    SelectStatement* ss;

    /**< Constructor */
    explicit FromClause(TableRefCommalist *n) : trc(n), ss(NULL) {}
    FromClause(SelectStatement *n) : trc(NULL), ss(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // FROM_CLAUSE_NODE_H
