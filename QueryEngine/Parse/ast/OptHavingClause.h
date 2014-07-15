#ifndef OPT_HAVING_CLAUSE_NODE_H
#define OPT_HAVING_CLAUSE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class OptHavingClause : public ASTNode {
    
public:
    SearchCondition* sc;
    
    /**< Constructor */
    explicit OptHavingClause(SearchCondition* n) : sc(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // OPT_HAVING_CLAUSE_NODE_H
