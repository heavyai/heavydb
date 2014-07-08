#ifndef OPT_GROUP_BY_CLAUSE_NODE_H
#define OPT_GROUP_BY_CLAUSE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class OptGroupByClause : public ASTNode {
    
public:
    ColumnRefCommalist* crc;
    
    /**< Constructor */
    explicit OptGroupByClause(ColumnRefCommalist* n) : crc(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // OPT_GROUP_BY_CLAUSE_NODE_H
