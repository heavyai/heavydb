#ifndef UPDATE_STATEMENT_POSITIONED_NODE_H
#define UPDATE_STATEMENT_POSITIONED_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class UpdateStatementPositioned : public ASTNode {
    
public:
    Table *tbl;
    AssignmentCommalist *ac;
    Cursor* c;
    
    /**< Constructor */
    explicit UpdateStatementPositioned(Table *n, AssignmentCommalist *n2, Cursor *n3) : tbl(n), ac(n2), c(n3) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // UPDATE_STATEMENT_POSITIONED_NODE_H
