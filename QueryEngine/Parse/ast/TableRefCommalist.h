#ifndef TABLE_REF_COMMALIST_NODE_H
#define TABLE_REF_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class TableRefCommalist : public ASTNode {
    
public:
    TableRef *tr;
    TableRefCommalist *trc;
    
    /**< Constructor */
    explicit TableRefCommalist(TableRef *n) : tr(n), trc(NULL) {}
    TableRefCommalist(TableRefCommalist *n1, TableRef *n2) 
        : trc(n1), tr(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // TABLE_REF_COMMALIST_NODE_H
