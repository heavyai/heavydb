#ifndef OPT_COLUMN_COMMALIST_NODE_H
#define OPT_COLUMN_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class OptColumnCommalist : public ASTNode {
    
public:
    ColumnCommalist* cc;
    
    /**< Constructor */
    explicit OptColumnCommalist(ColumnCommalist *n) : cc(n) {}
        
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // OPT_COLUMN_COMMALIST_NODE_H
