#ifndef COLUMN_REF_COMMALIST_NODE_H
#define COLUMN_REF_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ColumnRefCommalist : public ASTNode {
    
public:
    ColumnRef *cr;
    ColumnRefCommalist *crc;
    
    /**< Constructor */
    explicit ColumnRefCommalist(ColumnRef *n) : cr(n), crc(NULL) {}
    ColumnRefCommalist(ColumnRefCommalist *n1, ColumnRef *n2) 
        : crc(n1), cr(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // COLUMN_REF_COMMALIST_NODE_H
