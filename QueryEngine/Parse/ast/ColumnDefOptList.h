#ifndef COLUMN_DEF_OPT_LIST_NODE_H
#define COLUMN_DEF_OPT_LIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ColumnDefOptList : public ASTNode {
    
public:
    ColumnDefOpt *colDefOpt;
    ColumnDefOptList *colDefOptList;
    
    /**< Constructor */
    explicit ColumnDefOptList(ColumnDefOptList *n1, ColumnDefOpt *n2) 
        : colDefOptList(n1), colDefOpt(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // COLUMN_DEF_OPT_LIST_NODE_H
