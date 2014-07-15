#ifndef BASE_TABLE_ELEMENT_NODE_H
#define BASE_TABLE_ELEMENT_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class BaseTableElement : public ASTNode {
    
public:
    ColumnDef *colDef;
    TableConstraintDef *tblConDef;
    
    /**< Constructor */
    explicit BaseTableElement(ColumnDef *n) : colDef(n), tblConDef(NULL) {} 
    explicit BaseTableElement(TableConstraintDef *n) : colDef(NULL), tblConDef(n) {}
        
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // BASE_TABLE_ELEMENT_NODE_H
