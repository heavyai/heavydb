#ifndef COLUMN_DEF_NODE_H
#define COLUMN_DEF_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ColumnDef : public ASTNode {
    
public:

    Column *col;
    DataType *dType;
    ColumnDefOptList *colDefOptList;

    /* constructor */
    explicit ColumnDef(Column *col1, DataType *dType1, ColumnDefOptList *colDefOptList1) : col(col1), dType(dType1), colDefOptList(colDefOptList1) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // COLUMN_DEF_NODE_H
