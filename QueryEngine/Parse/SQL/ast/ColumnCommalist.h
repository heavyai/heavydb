#ifndef SQL_COLUMN_COMMALIST_NODE_H
#define SQL_COLUMN_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  ColumnCommalist : public ASTNode {
    
public:
    Column *col;
    ColumnCommalist *colCom;
    
    /**< Constructor */
    explicit ColumnCommalist(Column *n) : col(n), colCom(NULL) {}
    ColumnCommalist(ColumnCommalist *n1, Column *n2) 
        : colCom(n1), col(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // SQL_COLUMN_COMMALIST_NODE_H
