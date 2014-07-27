#ifndef SQL_COLUMN_COMMALIST_NODE_H
#define SQL_COLUMN_COMMALIST_NODE_H

#include <cstddef>
#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class ColumnCommalist : public ASTNode {
    
public:
    Column *col = NULL;
    ColumnCommalist *colCom = NULL;
    
    /**< Constructor */
    explicit ColumnCommalist(Column *n) {
        assert(n);
        this->col = n;
    }
    ColumnCommalist(ColumnCommalist *n1, Column *n2) {
        assert(n1 && n2);
        this->colCom = n1;
        this->col = n2;
    }
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // SQL_COLUMN_COMMALIST_NODE_H
