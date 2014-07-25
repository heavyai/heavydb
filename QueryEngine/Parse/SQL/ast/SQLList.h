#ifndef SQL_LIST_NODE_H
#define SQL_LIST_NODE_H

#include <cassert>
#include <iostream>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class SQLList : public ASTNode {
    
public:
    SQL *sql = NULL;
    SQLList *sqlList = NULL;
    
    /**< Constructor */
    explicit SQLList(SQL *n) {
        assert(n);
        this->sql = n;
    }

    SQLList(SQLList *n1, SQL *n2) {
        assert(n1 && n2);
        this->sqlList = n1;
        this->sql = n2;
    }
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // SQL_LIST_NODE_H
