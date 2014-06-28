#ifndef SQL_LIST_NODE_H
#define SQL_LIST_NODE_H

#include "ASTNode.h"
#include "SQL.h"
#include "../visitor/Visitor.h"

class SQLList : public ASTNode {
    
public:
    SQL *sql;
    SQLList *sqlList;
    
    /**< Constructor */
    explicit SQLList(SQL *n) : sql(n) {}
    SQLList(SQLList *n1, SQL *n2) : sqlList(n1), sql(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // SQL_LIST_NODE_H
