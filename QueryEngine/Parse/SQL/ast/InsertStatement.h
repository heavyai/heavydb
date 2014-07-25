#ifndef INSERT_STATEMENT_NODE_H
#define INSERT_STATEMENT_NODE_H

#include <cassert>
#include <cstddef>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class InsertStatement : public ASTNode {
    
public:
    Table *tbl = NULL;
    OptColumnCommalist *oCC = NULL;
    ValuesOrQuerySpec* voQS = NULL;
    
    /**< Constructor */
    InsertStatement(Table *n, OptColumnCommalist *n2, ValuesOrQuerySpec *n3) {
        assert(n && n2 && n3);
        this->tbl = n;
        this->oCC = n2;
        this->voQS = n3;
    }
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // INSERT_STATEMENT_NODE_H
