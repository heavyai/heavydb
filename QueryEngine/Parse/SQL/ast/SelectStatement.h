#ifndef SELECT_STATEMENT_NODE_H
#define SELECT_STATEMENT_NODE_H

#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class SelectStatement : public ASTNode {
    
public:
    OptAllDistinct *OAD = NULL;
    Selection *sel = NULL;
    TableExp *tblExp = NULL;
    
    /**< Constructor */
    explicit SelectStatement(OptAllDistinct *n, Selection *n2, TableExp *n3) {
        assert(n && n2 && n3);
        this->OAD = n;
        this->sel = n2;
        this->tblExp = n3;
    }
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // SELECT_STATEMENT_NODE_H
