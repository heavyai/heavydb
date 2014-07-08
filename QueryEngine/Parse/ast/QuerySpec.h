#ifndef QUERY_SPEC_NODE_H
#define QUERY_SPEC_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class QuerySpec : public ASTNode {
    
public:
    OptAllDistinct *OAD;
    Selection *sel;
    TableExp *tblExp;
    
    /**< Constructor */
    explicit QuerySpec(OptAllDistinct *n, Selection *n2, TableExp *n3) : OAD(n), sel(n2), tblExp(n3) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // QUERY_SPEC_NODE_H
