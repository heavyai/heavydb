#ifndef VALUES_OR_QUERY_SPEC_NODE_H
#define VALUES_OR_QUERY_SPEC_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ValuesOrQuerySpec : public ASTNode {
    
public:
    InsertAtomCommalist *iac;
    QuerySpec *qs;
    
    
    /**< Constructor */
    explicit ValuesOrQuerySpec(InsertAtomCommalist *n) : iac(n), qs(NULL) {}
    ValuesOrQuerySpec(QuerySpec *n) : qs(n), iac(NULL) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // VALUES_OR_QUERY_SPEC_NODE_H
