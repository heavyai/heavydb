#ifndef VALUES_OR_QUERY_SPEC_NODE_H
#define VALUES_OR_QUERY_SPEC_NODE_H

#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class ValuesOrQuerySpec : public ASTNode {
    
public:
    InsertAtomCommalist *iac = NULL;
    QuerySpec *qs = NULL;
    
    
    /**< Constructor */
    explicit ValuesOrQuerySpec(InsertAtomCommalist *n) {
        assert(n);
        this->iac = n;
    }
    
    explicit ValuesOrQuerySpec(QuerySpec *n) {
        assert(n);
        this->qs = n;
    }
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // VALUES_OR_QUERY_SPEC_NODE_H
