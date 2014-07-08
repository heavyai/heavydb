#ifndef MANIPULATIVE_STATEMENT_NODE_H
#define MANIPULATIVE_STATEMENT_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ManipulativeStatement : public ASTNode {
    
public:
    SelectStatement *selSta;
    /* UpdateStatementSearched *USS;
    UpdateStatementPosition *UPS; */
    InsertStatement *inSta;
    
    
    /**< Constructor */
    explicit ManipulativeStatement(SelectStatement *n) : selSta(n), inSta(NULL) {}
    ManipulativeStatement(InsertStatement *n) : inSta(n), selSta(NULL) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // MANIPULATIVE_STATEMENT_NODE_H
