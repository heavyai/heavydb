#ifndef MANIPULATIVE_STATEMENT_NODE_H
#define MANIPULATIVE_STATEMENT_NODE_H

#include <cassert>
#include <cstddef>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class ManipulativeStatement : public ASTNode {
    
public:
    SelectStatement *selSta = NULL;
    UpdateStatementSearched *USS = NULL;
    UpdateStatementPositioned *USP = NULL;
    InsertStatement *inSta = NULL;
   
    /// Constructor
    explicit ManipulativeStatement(SelectStatement *n) {
        assert(n);
        this->selSta = n;
    }

    /// Constructor
    explicit ManipulativeStatement(InsertStatement *n) {
        assert(n);
        this->inSta = n;
    }

    /// Constructor
    explicit ManipulativeStatement(UpdateStatementPositioned *n) {
        assert(n);
        this->USP = n;
    }

    /// Constructor
    explicit ManipulativeStatement(UpdateStatementSearched *n) {
        assert(n);
        this->USS = n;
    }
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // ManipulativeStatement

#endif // MANIPULATIVE_STATEMENT_NODE_H
