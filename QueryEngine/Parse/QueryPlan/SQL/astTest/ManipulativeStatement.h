#ifndef MANIPULATIVE_STATEMENT_NODE_H
#define MANIPULATIVE_STATEMENT_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  ManipulativeStatement : public ASTNode {
    
public:
    SelectStatement *selSta;
    UpdateStatementSearched *USS;
    UpdateStatementPositioned *USP;
    InsertStatement *inSta;
    
    
    /**< Constructor */
    explicit ManipulativeStatement(SelectStatement *n) : selSta(n), inSta(NULL), USS(NULL), USP(NULL) {}
    ManipulativeStatement(InsertStatement *n) : inSta(n), selSta(NULL), USS(NULL), USP(NULL) {}
    ManipulativeStatement(UpdateStatementPositioned *n) : inSta(NULL), selSta(NULL), USS(NULL), USP(n) {}
    ManipulativeStatement(UpdateStatementSearched *n) : inSta(NULL), selSta(NULL), USS(n), USP(NULL) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // MANIPULATIVE_STATEMENT_NODE_H
