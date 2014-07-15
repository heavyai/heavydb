#ifndef UPDATE_STATEMENT_SEARCHED_NODE_H
#define UPDATE_STATEMENT_SEARCHED_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  UpdateStatementSearched : public ASTNode {
    
public:
    Table *tbl;
    AssignmentCommalist *ac;
    OptWhereClause *owc;
    
    /**< Constructor */
    explicit UpdateStatementSearched(Table *n, AssignmentCommalist *n2, OptWhereClause *n3) : tbl(n), ac(n2), owc(n3) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // UPDATE_STATEMENT_SEARCHED_NODE_H
