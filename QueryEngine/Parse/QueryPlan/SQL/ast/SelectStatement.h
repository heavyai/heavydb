#ifndef SELECT_STATEMENT_NODE_H
#define SELECT_STATEMENT_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  SelectStatement : public ASTNode {
    
public:
    OptAllDistinct *OAD;
    Selection *sel;
    TableExp *tblExp;
    
    /**< Constructor */
    explicit SelectStatement(OptAllDistinct *n, Selection *n2, TableExp *n3) : OAD(n), sel(n2), tblExp(n3) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // SELECT_STATEMENT_NODE_H
