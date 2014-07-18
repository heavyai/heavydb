#ifndef INSERT_STATEMENT_NODE_H
#define INSERT_STATEMENT_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  InsertStatement : public ASTNode {
    
public:
    Table *tbl;
    OptColumnCommalist *oCC;
    ValuesOrQuerySpec* voQS;
    
    /**< Constructor */
    explicit InsertStatement(Table *n, OptColumnCommalist *n2, ValuesOrQuerySpec *n3) : tbl(n), oCC(n2), voQS(n3) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // INSERT_STATEMENT_NODE_H
