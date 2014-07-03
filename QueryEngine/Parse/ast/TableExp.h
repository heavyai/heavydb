#ifndef TABLE_EXP_NODE_H
#define TABLE_EXP_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class TableExp : public ASTNode {
    
public:
    FromClause *fc;
  //  OptWhereClause oWC;
   // OptGroupByClause oGBC;
    
    /**< Constructor */
    explicit TableExp(FromClause *n) : fc(n) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // TABLE_EXP_NODE_H
