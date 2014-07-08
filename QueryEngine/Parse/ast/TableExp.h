#ifndef TABLE_EXP_NODE_H
#define TABLE_EXP_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class TableExp : public ASTNode {
    
public:
    FromClause *fc;
  	OptWhereClause *owc;
    OptGroupByClause* ogbc;
    OptHavingClause* ohc;
    OptOrderByClause* oobc;
    OptLimitClause* olc;

    /**< Constructor */
    explicit TableExp(FromClause *n, OptWhereClause *n2, OptGroupByClause *n3, 
        OptHavingClause *n4, OptOrderByClause *n5, OptLimitClause *n6) :
    	 fc(n), owc(n2), ogbc(n3), ohc(n4), oobc(n5), olc(n6) {}
  
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // TABLE_EXP_NODE_H
