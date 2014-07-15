#ifndef OPT_LIMIT_CLAUSE_NODE_H
#define OPT_LIMIT_CLAUSE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  OptLimitClause : public ASTNode {
    
public:
    double lim1;
    double lim2;
    
    int rule_Flag;
    /* Rules:
    0 ','
    1 OFFSET */

    /**< Constructor */
    explicit OptLimitClause(double limit) : rule_Flag(-1), lim1(limit), lim2(-1) {}
    OptLimitClause(int rF, double limit1, double limit2) : rule_Flag(rF), lim1(limit1), lim2(limit2) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // OPT_LIMIT_CLAUSE_NODE_H
