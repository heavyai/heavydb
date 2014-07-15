#ifndef OPT_WHERE_CLAUSE_NODE_H
#define OPT_WHERE_CLAUSE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  OptWhereClause : public ASTNode {
    
public:
    SearchCondition *sc;
    
    /**< Constructor */
    explicit OptWhereClause(SearchCondition* n) : sc(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // OPT_WHERE_CLAUSE_NODE_H
