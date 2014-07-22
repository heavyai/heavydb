#ifndef OPT_GROUP_BY_CLAUSE_NODE_H
#define OPT_GROUP_BY_CLAUSE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  OptGroupByClause : public ASTNode {
    
public:
    ColumnRefCommalist* crc;
    GroupByList* gbl;

    /**< Constructor */
    explicit OptGroupByClause(ColumnRefCommalist* n) : crc(n), gbl(NULL) {}
    explicit OptGroupByClause(GroupByList* n) : gbl(n), crc(NULL) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // OPT_GROUP_BY_CLAUSE_NODE_H
