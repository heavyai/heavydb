#ifndef TABLE_REF_NODE_H
#define TABLE_REF_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  TableRef : public ASTNode {
    
public:
    Table *tbl;

    /**< Constructor */
    explicit TableRef(Table *n) : tbl(n) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // TABLE_REF_NODE_H
