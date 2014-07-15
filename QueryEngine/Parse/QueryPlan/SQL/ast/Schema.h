#ifndef SCHEMA_NODE_H
#define SCHEMA_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  Schema : public ASTNode {
    
public:
    BaseTableDef *btd;
    
    /**< Constructor */
    explicit Schema(BaseTableDef *n) : btd(n) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // SQL_NODE_H
