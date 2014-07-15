#ifndef BASE_TABLE_ELEMENT_COMMALIST_NODE_H
#define BASE_TABLE_ELEMENT_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  BaseTableElementCommalist : public ASTNode {
    
public:
    BaseTableElement *bte;
    BaseTableElementCommalist *btec;
    
    /**< Constructor */
    explicit BaseTableElementCommalist(BaseTableElement *n) : bte(n), btec(NULL) {}
    BaseTableElementCommalist(BaseTableElementCommalist *n1, BaseTableElement *n2) 
        : btec(n1), bte(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // BASE_TABLE_ELEMENT_COMMALIST_NODE_H
