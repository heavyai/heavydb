#ifndef AST_NODE_H
#define AST_NODE_H

#include "../visitor/Visitor.h"

class ASTNode {
    
public:
    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(Visitor &v) = 0;
};

#endif // AST_NODE_H