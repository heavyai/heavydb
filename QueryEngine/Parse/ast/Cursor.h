#ifndef AST_CURSOR_H
#define AST_CURSOR_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Cursor : public ASTNode {
    
public:
    std::string name1;
    
    /**< Constructor */
    explicit Cursor(const std::string &n1) : name1(n1) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // AST_CURSOR_H
