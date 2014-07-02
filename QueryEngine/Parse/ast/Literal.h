#ifndef AST_LITERAL_H
#define AST_LITERAL_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Literal : public ASTNode {
    
public:
    std::string name1;
    
    /**< Constructor */
    explicit Literal(const std::string &n1) : name1(n1) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // AST_LITERAL_H
