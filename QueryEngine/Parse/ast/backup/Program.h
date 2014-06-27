#ifndef AST_PROGRAM_H
#define AST_PROGRAM_H

#include "ASTNode.h"
#include "../visitor/SimplePrinterVisitor.h"

class Program : public ASTNode {
    
public:
    std::string test;
    
    /**< Constructor */
    explicit Program(const std::string n) : test(n) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // AST_PROGRAM_H
