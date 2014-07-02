#ifndef AST_TABLE_H
#define AST_TABLE_H

#include "ASTNode.h"
#include "../visitor/SimplePrinterVisitor.h"

class Table : public ASTNode {
    
public:
    std::string name1;
    std::string name2;
    
    /**< Constructor */
    explicit Table(const std::string &n1) : name1(n1) {}
    Table(const std::string &n1, const std::string &n2) : name1(n1), name2(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // AST_TABLE_H
