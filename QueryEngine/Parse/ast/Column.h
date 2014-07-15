#ifndef AST_COLUMN_H
#define AST_COLUMN_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Column : public ASTNode {
    
public:
    std::string name1;
    
    /**< Constructor */
    explicit Column(const std::string &n1) : name1(n1) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // AST_COLUMN_H
