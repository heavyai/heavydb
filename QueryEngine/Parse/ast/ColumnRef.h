#ifndef AST_COLUMN_REF_H
#define AST_COLUMN_REF_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ColumnRef : public ASTNode {
    
public:
	int args;
    std::string name1;
    std::string name2;
    std::string name3;
    
    int rule_Flag;
    /* Rules:
    0 '.'
    1 AS */

    /**< Constructor */
    explicit ColumnRef(const std::string &n1) : name1(n1), args(1), name2(""), name3("") {}
    ColumnRef(int rF, const std::string &n1, const std::string &n2) : rule_Flag(rF), name1(n1), name2(n2), name3(""), args(2) {}
    ColumnRef(const std::string &n1, const std::string &n2, const std::string &n3) : name1(n1), name2(n2), name3(n3), args(3) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // AST_COLUMN_REF_H
