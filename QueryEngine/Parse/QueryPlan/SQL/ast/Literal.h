#ifndef AST_LITERAL_H
#define AST_LITERAL_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  Literal : public ASTNode {
    
public:
    std::string name1;
    int int1;

    /**< Constructor */
    Literal(const std::string &n1) : name1(n1), int1(0) {}
    Literal(int n) : int1(n), name1("") {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // AST_LITERAL_H
