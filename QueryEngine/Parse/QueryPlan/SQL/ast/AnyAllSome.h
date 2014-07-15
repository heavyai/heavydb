#ifndef AST_ANY_ALL_SOME_H
#define AST_ANY_ALL_SOME_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  AnyAllSome : public ASTNode {
    
public:
    std::string anyAllSome;
    
    /**< Constructor */
    explicit AnyAllSome(const std::string &n1) : anyAllSome(n1) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // AST_ANY_ALL_SOME_H
