#ifndef AST_AMMSC_H
#define AST_AMMSC_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  Ammsc : public ASTNode {
    
public:
    std::string funcName;
    
    /**< Constructor */
    explicit Ammsc(const std::string &n1) : funcName(n1) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // AST_AMMSC_H
