#ifndef AST_OPT_ASC_DESC_NODE_H
#define AST_OPT_ASC_DESC_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  OptAscDesc : public ASTNode {
    
public:
    std::string ascDesc;
    
    /**< Constructor */
    explicit OptAscDesc(const std::string &n1) : ascDesc(n1) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // AST_OPT_ASC_DESC_NODE_H
