#ifndef AST_LITERAL_H
#define AST_LITERAL_H

#include "ASTNode.h"
#include <cassert>
#include "AbstractScalarExpr.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  Literal : public AbstractScalarExpr {
    
public:
    std::string name1 = "";
    int int1 = 0;

    /**< Constructor */
    Literal(const std::string &n1) {
        assert(n1 != "");
        name1 = n1; 
        this->setType(SCALAR_STRING);
    }
    Literal(int n) {
        int1 = n;
        this->setType(SCALAR_INT); 
    }

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // AST_LITERAL_H
