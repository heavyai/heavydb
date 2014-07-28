#ifndef AST_LITERAL_H
#define AST_LITERAL_H

#include "ASTNode.h"
#include "../../../../Shared/types.h"
#include <cassert>
#include "AbstractScalarExpr.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  Literal : public AbstractScalarExpr {
    
public:
    mapd_data_t type;
    std::string name1 = "";
    int int1 = 0;
    double f;

    /**< Constructor */
    Literal(const std::string &n1) {
        assert(n1 != "");
        //printf("str n =%s\n", n1.c_str());
        name1 = n1; 
        // @todo type = STR_TYPE;
    }
    Literal(long int n) {
        //printf("int n =%d\n", n);
        int1 = n;
        type = INT_TYPE;
    }

    Literal(double n) {
        //printf("double n =%d\n", n);
        this->f = n;
        type = FLOAT_TYPE;
    }

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

} // SQL_Namespace

#endif // AST_LITERAL_H
