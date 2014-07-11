#ifndef MATH_EXPR_NODE_H
#define MATH_EXPR_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class MathExpr : public ASTNode {
    
public:

	int rule_Flag;
	/* rules are:
	0 (MATH_EXPR)
    1 addition
	2 subtraction
	3 multiplication
	4 division */

    MathExpr* me1;
    MathExpr* me2;
    Attribute* attr;

    /* constructor */
    explicit MathExpr(int rF, MathExpr *n1, MathExpr* n2) : rule_Flag(rF), me1(n1), me2(n2), attr(NULL) {}
    MathExpr(int rF, MathExpr* n) : rule_Flag(rF), me1(n), me2(NULL), a(NULL){}
    MathExpr(Attribute *n) : rule_Flag(-1), me1(NULL), me2(NULL), a(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // MATH_EXPR_NODE_H
