#ifndef MATH_EXPR_NODE_H
#define MATH_EXPR_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class MathExpr : public RelAlgNode {
    
public:

	int rule_Flag;
	/* rules are:
	0 (MATH_EXPR)
    1 addition
	2 subtraction
	3 multiplication
	4 division 
    5 positive Math_Expr
    6 negative Math_Expr */

    MathExpr* me1;
    MathExpr* me2;
    Attribute* attr;
    Data* data;
    AggrExpr* agex;

    /* constructor */
    MathExpr(int rF, MathExpr *n1, MathExpr* n2) : rule_Flag(rF), me1(n1), me2(n2), attr(NULL), data(NULL), agex(NULL) {}
    MathExpr(int rF, MathExpr* n) : rule_Flag(rF), me1(n), me2(NULL), attr(NULL), data(NULL), agex(NULL) {}
    MathExpr(Attribute *n) : rule_Flag(-1), me1(NULL), me2(NULL), attr(n), data(NULL), agex(NULL) {}
    MathExpr(Data *n) : rule_Flag(-1), me1(NULL), me2(NULL), attr(NULL), data(n), agex(NULL) {}
    MathExpr(AggrExpr* n) : rule_Flag(-1), me1(NULL), me2(NULL), attr(NULL), data(NULL), agex(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // MATH_EXPR_NODE_H
