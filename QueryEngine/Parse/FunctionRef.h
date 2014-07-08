#ifndef FUNCTION_REF_NODE_H
#define FUNCTION_REF_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class FunctionRef : public ASTNode {
    
public:
    int rule_Flag;
    
    /* Rules:
    0 (ALL scalar_exp)
    1 (scalar_exp) */

    Ammsc* am;
    ColumnRef* cr;
    ScalarExp* se;

    /* constructor */
    explicit FunctionRef(Ammsc* n) : rule_Flag(-1), am(n), cr(NULL), se(NULL) {}
    FunctionRef(Ammsc* n1, ColumnRef* n2) : rule_Flag(-1), am(n1), cr(n2), se(NULL) {}
    FunctionRef(int rF, Ammsc *n1, ScalarExp* n2) : rule_Flag(rf), am(n1), se(n2), cr(NULL) {}

    /**< Accepts the given void visitor by calling v.visit(this) 
    void accept(Visitor &v) {
        v.visit(this);
    }*/
    
};
#endif // FUNCTION_REF_NODE_H
