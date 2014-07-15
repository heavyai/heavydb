#ifndef SCALAR_EXP_COMMALIST_NODE_H
#define SCALAR_EXP_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  ScalarExpCommalist : public ASTNode {
    
public:
    ScalarExp *se;
    ScalarExpCommalist *sec;
    
    /**< Constructor */
    explicit ScalarExpCommalist(ScalarExp *n) : se(n), sec(NULL) {}
    ScalarExpCommalist(ScalarExpCommalist *n1, ScalarExp *n2) 
        : sec(n1), se(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // SCALAR_EXP_COMMALIST_NODE_H
