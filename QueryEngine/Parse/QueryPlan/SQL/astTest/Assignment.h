#ifndef ASSIGNMENT_NODE_H
#define ASSIGNMENT_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  Assignment : public ASTNode {
    
public:
    Column *c;
    ScalarExp *se;
    
    /**< Constructor */
    explicit Assignment(Column *n1, ScalarExp *n2) : c(n1), se(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // ASSIGNMENT_NODE_H
