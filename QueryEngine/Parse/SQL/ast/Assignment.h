#ifndef SQL_ASSIGNMENT_NODE_H
#define SQL_ASSIGNMENT_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class Assignment : public ASTNode {
    
public:
    Column *c;
    ScalarExp *se;
    
    /**< Constructor */
    Assignment(Column *n1, ScalarExp *n2) : c(n1), se(n2) {}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // SQL_ASSIGNMENT_NODE_H
