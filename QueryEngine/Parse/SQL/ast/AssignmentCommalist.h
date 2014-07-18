#ifndef SQL_ASSIGNMENT_COMMALIST_NODE_H
#define SQL_ASSIGNMENT_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class AssignmentCommalist : public ASTNode {
    
public:
    Assignment *a;
    AssignmentCommalist *ac;
    
    /**< Constructor */
    explicit AssignmentCommalist(Assignment *n) : a(n), ac(NULL) {}
    AssignmentCommalist(AssignmentCommalist *n1, Assignment *n2) 
        : ac(n1), a(n2) {}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // SQL_ASSIGNMENT_COMMALIST_NODE_H
