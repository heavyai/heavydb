#ifndef ASSIGNMENT_COMMALIST_NODE_H
#define ASSIGNMENT_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class AssignmentCommalist : public ASTNode {
    
public:
    Assignment *a;
    AssignmentCommalist *ac;
    
    /**< Constructor */
    explicit AssignmentCommalist(Assignment *n) : a(n), ac(NULL) {}
    AssignmentCommalist(AssignmentCommalist *n1, Assignment *n2) 
        : ac(n1), a(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // ASSIGNMENT_COMMALIST_NODE_H
