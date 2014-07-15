#ifndef ATOM_COMMALIST_NODE_H
#define ATOM_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class AtomCommalist : public ASTNode {
    
public:
    Atom *a;
    AtomCommalist *ac;
    
    /**< Constructor */
    explicit AtomCommalist(Atom *n) : a(n), ac(NULL) {}
    AtomCommalist(AtomCommalist *n1, Atom *n2) 
        : ac(n1), a(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // ATOM_COMMALIST_NODE_H
