#ifndef INSERT_ATOM_COMMALIST_NODE_H
#define INSERT_ATOM_COMMALIST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class InsertAtomCommalist : public ASTNode {
    
public:
    InsertAtom *ia;
    InsertAtomCommalist *iac;
    
    /**< Constructor */
    explicit InsertAtomCommalist(InsertAtom *n) : ia(n), iac(NULL) {}
    InsertAtomCommalist(InsertAtomCommalist *n1, InsertAtom *n2) 
        : iac(n1), ia(n2) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // INSERT_ATOM_COMMALIST_NODE_H
