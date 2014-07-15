#ifndef INSERT_ATOM_NODE_H
#define INSERT_ATOM_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  InsertAtom : public ASTNode {
    
public:
    Atom *a;
    
    /**< Constructor */
    explicit InsertAtom(Atom *n) : a(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // INSERT_ATOM_NODE_H
