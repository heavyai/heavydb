#ifndef OPT_ESCAPE_NODE_H
#define OPT_ESCAPE_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class OptEscape : public ASTNode {
    
public:

	Atom* a;

    /* constructor */
    explicit OptEscape(Atom* n) : a(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // OPT_ESCAPE_NODE_H
