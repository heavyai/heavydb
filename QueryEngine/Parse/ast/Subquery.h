#ifndef SUBQUERY_NODE_H
#define SUBQUERY_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Subquery : public ASTNode {
    
public:
    OptAllDistinct* oad;
    Selection *s;
    TableExp *te;

    /* constructor */
    explicit Subquery(OptAllDistinct *n1, Selection *n2, TableExp *n3) : oad(n1), s(n2), te(n3) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // SUBQUERY_NODE_H
