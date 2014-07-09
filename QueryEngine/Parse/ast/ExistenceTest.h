#ifndef EXISTENCE_TEST_NODE_H
#define EXISTENCE_TEST_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ExistenceTest : public ASTNode {
    
public:
	Subquery* sq;

    /* constructor */
    explicit ExistenceTest(Subquery* n) : sq(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // EXISTENCE_TEST_NODE_H
