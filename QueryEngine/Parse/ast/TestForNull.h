#ifndef TEST_FOR_NULL_NODE_H
#define TEST_FOR_NULL_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class TestForNull : public ASTNode {
    
public:
	int rule_Flag;
	/* Rules:
	0 IS NULL
	1 IS NOT NULL */

    ColumnRef* cr;

    /* constructor */
    explicit TestForNull(int rF, ColumnRef* n) : rule_Flag(rF), cr(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // TEST_FOR_NULL_NODE_H
