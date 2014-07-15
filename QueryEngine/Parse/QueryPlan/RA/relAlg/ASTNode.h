#ifndef AST_NODE_H
#define AST_NODE_H

namespace RA_Namespace {
class ASTNode {
    
	public:

	    /**< Accepts the given void visitor by calling v.visit(this) */
	    virtual void accept(class Visitor &v) = 0;
	};
}

#endif // AST_NODE_H