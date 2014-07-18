#ifndef SQL_AST_NODE_H
#define SQL_AST_NODE_H

#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class ASTNode {
    
public:
	/**< Accepts the given void visitor by calling v.visit(this) */
	virtual void accept(class Visitor &v) = 0;
};

} // SQL_Namespace

#endif // SQL_AST_NODE_H
