#ifndef SQL_AST_ANY_ALL_SOME_H
#define SQL_AST_ANY_ALL_SOME_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class AnyAllSome : public ASTNode {
    
public:
    std::string anyAllSome = "";
    
    /// Constructor
    explicit AnyAllSome(const std::string &n1) {
    	assert(n1 != "");
    	anyAllSome = n1;
    }

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

} // SQL_Namespace

#endif // SQL_AST_ANY_ALL_SOME_H
