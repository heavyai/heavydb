#ifndef SQL_AST_GROUP_BY_LIST_H
#define SQL_AST_GROUP_BY_LIST_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class GroupByList : public ASTNode {
    
public:
    ScalarExp* se;
    GroupByList* gbl;
    OptAscDesc* oad;
    
    /// Constructor
    GroupByList(GroupByList* n, ScalarExp *n1, OptAscDesc *n2) : gbl(n), se(n1), oad(n2) {}
    GroupByList(ScalarExp *n1, OptAscDesc *n2) : gbl(NULL), se(n1), oad(n2) {}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

} // SQL_Namespace

#endif // SQL_AST_GROUP_BY_LIST_H
