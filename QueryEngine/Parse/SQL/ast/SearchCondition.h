#ifndef SEARCH_CONDITION_NODE_H
#define SEARCH_CONDITION_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  SearchCondition : public ASTNode {
    
public:

	int rule_Flag;
	/* rules are:
	0 OR
    1 AND
    2 NOT
    3 (search_condition) */

    SearchCondition* sc1;
    SearchCondition* sc2;
    Predicate* p;

    /* constructor */
    explicit SearchCondition(int rF, SearchCondition *n1, SearchCondition* n2) : rule_Flag(rF), sc1(n1), sc2(n2), p(NULL) {}
    SearchCondition(int rF, SearchCondition* n) : rule_Flag(rF), sc1(n), sc2(NULL), p(NULL) {}
    SearchCondition(Predicate *n) : rule_Flag(-1), sc1(NULL), sc2(NULL), p(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // SEARCH_CONDITION_NODE_H
