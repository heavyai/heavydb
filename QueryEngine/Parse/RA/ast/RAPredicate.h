#ifndef RA_PREDICATE_NODE_H
#define RA_PREDICATE_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
	class RAPredicate : public RelAlgNode {
    
public:
	
	int rule_Flag;
	/* rules:
	0 OR 
	1 AND
	2 NOT
	3 ( predicate )
	*/
	RAPredicate* p1;
	RAPredicate* p2;
	Comparison* c;

	RAPredicate(int rf, RAPredicate *n1, RAPredicate *n2) : rule_Flag(rf), p1(n1), p2(n2) {}
	RAPredicate(int rf, RAPredicate *n) : rule_Flag(rf), p1(n), p2(NULL) {}
	RAPredicate(Comparison* n) : c(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // RA_PREDICATE_NODE_H