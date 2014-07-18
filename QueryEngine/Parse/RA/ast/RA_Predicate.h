#ifndef RA_PREDICATE_NODE_H
#define RA_PREDICATE_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
	class RA_Predicate : public RelAlgNode {
    
public:
	
	int rule_Flag;
	/* rules:
	0 OR 
	1 AND
	2 NOT
	3 ( predicate )
	*/
	RA_Predicate* p1;
	RA_Predicate* p2;
	Comparison* c;

	RA_Predicate(int rf, RA_Predicate *n1, RA_Predicate *n2) : rule_Flag(rf), p1(n1), p2(n2) {}
	RA_Predicate(int rf, RA_Predicate *n) : rule_Flag(rf), p1(n), p2(NULL) {}
	RA_Predicate(Comparison* n) : c(n) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // RA_PREDICATE_NODE_H