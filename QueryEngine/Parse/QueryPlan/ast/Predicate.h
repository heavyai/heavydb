#ifndef PREDICATE_NODE_H
#define PREDICATE_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class Predicate : public RelAlgNode {
    
public:
	
	int rule_Flag;
	/* rules:
	0 OR 
	1 AND
	2 NOT
	3 ( predicate )
	*/
	Predicate* p1;
	Predicate* p2;
	Comparison* c;

	explicit Predicate(int rf, Predicate *n1, Predicate *n2) : rule_Flag(rf), p1(n1), p2(n2) {}
	Predicate(int rf, Predicate *n) : rule_Flag(rf), p1(n), p2(NULL) {}
	Predicate(Comparison* n) : c(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // PREDICATE_NODE_H