#ifndef COMPOP_NODE_H
#define COMPOP_NODE_H


#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class CompOp : public RelAlgNode {
    
public:

	const std::string comparator;

	CompOp(const std::string &n) : comparator(n) {}

		/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // COMPOP_NODE_H