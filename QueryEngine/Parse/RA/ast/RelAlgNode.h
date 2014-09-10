/**
 * @file	RelAlgNode.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_RELALGNODE_H
#define RA_RELALGNODE_H

#include "../visitor/Visitor.h"

namespace RA_Namespace {

enum OpType {  
    OP_GT,OP_LT,OP_GTE,OP_LTE,OP_NEQ,OP_EQ,OP_ADD,OP_SUBTRACT,OP_MULTIPLY,OP_DIVIDE,OP_AND,OP_OR,OP_NOT,OP_NOOP
};

class RelAlgNode {
    
public:
	/**< Accepts the given void visitor by calling v.visit(this) */
	virtual void accept(class Visitor &v) = 0;
};

} // RA_Namespace

#endif // RA_RELALGNODE_H
