/**
 * @file	RelAlgNode.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_RELALGNODE_H
#define RA_RELALGNODE_H

#include "../visitor/Visitor.h"

namespace RA_Namespace {

class RelAlgNode {
    
public:
	/**< Accepts the given void visitor by calling v.visit(this) */
	virtual void accept(class Visitor &v) = 0;
};

} // RA_Namespace

#endif // RA_RELALGNODE_H
