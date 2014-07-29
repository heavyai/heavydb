/**
 * @file	ASTNode.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_ASTNODE_H
#define SQL_ASTNODE_H

#include "../visitor/Visitor.h"

class ASTNode {

public:
	virtual void accept(Visitor &v) = 0;
};

#endif // SQL_ASTNODE_H
