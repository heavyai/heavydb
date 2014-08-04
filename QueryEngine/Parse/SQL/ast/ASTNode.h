/**
 * @file	ASTNode.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_ASTNODE_H
#define SQL_ASTNODE_H

#include <iostream>
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class ASTNode {

public:
	virtual void accept(SQL_Namespace::Visitor &v) = 0;
};

} // SQL_Namespace

#endif // SQL_ASTNODE_H
