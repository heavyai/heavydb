/**
 * @file	ASTNode.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_ASTNODE_H
#define SQL_ASTNODE_H

#include "../visitor/Visitor.h"
#include "../translator/SQL_RA_Translator.h"

namespace SQL_Namespace {

class ASTNode {

public:
	virtual void accept(Visitor &v) = 0;
	virtual void accept(SQL_RA_Translator &v) = 0;
};

} // SQL_Namespace

#endif // SQL_ASTNODE_H
