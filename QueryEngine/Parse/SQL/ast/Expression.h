/**
 * @file	Expression.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_EXPRESSION_NODE_H
#define SQL_EXPRESSION_NODE_H

#include "../../../../Shared/types.h"
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class Expression : public ASTNode {

public:
	/// All "Expression" nodes have a type (mapd_data_t)
	mapd_data_t type;

	virtual void accept(Visitor &v) = 0;
	virtual void accept(SQL_RA_Translator &v) = 0;
};

} // SQL_Namespace

#endif // SQL_EXPRESSION_NODE_H
