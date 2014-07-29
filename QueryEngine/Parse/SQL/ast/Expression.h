/**
 * @file	Expression.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_EXPRESSION_NODE_H
#define SQL_EXPRESSION_NODE_H

#include "../../../../Shared/types.h"
#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Expression : public ASTNode {

public:
	/// All "Expression" nodes have a type (mapd_data_t)
	mapd_data_t type;

	virtual void accept(Visitor &v) = 0;
};

#endif // SQL_EXPRESSION_NODE_H
