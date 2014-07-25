/**
 * @file	AbstractScalarExpr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef SQL_ABSTRACT_SCALR_EXPR_H
#define SQL_ABSTRACT_SCALR_EXPR_H

#include "ASTNode.h"
#include <cassert>

namespace SQL_Namespace {

enum ScalarExprType {
	SCALAR_INT,
	SCALAR_FLOAT,
	SCALAR_STRING
};

class AbstractScalarExpr : public ASTNode {

public:
	//virtual ~AbstractScalarExpr() = 0;

	virtual inline void setType(ScalarExprType t) {
		this->type = t;
	}

	virtual inline ScalarExprType getType() {
		return this->type;
	}

	virtual inline void setLineno(int lineno) {
		//assert(lineno > 0);
		this->lineno = lineno;
	}

	virtual inline int getLineno() {
		return this->lineno;
	}

	virtual inline void setColno(int colno) {
		//assert(colno > 0);
		this->colno = colno;
	}

	virtual inline int getColno() {
		return this->colno;
	}

private:
	
	ScalarExprType type;
	int lineno = 0;
	int colno = 0;
};

} // SQL_Namespace

#endif // SQL_ABSTRACT_SCALR_EXPR_H