/**
 * @file	AbstractScalarExpr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef SQL_ABSTRACT_SCALR_EXPR_H
#define SQL_ABSTRACT_SCALR_EXPR_H

#include <cassert>

namespace SQL_Namespace {

enum ScalarExprType {
	SCALAR_INT,
	SCALAR_FLOAT,
	SCALAR_STRING
}

class AbstractScalarExpr : public ASTNode {

public:
	virtual inline void type(ScalarExprType t) {
		this->type = t;
	}

	virtual inline ScalarExprType type() {
		return this->type;
	}

	virtual inline void lineno(int lineno) {
		assert(lineno > 0);
		this->lineno = lineno;
	}

	virtual inline int lineno() {
		return this->lineno;
	}

	virtual inline void colno(int colno) {
		assert(colno > 0);
		this->colno = colno;
	}

	virtual inline int colno() {
		return this->colno;
	}

private:
	virtual ~AbstractScalarExpr() = 0;
	ScalarExprType type;
	int lineno = 0;
	int colno = 0;
};

} // SQL_Namespace