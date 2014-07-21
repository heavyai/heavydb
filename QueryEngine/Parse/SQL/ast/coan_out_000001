/**
 * @file	AbstractScalarExpr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef SQL_ABSTRACT_SCALR_EXPR_H
#define SQL_ABSTRACT_SCALR_EXPR_H

namespace SQL_Namespace {

enum ScalarExprType {
	SCALAR_INT,
	SCALAR_FLOAT,
	SCALAR_STRING
}

class AbstractScalarExpr {

public:
	virtual inline void type(ScalarExprType t) {
		this->type = t;
	}

	virtual inline ScalarExprType type() {
		return this->type;
	}

private:
	virtual ~AbstractScalarExpr() = 0;
	ScalarExprType type;

};

} // SQL_Namespace