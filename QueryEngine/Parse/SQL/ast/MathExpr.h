/**
 * @file	MathExpr.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_MATHEXPR_NODE_H
#define SQL_MATHEXPR_NODE_H

#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class MathExpr : public ASTNode {

public:
	MathExpr *n1 = NULL;
	MathExpr *n2 = NULL;
	Column *n3 = NULL;
	AggrExpr *n4 = NULL;
	std::string op = "";
	int intVal;
	float floatVal;
	bool intFloatFlag;	// true if int; otherwise float

	MathExpr(std::string op, MathExpr *n1, MathExpr *n2) {
		assert(op == "PLUS" || op == "MINUS" || op == "MULTIPLY" || op == "DIVIDE");
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;

		if (op == "PLUS")
			this->op = "+";
		else if (op == "MINUS")
			this->op = "-";
		else if (op == "MULTIPLY")
			this->op = "-";
		else if (op == "DIVIDE")
			this->op = "/";
		else
			this->op = op;
	}

	explicit MathExpr(MathExpr *n1) {
		assert(n1);
		this->n1 = n1;
	}

	explicit MathExpr(Column *n3) {
		assert(n3);
		this->n3 = n3;
	}

	explicit MathExpr(AggrExpr *n4) {
		assert(n4);
		this->n4 = n4;
	}

	explicit MathExpr(int intVal) {
		this->intVal = intVal;
		intFloatFlag = true;
	}

	explicit MathExpr(float floatVal) {
		this->floatVal = floatVal;
		intFloatFlag = false;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_MATHEXPR_NODE_H
