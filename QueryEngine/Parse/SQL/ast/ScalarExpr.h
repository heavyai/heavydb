/**
 * @file	ScalarExpr.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef QUERYENGINE_PARSER_SQL_AST_SCALAREXPR_H
#define QUERYENGINE_PARSER_SQL_AST_SCALAREXPR_H

#include <cassert>
#include "Expression.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class ScalarExpr : public Expression {

public:
	ScalarExpr *n1 = nullptr;
	ScalarExpr *n2 = nullptr;
	Literal *n3 = nullptr;
	Column *n4 = nullptr;
    AggrExpr *n5 = nullptr;
	std::string op = "";

	ScalarExpr(std::string op, ScalarExpr *n1, ScalarExpr *n2) {
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

	explicit ScalarExpr(ScalarExpr *n1) {
		assert(n1);
		this->n1 = n1;
	}

	explicit ScalarExpr(Literal *n3) {
		assert(n3);
		this->n3 = n3;
	}

	explicit ScalarExpr(Column *n4) {
		assert(n4);
		this->n4 = n4;
	}

    explicit ScalarExpr(AggrExpr *n5) {
        assert(n5);
        this->n5 = n5;
    }
    
	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // QUERYENGINE_PARSER_SQL_AST_SCALAREXPR_H
