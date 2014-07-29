#ifndef SQL_SELECTSTMT_H
#define SQL_SELECTSTMT_H

#include <cassert>
#include "Statement.h"

class SelectStmt : public Statement {

public:

	OptAllDistinct* n1 = NULL;
	FromClause *n2 = NULL;
	OptWhere *n3 = NULL;
	OptGroupby *n4 = NULL;
	OptHaving *n5 = NULL;
	OptOrderby *n6 = NULL;
	OptLimit *n7 = NULL;

	explicit SelectStmt(OptAllDistinct *n1, FromClause *n2, OptWhere *n3,
		OptGroupby *n4, OptHaving *n5, OptOrderby *n6, OptLimit *n7)
	{
		assert(n2);
		this->n1 = n1;
		this->n2 = n2;
		this->n3 = n3;
		this->n4 = n4;
		this->n5 = n5;
		this->n6 = n6;
		this->n7 = n7;
	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_SELECTSTMT_H

