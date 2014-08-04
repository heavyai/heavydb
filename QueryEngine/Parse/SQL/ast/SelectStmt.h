#ifndef SQL_SELECTSTMT_H
#define SQL_SELECTSTMT_H

#include <cassert>
#include "Statement.h"

namespace SQL_Namespace {

class SelectStmt : public Statement {

public:

	OptAllDistinct* n1 = NULL;
	Selection *n2 = NULL;
	FromClause *n3 = NULL;
	OptWhere *n4 = NULL;
	OptGroupby *n5 = NULL;
	OptHaving *n6 = NULL;
	OptOrderby *n7 = NULL;
	OptLimit *n8 = NULL;

	explicit SelectStmt(OptAllDistinct *n1, Selection *n2, FromClause *n3, OptWhere *n4,
		OptGroupby *n5, OptHaving *n6, OptOrderby *n7, OptLimit *n8)
	{
		assert(n2 && n3);
		this->n1 = n1;
		this->n2 = n2;
		this->n3 = n3;
		this->n4 = n4;
		this->n5 = n5;
		this->n6 = n6;
		this->n7 = n7;
		this->n8 = n8;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_SELECTSTMT_H

