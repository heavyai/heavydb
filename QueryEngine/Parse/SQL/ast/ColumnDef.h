#ifndef SQL_COLUMNDEF_H
#define SQL_COLUMNDEF_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class ColumnDef : public ASTNode {

public:

	Column *n1 = NULL;
	MapdDataT *n2 = NULL;
	std::string s;

	ColumnDef(Column *n1, MapdDataT *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	ColumnDef(Column *n1, MapdDataT *n2, const std::string &s) {
		assert(n1 && n2);
		assert(s == "PRIMARY KEY" || s == "NULL" || s == "NOT NULL");
		this->n1 = n1;
		this->n2 = n2;
		this->s = s;
	}

	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

}

#endif // SQL_COLUMN_H
