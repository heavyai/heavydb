#ifndef QUERYENGINE_PARSE_SQL_AST_INSERTCOLUMNLIST_H
#define QUERYENGINE_PARSE_SQL_AST_INSERTCOLUMNLIST_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class InsertColumnList : public ASTNode {

public:

	InsertColumnList *n1 = NULL;
	std::string name = "";

	InsertColumnList(InsertColumnList *n1, const std::string name) {
		assert(n1 && name != "");
		this->n1 = n1;
		this->name = name;
	}

	explicit InsertColumnList(const std::string name) {
		assert(name != "");
		this->name = name;
	}

	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // QUERYENGINE_PARSE_SQL_AST_INSERTCOLUMNLIST_H
