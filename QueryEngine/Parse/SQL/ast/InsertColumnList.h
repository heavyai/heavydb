#ifndef QUERYENGINE_PARSE_SQL_AST_INSERTCOLUMNLIST_H
#define QUERYENGINE_PARSE_SQL_AST_INSERTCOLUMNLIST_H

#include <cassert>
#include "ASTNode.h"
#include "../../../../DataMgr/Metadata/Catalog.h"

using Metadata_Namespace::ColumnRow;

namespace SQL_Namespace {

/**
 * @class InsertColumnList
 *
 * This AST node either represents a column name or a list of column names. If name
 * is set to "" (which it is, by default), then this node represents a list; otherwise,
 * it represents a column name. The node can be further annotated by setting the
 * column id (columnId) to a nonnegative integer. The column id can be obtained during
 * a tree traversal by requesting it from the Catalog (@see NameWalker).
 *
 * (Note that the "Column" AST node was not used because, in the grammar for an insert
 * statement, NAME '.' NAME is not permitted.)
 */
class InsertColumnList : public ASTNode {

public:

	InsertColumnList *n1 = NULL;
	std::string name = "";
    ColumnRow metadata;

	InsertColumnList(InsertColumnList *n1, const std::string name) : metadata(name) {
		assert(n1 && name != "");
		this->n1 = n1;
		this->name = name;
	}

	explicit InsertColumnList(const std::string name) : metadata(name) {
		assert(name != "");
		this->name = name;
	}

	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // QUERYENGINE_PARSE_SQL_AST_INSERTCOLUMNLIST_H
