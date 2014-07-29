#ifndef SQL_TABLE_H
#define SQL_TABLE_H

#include <cassert>
#include "ASTNode.h"

class Table : public ASTNode {

public:
	std::string name1;
	std::string name2;

	// table_id is obtained from Catalog during semantic analysis
	int table_id = -1;

	/// Constructor
	explicit Table(const std::string &name1) {
		assert(name1 != "");
		this->name1 = name1;
	}

	/// Constructor
	Table(const std::string &name1, const std::string &name2) {
		assert(name1 != "" && name2 != "");
		this->name1 = name1;
		this->name2 = name2;
	}

	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_TABLE_H
