#ifndef SQL_TABLE_H
#define SQL_TABLE_H

#include <cassert>
#include "ASTNode.h"

class Table : public ASTNode {

public:
	std::string name1;
	std::string name2;

	explicit Table(const std::string &name1) {
		assert(name1 != "");
		this->name1 = name1;
	}

	Table(const std::string &name1, const std::string &name2) {
		assert(name1 != "" && name2 != "");
		this->name1 = name1;
		this->name2 = name2;
	}

	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_TABLE_H
