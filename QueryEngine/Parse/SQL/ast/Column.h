#ifndef SQL_COLUMN_H
#define SQL_COLUMN_H

#include <cassert>
#include "ASTNode.h"

class Column : public ASTNode {

public:

	std::string s1;
	std::string s2;

	explicit Column(const std::string &s1) {
		assert(s1 != "");
		this->s1 = s1;
	}

	Column(const std::string &s1, const std::string &s2) {
		assert(s1 != "");
		assert(s2 != "");
		this->s1 = s1;
		this->s2 = s2;
	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_COLUMN_H
