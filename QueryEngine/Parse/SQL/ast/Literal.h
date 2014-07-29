#ifndef SQL_LITERAL_H
#define SQL_LITERAL_H

#include <cassert>
#include "ASTNode.h"

class Literal : public ASTNode {

public:

	long int n1;
	double n2;
	std::string n3;

	explicit Literal(long int n1) {
		this->n1 = n1;
	}

	explicit Literal(double n2) {
		this->n2 = n2;
	}

	explicit Literal(const std::string &n3) {
		this->n3 = n3;
	}

	~Literal() {

	}
	
	virtual void accept(Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_LITERAL_H
