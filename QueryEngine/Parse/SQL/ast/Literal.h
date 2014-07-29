#ifndef SQL_LITERAL_H
#define SQL_LITERAL_H

#include <cassert>
#include "ASTNode.h"
#include "../../../../Shared/types.h"

class Literal : public ASTNode {

public:

	long int n1;
	double n2;
	std::string n3;
	mapd_data_t type;

	explicit Literal(long int n1) {
		this->n1 = n1;
		this->type = INT_TYPE;
	}

	explicit Literal(double n2) {
		this->n2 = n2;
		this->type = FLOAT_TYPE;
	}

	explicit Literal(const std::string &n3) {
		this->n3 = n3;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_LITERAL_H
