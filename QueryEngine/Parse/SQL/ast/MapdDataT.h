#ifndef SQL_MAPDDATAT_H
#define SQL_MAPDDATAT_H

#include <cassert>
#include "ASTNode.h"

#include "../../../../Shared/types.h"

class MapdDataT : public ASTNode {

public:

	mapd_data_t type;

	explicit MapdDataT(int type) {
		if (type == 0)
			this->type = INT_TYPE;
		else if (type == 1)
			this->type = FLOAT_TYPE;
		else if (type == 2)
			this->type == BOOLEAN_TYPE;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_MAPDDATAT_H
