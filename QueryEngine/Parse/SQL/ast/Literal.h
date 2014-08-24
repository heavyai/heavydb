#ifndef SQL_LITERAL_H
#define SQL_LITERAL_H

#include <cassert>
#include "ASTNode.h"
#include "../../../../Shared/types.h"

namespace SQL_Namespace {

class Literal : public ASTNode {

public:

    int intData;
    float realData;
    std::string strData;
	mapd_data_t type;

    Literal(float data, mapd_data_t type) {
		this->type = type;
        if (type == INT_TYPE)
            this->intData = (int)data;
        else if (type == FLOAT_TYPE)
            this->realData = data;
        else
            assert(NULL); // unsupported type
        // printf("intData=%d floatData=%f\n", intData, realData);
	}

	explicit Literal(const std::string &strData) {
        this->strData = strData;
        assert(NULL); // string data is not supported yet
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_LITERAL_H
