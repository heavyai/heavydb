#ifndef SQL_COLUMN_H
#define SQL_COLUMN_H

#include <cassert>
#include "ASTNode.h"
#include "../../../../Shared/types.h"

namespace SQL_Namespace {

class Column : public ASTNode {

public:
	std::string s1 = "";
	std::string s2 = "";

	// id and type is obtained from the Catalog during semantic analysis
	int column_id = -1;
	mapd_data_t column_type;

	/// Constructor
	explicit Column(const std::string &s1) {
		assert(s1 != "");
		this->s1 = s1;
	}

	/// Constructor
	Column(const std::string &s1, const std::string &s2) {
		assert(s1 != "");
		assert(s2 != "");
		this->s1 = s1;
		this->s2 = s2;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

    virtual void accept(class SQL_RA_Translator &v) {
        v.visit(this);
    }

};

} // SQL_Namespace

#endif // SQL_COLUMN_H
