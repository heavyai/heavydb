#ifndef SQL_COLUMN_H
#define SQL_COLUMN_H

#include <cassert>
#include <utility>
#include <string>
#include "ASTNode.h"
#include "../../../../Shared/types.h"

namespace SQL_Namespace {

class Column : public ASTNode {

public:
	std::pair<std::string, std::string> name;

	// id and type is obtained from the Catalog during semantic analysis
	int column_id = -1;
	mapd_data_t column_type;

	/// Constructor
	explicit Column(const std::string &s1) {
		assert(s1 != "");
		name.second = s1;
	}

	/// Constructor
	Column(const std::string &s1, const std::string &s2) {
		assert(s1 != "");
		assert(s2 != "");
		name.first = s1;
		name.second = s2;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_COLUMN_H
