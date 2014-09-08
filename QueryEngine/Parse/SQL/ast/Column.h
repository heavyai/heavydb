#ifndef SQL_COLUMN_H
#define SQL_COLUMN_H

#include <cassert>
#include <utility>
#include <string>
#include "ASTNode.h"
#include "../../../../Shared/types.h"
#include "../../../../DataMgr/Metadata/Catalog.h"

using Metadata_Namespace::ColumnRow;

namespace SQL_Namespace {

class Column : public ASTNode {

public:
	std::pair<std::string, std::string> name;
	mapd_data_t column_type;
    ColumnRow metadata;

	/// Constructor
	explicit Column(const std::string &s1) : metadata(s1) {
		assert(s1 != "");
		name.second = s1;
	}

	/// Constructor
	Column(const std::string &s1, const std::string &s2) : metadata(s2) {
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
