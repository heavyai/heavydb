#ifndef SQL_TABLE_H
#define SQL_TABLE_H

#include <cassert>
#include <utility>
#include <string>
#include "ASTNode.h"
#include "../../../../DataMgr/Metadata/Catalog.h"

using Metadata_Namespace::TableRow;

namespace SQL_Namespace {

class Table : public ASTNode {

public:
	std::pair<std::string, std::string> name;

	// table_id is obtained from Catalog during semantic analysis
	//int table_id = -1;
    TableRow metadata;

	/// Constructor
	explicit Table(const std::string &name) {
		assert(name != "");
		this->name.first = "";
		this->name.second = name;
	}

	/// Constructor
	Table(const std::string &name1, const std::string &name2) {
		assert(name1 != "" && name2 != "");
		this->name.first = name1;
		this->name.second = name2;
	}

	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_TABLE_H
