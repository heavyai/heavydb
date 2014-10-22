/**
 * @file	Relation.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_RELATION_NODE_H
#define RA_RELATION_NODE_H

#include <string>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"
#include "../../../../DataMgr/Metadata/Catalog.h"

using Metadata_Namespace::TableRow;

namespace RA_Namespace {

class Relation : public RelAlgNode {
    
public:
	std::string name;
    TableRow metadata; // metadata obtained during SQL to RA translation

	/// Constructor
	explicit Relation(const std::string &name) {
		this->name = name;
	}
    
    /// Constructor -- metadata only
	explicit Relation(const TableRow &metadata) : metadata(metadata) {
        this->name = metadata.tableName + "(" + std::to_string(metadata.tableId) + ")";
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

} // RA_Namespace

#endif // RA_RELATION_NODE_H
