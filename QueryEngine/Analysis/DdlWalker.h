/**
 * @file	DdlWalker.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef QUERYENGINE_ANALYSIS_DDLWALKER_H
#define QUERYENGINE_ANALYSIS_DDLWALKER_H

#include <iostream>
#include "../../Shared/types.h"
#include "../Parse/SQL/visitor/Visitor.h"
#include "../../DataMgr/Metadata/Catalog.h"
#include "../../DataMgr/Partitioner/TablePartitionMgr.h"
#include "../../DataMgr/Partitioner/Partitioner.h"

using namespace SQL_Namespace;
using namespace Metadata_Namespace;
using namespace Partitioner_Namespace;

namespace Analysis_Namespace {

/**
 * @class 	DdlWalker
 * @brief	This class is a visitor/executor for DDL statements in an SQL tree.
 */
class DdlWalker : public Visitor {

public:
	/// Constructor
	DdlWalker(Catalog *c, TablePartitionMgr *tpm) : c_(c), tpm_(tpm), errFlag_(false) {}

	/// Returns an error message if an error was encountered
	inline std::pair<bool, std::string> isError() { return std::pair<bool, std::string>(errFlag_, errMsg_); }

	/// @brief Visit an ColumnDefList node
	virtual void visit(Column *v);

	/// @brief Visit an ColumnDefList node
	virtual void visit(ColumnDef *v);

	/// @brief Visit an ColumnDefList node
	virtual void visit(ColumnDefList *v);

	/// @brief Visit an CreateStmt node
	virtual void visit(CreateStmt *v);

	/// @brief Visit an DdlStmt node
	virtual void visit(DdlStmt *v);

	/// @brief Visit an DropStmt node
	virtual void visit(DropStmt *v);

	/// @brief Visit an MapdDataT node
	virtual void visit(MapdDataT *v);

	/// @brief Visit an SqlStmt node
	virtual void visit(SqlStmt *v);

	/// @brief Visit an Table node
	virtual void visit(Table *v);

private:
	Catalog *c_;                /// a reference to a Catalog, which holds table/column metadata
    TablePartitionMgr *tpm_;    /// a reference to a TablePartitionMgr object
    
	std::string errMsg_;	/// holds an error message, if applicable; otherwise, it is ""
	bool errFlag_ = false;	/// indicates the existence of an error when true

	std::string tblName_; /// the name of a table
	std::vector<std::pair<std::string, std::string>> colNames_; /// the names of columns in a table
	std::vector<mapd_data_t> colTypes_; /// the types of columns in a table
};

} // Analysis_Namespace

#endif // QUERYENGINE_ANALYSIS_DDLWALKER_H
