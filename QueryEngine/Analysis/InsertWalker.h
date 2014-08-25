/**
 * @file	InsertWalker.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file contains the class specification for InsertWalker, which belongs
 * to the Analysis_Namespace.
 */
#ifndef QUERYENGINE_ANALYSIS_INSERTWALKER_H
#define QUERYENGINE_ANALYSIS_INSERTWALKER_H

#include <vector>
#include <string>
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
 * @class InsertWalker
 * @brief Parses and type-checks INSERT statements.
 *
 * The InsertWalker will traverse the SQL AST in order to parse statements of
 * the following form:
 *
 * INSERT INTO table (column1 [, column2, column3 ... ]) VALUES (value1 [, value2, value3 ... ])
 *
 * It verifies the existence of the table and column names, and it verifies that
 * the specified values are of the correct type for the corresponding column. If
 * not, then a local member called "errFlag_" is set to true, and "errMsg_" will
 * contain an appropriate error message. These members are accessible to the
 * client via the isError() method, which are returned as a pair<bool, string>.
 *
 * Example usage given an SQL AST node "parseRoot" and a Catalog "c":
 *
 *		InsertWalker iw(c);
 *		if (parseRoot != 0) {
 *          parseRoot->accept(iw); 
 *          std::pair<bool, std::string> insertErr = iw.isError();
 *          if (insertErr.first == true) {
 *              cout << "Error: " << insertErr.second << std::endl;
 *          }
 *		}
 *
 */
class InsertWalker : public Visitor {

public:
	/// Constructor
	InsertWalker(Catalog *c, TablePartitionMgr *tpm) : c_(c), tpm_(tpm), errFlag_(false) {}

	/// Returns an error message if an error was encountered
	inline std::pair<bool, std::string> isError() { return std::pair<bool, std::string>(errFlag_, errMsg_); }

	/// @brief Visit an SqlStmt node
	virtual void visit(SqlStmt *v);

	/// @brief Visit an DmlStmt node
	virtual void visit(DmlStmt *v);

	/// @brief Visit an InsertStmt node
	virtual void visit(InsertStmt *v);

	/// @brief Visit an ColumnList node
	virtual void visit(InsertColumnList *v);

	/// @brief Visit an LiteralList node
	virtual void visit(LiteralList *v);

	/// @brief Visit an Literal node
	virtual void visit(Literal *v);

private:
	Catalog *c_;                    /// a reference to a Catalog, which holds table/column metadata
    TablePartitionMgr *tpm_;        /// a reference to a TablePartitionMgr object
    std::vector<InsertColumnList*> colNodes_;
    std::vector<Literal*> literalNodes_;
    std::vector<mapd_data_t> literalTypes_;
    InsertData insertObj_;          /// represents the insert statement in a struct
    size_t byteCount_ = 0;          /// counts the number of bytes to allocate for the data being inserted
    
	std::string errMsg_;			/// holds an error message, if applicable; otherwise, it is ""
	bool errFlag_ = false;			/// indicates the existence of an error when true
};

} // Analysis_Namespace

#endif // QUERYENGINE_ANALYSIS_INSERTWALKER_H
