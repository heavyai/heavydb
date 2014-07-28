/**
 * @file	InsertWalker.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Steps through insert statements in the SQL AST.
 *
 */
#include <sstream>
#include "../../DataMgr/Metadata/Catalog.h"
#include "InsertWalker.h"
#include "../../Shared/types.h"
#include "../Parse/SQL/ast/Atom.h"
#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/ColumnCommalist.h"
#include "../Parse/SQL/ast/InsertAtom.h"
#include "../Parse/SQL/ast/InsertAtomCommalist.h"
#include "../Parse/SQL/ast/InsertStatement.h"  
#include "../Parse/SQL/ast/ManipulativeStatement.h" 
#include "../Parse/SQL/ast/OptColumnCommalist.h"
#include "../Parse/SQL/ast/Program.h"
#include "../Parse/SQL/ast/SQLList.h"
#include "../Parse/SQL/ast/SQL.h" 
#include "../Parse/SQL/ast/Table.h" 
#include "../Parse/SQL/ast/ValuesOrQuerySpec.h"  
#include "../Parse/SQL/ast/Literal.h"
 
namespace Analysis_Namespace {

void InsertWalker::visit(Program *v) {
	if (v->sqlList) v->sqlList->accept(*this);
}

void InsertWalker::visit(SQLList *v) {
	// only visits next insert stmt if no error messge has already been set
	if (!errFlag_ && v->sqlList)
		v->sqlList->accept(*this);
	if (!errFlag_ && v->sql)
		v->sql->accept(*this);
}

void InsertWalker::visit(SQL *v) {
	if (v->manSta) v->manSta->accept(*this);
}

void InsertWalker::visit(ManipulativeStatement *v) {
	if (v->inSta) v->inSta->accept(*this);
}

void InsertWalker::visit(InsertStatement *v) {
	if (v->tbl->name1 != "") {
		//printf("table = %s\n", v->tbl->name1.c_str());
		
		// visiting v->oCC will populate the colNames_ vector
		v->oCC->accept(*this);
		/*for (auto it = colNames_.begin(); it != colNames_.end(); ++it) {
			printf("column = %s\n", (*it).c_str());
		}*/

		// Request metadata for the columns
		std::vector<ColumnRow> colMetadata;
		mapd_err_t err = c_.getMetadataForColumns(v->tbl->name1, colNames_, colMetadata);

		if (err != MAPD_SUCCESS) {
			errFlag_ = true;
			if (err == MAPD_ERR_TABLE_DOES_NOT_EXIST)
				errMsg_ = "Table \"" + v->tbl->name1 + "\" does not exist";
			else if (err == MAPD_ERR_COLUMN_DOES_NOT_EXIST)
				errMsg_ = "Column \"" + colNames_[colMetadata.size()] + "\" does not exist";
			else if (err != MAPD_SUCCESS)
				errMsg_ = "Catalog returned an error.";
			printf("Error [%d]: %s\n", err, errMsg_.c_str());
		}
		else {

			// Obtain the types of each value for each column by visiting the child node
			v->voQS->accept(*this);

			// Check that the values in the insert statement are the right type
			if (colTypes_.size() == colNames_.size()) {
				for (int i = 0; i < colTypes_.size(); ++i) {
					// printf("colMetadata[i].columnName=\"%s\" colNames_[i]=\"%s\"\n", colMetadata[i].columnName.c_str(), colNames_[i].c_str());
					if (colMetadata[i].columnType != colTypes_[i]) {
						std::stringstream ss;
						ss << "Type mismatch at column \"" << colMetadata[i].columnName << "\"";
						errFlag_ = true;
						errMsg_ = ss.str();
						printf("%s\n", errMsg_.c_str());
						break;
					}
				}
			}
			else {
				errFlag_ = true;
				errMsg_ = "No values specified for indicated columns.";				
			}
		}

		colNames_.clear();
		colMetadata.clear();
	}
}

void InsertWalker::visit(OptColumnCommalist *v) {
	if (v->cc) v->cc->accept(*this);
}

void InsertWalker::visit(ColumnCommalist *v) {
	if (v->colCom) v->colCom->accept(*this);
	if (v->col) v->col->accept(*this);
}

void InsertWalker::visit(Column *v) {
	colNames_.push_back(v->name1);
}

void InsertWalker::visit(ValuesOrQuerySpec *v) {
	if (v->iac) v->iac->accept(*this);
}

void InsertWalker::visit(InsertAtomCommalist *v) {
	if (v->iac) v->iac->accept(*this);
	if (v->ia) v->ia->accept(*this);
}

void InsertWalker::visit(InsertAtom *v) {
	if (v->a) v->a->accept(*this);
}

void InsertWalker::visit(Atom *v) {
	if (v->lit) v->lit->accept(*this);
}

void InsertWalker::visit(Literal *v) {
	this->colTypes_.push_back(v->type);
}

} // Analysis_Namespace
