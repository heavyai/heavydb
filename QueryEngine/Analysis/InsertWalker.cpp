/**
 * @file	InsertWalker.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Steps through insert statements in the SQL AST.
 *
 */
#include "../../DataMgr/Metadata/Catalog.h"
#include "InsertWalker.h"
#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/ColumnCommalist.h"
#include "../Parse/SQL/ast/InsertStatement.h"  
#include "../Parse/SQL/ast/ManipulativeStatement.h" 
#include "../Parse/SQL/ast/OptColumnCommalist.h"
#include "../Parse/SQL/ast/Program.h"
#include "../Parse/SQL/ast/SQLList.h"
#include "../Parse/SQL/ast/SQL.h" 
#include "../Parse/SQL/ast/Table.h" 
 
namespace Analysis_Namespace {

void InsertWalker::visit(Program *v) {
	if (v->sqlList) v->sqlList->accept(*this);
}

void InsertWalker::visit(SQLList *v) {
	// only visits next insert stmt if no error messge has already been set
	if (errMsg_ == "") {
		if (v->sql) v->sql->accept(*this);
		if (v->sqlList) v->sqlList->accept(*this);
	}
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

		// Ask catalog to check column names
		std::vector<ColumnRow> colMetadata;
		mapd_err_t err = c_.getMetadataForColumns(v->tbl->name1, colNames_, colMetadata);

		if (err != MAPD_SUCCESS) {
			if (err == MAPD_ERR_TABLE_DOES_NOT_EXIST)
				errMsg_ = "Table \"" + v->tbl->name1 + "\" does not exist";
			else if (err == MAPD_ERR_COLUMN_DOES_NOT_EXIST)
				errMsg_ = "Column \"" + colNames_[colMetadata.size()] + "\" does not exist";
			else if (err != MAPD_SUCCESS)
				errMsg_ = "Catalog returned an error.";
			printf("Error [%d]: %s\n", err, errMsg_.c_str());
		}
		
		colNames_.clear();
		colMetadata.clear();

		//@todo number of columns should equal number of values
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


} // Analysis_Namespace
