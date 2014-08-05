/**
 * @file	InsertWalker.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Steps through insert statements in the SQL AST.
 *
 */
#include <sstream>
#include "InsertWalker.h"
#include "../../Shared/types.h"
#include "../Parse/SQL/ast/SqlStmt.h"
#include "../Parse/SQL/ast/DmlStmt.h"
#include "../Parse/SQL/ast/InsertStmt.h"
#include "../Parse/SQL/ast/Table.h"
#include "../Parse/SQL/ast/ColumnList.h"
#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/LiteralList.h"
#include "../Parse/SQL/ast/Literal.h"  
 
namespace Analysis_Namespace {

void InsertWalker::visit(SqlStmt *v) {
	if (v->n1) v->n1->accept(*this); // DmlStmt
}

void InsertWalker::visit(DmlStmt *v) {
	if (v->n1) v->n1->accept(*this); // InsertStmt
}

void InsertWalker::visit(InsertStmt *v) {
	assert(v->n1 && v->n2 && v->n3);
	if (v->n1->name1 != "") {
		//printf("table = %s\n", v->n1->name1.c_str());
		
		// visiting v->n2 will populate the colNames_ vector
		v->n2->accept(*this);
		/*for (auto it = colNames_.begin(); it != colNames_.end(); ++it) {
			printf("column = %s\n", (*it).c_str());
		}*/
		
		// Request metadata for the columns
		std::vector<ColumnRow> colMetadata;
		mapd_err_t err = c_.getMetadataForColumns(v->n1->name1, colNames_, colMetadata);

		// Check for error (table or column does not exists)
		if (err != MAPD_SUCCESS) {
			errFlag_ = true;
			if (err == MAPD_ERR_TABLE_DOES_NOT_EXIST)
				errMsg_ = "Table \"" + v->n1->name1 + "\" does not exist";
			else if (err == MAPD_ERR_COLUMN_DOES_NOT_EXIST)
				errMsg_ = "Column \"" + colNames_[colMetadata.size()] + "\" does not exist";
			else if (err != MAPD_SUCCESS)
				errMsg_ = "Catalog returned an error.";
			
			colNames_.clear();
			colMetadata.clear();
			return;
		}
		
		// Otherwise, check that the values match the column types
		v->n3->accept(*this);
		
		// Check that the values in the insert statement are the right type
		if (colTypes_.size() == colNames_.size()) {
			for (int i = 0; i < colTypes_.size(); ++i) {
				//printf("colMetadata[i].columnName=\"%s\" colNames_[i]=\"%s\" colMetadata[i].columnType=%d\n", colMetadata[i].columnName.c_str(), colNames_[i].c_str(), colMetadata[i].columnType);
				if (colMetadata[i].columnType == INT_TYPE && colTypes_[i] == FLOAT_TYPE) {
					std::stringstream ss;
					ss << "Type mismatch at column \"" << colMetadata[i].columnName << "\"";
					errFlag_ = true;
					errMsg_ = ss.str();
					break;
				}
			}
		}
		else {
			errFlag_ = true;
			errMsg_ = "Column count does not match value count.";				
		}
	}
}

void InsertWalker::visit(ColumnList *v) {
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void InsertWalker::visit(Column *v) {
	std::string colName;
	if (v->s2 != "")
		colName = v->s2 + "." + v->s1;
	else
		colName = v->s1;
	colNames_.push_back(colName);	
}

void InsertWalker::visit(LiteralList *v) {
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void InsertWalker::visit(Literal *v) {
	this->colTypes_.push_back(v->type);
}

} // Analysis_Namespace
