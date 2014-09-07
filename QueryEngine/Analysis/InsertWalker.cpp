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
#include "../Parse/SQL/ast/InsertColumnList.h"
#include "../Parse/SQL/ast/LiteralList.h"
#include "../Parse/SQL/ast/Literal.h"  
 
namespace Analysis_Namespace {

void InsertWalker::visit(SqlStmt *v) {
    insertObj_.numRows = 0;
    if (v->n1) v->n1->accept(*this); // DmlStmt
}

void InsertWalker::visit(DmlStmt *v) {
	if (v->n1) v->n1->accept(*this); // InsertStmt
}

void InsertWalker::visit(InsertStmt *v) {
	assert(v->n1 && v->n2 && v->n3);

    insertObj_.tableId = v->n1->metadata.tableId;
    insertObj_.numRows++;
    
    // visit the columns
    v->n2->accept(*this); // InsertColumnList
    
    // visit the literals
    v->n3->accept(*this); // LiteralList
    
    // set variables for number of columns and values
    size_t numColumns = colNodes_.size();
    size_t numValues = literalTypes_.size();
    assert(numColumns == insertObj_.columnIds.size());
    
    // check that the number of column matches the number of literals
    // (note that this isn't caught by the parser)
    if (numColumns > numValues) {
        errFlag_ = true;
        errMsg_ = "not enough values listed";
        return;
    }
    else if (numValues > numColumns) {
        errFlag_ = true;
        errMsg_ = "not enough columns listed";
        return;
    }
    
    // type check
    for (int i = 0; i < numColumns; ++i) {
        if (colNodes_[i]->metadata.columnType == INT_TYPE && literalTypes_[i] == FLOAT_TYPE) {
            errFlag_ = true;
            errMsg_ = "Type mismatch at column '" + colNodes_[i]->name +"' (cannot downcast float to int)";
            return;
        }
        else if (colNodes_[i]->metadata.columnType == FLOAT_TYPE && literalTypes_[i] == INT_TYPE) {
            literalNodes_[i]->realData = (float)literalNodes_[i]->intData;
            literalNodes_[i]->type = FLOAT_TYPE;
        }
    }

    // package the data
    for (int i = 0; i < numValues; ++i) {
        mapd_byte_t *data = new mapd_byte_t[byteCount_];
        mapd_byte_t *pData = data;
        mapd_size_t currentByte = 0;

        printf("[%d] int=%d float=%f\n", i, literalNodes_[i]->intData, literalNodes_[i]->realData);
        
        // for each possible type, the bytes are copied one at a time into
        // the bytes of the mapd_byte_t data pointer
        if (literalNodes_[i]->type == INT_TYPE) {
            mapd_byte_t *tmp = (mapd_byte_t*)&literalNodes_[i]->intData;
            for (int j = 0; j < sizeof(int); ++j, ++tmp, ++pData)
                *pData = *tmp;
            // printf("%d == %d\n", *((int*)(data+currentByte)), literalNodes_[i]->intData);
            assert(*((int*)(data+currentByte)) == literalNodes_[i]->intData);
            currentByte += sizeof(int);
        }
        else if (literalNodes_[i]->type == FLOAT_TYPE) {
            mapd_byte_t *tmp = (mapd_byte_t*)&literalNodes_[i]->realData;
            for (int j = 0; j < sizeof(float); ++j, ++tmp, ++pData)
                *pData = *tmp;
            // printf("%f == %f\n", *((float*)(data+currentByte)), literalNodes_[i]->realData);
            assert(*((float*)(data+currentByte)) == literalNodes_[i]->realData);
            currentByte += sizeof(int);
        }

        // push the packaged data into insertObj_
        insertObj_.data.push_back((void*)data);
    }
    
    // insert the data via the TablePartitionMgr object
    tpm_->insertData(insertObj_);
}

void InsertWalker::visit(InsertColumnList *v) {
	if (v->n1) v->n1->accept(*this);
    assert(v->metadata.tableId == insertObj_.tableId);
    insertObj_.columnIds.push_back(v->metadata.columnId);
    colNodes_.push_back(v);
}

void InsertWalker::visit(LiteralList *v) {
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void InsertWalker::visit(Literal *v) {
    literalNodes_.push_back(v);
    literalTypes_.push_back(v->type);
    
    assert(sizeof(int) == sizeof(float));
    byteCount_ += sizeof(float); // @todo clearly, we will need to know the sizes of any type
}

} // Analysis_Namespace
