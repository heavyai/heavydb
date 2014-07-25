#include "LinearTablePartitioner.h"

LinearTablePartitioner(const int tableId, const mapd_size_t maxFragmentSize, const vector <ColumnInfo> &columnInfoVec, BufferManager &bufferManager) : 
		tableId_(tableId), maxFragmentSize_(maxFragmentSize), bufferManager_(bufferManager) {
    // populate map with ColumnInfo structs
    for (vector <ColumnInfo>::const_iterator colIt = columnInfoVec.begin(); colIt != columnInfoVec.end(); ++colIt) {
        columnMap_[colIt -> columnId_] = *colIt; 
    }
}

LinearTablePartitioner::insertData (const vector <int> &columnIds, const vector <void *> &data, const int numRows) {
    if (currentInsertBufferSize_ + numRows > maxFragmentSize_) { // create new fragment - note that this as currently coded will leave empty tuplesspace at end of current buffer chunks in the case of an insert of multiple rows at a time 
        createNewFragment();
    }



    for (vector <int>::const_iterator colIdIt = columnIds.begin(); colIdIt != columnIds.end(); ++colIdIt) {
        map <int, ColumnInfo>::iterator colMapIt = columnMap_.find(*colIdIt);
        // This SHOULD be found and this iterator should not be end()
        // as SemanticChecker should have already checked each column reference
        // for validity
        assert(colMapIt != columnMap_.end());

        



        




    }
    


    



}


