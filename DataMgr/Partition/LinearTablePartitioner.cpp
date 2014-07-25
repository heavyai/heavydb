#include "LinearTablePartitioner.h"

LinearTablePartitioner(const int tableId,  vector <ColumnInfo> &columnInfoVec, BufferManager &bufferManager, const mapd_size_t maxFragmentRows, const mapd_size_t pageSize /*default 1MB*/) :
		tableId_(tableId), bufferManager_(bufferManager), maxFragmentRows_(maxFragmentRows), pageSize_(pageSize) {
    // populate map with ColumnInfo structs
    for (vector <ColumnInfo>::const_iterator colIt = columnInfoVec.begin(); colIt != columnInfoVec.end(); ++colIt) {
        columnMap_[colIt -> columnId_] = *colIt; 
    }
    // need to query table here to see how many rows are in each partition
}

void LinearTablePartitioner::insertData (const vector <int> &columnIds, const vector <void *> &data, const int numRows) {
    if (currentInsertBufferSize_ + numRows > maxFragmentRows_) { // create new fragment - note that this as currently coded will leave empty tuplesspace at end of current buffer chunks in the case of an insert of multiple rows at a time 
        createNewFragment();
    }

    //for (vector <int>::const_iterator colIdIt = columnIds.begin(); colIdIt != columnIds.end(); ++colIdIt) {
    for (int c = 0; c != columnIds.size(); ++c) {
        int columnId = columnIds[c];
        map <int, ColumnInfo>::iterator colMapIt = columnMap_.find(columnId);
        // This SHOULD be found and this iterator should not be end()
        // as SemanticChecker should have already checked each column reference
        // for validity
        assert(colMapIt != columnMap_.end());
        colMapIt -> second.insertBuffer_.append(columnMap_ -> second.bitSize_ * numRows, data[c]);
    }
}

void LinearTablePartitioner::createNewFragment() { 
    // also sets the new fragment as the insertBuffer_ for each column

    // iterate through all ColumnInfo structs in map, unpin previous insert buffer and
    // create new insert buffer
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        colMapIt -> second.insertBuffer_.unpin();
        ChunkKey chunkKey = {tableId_, currentInsertBufferFragmentId_,  colMapIt -> second.columnId_};
        // We will allocate enough pages to hold the maximum number of rows of
        // this type
        mapd_size_t numPages = ceiling(static_cast <float> (maxFragmentRows_) / (pageSize_ * 8 / colMapIt -> second.bitSize_); // number of pages for a fragment is celing maximum number of rows for a fragment divided by how many elements of this column type can fit on a page 
        colMapIt -> second.insertBuffer_ = bufferManager_.createChunk(chunkKey, numPages , pageSize_);
    }
}










