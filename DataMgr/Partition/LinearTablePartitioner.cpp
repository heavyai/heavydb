#include "LinearTablePartitioner.h"
#include "BufferMgr.h"
#include "Buffer.h"
#include <math.h>
#include <iostream>

#include <assert.h>
#include <boost/lexical_cast.hpp>

using Buffer_Namespace::Buffer;
using Buffer_Namespace::BufferMgr;

using namespace std;



LinearTablePartitioner::LinearTablePartitioner(const int partitionerId,  vector <ColumnInfo> &columnInfoVec, Buffer_Namespace::BufferMgr &bufferManager, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize /*default 1MB*/) :
		partitionerId_(partitionerId), bufferManager_(bufferManager), maxPartitionRows_(maxPartitionRows), pageSize_(pageSize), maxPartitionId_(-1), pgConnector_("mapd","mapd"), partitionerType_("linear"), isDirty_(false)/*, currentInsertBufferSize_(0) */ {
    // @todo Actually get user's name to feed to pgConnector_
    // populate map with ColumnInfo structs
    for (vector <ColumnInfo>::iterator colIt = columnInfoVec.begin(); colIt != columnInfoVec.end(); ++colIt) {
        columnMap_[colIt -> columnId] = *colIt; 
    }
    createStateTableIfDne();
    readState();

    // need to query table here to see how many rows are in each partition
    //createNewPartition();
    /*
    PartitionInfo fragInfo;
    fragInfo.partitionId = 0;
    fragInfo.numTuples = 0;
    partitionInfoVec_
    */

}

LinearTablePartitioner::~LinearTablePartitioner() {
    writeState();
}

//void LinearTablePartitioner::insertData (const vector <int> &columnIds, const vector <void *> &data, const int numRows) {
void LinearTablePartitioner::insertData (const InsertData &insertDataStruct) {
    if (maxPartitionId_ < 0 || partitionInfoVec_.back().numTuples + insertDataStruct.numRows > maxPartitionRows_) { // create new partition - note that this as currently coded will leave empty tuplesspace at end of current buffer chunks in the case of an insert of multiple rows at a time 
        // should we also do this if magPartitionId_ < 0 and allocate lazily?
        createNewPartition();
    }

    for (int c = 0; c != insertDataStruct.columnIds.size(); ++c) {
        int columnId =insertDataStruct.columnIds[c];
        map <int, ColumnInfo>::iterator colMapIt = columnMap_.find(columnId);
        // This SHOULD be found and this iterator should not be end()
        // as SemanticChecker should have already checked each column reference
        // for validity
        assert(colMapIt != columnMap_.end());
        //cout << "Insert buffer before insert: " << colMapIt -> second.insertBuffer << endl;
        //cout << "Insert buffer before insert length: " << colMapIt -> second.insertBuffer -> length() << endl;
        colMapIt -> second.insertBuffer -> append(colMapIt -> second.bitSize * insertDataStruct.numRows / 8, static_cast <mapd_addr_t> (insertDataStruct.data[c]));
    }
    //currentInsertBufferSize_ += numRows;
    partitionInfoVec_.back().numTuples += insertDataStruct.numRows;
    isDirty_ = true;
}



void LinearTablePartitioner::createNewPartition() { 
    // also sets the new partition as the insertBuffer for each column

    // iterate through all ColumnInfo structs in map, unpin previous insert buffer and
    // create new insert buffer
    maxPartitionId_++;
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        if (colMapIt -> second.insertBuffer != 0)
            colMapIt -> second.insertBuffer -> unpin();
        ChunkKey chunkKey = {partitionerId_, maxPartitionId_,  colMapIt -> second.columnId};
        // We will allocate enough pages to hold the maximum number of rows of
        // this type
        mapd_size_t numPages = ceil(static_cast <float> (maxPartitionRows_) / (pageSize_ * 8 / colMapIt -> second.bitSize)); // number of pages for a partition is celing maximum number of rows for a partition divided by how many elements of this column type can fit on a page 
        //cout << "NumPages: " << numPages << endl;
        colMapIt -> second.insertBuffer = bufferManager_.createChunk(chunkKey, numPages , pageSize_);
        //cout << "Insert buffer address after create: " << colMapIt -> second.insertBuffer << endl;
        //cout << "Length: " << colMapIt -> second.insertBuffer -> length () << endl;
        //if (!partitionInfoVec_.empty()) // if not first partition
        //    partitionInfoVec_.back().numTuples_ = currentInsertBufferSize_;
    }
    PartitionInfo newPartitionInfo;
    newPartitionInfo.partitionId = maxPartitionId_;
    newPartitionInfo.numTuples = 0; 
    partitionInfoVec_.push_back(newPartitionInfo);
}

void LinearTablePartitioner::getPartitionsForQuery(QueryInfo &queryInfo, const void *predicate) {
    queryInfo.partitionerId = partitionerId_;
    // right now we don't test predicate, so just return (copy of) all partitions 
    queryInfo.partitions = partitionInfoVec_;
    queryInfo.numTuples = 0;
    // now iterate over all partitions and add the 
    for (auto partIt = partitionInfoVec_.begin(); partIt != partitionInfoVec_.end(); ++partIt)
        queryInfo.numTuples += partIt -> numTuples;

}

void LinearTablePartitioner::createStateTableIfDne() {
     mapd_err_t status = pgConnector_.query("CREATE TABLE IF NOT EXISTS fragments(part_id INT, fragment_id INT, num_rows INT, PRIMARY KEY (part_id, fragment_id))");
     assert(status == MAPD_SUCCESS);
}


void LinearTablePartitioner::readState() {
    string partitionQuery ("select fragment_id, num_rows from fragments where part_id = " + boost::lexical_cast <string> (partitionerId_));
    partitionQuery += " order by fragment_id";
    mapd_err_t status = pgConnector_.query(partitionQuery);
    assert(status == MAPD_SUCCESS);
    size_t numRows = pgConnector_.getNumRows();
    partitionInfoVec_.resize(numRows);
    for (int r = 0; r < numRows; ++r) {
        partitionInfoVec_[r].partitionId = pgConnector_.getData<int>(r,0);  
        partitionInfoVec_[r].numTuples = pgConnector_.getData<int>(r,1);
    }
    if (numRows > 0) {
        maxPartitionId_ = partitionInfoVec_[numRows-1].partitionId; 
        getInsertBufferChunks();
    }
}

void LinearTablePartitioner::getInsertBufferChunks() {
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        if (colMapIt -> second.insertBuffer != 0)
            colMapIt -> second.insertBuffer -> unpin();
        ChunkKey chunkKey = {partitionerId_, maxPartitionId_,  colMapIt -> second.columnId};
        colMapIt -> second.insertBuffer = bufferManager_.getChunkBuffer(chunkKey);
        assert (colMapIt -> second.insertBuffer != NULL);
        //@todo change assert into throwing an exception
    }
}


void LinearTablePartitioner::writeState() {
    // do we want this to be fully durable or will allow ourselves
    // to delete existing rows for this table
    // out of convenience before adding the
    // newest state back in?
    // Will do latter for now as we do not 
    // plan on using postgres forever for metadata
    if (isDirty_) {
         string deleteQuery ("delete from fragments where partitioner_id = " + boost::lexical_cast <string> (partitionerId_));
         mapd_err_t status = pgConnector_.query(deleteQuery);
         assert(status == MAPD_SUCCESS);
        for (auto partIt = partitionInfoVec_.begin(); partIt != partitionInfoVec_.end(); ++partIt) {
            string insertQuery("INSERT INTO fragments (partitioner_id, fragment_id, num_rows) VALUES (" + boost::lexical_cast<string>(partitionerId_) + "," + boost::lexical_cast<string>(partIt -> partitionId) + "," + boost::lexical_cast<string>(partIt -> numTuples) + ")"); 
            status = pgConnector_.query(insertQuery);
             assert(status == MAPD_SUCCESS);
        }
    }
    isDirty_ = false;
}
