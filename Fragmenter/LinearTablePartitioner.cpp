#include "LinearTablePartitioner.h"
#include "AbstractDataMgr.h"
#include "AbstractBuffer.h"
#include <math.h>
#include <iostream>

#include <assert.h>
#include <boost/lexical_cast.hpp>

using Memory_Namespace::AbstractBuffer;
using Memory_Namespace::AbstractDataMgr;

using namespace std;

namespace Partitioner_Namespace {

LinearTablePartitioner::LinearTablePartitioner(const int partitionerId,  vector <ColumnInfo> &columnInfoVec, Memory_Namespace::AbstractDataMgr &bufferManager, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize /*default 1MB*/) :
		partitionerId_(partitionerId), bufferManager_(bufferManager), maxPartitionRows_(maxPartitionRows), pageSize_(pageSize), maxPartitionId_(-1), pgConnector_("mapd","mapd"), partitionerType_("linear"), isDirty_(false)/*, currentInsertBufferSize_(0) */ {
    // @todo Actually get user's name to feed to pgConnector_
    // populate map with ColumnInfo structs
    for (auto colIt = columnInfoVec.begin(); colIt != columnInfoVec.end(); ++colIt) {
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
    mapd_size_t numRowsLeft = insertDataStruct.numRows;
    mapd_size_t numRowsInserted = 0;
    if (maxPartitionId_ < 0 && numRowsLeft > 0) // if no partitions exist for table and there is > 1 row to insert
        createNewPartition();
    while (numRowsLeft > 0) { // may have to create multiple partitions for bulk insert
        // loop until done inserting all rows
        mapd_size_t rowsLeftInCurrentPartition = maxPartitionRows_ - partitionInfoVec_.back().numTuples;
        if (rowsLeftInCurrentPartition == 0) {
            createNewPartition();
            rowsLeftInCurrentPartition = maxPartitionRows_;
        }
        mapd_size_t numRowsToInsert = min(rowsLeftInCurrentPartition, numRowsLeft);

        // for each column, append the data in the appropriate insert buffer
        for (int i = 0; i < insertDataStruct.columnIds.size(); ++i) {
            int columnId = insertDataStruct.columnIds[i];
            auto colMapIt = columnMap_.find(columnId);
            assert(colMapIt != columnMap_.end());
            mapd_size_t colByteSize = colMapIt->second.bitSize / 8;
            
            // append the data (size of data is colBytesize * numRowsToInsert)
            AbstractBuffer *insertBuffer = colMapIt->second.insertBuffer;
            //insertBuffer->append(colByteSize * numRowsToInsert, static_cast<mapd_addr_t>(insertDataStruct.data[i]));
            insertBuffer->append(static_cast<mapd_addr_t>(insertDataStruct.data[i]),colByteSize*numRowsToInsert);
            //insertBuffer->print();
            //insertBuffer->print(colMapIt->second.columnType);
        }

        partitionInfoVec_.back().numTuples += numRowsToInsert;
        numRowsLeft -= numRowsToInsert;
        numRowsInserted += numRowsToInsert;
    }
    isDirty_ = true;
}

void LinearTablePartitioner::createNewPartition() { 
    // also sets the new partition as the insertBuffer for each column

    // iterate through all ColumnInfo structs in map, unpin previous insert buffer and
    // create new insert buffer
    maxPartitionId_++;
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        if (colMapIt -> second.insertBuffer != 0) {
            //colMapIt -> second.insertBuffer -> unPin();
        }
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
     mapd_err_t status = pgConnector_.query("CREATE TABLE IF NOT EXISTS fragments(partitioner_id INT, fragment_id INT, num_rows INT, PRIMARY KEY (partitioner_id, fragment_id))");
     assert(status == MAPD_SUCCESS);
     string createTableStatsQuery ("CREATE TABLE IF NOT EXISTS partitioner_" + boost::lexical_cast <string> (partitionerId_) + "_stats (fragment_id INT, ");  
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        int columnId = colMapIt -> first;
        string baseStatsColumnName = "col_" + boost::lexical_cast <string> (columnId);
        string typeName;
        switch(colMapIt -> second.columnType.type) {
            case INT_TYPE:
                typeName = " INT, ";
                break;
            case FLOAT_TYPE:
                typeName = " REAL, ";
                break;
            case BOOLEAN_TYPE:
                typeName = " BOOLEAN, ";
                break;
        }
        createTableStatsQuery += baseStatsColumnName = "_min" + typeName;
        createTableStatsQuery += baseStatsColumnName = "_max" + typeName;

    }
    createTableStatsQuery.replace(createTableStatsQuery.size() -1, 1, ")");
    status = pgConnector_.query(createTableStatsQuery);
}


void LinearTablePartitioner::readState() {
    string partitionQuery ("select fragment_id, num_rows from fragments where partitioner_id = " + boost::lexical_cast <string> (partitionerId_));
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
    string statsQuery ("select fragment_id");
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        int columnId = colMapIt -> first;
        string baseStatsColumnName = "col_" + boost::lexical_cast <string> (columnId);
        statsQuery += "," + baseStatsColumnName + "_min," + baseStatsColumnName + "_max";
    }
    statsQuery += " from partitioner_" + boost::lexical_cast <string> (partitionerId_) + "_stats";
    status = pgConnector_.query(statsQuery);
    assert(status == MAPD_SUCCESS);
    numRows = pgConnector_.getNumRows();
}

void LinearTablePartitioner::getInsertBufferChunks() {
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        if (colMapIt -> second.insertBuffer != NULL) {
            //colMapIt -> second.insertBuffer -> unPin();
        }
        ChunkKey chunkKey = {partitionerId_, maxPartitionId_,  colMapIt -> second.columnId};
        colMapIt -> second.insertBuffer = bufferManager_.getChunk(chunkKey);
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

} // Partitioner_Namespace
