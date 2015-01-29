#include "InsertOrderTablePartitioner.h"
#include "../DataMgr/DataMgr.h"
#include "../DataMgr/AbstractBuffer.h"
#include <math.h>
#include <iostream>
#include <thread>

#include <assert.h>
#include <boost/lexical_cast.hpp>

using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;

using namespace std;

namespace Partitioner_Namespace {


InsertOrderTablePartitioner::InsertOrderTablePartitioner(const vector <int> chunkKeyPrefix, vector <ColumnInfo> &columnInfoVec, Data_Namespace::DataMgr *dataMgr, const size_t maxPartitionRows, const size_t pageSize /*default 1MB*/, const Data_Namespace::MemoryLevel defaultInsertLevel) :
		chunkKeyPrefix_(chunkKeyPrefix), dataMgr_(dataMgr), maxPartitionRows_(maxPartitionRows), pageSize_(pageSize), maxPartitionId_(-1), partitionerType_("insert_order"), defaultInsertLevel_(defaultInsertLevel) {
    for (auto colIt = columnInfoVec.begin(); colIt != columnInfoVec.end(); ++colIt) {
        columnMap_[colIt -> columnDesc->columnId] = *colIt; 
    }
    getChunkMetadata();

}

InsertOrderTablePartitioner::~InsertOrderTablePartitioner() {

}

void InsertOrderTablePartitioner::getChunkMetadata() {
    std::vector<std::pair<ChunkKey,ChunkMetadata> > chunkMetadataVec;
    dataMgr_->getChunkMetadataVecForKeyPrefix(chunkMetadataVec,chunkKeyPrefix_);
    //dataMgr_->getChunkMetadataVec(chunkMetadataVec);

    // data comes like this - database_id, table_id, column_id, partition_id
    // but lets sort by database_id, table_id, partition_id, column_id

    int partitionSubKey = 3; 
    std::sort(chunkMetadataVec.begin(), chunkMetadataVec.end(),[&] (const std::pair<ChunkKey,ChunkMetadata> &pair1, const std::pair<ChunkKey,ChunkMetadata> &pair2) {
                return pair1.first[3] < pair2.first[3];
            });
                
    for (auto chunkIt = chunkMetadataVec.begin(); chunkIt != chunkMetadataVec.end(); ++chunkIt) {
        int curPartitionId = chunkIt->first[partitionSubKey];
        if (partitionInfoVec_.empty() || curPartitionId != partitionInfoVec_.back().partitionId) {
            partitionInfoVec_.push_back(PartitionInfo());
            partitionInfoVec_.back().partitionId = curPartitionId;
            partitionInfoVec_.back().numTuples = chunkIt->second.numElements;
            partitionInfoVec_.back().shadowNumTuples = partitionInfoVec_.back().numTuples;
        }
        else {
            if (chunkIt->second.numElements != partitionInfoVec_.back().numTuples) {
                throw std::runtime_error ("Inconsistency in num tuples within fragment");
            }
        }
        int columnId = chunkIt->first[2];
        partitionInfoVec_.back().chunkMetadataMap[columnId] = chunkIt->second; 
    }
    // Now need to get the insert buffers for each column - should be last
    // partition
    if (partitionInfoVec_.size() > 0) {
        int lastPartitionId = partitionInfoVec_.back().partitionId;
        for (auto colIt = columnMap_.begin(); colIt != columnMap_.end(); ++colIt) {
            ChunkKey insertKey = chunkKeyPrefix_; //database_id and table_id
            insertKey.push_back(colIt->first); // column id
            insertKey.push_back(lastPartitionId); // partition id
            colIt->second.insertBuffer=dataMgr_->getChunk(defaultInsertLevel_,insertKey);
        }
    }
}




void InsertOrderTablePartitioner::insertData (const InsertData &insertDataStruct) {
    boost::lock_guard<boost::mutex> insertLock (insertMutex_); // prevent two threads from trying to insert into the same table simultaneously

    size_t numRowsLeft = insertDataStruct.numRows;
    size_t numRowsInserted = 0;
    vector<int8_t *> dataCopy = insertDataStruct.data; // bc append data will move ptr forward and this violates constness of InsertData
    //vector <PartitionInfo *> partitionsToBeUpdated;  
    //std::vector<PartitionInfo>::iterator partIt = partitionInfoVec_.back();
    if (numRowsLeft <= 0) {
        return;
    }

    PartitionInfo *currentPartition=0;

    if (partitionInfoVec_.empty()) { // if no partitions exist for table 
        currentPartition = createNewPartition(defaultInsertLevel_);
    }
    else {
        currentPartition = &(partitionInfoVec_.back());
    }
    size_t startPartition = partitionInfoVec_.size() - 1;

    while (numRowsLeft > 0) { // may have to create multiple partitions for bulk insert
        // loop until done inserting all rows
        size_t rowsLeftInCurrentPartition = maxPartitionRows_ - currentPartition->shadowNumTuples;
        if (rowsLeftInCurrentPartition == 0) {
            currentPartition = createNewPartition(); 
            rowsLeftInCurrentPartition = maxPartitionRows_;
        }
        size_t numRowsToInsert = min(rowsLeftInCurrentPartition, numRowsLeft);
        // for each column, append the data in the appropriate insert buffer
        for (int i = 0; i < insertDataStruct.columnIds.size(); ++i) {
            int columnId = insertDataStruct.columnIds[i];
            auto colMapIt = columnMap_.find(columnId);
            assert(colMapIt != columnMap_.end());
            //AbstractBuffer *insertBuffer = colMapIt->second.insertBuffer;
            currentPartition->shadowChunkMetadataMap[columnId] = colMapIt->second.insertBuffer->encoder->appendData(dataCopy[i],numRowsToInsert);
            //partitionInfoVec_.back().shadowChunkMetadataMap[columnId] = colMapIt->second.insertBuffer->encoder->appendData(static_cast<int8_t *>(insertDataStruct.data[i]),numRowsToInsert);
        }

        currentPartition->shadowNumTuples = partitionInfoVec_.back().numTuples + numRowsToInsert;
        //partitionInfoVec_.back().shadowNumTuples = partitionInfoVec_.back().numTuples + numRowsToInsert;
        //cout << "Shadow tuples"  << partitionInfoVec_.back().shadowNumTuples << endl;
        //partitionsToBeUpdated.push_back(&(partitionInfoVec_.back()));
        numRowsLeft -= numRowsToInsert;
        numRowsInserted += numRowsToInsert;
    }
    boost::unique_lock < boost::shared_mutex > writeLock (partitionInfoMutex_);
    //for (auto partIt = partitionsToBeUpdated.begin(); partIt != partitionsToBeUpdated.end(); ++partIt) {
    for (auto partIt = partitionInfoVec_.begin() + startPartition; partIt != partitionInfoVec_.end(); ++partIt) { 
        partIt->numTuples = partIt->shadowNumTuples;
        partIt->chunkMetadataMap=partIt->shadowChunkMetadataMap;
    }
}

PartitionInfo * InsertOrderTablePartitioner::createNewPartition(const Data_Namespace::MemoryLevel memoryLevel) { 
    // also sets the new partition as the insertBuffer for each column

    // iterate through all ColumnInfo structs in map, unpin previous insert buffer and
    // create new insert buffer
    maxPartitionId_++;
    //cout << "Create new partition: " << maxPartitionId_ << endl;
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        if (colMapIt -> second.insertBuffer != 0) {
            colMapIt -> second.insertBuffer -> unPin();
        }
        ChunkKey chunkKey =  chunkKeyPrefix_;
        chunkKey.push_back(colMapIt->second.columnDesc->columnId);
        chunkKey.push_back(maxPartitionId_);
        colMapIt->second.insertBuffer = dataMgr_->createChunk(memoryLevel,chunkKey);
        //cout << "Creating chunk with encodingType: " << colMapIt->second.columnDesc->compression << endl;
        colMapIt->second.insertBuffer->initEncoder(colMapIt->second.columnDesc->columnType.type,colMapIt->second.columnDesc->compression,colMapIt->second.columnDesc->comp_param);
    }
    PartitionInfo newPartitionInfo;
    newPartitionInfo.partitionId = maxPartitionId_;
    newPartitionInfo.shadowNumTuples = 0; 
    newPartitionInfo.numTuples = 0; 

    boost::unique_lock < boost::shared_mutex > writeLock (partitionInfoMutex_);
    partitionInfoVec_.push_back(newPartitionInfo);
    return &(partitionInfoVec_.back());
}

void InsertOrderTablePartitioner::getPartitionsForQuery(QueryInfo &queryInfo) {
    queryInfo.chunkKeyPrefix = chunkKeyPrefix_;
    // right now we don't test predicate, so just return (copy of) all partitions 
    {
        {
            boost::shared_lock < boost::shared_mutex > readLock (partitionInfoMutex_);
            queryInfo.partitions = partitionInfoVec_; //makes a copy
        }
        queryInfo.numTuples = 0;
        for (auto partIt = queryInfo.partitions.begin(); partIt != queryInfo.partitions.end(); ++partIt) {
            queryInfo.numTuples += partIt -> numTuples;  
        }
    }
}


void InsertOrderTablePartitioner::getInsertBufferChunks() {
    for (map<int, ColumnInfo>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
        assert (colMapIt -> second.insertBuffer == NULL);
        if (colMapIt -> second.insertBuffer != NULL) {
            // should really always be null - we should just be using this
            // method up front
            colMapIt -> second.insertBuffer -> unPin();
        }
        ChunkKey chunkKey = {partitionerId_, maxPartitionId_,  colMapIt -> second.columnDesc->columnId};
        colMapIt -> second.insertBuffer = dataMgr_->getChunk(Data_Namespace::DISK_LEVEL,chunkKey);
    }
}
/*
void InsertOrderTablePartitioner::readState() {
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
void InsertOrderTablePartitioner::writeState() {
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
*/

} // Partitioner_Namespace
