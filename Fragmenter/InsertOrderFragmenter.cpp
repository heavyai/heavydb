#include "InsertOrderFragmenter.h"
#include "../DataMgr/DataMgr.h"
#include "../DataMgr/AbstractBuffer.h"
#include <math.h>
#include <iostream>
#include <thread>

#include <assert.h>
#include <boost/lexical_cast.hpp>

using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;
using Chunk_NS::Chunk;

using namespace std;

namespace Fragmenter_Namespace {


InsertOrderFragmenter::InsertOrderFragmenter(const vector <int> chunkKeyPrefix, vector <Chunk> &chunkVec, Data_Namespace::DataMgr *dataMgr, const size_t maxFragmentRows, const size_t pageSize /*default 1MB*/, const Data_Namespace::MemoryLevel defaultInsertLevel) :
		chunkKeyPrefix_(chunkKeyPrefix), dataMgr_(dataMgr), maxFragmentRows_(maxFragmentRows), pageSize_(pageSize), maxFragmentId_(-1), fragmenterType_("insert_order"), defaultInsertLevel_(defaultInsertLevel) {

    for (auto colIt = chunkVec.begin(); colIt != chunkVec.end(); ++colIt) {
        columnMap_[colIt ->get_column_desc()->columnId] = *colIt; 
    }
    getChunkMetadata();

}

InsertOrderFragmenter::~InsertOrderFragmenter() {

}

void InsertOrderFragmenter::getChunkMetadata() {
    std::vector<std::pair<ChunkKey,ChunkMetadata> > chunkMetadataVec;
    dataMgr_->getChunkMetadataVecForKeyPrefix(chunkMetadataVec,chunkKeyPrefix_);
    //dataMgr_->getChunkMetadataVec(chunkMetadataVec);

    // data comes like this - database_id, table_id, column_id, fragment_id
    // but lets sort by database_id, table_id, fragment_id, column_id

    int fragmentSubKey = 3; 
    std::sort(chunkMetadataVec.begin(), chunkMetadataVec.end(),[&] (const std::pair<ChunkKey,ChunkMetadata> &pair1, const std::pair<ChunkKey,ChunkMetadata> &pair2) {
                return pair1.first[3] < pair2.first[3];
            });
                
    for (auto chunkIt = chunkMetadataVec.begin(); chunkIt != chunkMetadataVec.end(); ++chunkIt) {
        int curFragmentId = chunkIt->first[fragmentSubKey];
        if (fragmentInfoVec_.empty() || curFragmentId != fragmentInfoVec_.back().fragmentId) {
            fragmentInfoVec_.push_back(FragmentInfo());
            fragmentInfoVec_.back().fragmentId = curFragmentId;
            fragmentInfoVec_.back().numTuples = chunkIt->second.numElements;
            for (const auto levelSize: dataMgr_->levelSizes_) { 
                fragmentInfoVec_.back().deviceIds.push_back(curFragmentId  % levelSize);
            }
            fragmentInfoVec_.back().shadowNumTuples = fragmentInfoVec_.back().numTuples;
        }
        else {
            if (chunkIt->second.numElements != fragmentInfoVec_.back().numTuples) {
                throw std::runtime_error ("Inconsistency in num tuples within fragment");
            }
        }
        int columnId = chunkIt->first[2];
        fragmentInfoVec_.back().chunkMetadataMap[columnId] = chunkIt->second; 
    }
    // Now need to get the insert buffers for each column - should be last
    // fragment
    if (fragmentInfoVec_.size() > 0) {
        int lastFragmentId = fragmentInfoVec_.back().fragmentId;
        int deviceId = fragmentInfoVec_.back().deviceIds[static_cast<int>(defaultInsertLevel_)];
        for (auto colIt = columnMap_.begin(); colIt != columnMap_.end(); ++colIt) {
            ChunkKey insertKey = chunkKeyPrefix_; //database_id and table_id
            insertKey.push_back(colIt->first); // column id
            insertKey.push_back(lastFragmentId); // fragment id
						colIt->second.getChunkBuffer(dataMgr_, insertKey, defaultInsertLevel_, deviceId);
        }
    }
}




void InsertOrderFragmenter::insertData (const InsertData &insertDataStruct) {
    boost::lock_guard<boost::mutex> insertLock (insertMutex_); // prevent two threads from trying to insert into the same table simultaneously

    size_t numRowsLeft = insertDataStruct.numRows;
    size_t numRowsInserted = 0;
    vector<DataBlockPtr> dataCopy = insertDataStruct.data; // bc append data will move ptr forward and this violates constness of InsertData
    if (numRowsLeft <= 0) {
        return;
    }

    FragmentInfo *currentFragment=0;

    if (fragmentInfoVec_.empty()) { // if no fragments exist for table 
        currentFragment = createNewFragment(defaultInsertLevel_);
    }
    else {
        currentFragment = &(fragmentInfoVec_.back());
    }
    size_t startFragment = fragmentInfoVec_.size() - 1;

    while (numRowsLeft > 0) { // may have to create multiple fragments for bulk insert
        // loop until done inserting all rows
        size_t rowsLeftInCurrentFragment = maxFragmentRows_ - currentFragment->shadowNumTuples;
        if (rowsLeftInCurrentFragment == 0) {
            currentFragment = createNewFragment(); 
            rowsLeftInCurrentFragment = maxFragmentRows_;
        }
        size_t numRowsToInsert = min(rowsLeftInCurrentFragment, numRowsLeft);
        // for each column, append the data in the appropriate insert buffer
        for (int i = 0; i < insertDataStruct.columnIds.size(); ++i) {
            int columnId = insertDataStruct.columnIds[i];
            auto colMapIt = columnMap_.find(columnId);
            assert(colMapIt != columnMap_.end());
            currentFragment->shadowChunkMetadataMap[columnId] = colMapIt->second.appendData(dataCopy[i],numRowsToInsert, numRowsInserted);
        }

        currentFragment->shadowNumTuples = fragmentInfoVec_.back().numTuples + numRowsToInsert;
        //fragmentInfoVec_.back().shadowNumTuples = fragmentInfoVec_.back().numTuples + numRowsToInsert;
        //cout << "Shadow tuples"  << fragmentInfoVec_.back().shadowNumTuples << endl;
        //fragmentsToBeUpdated.push_back(&(fragmentInfoVec_.back()));
        numRowsLeft -= numRowsToInsert;
        numRowsInserted += numRowsToInsert;
    }
    boost::unique_lock < boost::shared_mutex > writeLock (fragmentInfoMutex_);
    //for (auto partIt = fragmentsToBeUpdated.begin(); partIt != fragmentsToBeUpdated.end(); ++partIt) {
    for (auto partIt = fragmentInfoVec_.begin() + startFragment; partIt != fragmentInfoVec_.end(); ++partIt) { 
        partIt->numTuples = partIt->shadowNumTuples;
        partIt->chunkMetadataMap=partIt->shadowChunkMetadataMap;
    }
    dataMgr_->checkpoint();
}

FragmentInfo * InsertOrderFragmenter::createNewFragment(const Data_Namespace::MemoryLevel memoryLevel) { 
    // also sets the new fragment as the insertBuffer for each column

    // iterate through all Chunk's in map, unpin previous insert buffer and
    // create new insert buffer
    maxFragmentId_++;
    //cout << "Create new fragment: " << maxFragmentId_ << endl;
    FragmentInfo newFragmentInfo;
    newFragmentInfo.fragmentId = maxFragmentId_;
    newFragmentInfo.shadowNumTuples = 0; 
    newFragmentInfo.numTuples = 0; 
    for (const auto levelSize: dataMgr_->levelSizes_) { 
        newFragmentInfo.deviceIds.push_back(newFragmentInfo.fragmentId % levelSize);
    }

    for (map<int, Chunk>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
				// colMapIt->second.unpin_buffer();
        ChunkKey chunkKey =  chunkKeyPrefix_;
        chunkKey.push_back(colMapIt->second.get_column_desc()->columnId);
        chunkKey.push_back(maxFragmentId_);
				colMapIt->second.createChunkBuffer(dataMgr_, chunkKey, memoryLevel, newFragmentInfo.deviceIds[static_cast<int>(memoryLevel)]);
				colMapIt->second.init_encoder();
    }

    boost::unique_lock < boost::shared_mutex > writeLock (fragmentInfoMutex_);
    fragmentInfoVec_.push_back(newFragmentInfo);
    return &(fragmentInfoVec_.back());
}

void InsertOrderFragmenter::getFragmentsForQuery(QueryInfo &queryInfo) {
    queryInfo.chunkKeyPrefix = chunkKeyPrefix_;
    // right now we don't test predicate, so just return (copy of) all fragments 
    {
        {
            boost::shared_lock < boost::shared_mutex > readLock (fragmentInfoMutex_);
            queryInfo.fragments = fragmentInfoVec_; //makes a copy
        }
        queryInfo.numTuples = 0;
        for (auto partIt = queryInfo.fragments.begin(); partIt != queryInfo.fragments.end(); ++partIt) {
            queryInfo.numTuples += partIt -> numTuples;  
        }
    }
}


void InsertOrderFragmenter::getInsertBufferChunks() {
    for (map<int, Chunk>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
				// colMapIt->second.unpin_buffer();
        ChunkKey chunkKey = {fragmenterId_, maxFragmentId_,  colMapIt -> second.get_column_desc()->columnId};
				colMapIt->second.getChunkBuffer(dataMgr_, chunkKey, defaultInsertLevel_);
    }
}

} // Fragmenter_Namespace
