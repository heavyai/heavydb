/**
 * @file	InsertOrderTablePartitioner.h
 * @author	Todd Mostak <todd@mapd.com>
 */
#ifndef INSERT_ORDER_TABLE_PARTITIONER_H
#define INSERT_ORDER_TABLE_PARTITIONER_H

#include "../Shared/types.h"
#include "AbstractTablePartitioner.h"
#include "../DataMgr/MemoryLevel.h"

#include <vector>
#include <map>
#include <boost/thread.hpp>

#include <mutex>

namespace Data_Namespace {
    class DataMgr; 
}

#define DEFAULT_FRAGMENT_SIZE		1000000 // in tuples
#define DEFAULT_PAGE_SIZE				1048576 // in bytes

namespace Partitioner_Namespace {

/**
 * @type InsertOrderTablePartitioner
 * @brief	The InsertOrderTablePartitioner is a child class of
 * AbstractTablePartitioner, and partitions data in insert 
 * order. Likely the default partitioner
 */

class InsertOrderTablePartitioner : public AbstractTablePartitioner {

public:

    InsertOrderTablePartitioner(const std::vector <int> chunkKeyPrefix, std::vector <ColumnInfo> &columnInfoVec, Data_Namespace::DataMgr *dataMgr, const size_t maxPartitionRows = DEFAULT_FRAGMENT_SIZE, const size_t pageSize = DEFAULT_PAGE_SIZE /*default 1MB*/, const Data_Namespace::MemoryLevel defaultInsertLevel = Data_Namespace::DISK_LEVEL);

    virtual ~InsertOrderTablePartitioner();
    /**
     * @brief returns (inside QueryInfo) object all 
     * ids and row sizes of partitions 
     *
     */

    //virtual void getPartitionsForQuery(QueryInfo &queryInfo, const void *predicate = 0);
    virtual void getPartitionsForQuery(QueryInfo &queryInfo);

    /**
     * @brief appends data onto the most recently occuring
     * partition, creating a new one if necessary
     * 
     * @todo be able to fill up current partition in
     * multi-row insert before creating new partition
     */
    virtual void insertData (const InsertData &insertDataStruct);
    /**
     * @brief get partitioner's id
     */

    inline int getPartitionerId () {return  chunkKeyPrefix_.back();}
    /**
     * @brief get partitioner's type (as string
     */
    inline std::string getPartitionerType () {return partitionerType_;}

private:

	int partitionerId_; /**< Stores the id of the partitioner - passed to constructor */
    std::vector<int> chunkKeyPrefix_;
    std::map <int, ColumnInfo> columnMap_; /**< stores a map of column id to metadata about that column */ 
    std::vector<PartitionInfo> partitionInfoVec_; /**< data about each partition stored - id and number of rows */  
    //int currentInsertBufferPartitionId_;
    Data_Namespace::DataMgr *dataMgr_;
	size_t maxPartitionRows_;
    size_t pageSize_; /* Page size in bytes of each page making up a given chunk - passed to BufferMgr in createChunk() */
    int maxPartitionId_;
    std::string partitionerType_;
    boost::shared_mutex partitionInfoMutex_; // to prevent read-write conflicts for partitionInfoVec_
    boost::mutex insertMutex_; // to prevent race conditions on insert - only one insert statement should be going to a table at a time
    Data_Namespace::MemoryLevel defaultInsertLevel_;
    
    

    /**
     * @brief creates new partition, calling createChunk()
     * method of BufferMgr to make a new chunk for each column
     * of the table.
     *
     * Also unpins the chunks of the previous insert buffer
     */

    PartitionInfo * createNewPartition(const Data_Namespace::MemoryLevel memoryLevel = Data_Namespace::DISK_LEVEL);

    /**
     * @brief Called at readState to associate chunks of 
     * partition with max id with pointer into buffer pool
     */

    void getInsertBufferChunks(); 
    void getChunkMetadata();
	
	InsertOrderTablePartitioner(const InsertOrderTablePartitioner&);
	InsertOrderTablePartitioner& operator=(const InsertOrderTablePartitioner&);

};

} // Partitioner_Namespace

 #endif // INSERT_ORDER_TABLE_PARTITIONER_H

