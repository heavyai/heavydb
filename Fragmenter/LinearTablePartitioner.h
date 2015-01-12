/**
 * @file	LinearTablePartitioner.h
 * @author	Todd Mostak <todd@map-d.com>
 */
#ifndef _LINEAR_TABLE_PARTITIONER_H
#define _LINEAR_TABLE_PARTITIONER_H

#include "../../Shared/types.h"
#include "../PgConnector/PgConnector.h"
#include "AbstractTablePartitioner.h"

#include <vector>
#include <map>

namespace Partitioner_Namespace {

/**
 * @type LinearTablePartitioner
 * @brief	The LinearTablePartitioner is a child class of
 * AbstractTablePartitioner, and partitions data in insert 
 * order. Likely the default partitioner
 */

class LinearTablePartitioner : public AbstractTablePartitioner {

public:

    LinearTablePartitioner(const int partitionerId,  std::vector <ColumnInfo> &columnInfoVec, Memory_Namespace::AbstractDataMgr &bufferManager, const mapd_size_t maxPartitionRows =1048576, const mapd_size_t pageSize = 1048576 /*default 1MB*/);

    virtual ~LinearTablePartitioner();
    /**
     * @brief returns (inside QueryInfo) object all 
     * ids and row sizes of partitions that could potentially
     * satisify the given optional predicate.
     *
     * Note that just because this partitioner partitions
     * data only on insert order, doesn't mean that checking
     * the predicate will not elimate the need to scan certain
     * partitions (for example - in the case of 
     * data inserted in time order)
     */

    virtual void getPartitionsForQuery(QueryInfo &queryInfo, const void *predicate = 0);

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

    inline int getPartitionerId () {return partitionerId_;}
    /**
     * @brief get partitioner's type (as string
     */
    inline std::string getPartitionerType () {return partitionerType_;}

private:

	int partitionerId_; /**< Stores the id of the partitioner - passed to constructor */
    std::string partitionerType_;
	mapd_size_t maxPartitionRows_;
    int maxPartitionId_;
    mapd_size_t pageSize_; /* Page size in bytes of each page making up a given chunk - passed to BufferMgr in createChunk() */
    std::map <int, ColumnInfo> columnMap_; /**< stores a map of column id to metadata about that column */ 
    std::vector<PartitionInfo> partitionInfoVec_; /**< data about each partition stored - id and number of rows */  
    //int currentInsertBufferPartitionId_;
    Memory_Namespace::AbstractDataMgr &bufferManager_;
    PgConnector pgConnector_;
    bool isDirty_;  /**< Specifies if the LinearTablePartitioner has been modified in memory since the last flush to file - no need to rewrite file if this is false. */
   
    /**
     * @brief creates new partition, calling createChunk()
     * method of BufferMgr to make a new chunk for each column
     * of the table.
     *
     * Also unpins the chunks of the previous insert buffer
     */

    void createNewPartition();

    void createStateTableIfDne();
    void readState();
    void writeState();

    /**
     * @brief Called at readState to associate chunks of 
     * partition with max id with pointer into buffer pool
     */

    void getInsertBufferChunks(); 
	
	LinearTablePartitioner(const LinearTablePartitioner&);
	LinearTablePartitioner& operator=(const LinearTablePartitioner&);

};

} // Partitioner_Namespace

 #endif // LINEAR_TABLE_PARTITIONER_H

