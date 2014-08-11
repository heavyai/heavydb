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


/**
 * @brief	The LinearTablePartitioner maps partial keys to partition ids.
 *
 * The LinearTablePartitioner maps partial keys to partition ids. It's principle method
 * is getPartitionIds(), which returns a vector of ids. 
 *
 */
class LinearTablePartitioner : public AbstractTablePartitioner { // implements

public:
    LinearTablePartitioner(const int partitionerId,  std::vector <ColumnInfo> &columnInfoVec, Buffer_Namespace::BufferMgr &bufferManager, const mapd_size_t maxPartitionRows =1048576, const mapd_size_t pageSize = 1048576 /*default 1MB*/);

    virtual ~LinearTablePartitioner();

    virtual void getPartitionsForQuery(QueryInfo &queryInfo, const void *predicate = 0);

    //virtual void insertData (const std::vector <int> &columnIds, const std::vector <void *> &data, const int numRows);
    virtual void insertData (const InsertData &insertDataStruct);
    //mapd_size_t currentInsertBufferSize_;
    inline int getPartitionerId () {return partitionerId_;}
    inline std::string getPartitionerType () {return partitionerType_;}

private:
	int partitionerId_;
    std::string partitionerType_;
	mapd_size_t maxPartitionRows_;
    int maxPartitionId_;
    mapd_size_t pageSize_;
    std::map <int, ColumnInfo> columnMap_; 
    std::vector<PartitionInfo> partitionInfoVec_; // do we assume this kept in order
    //int currentInsertBufferPartitionId_;
    Buffer_Namespace::BufferMgr &bufferManager_;
    PgConnector pgConnector_;
    bool isDirty_;  /**< Specifies if the LinearTablePartitioner has been modified in memory since the last flush to file - no need to rewrite file if this is false. */
    
    void createNewPartition();
    void createStateTableIfDne();
    void readState();
    void writeState();
    void getInsertBufferChunks();
	
	LinearTablePartitioner(const LinearTablePartitioner&);
	LinearTablePartitioner& operator=(const LinearTablePartitioner&);

};

 #endif // LINEAR_TABLE_PARTITIONER_H
