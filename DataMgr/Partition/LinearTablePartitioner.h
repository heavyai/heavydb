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
    LinearTablePartitioner(const int tableId,  std::vector <ColumnInfo> &columnInfoVec, Buffer_Namespace::BufferMgr &bufferManager, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize = 1048576 /*default 1MB*/);

    ~LinearTablePartitioner();

    virtual void getPartitionsForQuery(std::vector <PartitionInfo> &partitionIds, const void *predicate = 0);

    //virtual void insertData (const std::vector <int> &columnIds, const std::vector <void *> &data, const int numRows);
    virtual void insertData (const InsertData &insertDataStruct);
    //mapd_size_t currentInsertBufferSize_;

private:
	int tableId_;
	mapd_size_t maxPartitionRows_;
    int maxPartitionId_;
    mapd_size_t pageSize_;
    std::map <int, ColumnInfo> columnMap_; 
    std::vector<PartitionInfo> partitionInfoVec_; // do we assume this kept in order
    //int currentInsertBufferPartitionId_;
    Buffer_Namespace::BufferMgr &bufferManager_;
    PgConnector pgConnector_;
    
    void createNewPartition();
    void createStateTableIfDne();
    void readState();
    void writeState();
    void getInsertBufferChunks();
	
	LinearTablePartitioner(const LinearTablePartitioner&);
	LinearTablePartitioner& operator=(const LinearTablePartitioner&);

};

 #endif // LINEAR_TABLE_PARTITIONER_H
