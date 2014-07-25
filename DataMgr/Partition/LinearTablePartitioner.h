/**
 * @file	LinearTablePartitioner.h
 * @author	Todd Mostak <todd@map-d.com>
 */
#ifndef _LINEAR_TABLE_PARTITIONER_H
#define _LINEAR_TABLE_PARTITIONER_H
#include "AbstractPartitioner.h"
#include "../../Shared/types.h"

class Buffer;
class BufferManager;

struct ColumnInfo {
    int columnId_; // in case we iterate over all structs of ColumnInfo instead of using a map
    DataType columnType_; 
    int bitSize_;
    Buffer * insertBuffer_; // a pointer so can be null
    mapd_size_t insertBufferNumTuples;
    ColumnInfo(int columnId, int bitSize): columnId_(columnId), bitSize_(bitSize) {}
};

struct FragmentInfo {
    int fragmentId_;
    mapd_size_t numTuples;
};

    


/**
 * @brief	The LinearTablePartitioner maps partial keys to fragment ids.
 *
 * The LinearTablePartitioner maps partial keys to fragment ids. It's principle method
 * is getPartitionIds(), which returns a vector of ids. 
 *
 * @todo 	The LinearTablePartitioner should have a reference to a PartitionScheme;
 *			this will be added in an upcoming release.
 */
class LinearTablePartitioner : public AbstractPartitioner { // implements

public:
	LinearTablePartitioner(int tableId, mapd_size_t maxFragmentSize, vector <ColumnInfo> &columnInfoVec, BufferManager &bufferManager) :
		tableId_(tableId), insertBufferSize_(insertBufferSize), columnInfoVec_(columnInfoVec), bufferManager_(bufferManager) {}

	//virtual ~LinearTablePartitioner();

	virtual bool getPartitionIds(const void *predicate, std::vector<int> &result);

    virtual void insertData (const vector <int> &columnIds, const vector <void *> &data, const int numRows);

private:
	int tableId_;
	mapd_size_t maxFragmentSize_;
    map <int, ColumnInfo> columnMap_; 
    std::vector<FragmentInfo> fragmentInfoVec_;
    mapd_size_t currentInsertBufferSize_;
    Buffer_Namespace::BufferManager &bufferManager_;
	
	LinearTablePartitioner(const LinearTablePartitioner&);
	LinearTablePartitioner& operator=(const LinearTablePartitioner&);

};

 #endif // _INSERT_ORDER_PARTITIONER_H
