/**
 * @file	LinearTablePartitioner.h
 * @author	Todd Mostak <todd@map-d.com>
 */
#ifndef _LINEAR_TABLE_PARTITIONER_H
#define _LINEAR_TABLE_PARTITIONER_H
//#include "AbstractPartitioner.h"
#include "../../Shared/types.h"
#include "../PgConnector/PgConnector.h"

#include <vector>
#include <map>

namespace Buffer_Namespace {
    class Buffer;
    class BufferMgr;
};

struct ColumnInfo {
    int columnId_; // for when we iterate over all structs of ColumnInfo instead of using a map
    mapd_data_t columnType_; 
    int bitSize_;
    Buffer_Namespace::Buffer * insertBuffer_; // a pointer so can be null

    //ColumnInfo(const int columnId, const mapd_data_t columnType, const int bitSize): columnId_(columnId), columnType_(columnType), bitSize_(bitSize) {}
	//ColumnInfo& operator=(const ColumnInfo&);
};

struct FragmentInfo {
    int fragmentId_;
    mapd_size_t numTuples_;
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
class LinearTablePartitioner { //: public AbstractPartitioner { // implements

public:
    LinearTablePartitioner(const int tableId,  std::vector <ColumnInfo> &columnInfoVec, Buffer_Namespace::BufferMgr &bufferManager, const mapd_size_t maxFragmentRows, const mapd_size_t pageSize = 1048576 /*default 1MB*/);

	//virtual ~LinearTablePartitioner();

	/*virtual*/ bool getPartitionIds(const void *predicate, std::vector<int> &result);

    /*virtual*/ void insertData (const std::vector <int> &columnIds, const std::vector <void *> &data, const int numRows);

private:
	int tableId_;
	mapd_size_t maxFragmentRows_;
    int maxFragmentId_;
    mapd_size_t pageSize_;
    std::map <int, ColumnInfo> columnMap_; 
    std::vector<FragmentInfo> fragmentInfoVec_;
    mapd_size_t currentInsertBufferSize_;
    int currentInsertBufferFragmentId_;
    Buffer_Namespace::BufferMgr &bufferManager_;

    PgConnector pgConnector_;
    
    void createNewFragment();
	
	LinearTablePartitioner(const LinearTablePartitioner&);
	LinearTablePartitioner& operator=(const LinearTablePartitioner&);

};

 #endif // _INSERT_ORDER_PARTITIONER_H
