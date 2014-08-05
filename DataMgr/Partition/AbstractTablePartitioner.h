/**
 * @file    AbstractTablePartitioner.h
 * @author  Todd Mostak <todd@map-d.com
 */

#ifndef _ABSTRACT_TABLE_PARTITIONER_H
#define _ABSTRACT_TABLE_PARTITIONER_H

#include "../../Shared/types.h"
#include <vector>

// Should the ColumnInfo and PartitionInfo structs be in
// AbstractTablePartitioner?

namespace Buffer_Namespace {
    class Buffer;
    class BufferMgr;
};

struct ColumnInfo {
    int columnId; // for when we iterate over all structs of ColumnInfo instead of using a map
    mapd_data_t columnType; 
    int bitSize;
    Buffer_Namespace::Buffer * insertBuffer; // a pointer so can be null

    //ColumnInfo(const int columnId, const mapd_data_t columnType, const int bitSize): columnId_(columnId), columnType_(columnType), bitSize_(bitSize) {}
	//ColumnInfo& operator=(const ColumnInfo&);
};

struct PartitionInfo {
    int partitionId;
    mapd_size_t numTuples;
};

struct QueryInfo {
    int partitionerId;
    std::vector<PartitionInfo> partitions;
    mapd_size_t numTuples; 
};

class AbstractTablePartitioner { 

    public:

        virtual void getPartitionsForQuery(QueryInfo &queryInfo, const void *predicate = 0) = 0;

        virtual void insertData (const std::vector <int> &columnIds, const std::vector <void *> &data, const int numRows) = 0;

};

#endif // _ABSTRACT_TABLE_PARTITIONER 
