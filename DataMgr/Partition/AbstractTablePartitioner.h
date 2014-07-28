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
    int columnId_; // for when we iterate over all structs of ColumnInfo instead of using a map
    mapd_data_t columnType_; 
    int bitSize_;
    Buffer_Namespace::Buffer * insertBuffer_; // a pointer so can be null

    //ColumnInfo(const int columnId, const mapd_data_t columnType, const int bitSize): columnId_(columnId), columnType_(columnType), bitSize_(bitSize) {}
	//ColumnInfo& operator=(const ColumnInfo&);
};

struct PartitionInfo {
    int partitionId_;
    mapd_size_t numTuples_;
};

class AbstractTablePartitioner { 

    public:

        virtual void getPartitionsForQuery(std::vector <PartitionInfo> &partitions, const void *predicate = 0) = 0;

        virtual void insertData (const std::vector <int> &columnIds, const std::vector <void *> &data, const int numRows) = 0;

};

#endif // _ABSTRACT_TABLE_PARTITIONER 
