/**
 * @file    AbstractTablePartitioner.h
 * @author  Todd Mostak <todd@map-d.com
 */

#ifndef _ABSTRACT_TABLE_PARTITIONER_H
#define _ABSTRACT_TABLE_PARTITIONER_H

#include "../../Shared/types.h"
#include <vector>
#include <string>

// Should the ColumnInfo and PartitionInfo structs be in
// AbstractTablePartitioner?

namespace Buffer_Namespace {
    class Buffer;
    class BufferMgr;
};

enum PartitionerType {
    LINEAR
};

struct ColumnInfo {
    int columnId; // for when we iterate over all structs of ColumnInfo instead of using a map
    mapd_data_t columnType; 
    int bitSize;
    Buffer_Namespace::Buffer * insertBuffer; // a pointer so can be null

    //ColumnInfo(const int columnId, const mapd_data_t columnType, const int bitSize): columnId_(columnId), columnType_(columnType), bitSize_(bitSize) {}
	//ColumnInfo& operator=(const ColumnInfo&);
};

struct InsertData {
	int tableId;						/// identifies the table into which the data is being inserted
	std::vector<int> columnIds;				/// a vector of column ids for the row(s) being inserted
	mapd_size_t numRows;				/// the number of rows being inserted
    std::vector <void *> data;							/// points to the start of the data for the row(s) being inserted
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
        virtual ~AbstractTablePartitioner() {}
        virtual void getPartitionsForQuery(QueryInfo &queryInfo, const void *predicate = 0) = 0;
        virtual void insertData (const InsertData &insertDataStruct) = 0;
        virtual int getPartitionerId() = 0;
        virtual std::string getPartitionerType() = 0;

        //virtual void insertData (const std::vector <int> &columnIds, const std::vector <void *> &data, const int numRows) = 0;

};

#endif // _ABSTRACT_TABLE_PARTITIONER 
