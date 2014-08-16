/**
 * @file    AbstractTablePartitioner.h
 * @author  Todd Mostak <todd@map-d.com
 */

#ifndef _ABSTRACT_TABLE_PARTITIONER_H
#define _ABSTRACT_TABLE_PARTITIONER_H

#include "../../Shared/types.h"
#include "PartitionIncludes.h"
#include <vector>
#include <string>

// Should the ColumnInfo and PartitionInfo structs be in
// AbstractTablePartitioner?

namespace Buffer_Namespace {
    class Buffer;
    class BufferMgr;
};

namespace Partition_Namespace {

/**
 * @type ColumnInfo
 * @brief data structure to store id, type, bitsize
 * and insert buffer (if applicable) of a given column
 * managed by the partitioner
 */

struct ColumnInfo {
    int columnId; // for when we iterate over all structs of ColumnInfo instead of using a map
    mapd_data_t columnType; 
    int bitSize;
    Buffer_Namespace::Buffer * insertBuffer; // a pointer so can be null
    //@todo get the constructor for ColumnInfo compiling
    //ColumnInfo(const int columnId, const mapd_data_t columnType, const int bitSize): columnId_(columnId), columnType_(columnType), bitSize_(bitSize) {}
	//ColumnInfo& operator=(const ColumnInfo&);
};


/*
 * @type AbstractTablePartitioner
 * @brief abstract base class for all table partitioners
 *
 * The virtual methods of this class provide an interface
 * for an interface for getting the id and type of a 
 * partitioner, inserting data into a partitioner, and
 * getting the partitions (fragments) managed by a
 * partitioner that must be queried given a predicate
 */

class AbstractTablePartitioner { 

    public:
        virtual ~AbstractTablePartitioner() {}

        /**
         * @brief Should get the partitions(fragments) 
         * where at least one tuple could satisfy the
         * (optional) provided predicate, given any 
         * statistics on data distribution the partitioner
         * keeps. May also prune the predicate.
         */

        virtual void getPartitionsForQuery(QueryInfo &queryInfo, const void *predicate = 0) = 0;

        /**
         * @brief Given data wrapped in an InsertData struct,
         * inserts it into the correct partitions
         */

        virtual void insertData (const InsertData &insertDataStruct) = 0;

        /**
         * @brief Gets the id of the partitioner
         */
        virtual int getPartitionerId() = 0;

        /**
         * @brief Gets the string type of the partitioner
         * @todo have a method returning the enum type?
         */

        virtual std::string getPartitionerType() = 0;

};

} // Partition_Namespace

#endif // _ABSTRACT_TABLE_PARTITIONER 
