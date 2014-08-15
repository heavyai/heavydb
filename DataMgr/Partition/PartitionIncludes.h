#ifndef PARTITIONERINCLUDES_H
#define PARTITIONERINCLUDES_H

#include "../../Shared/types.h"
#include <vector>
namespace Partition_Namespace {
    enum PartitionerType {
        LINEAR
    };

    /**
     * @struct InsertData
     * @brief The data to be inserted using the partition manager.
     *
     * The data being inserted is assumed to be in columnar format, and so the offset
     * to the beginning of each column can be calculated by multiplying the column size
     * by the number of rows.
     *
     * @todo support for variable-length data types
     */


    struct InsertData {
        int tableId;						/// identifies the table into which the data is being inserted
        std::vector<int> columnIds;				/// a vector of column ids for the row(s) being inserted
        mapd_size_t numRows;				/// the number of rows being inserted
        std::vector <void *> data;							/// points to the start of the data for the row(s) being inserted
    };
}

struct PartitionInfo {
    int partitionId;
    mapd_size_t numTuples;
};

struct QueryInfo {
    int partitionerId;
    std::vector<PartitionInfo> partitions;
    mapd_size_t numTuples; 
};

#endif
