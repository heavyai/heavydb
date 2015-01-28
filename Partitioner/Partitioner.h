#ifndef PARTITIONER_PARTITIONER_H
#define PARTITIONER_PARTITIONER_H

#include <map>
#include <vector>
#include <list>
#include "../Shared/types.h"
#include "../DataMgr/ChunkMetadata.h"
#include "../Catalog/ColumnDescriptor.h"

namespace Data_Namespace {
    class AbstractBuffer;
}

namespace Partitioner_Namespace {
    
    /**
     * @enum PartitionerType
     * stores the type of a child class of
     * AbstractTablePartitioner
     */
    
    enum PartitionerType {
        INSERT_ORDER = 0 // these values persist in catalog.  make explicit
    };
    
    /**
     * @type ColumnInfo
     * @brief data structure to store id, type, bitsize
     * and insert buffer (if applicable) of a given column
     * managed by the partitioner
     */

    struct ColumnInfo {
				const ColumnDescriptor *columnDesc;
        Data_Namespace::AbstractBuffer * insertBuffer; // a pointer so can be null
				ColumnInfo() : columnDesc(nullptr), insertBuffer(nullptr) {}
				ColumnInfo(const ColumnDescriptor *c) : columnDesc(c), insertBuffer(nullptr) {}
				static void translateColumnDescriptorsToColumnInfoVec(const std::list<const ColumnDescriptor*> &colDescs, std::vector<ColumnInfo> &columnInfoVec) {
					for (auto cd : colDescs)
						columnInfoVec.push_back(ColumnInfo(cd));
				}
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
        int databaseId;						/// identifies the database into which the data is being inserted
        int tableId;						/// identifies the table into which the data is being inserted
        std::vector<int> columnIds;				/// a vector of column ids for the row(s) being inserted
        size_t numRows;				/// the number of rows being inserted
        std::vector <int8_t *> data;							/// points to the start of the data for the row(s) being inserted
    };
   
    /**
     * @struct PartitionInfo
     * @brief Used by Partitioner classes to store info about each
     * partition - the partition id and number of tuples(rows)
     * currently stored by that partition
     */
    
    struct PartitionInfo {
        //std::vector<int>partitionKeys;
        int partitionId;
        size_t numTuples;
        size_t shadowNumTuples;
        std::vector<int> deviceIds;
        std::map <int, ChunkMetadata> chunkMetadataMap; 
        std::map <int, ChunkMetadata> shadowChunkMetadataMap; 

        PartitionInfo(): partitionId(-1), numTuples(0),shadowNumTuples(0) {}
    };
    
    /**
     * @struct QueryInfo
     * @brief returned by Partitioner classes in
     * getPartitionsForQuery - tells Executor which
     * partitions to scan from which partitioner
     * (partitioner id and partition id needed for building
     * ChunkKey)
     */
    
    struct QueryInfo {
        std::vector <int> chunkKeyPrefix; 
        std::vector<PartitionInfo> partitions;
        size_t numTuples;
    };
   
} // Partitioner_Namespace

#endif // DATAMGR_PARTITIONER_PARTITIONER_H
