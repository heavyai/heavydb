/**
 * @file	TablePartitionMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 */
#ifndef _TABLE_PARTITION_MGR_H
#define _TABLE_PARTITION_MGR_H

#include "Partitioner.h"
#include "AbstractTablePartitioner.h"
#include "../../Shared/types.h"
#include "../PgConnector/PgConnector.h"

#include <map>
#include <vector>

// forward declaration(s)
namespace Buffer_Namespace {
    class BufferMgr;
};

namespace Metadata_Namespace {
    class Catalog;
    struct ColumnRow;
}

namespace Partitioner_Namespace {
    
    /**
     * @class TablePartitionMgr
     * @brief A partition manager for tables in the relational data model.
     *
     */
    
    class TablePartitionMgr {
        
        friend class TablePartitionerTest;
        
    public:
        /**
         * @brief Constructor - TablePartitionMgr requires
         * references to already instanciated catalog
         * and buffer manager objects
         *
         * @param catalog reference to instanciated
         * Metadata_Namespace::Catalog object
         * @param bufferMgr  reference to instanciated
         * Buffer_Namespace::BufferMgr object
         */
        
        TablePartitionMgr(Metadata_Namespace::Catalog &catalog, Buffer_Namespace::BufferMgr &bufferMgr);
        
        /**
         * @brief Destructor - writes metadata to storage and
         * deletes partitioners allocated on heap
         */
        
        ~TablePartitionMgr();
        
        /**
         * @brief Called by executor with an optional predicate
         * to get QueryInfo object back (with ids and # tuples of
         * each partition that needs to be scanned)
         *
         * @see QueryInfo
         *
         * @param tableId id of table that query is to be run on
         * @param queryInfo filled in by method with data on
         * partitions to be scanned
         * @param predicate pointer to boolean predicate that is evauluated
         * to determine which partitions and columns thereof need
         * to be scanned.  Default NULL.
         */
        
        void getQueryPartitionInfo(const int tableId, QueryInfo &queryInfo, const void *predicate = 0);
        
        /**
         * @brief creates a partitioner for a table specified by
         * name
         *
         * Table does not have to exist in TablePartitionMgr
         * Queries catalog to get metadata about table (including
         * its tableId and types and ids of its columns
         *
         * @param tableName name of table for which partitioner
         * is to be created
         * @param partitionerType enum type of partitioner
         * to be created
         * @param maxPartitionRows maximum number of rows in
         * each partition for this partitioner (default 1,048,576)
         * @param pageSize Page/Block size for partitions of this
         * partitioner (default 1MB)
         *
         * @todo add createPartitionerForTable taking tableId instead of tableName
         * @see Catalog
         */
        
        void createPartitionerForTable (const std::string &tableName, const PartitionerType partititonerType = LINEAR, const mapd_size_t maxPartitionRows = 1048576, const mapd_size_t pageSize = 1048576);
        
        /**
         * @brief Insert data (insertDataStruct) into the table
         *
         * @param insertDataStruct metadata on the data to be
         * inserted - can handle multiple rows at once
         *
         * @see InsertData
         */
        void insertData(const InsertData &insertDataStruct);
        
        //@todo make deletePartitioner function
        
    private:
        
        /**
         * @brief given a data type (enum) - returns its size in bits
         *
         * This bitsize is needed by partitioner during data insertion
         * @param dataType gets size in bits of this data type
         */
        
        inline mapd_size_t getBitSizeForType(const mapd_data_t dataType) {
            switch (dataType) {
                case INT_TYPE:
                case FLOAT_TYPE:
                    return 32;
                    break;
                case BOOLEAN_TYPE:
                    return 1;
                    break;
            }
        }
        
        /**
         * @brief Creates partitioners table (curreintly in Postgres)
         */
        
        void createStateTableIfDne();
        
        /**
         * @brief reads metadata about partitioners from table
         * into memory * (currently partitioners table in
         * Postgres) and uses this data to recreate
         * the partitioners.
         *
         */
        void readState();
        
        /**
         * @brief updates metadata on disk - currently uses
         * postgres, wiping table and rewriting
         */
        
        void writeState();
        
        /**
         * @brief Iterate over all entries in columnRows
         * and translate to columnInfoVec needed by partitioner
         * being created
         *
         * @param columnRows vector of columnRows (generated by
         * Catalog) to be translated
         * @param columnInfoVec vector of ColumnInfo structs that
         * will be generated by translation from ColumnRow -
         * should be passed as empty vector
         *
         * @see ColumnRow
         * @see ColumnInfo
         */
        
        // columnInfoVec needed  by partitioner
        void translateColumnRowsToColumnInfoVec (std::vector <Metadata_Namespace::ColumnRow> &columnRows, std::vector<ColumnInfo> &columnInfoVec);
        
        
        int maxPartitionerId_; /**< Since each new partitioner is
                                assigned a monotonically increasing
                                id - we keep track of the maximum
                                id already assigned */
        
        /**
         * @type tableToPartitionerMMap_
         * @brief Maps table ids to TablePartitioner objects.
         *
         * The TablePartitionMgr uses this multimap in order to map a table id to multiple
         * possible TablePartitioner objects. (Note that a multimap permits the mapping of
         * a key to multiple values.)
         */
        std::multimap<int, AbstractTablePartitioner*> tableToPartitionerMMap_;
        
        Metadata_Namespace::Catalog &catalog_; /**< reference to Catalog object - must be queried to get metadata for tables and columns before partitioner creation */
        
        Buffer_Namespace::BufferMgr & bufferMgr_;									/**< reference to the buffer manager object*/
        PgConnector pgConnector_; /**<object that connects to postgres to allow metadata storage */
        
        bool isDirty_;  /**< Specifies if the TablePartitionMgr has been modified in memory since the last flush to file - no need to rewrite state if this is false. */
    };
    
} // Partitioner_Namespace

#endif // _TABLE_PARTITION_MGR_H
