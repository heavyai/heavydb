/**
 * @file	TablePartitionMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 */
#include "TablePartitionMgr.h"
#include "Catalog.h"
#include "AbstractDataMgr.h"
#include "LinearTablePartitioner.h"

#include <limits>
#include <boost/lexical_cast.hpp>

using std::vector;
using std::string;

namespace Partitioner_Namespace {

/// Searches for the table's partitioner and calls its getPartitionIds() method

TablePartitionMgr::TablePartitionMgr(Catalog_Namespace::Catalog &catalog, Memory_Namespace::AbstractDataMgr &bufferMgr): catalog_(catalog), bufferMgr_(bufferMgr), maxPartitionerId_(-1), pgConnector_("mapd","mapd"), isDirty_(false) {
    createStateTableIfDne();
    readState();
}

TablePartitionMgr::~TablePartitionMgr() {
    writeState();
    // now delete the Partitioners allocated on heap
    for (auto partMapIt = tableToPartitionerMMap_.begin(); partMapIt != tableToPartitionerMMap_.end(); ++partMapIt)
        delete partMapIt->second;
}


/// Notes: predicate can be null; queryInfo should hold the metadata needed for
/// the executor for the most optimal partitioner for this query
void TablePartitionMgr::getQueryPartitionInfo(const int tableId, QueryInfo &queryInfo, const void *predicate) {
    // obtain iterator over partitions for the given tableId
    auto mapIt = tableToPartitionerMMap_.find(tableId);
    assert (mapIt != tableToPartitionerMMap_.end());

    // set numTuples to maximum allowable value given its type
    // to allow finding the partitioner that makes us scan the 
    // least number of tuples
    queryInfo.numTuples = std::numeric_limits<mapd_size_t>::max();

    // iterate over each partitioner that exists for the table,
    // obtaining a QueryInfo object for the least number of tuples
    for (; mapIt != tableToPartitionerMMap_.end(); ++mapIt) {
        assert(tableId == mapIt->first);
        QueryInfo tempQueryInfo;
        AbstractTablePartitioner *abstractTablePartitioner = mapIt->second;
        abstractTablePartitioner->getPartitionsForQuery(tempQueryInfo, predicate);
        if (tempQueryInfo.numTuples <= queryInfo.numTuples)
            queryInfo = tempQueryInfo;
    }
}

void TablePartitionMgr::createPartitionerForTable(const string &tableName, const PartitionerType partitionerType, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize) {
    const Catalog_Namespace::TableDescriptor *tableDescriptor = catalog_.getMetadataForTable(tableName);
    int32_t tableId = tableDescriptor -> tableId;



    // need to query catalog for needed metadata
    vector <const Catalog_Namespace::ColumnDescriptor * > columnDescriptors = catalog_.getAllColumnMetadataForTable(tableId);
    vector<ColumnInfo> columnInfoVec;
    translateColumnDescriptorsToColumnInfoVec(columnDescriptors, columnInfoVec);
    maxPartitionerId_++; // id for the new partitioner
    AbstractTablePartitioner *partitioner = nullptr;
    if (partitionerType == LINEAR) {
        partitioner = new LinearTablePartitioner(maxPartitionerId_, columnInfoVec, bufferMgr_);
    }
    assert (partitionerType == LINEAR); // only LINEAR is currently supported
    
    auto partMapIt = tableToPartitionerMMap_.find(tableId);
    if (partMapIt != tableToPartitionerMMap_.end()) {
        //@todo we need to copy existing partitioner's data to this partitioner
    }
    tableToPartitionerMMap_.insert(std::pair<int, AbstractTablePartitioner*>(tableId, partitioner));
    
    // metadata has changed so needs to be written to disk by next checkpoint
    isDirty_ = true;
}

void TablePartitionMgr::insertData(const InsertData &insertDataStruct) {
	auto mapIt = tableToPartitionerMMap_.find(insertDataStruct.tableId);
    printf("# of partitioners = %lu\n", tableToPartitionerMMap_.count(insertDataStruct.tableId));
    assert (mapIt != tableToPartitionerMMap_.end());
    
    // Now iterate over all partitioners for given table
    for (; mapIt != tableToPartitionerMMap_.end(); ++mapIt) {
        AbstractTablePartitioner *abstractTablePartitioner = mapIt->second;
        abstractTablePartitioner->insertData(insertDataStruct);
    }
}

void TablePartitionMgr::createStateTableIfDne() {
     mapd_err_t status = pgConnector_.query("CREATE TABLE IF NOT EXISTS partitioners(table_id INT, partitioner_id INT, partitioner_type text, PRIMARY KEY (table_id, partitioner_id))");
     assert(status == MAPD_SUCCESS);
}

void TablePartitionMgr::readState() {
    vector <AbstractTablePartitioner*> partitionerVec_;
    vector<ColumnInfo> columnInfoVec;

    // query to obtain table partitioner information
    string partitionerQuery("SELECT table_id, partitioner_id, partitioner_type FROM partitioners ORDER BY table_id, partitioner_id");
    mapd_err_t status = pgConnector_.query(partitionerQuery);
    assert(status == MAPD_SUCCESS);
    size_t numRows = pgConnector_.getNumRows();

    // traverse query results, inserting tableId/partitioner entries into tableToPartitionerMMap_
    for (int r = 0; r < numRows; ++r) {
        
        // read results of query into local variables
        int tableId = pgConnector_.getData<int>(r,0);
        int partitionerId = pgConnector_.getData<int>(r,1);
        string partitionerType = pgConnector_.getData<string>(r,2);
        
        // update max partition id based on those read in from the query
        maxPartitionerId_ = std::max(maxPartitionerId_, partitionerId);

        // obtain metadata for the columns of the table
        vector<const Catalog_Namespace::ColumnDescriptor *> columnDescriptors = catalog_.getAllColumnMetadataForTable(tableId);
        columnInfoVec.clear();
        translateColumnDescriptorsToColumnInfoVec(columnDescriptors, columnInfoVec);
        
        // instantitate the table partitioner object for the given tableId and partitionerType
        AbstractTablePartitioner *partitioner;
        if (partitionerType == "linear")
            partitioner = new LinearTablePartitioner(partitionerId, columnInfoVec, bufferMgr_);
        assert(partitionerType == "linear");
        
        // insert entry into tableToPartitionerMMap_ for the given tableId/partitioner pair
        tableToPartitionerMMap_.insert(std::pair<int, AbstractTablePartitioner*>(tableId, partitioner));

    }
}

void TablePartitionMgr::writeState() {
    if (isDirty_) { // only need to rewrite state if we've made modifications since last write
        
        // submit query to clear the partitioners table
        string deleteQuery ("DELETE FROM partitioners");
        mapd_err_t status = pgConnector_.query(deleteQuery);
        assert(status == MAPD_SUCCESS);
        
        // traverse the partitioners stored in the multimap
        for (auto it = tableToPartitionerMMap_.begin(); it != tableToPartitionerMMap_.end(); ++it) {
            
            // gather values for the partitioner's insert query
            string tableId = boost::lexical_cast<string>(it->first);
            AbstractTablePartitioner *abstractTablePartitioner = it->second;
            int partitionerId = abstractTablePartitioner->getPartitionerId();
            string partitionerType = abstractTablePartitioner->getPartitionerType();
            
            // submit query to insert record
            string insertQuery("INSERT INTO partitioners (table_id, partitioner_id, partitioner_type) VALUES (" + tableId + "," + boost::lexical_cast<string>(partitionerId) + ",'" + partitionerType + "')");
            mapd_err_t status = pgConnector_.query(insertQuery);
            assert (status == MAPD_SUCCESS);
            
        }
        isDirty_ = false; // now that we've written our state to disk, our metadata isn't dirty anymore
    }
}

void TablePartitionMgr::translateColumnDescriptorsToColumnInfoVec (vector <const Catalog_Namespace::ColumnDescriptor *> &columnDescriptors, vector<ColumnInfo> &columnInfoVec) {
    // Iterate over all entries in columnRows and translate to 
    // columnInfoVec needed  by partitioner
    for (auto colDescIt = columnDescriptors.begin(); colDescIt != columnDescriptors.end(); ++colDescIt) {
        ColumnInfo columnInfo;
        columnInfo.columnId = (*colDescIt) -> columnId;
        columnInfo.columnType = (*colDescIt) -> columnType;
        columnInfo.bitSize = getBitSizeForType(columnInfo.columnType);  
        columnInfo.insertBuffer = NULL; // set as NULL
        //ColumnInfo columnInfo (colDescIt -> columnId, colDescIt -> columnType, getBitSizeForType(columnInfo.columnType));
        columnInfoVec.push_back(columnInfo);
    }
}

} // Partitioner_Namespace
