/**
 * @file	TablePartitionerMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 */
#include "TablePartitionerMgr.h"
//#include "Catalog.h"
#include "../DataMgr/DataMgr.h"
#include "InsertOrderTablePartitioner.h"
#include "../Catalog/TableDescriptor.h"
#include "../Catalog/ColumnDescriptor.h"

#include <limits>
#include <boost/lexical_cast.hpp>
#include <iostream>

using std::vector;
using std::list;
using std::string;
using std::pair;
using std::cout;
using std::endl;

namespace Partitioner_Namespace {

/// Searches for the table's partitioner and calls its getPartitionIds() method

TablePartitionerMgr::TablePartitionerMgr(Data_Namespace::DataMgr *dataMgr): dataMgr_(dataMgr) {
    //dataMgr_-> getChunkMetadataVec(chunkMetadataVec);
    /*
    ChunkKey lastChunkKey;
    vector <int> lastPartitionerKey;
    //Chunk key should be database_id, table_id, partitioner_id, column_id, fragment_id (fragment_id could be 2 keys)
    // first three keys (database_id, table_id, partitioner_id make up
    // a partitioner
    vector <int> lastPartitionerKey;
    vector <ColumnInfo> columnInfoVec;
    int lastColumnId = -1;

    for (auto chunkIt = chunkMetadataVec.begin(); chunkIt != chunkMetadataVec.end(); ++chunkIt) {
        ChunkKey partitionerKey = vector <int> (chunkIt -> first -> begin(); chunkIt -> first -> begin() + 3); // tableKey will be database id and tableid
        int columnId =   
        if (partitionerKey != lastPartitionerKey) {


        }
    }
    */
}

TablePartitionerMgr::~TablePartitionerMgr() {
    //writeState();
    // now delete the Partitioners allocated on heap
    /*
    for (auto partMapIt = tableToPartitionerMMap_.begin(); partMapIt != tableToPartitionerMMap_.end(); ++partMapIt)
        delete partMapIt->second;
    */
}

/// Notes: predicate can be null; queryInfo should hold the metadata needed for
/// the executor for the most optimal partitioner for this query
void TablePartitionerMgr::getQueryPartitionInfo(const int databaseId, const int tableId, QueryInfo &queryInfo/*, const void *predicate*/) {
    // obtain iterator over partitions for the given tableId
    ChunkKey tableKey = {databaseId, tableId};
    //auto mapIt = tableToPartitionerMMap_.find(tableKey);
    auto mapRangeIt = tableToPartitionerMMap_.equal_range(tableKey); 
    //assert (mapIt != tableToPartitionerMMap_.end());

    // set numTuples to maximum allowable value given its type
    // to allow finding the partitioner that makes us scan the 
    // least number of tuples
    queryInfo.numTuples = std::numeric_limits<mapd_size_t>::max();

    // iterate over each partitioner that exists for the table,
    // obtaining a QueryInfo object for the least number of tuples
    for (auto mapIt = mapRangeIt.first; mapIt != mapRangeIt.second; ++mapIt) {
        //cout << "Table Key: " << mapIt->first[0] << endl;
        assert(tableKey == mapIt->first);
        QueryInfo tempQueryInfo;
        AbstractTablePartitioner *abstractTablePartitioner = mapIt->second;
        abstractTablePartitioner->getPartitionsForQuery(tempQueryInfo/*, predicate*/);
        if (tempQueryInfo.numTuples < queryInfo.numTuples) {
            queryInfo = tempQueryInfo;
        }
    }
}

void TablePartitionerMgr::createPartitionerForTable (const int databaseId, const TableDescriptor *tableDescriptor, const list<const ColumnDescriptor*> &columnDescriptors) {
    int32_t tableId = tableDescriptor -> tableId;
    vector<ColumnInfo> columnInfoVec;
    translateColumnDescriptorsToColumnInfoVec(columnDescriptors, columnInfoVec);
    AbstractTablePartitioner *partitioner = nullptr;
    ChunkKey tablePrefix = {databaseId, tableId};
    ChunkKey chunkKeyPrefix = tablePrefix;
    if (tableDescriptor->fragType == INSERT_ORDER) {
        partitioner = new InsertOrderTablePartitioner(chunkKeyPrefix, columnInfoVec, dataMgr_);
    }
    assert (tableDescriptor->fragType == INSERT_ORDER); // only INSERT_ORDER is currently supported
    
    auto partMapIt = tableToPartitionerMMap_.find(tablePrefix);
    if (partMapIt != tableToPartitionerMMap_.end()) {
        //@todo we need to copy existing partitioner's data to this partitioner
    }
    tableToPartitionerMMap_.insert(std::pair<ChunkKey, AbstractTablePartitioner*>(tablePrefix, partitioner));
}

void TablePartitionerMgr::insertData(const InsertData &insertDataStruct) {
    ChunkKey tableKey = {insertDataStruct.databaseId, insertDataStruct.tableId};
	auto mapIt = tableToPartitionerMMap_.find(tableKey);
    printf("# of partitioners = %lu\n", tableToPartitionerMMap_.count(tableKey));
    assert (mapIt != tableToPartitionerMMap_.end());
    
    // Now iterate over all partitioners for given table
    for (; mapIt != tableToPartitionerMMap_.end(); ++mapIt) {
        AbstractTablePartitioner *abstractTablePartitioner = mapIt->second;
        abstractTablePartitioner->insertData(insertDataStruct);
    }
}

void TablePartitionerMgr::translateColumnDescriptorsToColumnInfoVec (const list <const ColumnDescriptor *> &columnDescriptors, vector<ColumnInfo> &columnInfoVec) {
    // Iterate over all entries in columnRows and translate to 
    // columnInfoVec needed  by partitioner
    for (auto colDescIt = columnDescriptors.begin(); colDescIt != columnDescriptors.end(); ++colDescIt) {
        ColumnInfo columnInfo;
        columnInfo.columnId = (*colDescIt)->columnId;
        columnInfo.columnType = (*colDescIt)->columnType.type;
        columnInfo.encodingType = (*colDescIt)->compression;
        columnInfo.encodingBits = (*colDescIt)->comp_param;
        //columnInfo.bitSize = getBitSizeForType(columnInfo.columnType);  
        columnInfo.insertBuffer = NULL; // set as NULL
        //ColumnInfo columnInfo (colDescIt -> columnId, colDescIt -> columnType, getBitSizeForType(columnInfo.columnType));
        columnInfoVec.push_back(columnInfo);
    }
}



//void TablePartitionerMgr::translateChunkMetadataVectoColumnInfoVec(const std::vector<std::pair <ChunkKey,ChunkMetadata> &chunkMetadataVec, vector<ColumnInfo> &columnInfoVec) {
//    for (auto chunkIt = chunkMetadataVec.begin(); chunkIt != chunkMetadataVec.end(); ++chunkIt) {
//        ColumnInfo columnInfo;
//        columnInfo.columnId = 
//
//    }
//
//}





//void TablePartitionerMgr::createStateTableIfDne() {
//     mapd_err_t status = pgConnector_.query("CREATE TABLE IF NOT EXISTS partitioners(table_id INT, partitioner_id INT, partitioner_type text, PRIMARY KEY (table_id, partitioner_id))");
//     assert(status == MAPD_SUCCESS);
//}
//
//void TablePartitionerMgr::readState() {
//    vector <AbstractTablePartitioner*> partitionerVec_;
//    vector<ColumnInfo> columnInfoVec;
//
//    // query to obtain table partitioner information
//    string partitionerQuery("SELECT table_id, partitioner_id, partitioner_type FROM partitioners ORDER BY table_id, partitioner_id");
//    mapd_err_t status = pgConnector_.query(partitionerQuery);
//    assert(status == MAPD_SUCCESS);
//    size_t numRows = pgConnector_.getNumRows();
//
//    // traverse query results, inserting tableId/partitioner entries into tableToPartitionerMMap_
//    for (int r = 0; r < numRows; ++r) {
//        
//        // read results of query into local variables
//        int tableId = pgConnector_.getData<int>(r,0);
//        int partitionerId = pgConnector_.getData<int>(r,1);
//        string partitionerType = pgConnector_.getData<string>(r,2);
//        
//        // update max partition id based on those read in from the query
//        maxPartitionerId_ = std::max(maxPartitionerId_, partitionerId);
//
//        // obtain metadata for the columns of the table
//        vector<const Catalog_Namespace::ColumnDescriptor *> columnDescriptors = catalog_.getAllColumnMetadataForTable(tableId);
//        columnInfoVec.clear();
//        translateColumnDescriptorsToColumnInfoVec(columnDescriptors, columnInfoVec);
//        
//        // instantitate the table partitioner object for the given tableId and partitionerType
//        AbstractTablePartitioner *partitioner;
//        if (partitionerType == "linear")
//            partitioner = new LinearTablePartitioner(partitionerId, columnInfoVec, bufferMgr_);
//        assert(partitionerType == "linear");
//        
//        // insert entry into tableToPartitionerMMap_ for the given tableId/partitioner pair
//        tableToPartitionerMMap_.insert(std::pair<int, AbstractTablePartitioner*>(tableId, partitioner));
//
//    }
//}
//
//void TablePartitionerMgr::writeState() {
//    if (isDirty_) { // only need to rewrite state if we've made modifications since last write
//        
//        // submit query to clear the partitioners table
//        string deleteQuery ("DELETE FROM partitioners");
//        mapd_err_t status = pgConnector_.query(deleteQuery);
//        assert(status == MAPD_SUCCESS);
//        
//        // traverse the partitioners stored in the multimap
//        for (auto it = tableToPartitionerMMap_.begin(); it != tableToPartitionerMMap_.end(); ++it) {
//            
//            // gather values for the partitioner's insert query
//            string tableId = boost::lexical_cast<string>(it->first);
//            AbstractTablePartitioner *abstractTablePartitioner = it->second;
//            int partitionerId = abstractTablePartitioner->getPartitionerId();
//            string partitionerType = abstractTablePartitioner->getPartitionerType();
//            
//            // submit query to insert record
//            string insertQuery("INSERT INTO partitioners (table_id, partitioner_id, partitioner_type) VALUES (" + tableId + "," + boost::lexical_cast<string>(partitionerId) + ",'" + partitionerType + "')");
//            mapd_err_t status = pgConnector_.query(insertQuery);
//            assert (status == MAPD_SUCCESS);
//            
//        }
//        isDirty_ = false; // now that we've written our state to disk, our metadata isn't dirty anymore
//    }
//}


} // Partitioner_Namespace
