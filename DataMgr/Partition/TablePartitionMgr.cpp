/**
 * @file	TablePartitionMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 */
#include "TablePartitionMgr.h"
#include "Catalog.h"
#include <limits>

/// Searches for the table's partitioner and calls its getPartitionIds() method

TablePartitionMgr(Catalog &catalog, BufferMgr &bufferMgr): catalog_(catalog), bufferMgr_(bufferMgr), maxPartitionerId_(-1), pgConnector_("mapd","tmostak") {
    createStateTableIfDne();

}



void TablePartitionMgr::getQueryPartitionInfo(const int tableId, QueryInfo &queryInfo, const void *predicate) {
    // predicate can be null
	auto mapIt = tableToPartitionerMap_.find(tableId);
    assert (mapIt != tableToPartitionerMap_.end());

    // set numTuples to maximum allowable value given its type
    // to allow finding the partitioner that makes us scan the 
    // least number of tuples
    
    queryinfo.numTuples = std::numeric_limits<mapd_size_t>::max();


    // Iterate over each partitioner that exists for the table
    for (auto vecIt = mapIt -> second.begin(); vecIt != mapIt -> second.end(); ++vecIt) {
        QueryInfo tempQueryInfo;
        vecIt -> getPartitionsForQuery(tempQueryInfo, predicate);
        if (tempQueryInfo.numTuples < queryInfo.numTuples) 
            queryInfo = tempQueryInfo;
    }
    // At the end of this loop queryInfo should hold the metadata needed for
    // the executor for the most optimal partitioner for this query
}



void TablePartitionMgr::createPartitionerForTable(const int tableId, const PartitionType partititonType, std::vector <ColumnInfo> &columnInfoVec, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize = 1048576);

void TablePartitionMgr::insertData(const InsertData &insertDataStruct) {
	auto mapIt = tableToPartitionerMap_.find(insertDataStruct.tableId);
    assert (mapIt != tableToPartitionerMap_.end());
    // Now iterate over all partitioners for given table
    for (auto vecIt = mapIt -> second.begin(); vecIt != mapIt -> second.end(); ++vecIt) {
        vecIt -> insertData(insertDataStruct);
    }
}

void LinearTablePartitioner::createStateTableIfDne() {
     mapd_err_t status = pgConnector_.query("CREATE TABLE IF NOT EXISTS partitioners(table_id INT, part_id INT, PRIMARY KEY (table_id, part_id))");
     assert(status == MAPD_SUCCESS);
}





