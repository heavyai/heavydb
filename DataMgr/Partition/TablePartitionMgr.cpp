/**
 * @file	TablePartitionMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 */
#include "TablePartitionMgr.h"
#include "TablePartitioner.h"

/// Searches for the table's partitioner and calls its getPartitionIds() method
void TablePartitionMgr::getPartitionIds(const int tableId,  std::vector<int> &partitionIds, const void *predicate) {
    // predicate can be null
	auto it = tableToPartitionerMap_.find(tableId);
    assert (it != tableToPartitionerMap_.end());
	//if (it == tableToPart_.end())
		//return false;
	it->second.getPartitionIds(partitionIds, predicate);
}

void TablePartitionMgr::createPartitionerForTable(const int tableId, const PartitionType partititonType, std::vector <ColumnInfo> &columnInfoVec, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize = 1048576);

void TablePartitionMgr::insertData(const InsertData &insertDataStruct) {
	auto it = tableToPartitionerMap_.find(insertDataStruct.tableId);
    assert (it != tableToPartitionerMap_.end());
    it -> second.insertData(insertDataStruct);
}





