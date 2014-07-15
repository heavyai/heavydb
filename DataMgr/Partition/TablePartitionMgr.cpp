/**
 * @file	TablePartitionMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 */
#include "TablePartitionMgr.h"
#include "TablePartitioner.h"

/// Searches for the table's partitioner and calls its getPartitionIds() method
bool TablePartitionMgr::getPartitionIds(const int tableId, const void *predicate, std::vector<int> &result) {
	auto it = tableToPart_.find(tableId);
	if (it == tableToPart_.end())
		return false;
	return it->second->getPartitionIds(predicate, result);
}
