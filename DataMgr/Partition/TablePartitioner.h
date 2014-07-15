/**
 * @file	TablePartitioner.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef _TABLE_PARTITIONER_H
#define _TABLE_PARTITIONER_H
#include "AbstractPartitioner.h"
#include "../../Shared/types.h"

/**
 * @brief	The TablePartitioner maps partial keys to fragment ids.
 *
 * The TablePartitioner maps partial keys to fragment ids. It's principle method
 * is getPartitionIds(), which returns a vector of ids. 
 *
 * @todo 	The TablePartitioner should have a reference to a PartitionScheme;
 *			this will be added in an upcoming release.
 */
class TablePartitioner : public AbstractPartitioner { // implements

public:
	TablePartitioner(int tableId, mapd_size_t insertBufferSize) :
		tableId_(tableId), insertBufferSize_(insertBufferSize) {}

	virtual ~TablePartitioner();

	virtual bool getPartitionIds(const void *predicate, std::vector<int> &result);

private:
	int tableId_;
	mapd_size_t insertBufferSize_;
	// PartitionSchema sch;
	
	TablePartitioner(const TablePartitioner&);
	TablePartitioner& operator=(const TablePartitioner&);

};

 #endif // _TABLE_PARTITIONER_H
