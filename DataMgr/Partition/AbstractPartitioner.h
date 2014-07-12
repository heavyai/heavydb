/**
 * @file	AbstractPartitionMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef _ABSTRACT_PARTITIONER_H
#define _ABSTRACT_PARTITIONER_H

#include <vector>

class AbstractPartitioner {

public:
	/// Destructor
	virtual ~AbstractPartitioner() = 0;

	/**
 	 * Returns the ids of partitions given the partialKey. The partialKey is the
 	 * client's way of querying the Partitioner object.
 	 *
 	 * @param ids	A vector of int of column identifirs
	 */
	virtual bool getPartitionIds(const void *predicate, std::vector<int> &result);
};

 #endif // _ABSTRACT_PARTITIONER_H
