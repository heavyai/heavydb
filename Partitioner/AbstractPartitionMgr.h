/**
 * @file	AbstractPartitionMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef DATAMGR_PARTITION_ABSTRACTPARTITIONER_H
#define DATAMGR_PARTITION_ABSTRACTPARTITIONER_H

namespace Partitioner_Namespace { 

/**
 * @class AbstractPartitionMgr
 * @brief An abstract class for a partition manager.
 *
 * This is an abstract class that specifies the basic interface for a
 * partition manager. Concrete partition manager classes, which are
 * specialized for some data model used by Map-D, will implement this
 * interface. The use of pure virtual methods (i.e., "= 0") ensures
 * that any concrete class implementing this interface will be forced
 * by the compiler to implement these methods.
 *
 */
class AbstractPartitionMgr {

public:
	/// Destructor
	virtual ~AbstractPartitionMgr() = 0;

	/// Insert data via a partition manager
	virtual void insert(const void *insertData) = 0;

	/**
	 * Obtain partition ids via a partition manager.
	 * @entityId	The id of an data entity being managed
	 * @predicate	A pointer to a predicate by which the data is partitioned
	 */
	virtual void getPartitionIds(const int entityId, const void *predicate, std::vector<int> &result) = 0;
};

} // Partitioner_Namespace

#endif // DATAMGR_PARTITION_ABSTRACTPARTITIONER_H
