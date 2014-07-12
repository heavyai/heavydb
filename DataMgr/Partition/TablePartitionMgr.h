/**
 * @file	TablePartitionMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef _TABLE_PARTITION_MGR_H
#define _TABLE_PARTITION_MGR_H

#include <map>
#include <vector>
#include "../../Shared/types.h"

// forward declaration(s)
class BufferMgr;
class TablePartitioner;

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
	int tableId;					/// identifies the table into which the data is being inserted
	std::vector<int> colId;			/// a vector of column ids for the row(s) being inserted
	std::vector<mapd_size_t> colSize; /// the size (in bytes) of each column
	mapd_size_t numRows;			/// the number of rows being inserted
	void *data;						/// points to the start of the data for the row(s) being inserted
};

/**
 * @class TablePartitionMgr
 * @brief A partition manager for tables in the relational data model.
 *
 * An concrete partition manager that implements the AbstractPartitionMgr
 * interface for the relational table model.
 */
class TablePartitionMgr {

public:
	/// Constructor
	TablePartitionMgr(BufferMgr *bm) : bm_(bm) {}

	/// Destructor
	virtual ~TablePartitionMgr();

	/// Insert the data (insertData) into the table
	virtual void insert(const InsertData *insertData);

	/**
	 * Obtain partition ids via a partition manager.
	 * @entityId	The id of an data entity being managed
	 * @predicate	A pointer to a predicate by which the data is partitioned
	 *
	 * @todo The type of predicate should be changed to point to the subtree of the predicate in the AST
	 */
	virtual bool getPartitionIds(const int tableId, const void *predicate, std::vector<int> &result);

private:
	std::map<int, TablePartitioner*> tableToPart_; 	/// maps table ids to TablePartitioner objects
	BufferMgr *bm_;							/// pointer to the buffer manager object
};

#endif // _TABLE_PARTITION_MGR_H
