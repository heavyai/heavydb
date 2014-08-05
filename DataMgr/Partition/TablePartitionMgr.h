/**
 * @file	TablePartitionMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 */
#ifndef _TABLE_PARTITION_MGR_H
#define _TABLE_PARTITION_MGR_H

#include "AbstractTablePartitioner.h"
#include "../../Shared/types.h"
#include "../PgConnector/PgConnector.h"

#include <map>
#include <vector>

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
	int tableId;						/// identifies the table into which the data is being inserted
	std::vector<int> columnIds;				/// a vector of column ids for the row(s) being inserted
	mapd_size_t numRows;				/// the number of rows being inserted
	vector <void *> data;							/// points to the start of the data for the row(s) being inserted
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
	TablePartitionMgr(Catalog &catalog, BufferMgr &bm);

	/// Destructor
	~TablePartitionMgr();

    void createPartitionerForTable (const int tableId, const PartitionType partititonType, std::vector <ColumnInfo> &columnInfoVec, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize = 1048576);
	/// Insert the data (insertData) into the table
	void insertData(const InsertData &insertDataStruct);

	/**
	 * Obtain partition ids via a partition manager.
	 * @entityId	The id of an data entity being managed
	 * @predicate	A pointer to a predicate by which the data is partitioned
	 *
	 * @todo The type of predicate should be changed to point to the subtree of the predicate in the AST
	 */
	void getPartitionIds(const int tableId,  std::vector<int> &partitionIds, const void *predicate = 0);

private:
    int maxPartitionerId_;
	std::map<int, vector <AbstractTablePartitioner &> > tableToPartitionerMap_; 	/// maps table ids to TablePartitioner objects
    Catalog &catalog_;
	BufferMgr & bufferMgr_;									/// pointer to the buffer manager object
    PgConnector pgConnector_;
};

#endif // _TABLE_PARTITION_MGR_H
