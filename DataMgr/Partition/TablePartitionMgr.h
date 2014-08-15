/**
 * @file	TablePartitionMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 */
#ifndef _TABLE_PARTITION_MGR_H
#define _TABLE_PARTITION_MGR_H

#include "PartitionIncludes.h"
#include "AbstractTablePartitioner.h"
#include "../../Shared/types.h"
#include "../PgConnector/PgConnector.h"

#include <map>
#include <vector>

// forward declaration(s)
namespace Buffer_Namespace {
    class BufferMgr;
};

namespace Metadata_Namespace {
    class Catalog;
    struct ColumnRow;
}

namespace Partition_Namespace {

/**
 * @class TablePartitionMgr
 * @brief A partition manager for tables in the relational data model.
 *
 * An concrete partition manager that implements the AbstractPartitionMgr
 * interface for the relational table model.
 */

class TablePartitionMgr {

friend class TablePartitionerTest;

public:
	/// Constructor
	TablePartitionMgr(Metadata_Namespace::Catalog &catalog, Buffer_Namespace::BufferMgr &bm);

	/// Destructor
	~TablePartitionMgr();

    void getQueryPartitionInfo(const int tableId, QueryInfo &queryInfo, const void *predicate);
    void createPartitionerForTable (const std::string &tableName, const PartitionerType partititonerType, const mapd_size_t maxPartitionRows = 1048576, const mapd_size_t pageSize = 1048576);
	/// Insert the data (insertData) into the table
	void insertData(const InsertData &insertDataStruct);


private:

    inline mapd_size_t getBitSizeForType(const mapd_data_t dataType) {
        switch (dataType) {
            case INT_TYPE:
            case FLOAT_TYPE:
                return 32;
                break;
            case BOOLEAN_TYPE:
                return 1;
                break;
        }
    }
    void createStateTableIfDne();
    void readState();
    void writeState();
    void translateColumnRowsToColumnInfoVec (std::vector <Metadata_Namespace::ColumnRow> &columnRows, std::vector<ColumnInfo> &columnInfoVec);


    int maxPartitionerId_;
	std::map<int, std::vector <AbstractTablePartitioner *> > tableToPartitionerMap_; 	/// maps table ids to TablePartitioner objects
    Metadata_Namespace::Catalog &catalog_;
    Buffer_Namespace::BufferMgr & bufferMgr_;									/// pointer to the buffer manager object
    PgConnector pgConnector_;
    bool isDirty_;  /**< Specifies if the TablePartitionMgr has been modified in memory since the last flush to file - no need to rewrite state if this is false. */
};

} // Partition_Namespace

#endif // _TABLE_PARTITION_MGR_H
