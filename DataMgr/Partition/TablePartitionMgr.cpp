/**
 * @file	TablePartitionMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 */
#include "TablePartitionMgr.h"
#include "Catalog.h"
#include "BufferMgr.h"
#include "LinearTablePartitioner.h"

#include <limits>
#include <boost/lexical_cast.hpp>

using std::vector;
using std::string;

/// Searches for the table's partitioner and calls its getPartitionIds() method

TablePartitionMgr::TablePartitionMgr(Metadata_Namespace::Catalog &catalog, Buffer_Namespace::BufferMgr &bufferMgr): catalog_(catalog), bufferMgr_(bufferMgr), maxPartitionerId_(-1), pgConnector_("mapd","tmostak"), isDirty_(false) {
    createStateTableIfDne();
    readState();
}

TablePartitionMgr::~TablePartitionMgr() {
    writeState();
    // now delete the Partitioners allocated on heap
    for (auto partMapIt = tableToPartitionerMap_.begin(); partMapIt != tableToPartitionerMap_.end(); ++partMapIt) {
        for (auto partIt = (partMapIt -> second).begin(); partIt != (partMapIt -> second).end(); ++partIt) {
            delete *partIt;
        }
    }
}

void TablePartitionMgr::getQueryPartitionInfo(const int tableId, QueryInfo &queryInfo, const void *predicate) {
    // predicate can be null
	auto mapIt = tableToPartitionerMap_.find(tableId);
    assert (mapIt != tableToPartitionerMap_.end());

    // set numTuples to maximum allowable value given its type
    // to allow finding the partitioner that makes us scan the 
    // least number of tuples
    
    queryInfo.numTuples = std::numeric_limits<mapd_size_t>::max();


    // Iterate over each partitioner that exists for the table
    for (auto vecIt = mapIt -> second.begin(); vecIt != mapIt -> second.end(); ++vecIt) {
        QueryInfo tempQueryInfo;
        AbstractTablePartitioner * abstractTablePartitioner = *vecIt;
        abstractTablePartitioner -> getPartitionsForQuery(tempQueryInfo, predicate);
        if (tempQueryInfo.numTuples < queryInfo.numTuples) 
            queryInfo = tempQueryInfo;
    }
    // At the end of this loop queryInfo should hold the metadata needed for
    // the executor for the most optimal partitioner for this query
}



void TablePartitionMgr::createPartitionerForTable(const int tableId, const PartitionerType partitionerType, const mapd_size_t maxPartitionRows, const mapd_size_t pageSize) {
    // need to query catalog for needed metadata
    vector <Metadata_Namespace::ColumnRow> columnRows;
    mapd_err_t status = catalog_.getAllColumnMetadataForTable(tableId, columnRows);
    assert(status == MAPD_SUCCESS);
    vector<ColumnInfo> columnInfoVec;
    translateColumnRowsToColumnInfoVec(columnRows, columnInfoVec);

    maxPartitionerId_++;
    AbstractTablePartitioner * partitioner;
    assert (partitionerType == LINEAR);
    if (partitionerType == LINEAR) {
        partitioner = new LinearTablePartitioner(maxPartitionerId_, columnInfoVec, bufferMgr_);
    }
    auto partMapIt = tableToPartitionerMap_.find(tableId);
    if (partMapIt != tableToPartitionerMap_.end()) {
        (partMapIt -> second).push_back(partitioner);
    }
    else { // first time we've seen this table id
        vector <AbstractTablePartitioner *> partitionerVec;
        partitionerVec.push_back(partitioner);
        tableToPartitionerMap_[tableId] = partitionerVec;
    }
    isDirty_ = true;
}

void TablePartitionMgr::insertData(const InsertData &insertDataStruct) {
	auto mapIt = tableToPartitionerMap_.find(insertDataStruct.tableId);
    assert (mapIt != tableToPartitionerMap_.end());
    // Now iterate over all partitioners for given table
    for (auto vecIt = mapIt -> second.begin(); vecIt != mapIt -> second.end(); ++vecIt) {
        AbstractTablePartitioner * abstractTablePartitioner = *vecIt;

        abstractTablePartitioner -> insertData(insertDataStruct);
    }
}

void TablePartitionMgr::createStateTableIfDne() {
     mapd_err_t status = pgConnector_.query("CREATE TABLE IF NOT EXISTS partitioners(table_id INT, partitioner_id INT, partitioner_type text, PRIMARY KEY (table_id, partitioner_id))");
     assert(status == MAPD_SUCCESS);
}

void TablePartitionMgr::readState() {
    string partitionerQuery("SELECT table_id, partitioner_id, partitioner_type FROM partitioners ORDER BY table_id, partitioner_id");
    mapd_err_t status = pgConnector_.query(partitionerQuery);
    assert(status == MAPD_SUCCESS);
    size_t numRows = pgConnector_.getNumRows();
    int prevTableId = -1;
    vector <AbstractTablePartitioner *> partitionerVec_;
    vector<ColumnInfo> columnInfoVec;
    for (int r = 0; r < numRows; ++r) {
        int tableId = pgConnector_.getData<int>(r,0);
        if (tableId != prevTableId && partitionerVec_.size() > 0) {
            tableToPartitionerMap_[prevTableId] = partitionerVec_;
            partitionerVec_.clear(); // will this not delete 
            prevTableId = tableId;
            vector <Metadata_Namespace::ColumnRow> columnRows;
            catalog_.getAllColumnMetadataForTable(tableId, columnRows);
            columnInfoVec.clear();
            translateColumnRowsToColumnInfoVec(columnRows, columnInfoVec);
        }
        int partitionerId = pgConnector_.getData<int>(r,1);
        if (partitionerId > maxPartitionerId_)
            maxPartitionerId_ = partitionerId;
        string partitionerType = pgConnector_.getData<string>(r,2);
        if (partitionerType == "linear") {
            partitionerVec_.push_back(new LinearTablePartitioner(partitionerId, columnInfoVec, bufferMgr_)); 
        }
    }
}

void TablePartitionMgr::writeState() {
    if (isDirty_) { // only need to rewrite state if we've made modifications since last write
        string deleteQuery ("DELETE FROM partitioners");
        mapd_err_t status = pgConnector_.query(deleteQuery);
        assert(status == MAPD_SUCCESS);
        for (auto partMapIt = tableToPartitionerMap_.begin(); partMapIt != tableToPartitionerMap_.end(); ++partMapIt) {
            string tableId = boost::lexical_cast<string>(partMapIt -> first);
            for (auto partIt = (partMapIt -> second).begin(); partIt != (partMapIt -> second).end(); ++partIt) {
                AbstractTablePartitioner *abstractTablePartitioner = *partIt;

                int partitionerId = abstractTablePartitioner -> getPartitionerId();
                string partitionerType = abstractTablePartitioner -> getPartitionerType();
                string insertQuery("INSERT INTO partitioners (table_id, partitioner_id, partitioner_type) VALUES (" + boost::lexical_cast<string>(tableId) + "," + boost::lexical_cast<string>(partitionerId) + ",'" + partitionerType + "'");
                mapd_err_t status = pgConnector_.query(insertQuery);
                assert (status == MAPD_SUCCESS);
            }
        }
        isDirty_ = false;
    }
}

void TablePartitionMgr::translateColumnRowsToColumnInfoVec (vector <Metadata_Namespace::ColumnRow> &columnRows, vector<ColumnInfo> &columnInfoVec) {
    for (auto colRowIt = columnRows.begin(); colRowIt != columnRows.end(); ++colRowIt) {
        ColumnInfo columnInfo;
        columnInfo.columnId = colRowIt -> columnId;
        columnInfo.columnType = colRowIt -> columnType;
        columnInfo.bitSize = getBitSizeForType(columnInfo.columnType);  
        columnInfo.insertBuffer = 0;
        columnInfoVec.push_back(columnInfo);
    }
}
