/**
 * @file	TablePartitionerTest 
 * @author	Todd Mostak <todd@map-d.com>
 */

#include "Partitioner.h"
#include "TablePartitionMgr.h"
#include "../File/FileMgr.h"
#include "../Buffer/BufferMgr.h"
#include "../Metadata/Catalog.h"

#include "../../Shared/errors.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"
#include "../../Shared/types.h"

#include <stdlib.h>

using namespace Testing;
using namespace std;

namespace Partitioner_Namespace { 

class TablePartitionerTest { 


    public:

        ~TablePartitionerTest() {
            delete tablePartitionMgr_;
            delete catalog_;
            delete bufferMgr_;
            delete fileMgr_;
        }

        void reset() {
            system("dropdb mapd");
            system("createdb mapd");
            system("rm -rf data");
            system("mkdir data");
        }

        bool instanciate() {
            fileMgr_ = new File_Namespace::FileMgr ("data");
            bufferMgr_ = new Buffer_Namespace::BufferMgr (128*1048576, fileMgr_);
            catalog_ = new Metadata_Namespace::Catalog ("data");
            tablePartitionMgr_ = new TablePartitionMgr (*catalog_, *bufferMgr_);
            Testing::pass++;
            return true;
        }

        bool createTableAndPartitioners() {
            vector <Metadata_Namespace::ColumnRow *> columnRows;
            columnRows.push_back(new Metadata_Namespace::ColumnRow("a", INT_TYPE, true));
            columnRows.push_back(new Metadata_Namespace::ColumnRow("b", INT_TYPE, false));
            columnRows.push_back(new Metadata_Namespace::ColumnRow("c", FLOAT_TYPE, false));
            mapd_err_t status = catalog_ -> addTableWithColumns("test1", columnRows);
            if (status != MAPD_SUCCESS) 
                return false;
            tablePartitionMgr_ -> createPartitionerForTable("test1",LINEAR);
            Testing::pass++;

            // now add another
            tablePartitionMgr_ -> createPartitionerForTable("test1",LINEAR);
            Metadata_Namespace::TableRow tableRow;
            status = catalog_ -> getMetadataForTable("test1",tableRow);
            if (status != MAPD_SUCCESS)
                return false;
            int tableId = tableRow.tableId; 
            
            if (tablePartitionMgr_->tableToPartitionerMMap_.count(tableId) != 2)
                return false;
            Testing::pass++;
            
            // Verify partitioner ids are different
            auto tableIt = tablePartitionMgr_->tableToPartitionerMMap_.find(tableId);
            int partId1 = tableIt->second->getPartitionerId();
            int partId2 = (++tableIt)->second->getPartitionerId();
            if (partId1 == partId2)
                return false;
            --tableIt;
            Testing::pass++;

            // Verify both partitioners are linear
            string partType1 = tableIt->second->getPartitionerType();
            string partType2 = (++tableIt)->second->getPartitionerType();
            if (partType1 != "linear" || partType2 != "linear")
                return false;
            Testing::pass++;
            return true;
    }

    bool insertIntoPartitioners() {
        vector <string> tableNames;
        tableNames.push_back("test1");
        vector <pair <string,string> > columnNames;
        columnNames.push_back(make_pair("","a"));
        columnNames.push_back(make_pair("","b"));
        columnNames.push_back(make_pair("","c"));
        vector <Metadata_Namespace::ColumnRow> columnRows;
        catalog_ -> getMetadataForColumns(tableNames, columnNames, columnRows);
        InsertData insertData;
        for (auto colRowIt = columnRows.begin(); colRowIt != columnRows.end(); ++colRowIt) {
            insertData.tableId = colRowIt -> tableId;
            insertData.columnIds.push_back(colRowIt -> columnId);
        }
        insertData.numRows = 1;
        int col0Data = 4;
        int col1Data = 6;
        float col2Data = 3.5;
        insertData.data.push_back(static_cast <void *> (&col0Data));
        insertData.data.push_back(static_cast <void *> (&col1Data));
        insertData.data.push_back(static_cast <void *> (&col2Data));
        for (int i = 0; i != 5000000; ++i)
            tablePartitionMgr_ -> insertData(insertData);
        Testing::pass++;

        return true;
    }

    
    private:
        File_Namespace::FileMgr *fileMgr_;
        Buffer_Namespace::BufferMgr * bufferMgr_;
        Metadata_Namespace::Catalog * catalog_;
        TablePartitionMgr * tablePartitionMgr_;
};



} // Partitioner_Namespace

int main(void) {
    Partitioner_Namespace::TablePartitionerTest tablePartitionerTest;
    tablePartitionerTest.reset();
    tablePartitionerTest.instanciate() ?
        PPASS("Instanciate") : PFAIL("Instanciate");
    tablePartitionerTest.createTableAndPartitioners() ?
        PPASS("Create Table and partitioners") : PFAIL("Create Table and partitioners");
    tablePartitionerTest.insertIntoPartitioners() ?
        PPASS("Insert into partitioners") : PFAIL("Insert into partitioners");

    return 0;
}
