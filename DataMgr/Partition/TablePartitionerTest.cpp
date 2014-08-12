/**
 * @file	TablePartitionerTest 
 * @author	Todd Mostak <todd@map-d.com>
 */

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

        bool createTable() {
            vector <Metadata_Namespace::ColumnRow *> columnRows;
            columnRows.push_back(new Metadata_Namespace::ColumnRow("a", INT_TYPE, true));
            columnRows.push_back(new Metadata_Namespace::ColumnRow("b", INT_TYPE, false));
            columnRows.push_back(new Metadata_Namespace::ColumnRow("c", FLOAT_TYPE, false));
            mapd_err_t status = catalog_ -> addTableWithColumns("test1", columnRows);
            if (status != MAPD_SUCCESS) 
                return false;
            tablePartitionMgr_ -> createPartitionerForTable("test1",LINEAR);
            Testing::pass++;
            return true;
        }



    private:
        File_Namespace::FileMgr *fileMgr_;
        Buffer_Namespace::BufferMgr * bufferMgr_;
        Metadata_Namespace::Catalog * catalog_;
        TablePartitionMgr * tablePartitionMgr_;
};


int main(void) {
    TablePartitionerTest tablePartitionerTest;
    tablePartitionerTest.reset();
    tablePartitionerTest.instanciate() ?
        PPASS("Instanciate") : PFAIL("Instanciate");
    tablePartitionerTest.createTable() ?
        PPASS("Create Table") : PFAIL("Create Table");
    /*
    File_Namespace::FileMgr * fileMgr = new File_Namespace::FileMgr ("data");
    Buffer_Namespace::BufferMgr bufferMgr (128*1048576, fileMgr);
    Metadata_Namespace::Catalog catalog ("data");
    TablePartitionMgr tablePartitionMgr(catalog, bufferMgr);
    vector <Metadata_Namespace::ColumnRow *> columnRows;
    columnRows.push_back(new Metadata_Namespace::ColumnRow("a", INT_TYPE, true));
    columnRows.push_back(new Metadata_Namespace::ColumnRow("b", INT_TYPE, false));
    columnRows.push_back(new Metadata_Namespace::ColumnRow("c", FLOAT_TYPE, false));
    mapd_err_t status = catalog.addTableWithColumns("test1", columnRows);
    if (status != MAPD_SUCCESS) 
        return false;
    tablePartitionMgr.createPartitionerForTable("test1",LINEAR);
    vector <int> columnIds; 
    vector <void *> data;
    columnIds.push_back(0);
    columnIds.push_back(1);
    int intData = 3;
    data.push_back(static_cast <void *> (&intData));
    float floatData = 7.2;
    data.push_back(static_cast <void *> (&floatData));
    */
}
