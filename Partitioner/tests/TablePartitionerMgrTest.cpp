/**
 * @file	TablePartitionerMgrTest 
 * @author	Todd Mostak <todd@mapd.com>
 */

#include "gtest/gtest.h"
#include "../../DataMgr/DataMgr.h"
#include "../TablePartitionerMgr.h"
#include "../InsertOrderTablePartitioner.h"
#include "../../Catalog/TableDescriptor.h"
#include "../../Catalog/ColumnDescriptor.h"


#include <iostream>
#include <list>
#include <math.h>
#include <boost/filesystem.hpp>

using namespace std;

#define TABLEPARTITIONERMGR_UNIT_TESTING

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace Partitioner_Namespace {

    class TablePartitionerMgrTest : public ::testing::Test {
        protected:
            virtual void SetUp() {
                maxPartitionRows = 1000000;
                deleteData("data");
                dataMgr = new Data_Namespace::DataMgr(2,"data");
                ChunkKey chunkKeyPrefix = {0,1,2};

                tableDesc0.tableId = 0;
                tableDesc0.tableName = "table_0";
                tableDesc0.nColumns = 2;

                tableDesc1.tableId = 1;
                tableDesc1.tableName = "table_1";
                tableDesc1.nColumns = 3;
                SQLTypeInfo sqlTypeInfo;
                sqlTypeInfo.type = kINT;
                columnDescsTable0.push_back(new ColumnDescriptor(0,0,"col0_0",sqlTypeInfo,kENCODING_FIXED,8)); 
                sqlTypeInfo.type = kFLOAT;
                columnDescsTable0.push_back(new ColumnDescriptor(0,1,"col0_1",sqlTypeInfo,kENCODING_NONE)); 

                sqlTypeInfo.type = kDOUBLE;
                columnDescsTable1.push_back(new ColumnDescriptor(1,0,"col1_0",sqlTypeInfo,kENCODING_NONE)); 
                sqlTypeInfo.type = kBIGINT;
                columnDescsTable1.push_back(new ColumnDescriptor(1,1,"col1_1",sqlTypeInfo,kENCODING_FIXED,32)); 
                sqlTypeInfo.type = kSMALLINT;
                columnDescsTable1.push_back(new ColumnDescriptor(1,2,"col1_2",sqlTypeInfo,kENCODING_NONE)); 

                tablePartitionerMgr = new TablePartitionerMgr(dataMgr);
            }

            virtual void TearDown() {
                delete tablePartitionerMgr;
                delete dataMgr;
                for (auto colDesc0It = columnDescsTable0.begin(); colDesc0It != columnDescsTable0.end(); ++ colDesc0It) { 
                    delete *colDesc0It;
                }
                for (auto colDesc1It = columnDescsTable1.begin(); colDesc1It != columnDescsTable1.end(); ++ colDesc1It) { 
                    delete *colDesc1It;
                }
            }

            void deleteData(const std::string &dirName) {
                boost::filesystem::remove_all(dirName);
            }

            Data_Namespace::DataMgr *dataMgr;
            TablePartitionerMgr *tablePartitionerMgr;
            InsertOrderTablePartitioner *insertOrderTablePartitioner;
            TableDescriptor tableDesc0;
            TableDescriptor tableDesc1;
            list <const ColumnDescriptor *> columnDescsTable0;
            list <const ColumnDescriptor *> columnDescsTable1;
            int64_t maxPartitionRows;
    };

    TEST_F (TablePartitionerMgrTest, createPartitionerTest) {
        const TableDescriptor * tableDesc = &tableDesc0;
        tablePartitionerMgr->createPartitionerForTable(0,tableDesc,columnDescsTable0);
    }

} // Partitioner_Namespace
