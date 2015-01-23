/**
 * @file	LinearTablePartitionerTest 
 * @author	Todd Mostak <todd@map-d.com>
 */

#include "gtest/gtest.h"
#include "../../DataMgr/DataMgr.h"
#include "../LinearTablePartitioner.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <boost/filesystem.hpp>

using namespace std;

#define LINEARTABLEPARTITIONER_UNIT_TESTING

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace Partitioner_Namespace {

    class LinearTablePartitionerTest : public ::testing::Test {
        protected:
            virtual void SetUp() {
                maxPartitionRows = 1000000;
                deleteData("data");
                dataMgr = new Data_Namespace::DataMgr(2,"data");
                ChunkKey chunkKeyPrefix = {0,1,2};
                vector <ColumnInfo> columnInfoVec;
                ColumnInfo colInfo;
                colInfo.columnId = 0;
                colInfo.columnType = kINT; 
                colInfo.encodingType = kENCODING_FIXED; 
                colInfo.encodingBits = 8;
                colInfo.insertBuffer = 0;
                columnInfoVec.push_back(colInfo);
                colInfo.columnId = 1;
                colInfo.columnType = kFLOAT; 
                colInfo.encodingType = kENCODING_NONE; 
                colInfo.encodingBits = 8;
                colInfo.insertBuffer = 0;
                columnInfoVec.push_back(colInfo);
                linearTablePartitioner = new LinearTablePartitioner(chunkKeyPrefix,columnInfoVec,dataMgr,maxPartitionRows);
            }

            virtual void TearDown() {
                delete linearTablePartitioner;
                delete dataMgr;
            }

            void deleteData(const std::string &dirName) {
                boost::filesystem::remove_all(dirName);
            }

            Data_Namespace::DataMgr *dataMgr;
            LinearTablePartitioner *linearTablePartitioner;
            int64_t maxPartitionRows;
    };

    TEST_F (LinearTablePartitionerTest, insert) {

        int numRows = 50000000;
        int * intData = new int[numRows];
        float * floatData = new float[numRows];
        for (int i = 0; i < numRows; ++i) {
            intData[i] = i % 100;
            floatData[i] = M_PI * i;
        }
        InsertData insertData;
        insertData.databaseId = 0;
        insertData.tableId = 1;
        insertData.columnIds.push_back(0);
        insertData.columnIds.push_back(1);
        insertData.data.push_back((mapd_addr_t)intData);
        insertData.data.push_back((mapd_addr_t)floatData);
        insertData.numRows = numRows;
        linearTablePartitioner->insertData(insertData); 
        dataMgr->checkpoint();
        QueryInfo queryInfo;
        linearTablePartitioner->getPartitionsForQuery(queryInfo);
        EXPECT_EQ(0,queryInfo.chunkKeyPrefix[0]);
        EXPECT_EQ(1,queryInfo.chunkKeyPrefix[1]);
        EXPECT_EQ(2,queryInfo.chunkKeyPrefix[2]);
        EXPECT_EQ((numRows-1)/maxPartitionRows+1,queryInfo.partitions.size());
        size_t numPartitions = queryInfo.partitions.size();
        for (size_t p = 0; p != numPartitions; ++p) {
            cout << "Num tuples: " << queryInfo.partitions[p].numTuples << endl;
            size_t expectedRows = (p == numPartitions - 1 && numRows % maxPartitionRows != 0) ? numRows % maxPartitionRows : maxPartitionRows;
            EXPECT_EQ(expectedRows,queryInfo.partitions[p].numTuples);
        }

        delete [] intData;
        delete [] floatData;

    }

} // Partitioner_Namespace
