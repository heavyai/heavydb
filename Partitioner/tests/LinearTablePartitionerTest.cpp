/**
 * @file	LinearTablePartitionerTest 
 * @author	Todd Mostak <todd@map-d.com>
 */

#include "gtest/gtest.h"
#include "../../DataMgr/DataMgr.h"

#include "LinearTablePartitioner.h"
#include <iostream>
#include <vector>
#include <math.h>

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
                colInfo.insertBuffer_ = 0;
                columnInfoVec.push_back(colInfo);
                linearTablePartitioner = new LinearTablePartitioner(chunkKeyPrefix,columnInfoVec,dataMgr);
            }

            virtual void TearDown() {
                delete linearTablePartitioner;
                delete dataMgr;
            }

            void deleteData(const std::string &dirName) {
                boost::filesystem::remove_all(dirName);
            }

            DataMgr *dataMgr;
            LinearTablePartitioner *linearTablePartitioner;
    };

    TEST_F (LinearTablePartitionerTest, insert) {

        int numRows = 100;
        int * intData = new int[numRows];
        int * floatData = new float[numRows];
        for (int i = 0; i < numRows; ++i) {
            intData[i] = i;
            floatData[i] = M_PI * i;
        }
        InsertData insertData;
        insertData.databaseId = 0;
        insertData.tableId = 1;
        insertData.columnIds.push_back(0);
        insertData.columnIds.push_back(1);
        insertData.data.push_back((void*)intData);
        insertData.data.push_back((void*)floatData);
        insertData.numRows = numRows;
        linearTablePartitioner->insertData(insertData); 
    }





int main() {
   
    File_Namespace::FileMgr * fileMgr = new File_Namespace::FileMgr ("data");
    Buffer_Namespace::BufferMgr bufferMgr (128*1048576, fileMgr);
    vector <ColumnInfo> columnInfoVec;
    ColumnInfo colInfo;
    colInfo.columnId_ = 0;
    colInfo.columnType_ = INT_TYPE; 
    colInfo.bitSize_ = 32;
    colInfo.insertBuffer_ = 0;
    columnInfoVec.push_back(colInfo);
    colInfo.columnId_ = 1;
    colInfo.columnType_ = FLOAT_TYPE; 
    colInfo.bitSize_ = 32;
    colInfo.insertBuffer_ = 0;
    columnInfoVec.push_back(colInfo);
    LinearTablePartitioner linearTablePartitioner(0, columnInfoVec, bufferMgr, 1048576 );

    vector <int> columnIds; 
    vector <void *> data;
    columnIds.push_back(0);
    columnIds.push_back(1);
    int intData = 3;
    data.push_back(static_cast <void *> (&intData));
    float floatData = 7.2;
    data.push_back(static_cast <void *> (&floatData));

    for (int i = 0; i < 40; ++i) {
        for (int r = 0; r < 100000; ++r) { 
            linearTablePartitioner.insertData(columnIds,data,1);
        }
        vector <PartitionInfo> partitions;
        linearTablePartitioner.getPartitionsForQuery(partitions);
        cout << endl << "Query " << i << endl;
        for (int p = 0; p < partitions.size(); ++p)
            cout << partitions[p].partitionId_ << " " << partitions[p].numTuples_ << endl;
    }




    //linearTablePartitioner.insertData








}

