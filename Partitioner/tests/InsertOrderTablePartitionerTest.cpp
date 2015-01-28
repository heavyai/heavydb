/**
 * @file	InsertOrderTablePartitionerTest 
 * @author	Todd Mostak <todd@map-d.com>
 */

#include "gtest/gtest.h"
#include "../../DataMgr/DataMgr.h"
#include "../InsertOrderTablePartitioner.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <boost/filesystem.hpp>

using namespace std;

#define INSERTORDERTABLEPARTITIONER_UNIT_TESTING


GTEST_API_ int main(int argc, char **argv) {
    cout << "Deleting" << endl;
    boost::filesystem::remove_all("data");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace Partitioner_Namespace {

    class InsertOrderTablePartitionerTest : public ::testing::Test {
        public:
            InsertOrderTablePartitionerTest() {
            }

            virtual void SetUp() {
                maxPartitionRows = 1000000;
                dataMgr = new Data_Namespace::DataMgr(2,"data");
                ChunkKey chunkKeyPrefix = {0,1};
                vector <ColumnInfo> columnInfoVec;
                SQLTypeInfo sqlTypeInfo;
                sqlTypeInfo.type = kINT;
                ColumnInfo colInfo;
                colInfo.columnDesc = new ColumnDescriptor(1,0,"col1",sqlTypeInfo,kENCODING_FIXED,8);
                colInfo.insertBuffer = 0;
                columnInfoVec.push_back(colInfo);
                sqlTypeInfo.type = kFLOAT;
                colInfo.columnDesc = new ColumnDescriptor(1,1,"col2",sqlTypeInfo,kENCODING_NONE,0);
                columnInfoVec.push_back(colInfo);
                insertOrderTablePartitioner = new InsertOrderTablePartitioner(chunkKeyPrefix,columnInfoVec,dataMgr,maxPartitionRows);
            }

            virtual void TearDown() {
                delete insertOrderTablePartitioner;
                delete dataMgr;
            }
            /*
            void deleteData(const std::string &dirName) {
                boost::filesystem::remove_all(dirName);
            }
            */

            Data_Namespace::DataMgr *dataMgr;
            InsertOrderTablePartitioner *insertOrderTablePartitioner;
            int64_t maxPartitionRows;
    };

    TEST_F (InsertOrderTablePartitionerTest, insert) {

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
        insertData.data.push_back((int8_t *)intData);
        insertData.data.push_back((int8_t *)floatData);
        insertData.numRows = numRows;
        insertOrderTablePartitioner->insertData(insertData); 
        dataMgr->checkpoint();
        QueryInfo queryInfo;
        insertOrderTablePartitioner->getPartitionsForQuery(queryInfo);
        EXPECT_EQ(0,queryInfo.chunkKeyPrefix[0]);
        EXPECT_EQ(1,queryInfo.chunkKeyPrefix[1]);
        EXPECT_EQ((numRows-1)/maxPartitionRows+1,queryInfo.partitions.size());
        size_t numPartitions = queryInfo.partitions.size();
        for (size_t p = 0; p != numPartitions; ++p) {
            cout << "Num tuples: " << queryInfo.partitions[p].numTuples << endl;
            size_t expectedRows = (p == numPartitions - 1 && numRows % maxPartitionRows != 0) ? numRows % maxPartitionRows : maxPartitionRows;
            EXPECT_EQ(expectedRows,queryInfo.partitions[p].numTuples);
            EXPECT_EQ(p,queryInfo.partitions[p].partitionId);
            EXPECT_EQ(kINT,queryInfo.partitions[p].chunkMetadataMap[0].sqlType);
            EXPECT_EQ(kFLOAT,queryInfo.partitions[p].chunkMetadataMap[1].sqlType);
            EXPECT_EQ(kENCODING_FIXED,queryInfo.partitions[p].chunkMetadataMap[0].encodingType);
            EXPECT_EQ(kENCODING_NONE,queryInfo.partitions[p].chunkMetadataMap[1].encodingType);
            EXPECT_EQ(expectedRows,queryInfo.partitions[p].chunkMetadataMap[0].numElements);
            EXPECT_EQ(expectedRows,queryInfo.partitions[p].chunkMetadataMap[1].numElements);
            EXPECT_EQ(8,queryInfo.partitions[p].chunkMetadataMap[0].encodingBits);
            EXPECT_EQ(expectedRows*sizeof(int8_t),queryInfo.partitions[p].chunkMetadataMap[0].numBytes);
            EXPECT_EQ(expectedRows*sizeof(float),queryInfo.partitions[p].chunkMetadataMap[1].numBytes);
            EXPECT_EQ(0,queryInfo.partitions[p].chunkMetadataMap[0].chunkStats.min.intval);
            EXPECT_EQ(99,queryInfo.partitions[p].chunkMetadataMap[0].chunkStats.max.intval);
            ChunkKey intChunkKey = {0,1,0,queryInfo.partitions[p].partitionId}; 
            ChunkKey floatChunkKey = {0,1,1,queryInfo.partitions[p].partitionId}; 
            Data_Namespace::AbstractBuffer *intBuffer = dataMgr->getChunk(Data_Namespace::CPU_LEVEL,intChunkKey); 
            Data_Namespace::AbstractBuffer *floatBuffer = dataMgr->getChunk(Data_Namespace::CPU_LEVEL,floatChunkKey); 
            int8_t *intPtr = intBuffer -> getMemoryPtr();
            float *floatPtr = reinterpret_cast<float *> (floatBuffer -> getMemoryPtr());
            int minIntVal = std::numeric_limits<int>::max(); 
            int maxIntVal = std::numeric_limits<int>::min(); 
            float minFloatVal = std::numeric_limits<float>::max(); 
            float maxFloatVal = std::numeric_limits<float>::min(); 
            for (size_t i = 0; i < queryInfo.partitions[p].numTuples; ++i) {

                if (intPtr[i] < minIntVal)
                    minIntVal = intPtr[i];
                if (intPtr[i] > maxIntVal)
                    maxIntVal = intPtr[i];
                if (floatPtr[i] < minFloatVal)
                    minFloatVal = floatPtr[i];
                if (floatPtr[i] > maxFloatVal)
                    maxFloatVal = floatPtr[i];
            }

            cout << "Min int val: " << minIntVal << endl;
            cout << "Max int val: " << maxIntVal << endl;
            cout << "Min float val: " << minFloatVal << endl;
            cout << "Max float val: " << maxFloatVal << endl;
            EXPECT_EQ(minIntVal,queryInfo.partitions[p].chunkMetadataMap[0].chunkStats.min.intval);
            EXPECT_EQ(maxIntVal,queryInfo.partitions[p].chunkMetadataMap[0].chunkStats.max.intval);
            EXPECT_EQ(minFloatVal,queryInfo.partitions[p].chunkMetadataMap[1].chunkStats.min.floatval);
            EXPECT_EQ(maxFloatVal,queryInfo.partitions[p].chunkMetadataMap[1].chunkStats.max.floatval);
        }
        dataMgr->checkpoint();

        delete [] intData;
        delete [] floatData;
    }
    TEST_F (InsertOrderTablePartitionerTest, reopen) {

    }
    /*

    TEST_F (InsertOrderTablePartitionerTest, reopen) {
        int numRows = 50000000;
        int * intData = new int[numRows];
        float * floatData = new float[numRows];
        for (int i = 0; i < numRows; ++i) {
            intData[i] = i % 100;
            floatData[i] = M_PI * i;
        }
        QueryInfo queryInfo;
        insertOrderTablePartitioner->getPartitionsForQuery(queryInfo);
        EXPECT_EQ(0,queryInfo.chunkKeyPrefix[0]);
        EXPECT_EQ(1,queryInfo.chunkKeyPrefix[1]);
        size_t numPartitions = queryInfo.partitions.size();
        cout << "NumPartitions: " << numPartitions << endl;

        for (size_t p = 0; p != numPartitions; ++p) {
            cout << "Num tuples: " << queryInfo.partitions[p].numTuples << endl;
            size_t expectedRows = (p == numPartitions - 1 && numRows % maxPartitionRows != 0) ? numRows % maxPartitionRows : maxPartitionRows;
            EXPECT_EQ(expectedRows,queryInfo.partitions[p].numTuples);
            EXPECT_EQ(p,queryInfo.partitions[p].partitionId);
            EXPECT_EQ(kINT,queryInfo.partitions[p].chunkMetadataMap[0].sqlType);
            EXPECT_EQ(kFLOAT,queryInfo.partitions[p].chunkMetadataMap[1].sqlType);
            EXPECT_EQ(kENCODING_FIXED,queryInfo.partitions[p].chunkMetadataMap[0].encodingType);
            EXPECT_EQ(kENCODING_NONE,queryInfo.partitions[p].chunkMetadataMap[1].encodingType);
            EXPECT_EQ(expectedRows,queryInfo.partitions[p].chunkMetadataMap[0].numElements);
            EXPECT_EQ(expectedRows,queryInfo.partitions[p].chunkMetadataMap[1].numElements);
            EXPECT_EQ(8,queryInfo.partitions[p].chunkMetadataMap[0].encodingBits);
            EXPECT_EQ(expectedRows*sizeof(int8_t),queryInfo.partitions[p].chunkMetadataMap[0].numBytes);
            EXPECT_EQ(expectedRows*sizeof(float),queryInfo.partitions[p].chunkMetadataMap[1].numBytes);
            EXPECT_EQ(0,queryInfo.partitions[p].chunkMetadataMap[0].chunkStats.min.intval);
            EXPECT_EQ(99,queryInfo.partitions[p].chunkMetadataMap[0].chunkStats.max.intval);
            ChunkKey intChunkKey = {0,1,0,queryInfo.partitions[p].partitionId}; 
            ChunkKey floatChunkKey = {0,1,1,queryInfo.partitions[p].partitionId}; 
            Data_Namespace::AbstractBuffer *intBuffer = dataMgr->getChunk(Data_Namespace::CPU_LEVEL,intChunkKey); 
            Data_Namespace::AbstractBuffer *floatBuffer = dataMgr->getChunk(Data_Namespace::CPU_LEVEL,floatChunkKey); 
            int8_t *intPtr = intBuffer -> getMemoryPtr();
            float *floatPtr = reinterpret_cast<float *> (floatBuffer -> getMemoryPtr());
            int minIntVal = std::numeric_limits<int>::max(); 
            int maxIntVal = std::numeric_limits<int>::min(); 
            float minFloatVal = std::numeric_limits<float>::max(); 
            float maxFloatVal = std::numeric_limits<float>::min(); 
            for (size_t i = 0; i < queryInfo.partitions[p].numTuples; ++i) {

                if (intPtr[i] < minIntVal)
                    minIntVal = intPtr[i];
                if (intPtr[i] > maxIntVal)
                    maxIntVal = intPtr[i];
                if (floatPtr[i] < minFloatVal)
                    minFloatVal = floatPtr[i];
                if (floatPtr[i] > maxFloatVal)
                    maxFloatVal = floatPtr[i];
            }

            cout << "Min int val: " << minIntVal << endl;
            cout << "Max int val: " << maxIntVal << endl;
            cout << "Min float val: " << minFloatVal << endl;
            cout << "Max float val: " << maxFloatVal << endl;
            EXPECT_EQ(minIntVal,queryInfo.partitions[p].chunkMetadataMap[0].chunkStats.min.intval);
            EXPECT_EQ(maxIntVal,queryInfo.partitions[p].chunkMetadataMap[0].chunkStats.max.intval);
            EXPECT_EQ(minFloatVal,queryInfo.partitions[p].chunkMetadataMap[1].chunkStats.min.floatval);
            EXPECT_EQ(maxFloatVal,queryInfo.partitions[p].chunkMetadataMap[1].chunkStats.max.floatval);
        }
        delete [] intData;
        delete [] floatData;
    }
    */



} // Partitioner_Namespace
