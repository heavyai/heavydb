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

                tableDesc0.tableId = 0;
                tableDesc0.tableName = "table_0";
                tableDesc0.nColumns = 2;

                tableDesc1.tableId = 1;
                tableDesc1.tableName = "table_1";
                tableDesc1.nColumns = 3;
                SQLTypeInfo sqlTypeInfo;
                sqlTypeInfo.type = kINT;
                columnDescsTable0.push_back(new ColumnDescriptor(0,0,"col0_0",sqlTypeInfo,kENCODING_FIXED,8)); 
                cout << "Compression back: " << columnDescsTable0.back() -> compression << endl; 
                sqlTypeInfo.type = kFLOAT;
                columnDescsTable0.push_back(new ColumnDescriptor(0,1,"col0_1",sqlTypeInfo,kENCODING_NONE)); 
                /*

                sqlTypeInfo.type = kDOUBLE;
                columnDescsTable1.push_back(new ColumnDescriptor(1,0,"col1_0",sqlTypeInfo,kENCODING_NONE)); 
                sqlTypeInfo.type = kBIGINT;
                columnDescsTable1.push_back(new ColumnDescriptor(1,1,"col1_1",sqlTypeInfo,kENCODING_FIXED,32)); 
                sqlTypeInfo.type = kSMALLINT;
                columnDescsTable1.push_back(new ColumnDescriptor(1,2,"col1_2",sqlTypeInfo,kENCODING_NONE)); 
                */

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
                boost::filesystem::remove_all("mapd_partitions");
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
        int numRows = 2000000;
        const TableDescriptor * tableDesc0Ptr = &tableDesc0;
        tablePartitionerMgr->createPartitionerForTable(0,tableDesc0Ptr,columnDescsTable0);
        /*
        const TableDescriptor * tableDesc1Ptr = &tableDesc1;
        tablePartitionerMgr->createPartitionerForTable(1,tableDesc1Ptr,columnDescsTable1);
        */
        int * intData = new int[numRows];
        float * floatData = new float[numRows];
        for (int i = 0; i < numRows; ++i) {
            intData[i] = i % 100;
            floatData[i] = M_PI * i;
        }
        InsertData insertData;
        insertData.databaseId = 0;
        insertData.tableId = 0;
        insertData.columnIds.push_back(0);
        insertData.columnIds.push_back(1);
        insertData.data.push_back((int8_t *)intData);
        insertData.data.push_back((int8_t *)floatData);
        insertData.numRows = numRows;
        tablePartitionerMgr->insertData(insertData);
        dataMgr->checkpoint();
        QueryInfo queryInfo;
        tablePartitionerMgr->getQueryPartitionInfo(0,0,queryInfo);
        EXPECT_EQ(0,queryInfo.chunkKeyPrefix[0]);
        EXPECT_EQ(0,queryInfo.chunkKeyPrefix[1]);
        //EXPECT_EQ(0,queryInfo.chunkKeyPrefix[2]);
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
            ChunkKey intChunkKey = {0,0,0,0,queryInfo.partitions[p].partitionId}; 
            ChunkKey floatChunkKey = {0,0,0,1,queryInfo.partitions[p].partitionId}; 
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

} // Partitioner_Namespace
