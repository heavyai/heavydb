#include "gtest/gtest.h"
#include "../DataMgr.h"

#include <boost/filesystem.hpp>

#include <iostream>
#include <math.h>

using namespace std;

#define DATAMGR_UNIT_TESTING


GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace Data_Namespace {

    class DataMgrTest : public ::testing::Test {

        protected:
            virtual void SetUp() {
                deleteData("data");
                dataMgr = new DataMgr(2,"data");
            }

            virtual void TearDown() {
                delete dataMgr;
            }

            void deleteData(const std::string &dirName) {
                boost::filesystem::remove_all(dirName);
            }

            DataMgr *dataMgr;
    };

    TEST_F (DataMgrTest, pushWriteAndRead) {
        ChunkKey key;
        for (int i = 0; i < 3; ++i) {
            key.push_back(i);
        }
        AbstractBuffer * gpuBuffer = dataMgr->createChunk(key,GPU_LEVEL);
        const int numInts = 10000;
        int * data1 = new int [numInts];
        for (size_t i = 0; i < numInts; ++i) {
            data1[i] = i;
        }
        gpuBuffer->append((int8_t *)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        // will put gpu chunk to cpu chunk, and then cpu chunk to file chunk
        dataMgr->checkpoint();
        // will read File chunk into Cpu chunk
        AbstractBuffer * cpuBuffer = dataMgr->getChunk(key,CPU_LEVEL);
        int *data2 = new int [numInts];
        cpuBuffer->read((int8_t *)data2,numInts*sizeof(int),Data_Namespace::CPU_BUFFER,0);
        for (size_t i = 0; i < numInts; ++i) {
            EXPECT_EQ(data1[i],data2[i]);
        }

        delete [] data1;
        delete [] data2;
    }

    TEST_F (DataMgrTest, deleteChunk) {
        ChunkKey key;
        for (int i = 0; i < 3; ++i) {
            key.push_back(i);
        }
        AbstractBuffer * cpuChunk = dataMgr->createChunk(key,CPU_LEVEL);
        const int numInts = 10000;
        int * data1 = new int [numInts];
        for (size_t i = 0; i < numInts; ++i) {
            data1[i] = i;
        }
        cpuChunk->append((int8_t *)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        dataMgr->checkpoint();
        EXPECT_NO_THROW(dataMgr->getChunk(key,GPU_LEVEL));
        EXPECT_NO_THROW(dataMgr->getChunk(key,CPU_LEVEL));
        EXPECT_NO_THROW(dataMgr->getChunk(key,DISK_LEVEL));
        dataMgr->deleteChunk(key);
        EXPECT_THROW(dataMgr->getChunk(key,GPU_LEVEL), std::runtime_error);
        EXPECT_THROW(dataMgr->getChunk(key,CPU_LEVEL), std::runtime_error);
        EXPECT_THROW(dataMgr->getChunk(key,DISK_LEVEL), std::runtime_error);
        delete [] data1;
    }

    TEST_F (DataMgrTest, buffer) {
        AbstractBuffer *cpuBuffer = dataMgr->createBuffer(CPU_LEVEL,0,4096);
        const int numInts = 10000;
        int * data1 = new int [numInts];
        for (size_t i = 0; i < numInts; ++i) {
            data1[i] = i;
        }
        cpuBuffer->append((int8_t *)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        //Checkpoint should not flush a buffer - only real chunks 
        dataMgr->checkpoint();

        EXPECT_EQ(dataMgr->bufferMgrs_[1][0]->getNumChunks(),1);
        EXPECT_EQ(dataMgr->bufferMgrs_[0][0]->getNumChunks(),0);
        delete [] data1;
    }


    TEST_F (DataMgrTest, deletePrefix) {
        ChunkKey key1 = {1,2,3};
        ChunkKey key2 = {2,2,4};
        ChunkKey key3 = {1,2,4};
        ChunkKey key4 = {1,3,4};

        const int numInts = 100000;
        AbstractBuffer *chunk1 = dataMgr->createChunk(key1,CPU_LEVEL);
        AbstractBuffer *chunk2 = dataMgr->createChunk(key2,CPU_LEVEL);
        AbstractBuffer *chunk3 = dataMgr->createChunk(key3,CPU_LEVEL);
        AbstractBuffer *chunk4 = dataMgr->createChunk(key4,CPU_LEVEL);

        int * data1 = new int [numInts];
        for (size_t i = 0; i < numInts; ++i) {
            data1[i] = i;
        }
        chunk1->append((int8_t *)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        chunk2->append((int8_t *)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        chunk3->append((int8_t *)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        chunk4->append((int8_t *)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        EXPECT_EQ(4,dataMgr->bufferMgrs_[1][0]->getNumChunks());
        //EXPECT_EQ(2,dataMgr->bufferMgrs_[2][0]->getNumChunks());
        cout << "Before checkpoint" << endl;
        dataMgr->checkpoint();
        ChunkKey deletePrefix = {1,2};
        dataMgr->deleteChunksWithPrefix(deletePrefix);
        cout << "After delete" << endl;
        EXPECT_EQ(2,dataMgr->bufferMgrs_[1][0]->getNumChunks());
        //EXPECT_EQ(1,dataMgr->bufferMgrs_[2][0]->getNumChunks());
        cout << "After expect" << endl;

        EXPECT_ANY_THROW(dataMgr->getChunk(key3,GPU_LEVEL));
        EXPECT_ANY_THROW(dataMgr->getChunk(key1,GPU_LEVEL));
        EXPECT_NO_THROW(dataMgr->getChunk(key2,GPU_LEVEL));
        EXPECT_NO_THROW(dataMgr->getChunk(key4,GPU_LEVEL));
        delete [] data1;
    }

    TEST_F (DataMgrTest, encoding) {
        ChunkKey key1 = {1,2,3};
        ChunkKey key2 = {4,5,6,7};

        AbstractBuffer *gpuChunk1 = dataMgr->createChunk(key1,GPU_LEVEL);
        AbstractBuffer *gpuChunk2 = dataMgr->createChunk(key2,GPU_LEVEL);
        gpuChunk1->initEncoder(kINT,kENCODING_FIXED,8);
        EXPECT_EQ(kINT,gpuChunk1->sqlType);
        EXPECT_EQ(kENCODING_FIXED,gpuChunk1->encodingType);
        EXPECT_EQ(8,gpuChunk1->encodingBits);
        int numElems = 10000;
        int * data1 = new int [numElems];
        float * data2 = new float [numElems];
        for (size_t i = 0; i < numElems; ++i) {
            data1[i] = i % 100; // so fits in one byte
            data2[i] = M_PI * i;
        }
        int8_t *tmpPtr = reinterpret_cast<int8_t *>(data1);
        gpuChunk1->encoder->appendData(tmpPtr,numElems);
        EXPECT_EQ(numElems,gpuChunk1->size());
        EXPECT_EQ(numElems,gpuChunk1->encoder->numElems);
        dataMgr->checkpoint();
        AbstractBuffer *fileChunk1 = dataMgr->getChunk(key1,DISK_LEVEL);
        EXPECT_EQ(kINT,fileChunk1->sqlType);
        EXPECT_EQ(kENCODING_FIXED,fileChunk1->encodingType);
        EXPECT_EQ(8,fileChunk1->encodingBits);
        EXPECT_EQ(numElems,fileChunk1->size());
        EXPECT_EQ(numElems,fileChunk1->encoder->numElems);
        dataMgr->checkpoint();

        // Now lets test getChunkMetadataVec

        vector <std::pair <ChunkKey,ChunkMetadata> > chunkMetadataVec;
        dataMgr->getChunkMetadataVec(chunkMetadataVec);
        EXPECT_EQ(1,chunkMetadataVec.size());
        EXPECT_EQ(key1, chunkMetadataVec[0].first);
        ChunkMetadata chunk1Metadata = chunkMetadataVec[0].second;
        EXPECT_EQ(kINT, chunk1Metadata.sqlType);
        EXPECT_EQ(kENCODING_FIXED, chunk1Metadata.encodingType);
        EXPECT_EQ(8, chunk1Metadata.encodingBits);
        EXPECT_EQ(numElems*sizeof(int8_t), chunk1Metadata.numBytes);
        EXPECT_EQ(numElems, chunk1Metadata.numElements);
        EXPECT_EQ(0, chunk1Metadata.chunkStats.min.smallintval);
        EXPECT_EQ(99, chunk1Metadata.chunkStats.max.smallintval);


        delete [] data1;
        delete [] data2;

    }

}















