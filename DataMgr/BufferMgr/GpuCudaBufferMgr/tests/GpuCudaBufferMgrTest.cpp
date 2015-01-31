#include "gtest/gtest.h"
#include "../GpuCudaBufferMgr.h"
#include "../../CpuBufferMgr/CpuBufferMgr.h"
#include "../../../../CudaMgr/CudaMgr.h"

#include <boost/timer/timer.hpp>

#include <iostream>

using namespace Buffer_Namespace;
using namespace std;

#define BUFFERMGR_UNIT_TESTING

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class GpuCudaBufferMgrTest : public ::testing::Test {
    
protected:
    virtual void SetUp() {
        cudaMgr = new CudaMgr_Namespace::CudaMgr();
        cpuBufferMgr = new CpuBufferMgr(memSize,CUDA_HOST,cudaMgr);
        gpuBufferMgr = new GpuCudaBufferMgr(memSize,0,cudaMgr, slabSize,pageSize,cpuBufferMgr);
    }
    
    virtual void TearDown() {
        delete gpuBufferMgr;
        delete cpuBufferMgr;
    }
    
    size_t memSize = 1 << 31; 
    size_t slabSize = 1 << 27; 
    size_t pageSize = 512;
    GpuCudaBufferMgr *gpuBufferMgr;
    CpuBufferMgr *cpuBufferMgr;
    CudaMgr_Namespace::CudaMgr *cudaMgr;

};


TEST_F(GpuCudaBufferMgrTest, Constructor)
{
    ASSERT_EQ(gpuBufferMgr->size(), 0);
}

TEST_F(GpuCudaBufferMgrTest, createChunk)
{
    gpuBufferMgr->clear();
    
    ChunkKey key;

    for (int i = 0; i < 3; ++i)
        key.push_back(i);
    gpuBufferMgr->printSegs(); 
    gpuBufferMgr->createChunk(key, pageSize,4096);
    //gpuBufferMgr->printSegs(); 
    gpuBufferMgr->printMap();
    // should not be able to call createChunk with the same key again
    EXPECT_THROW(gpuBufferMgr->createChunk(key, pageSize), std::runtime_error);
    //gpuBufferMgr->printSegs(); 
    
    // should be able to delete the Chunk and then create it again
    EXPECT_NO_THROW(gpuBufferMgr->deleteChunk(key));
    gpuBufferMgr->printSegs(); 
    gpuBufferMgr->printMap();
    EXPECT_NO_THROW(gpuBufferMgr->createChunk(key, pageSize));
    gpuBufferMgr->printSegs(); 
    
}

TEST_F(GpuCudaBufferMgrTest, createChunks)
{
    gpuBufferMgr->clear();
    
    for (int c = 0; c < 1000; c++) {
        ChunkKey key;
        for (int i = 0; i < 3; ++i) {
            key.push_back(c+i);
        }
        gpuBufferMgr->createChunk(key,pageSize,4096);
    }
    ChunkKey key1 {1000,1001,1002};
    gpuBufferMgr->createChunk(key1,pageSize,2048);
    ChunkKey key2 {2000,2001,2002};
    gpuBufferMgr->createChunk(key2,pageSize,8704);
    ChunkKey key3 {3000,3001,3002};
    gpuBufferMgr->createChunk(key3,pageSize,2500);
    ChunkKey key0 {0,1,2};
    //EXPECT_THROW(gpuBufferMgr->deleteChunk(key0),std::runtime_error);
    gpuBufferMgr->printSegs(); 
    gpuBufferMgr->printMap();
}

TEST_F(GpuCudaBufferMgrTest, readAndWrite) {
    gpuBufferMgr->clear();
    ChunkKey chunkKey1 = {1,2,3,4};
    //ChunkKey chunkKey2 = {2,3,4,5};
    gpuBufferMgr -> createChunk(chunkKey1,pageSize);
    //gpuBufferMgr -> createChunk(chunkKey2,pageSize,4096*100);
    cout << "After create" << endl;
    const size_t numInts = 12500000;
    int * data1 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    cout << "After initializing data" << endl;
    AbstractBuffer *chunk1 = gpuBufferMgr -> getChunk(chunkKey1);
    cout << "After gotChunk" << endl;
    const boost::timer::nanosecond_type oneSecond(1000000000LL);
    {
        boost::timer::cpu_timer cpuTimer;
        chunk1 -> write((int8_t *)data1,numInts*sizeof(int),CPU_BUFFER,0);
        double elapsedTime = double(cpuTimer.elapsed().wall) / oneSecond;
        double bandwidth = numInts * sizeof(int) / elapsedTime / 1000000000.0;
        cout << "Write Bandwidth: " << bandwidth << " GB/sec" << endl;
    }
    int * data2 = new int [numInts];
    {
        boost::timer::cpu_timer cpuTimer;
        chunk1 -> read((int8_t *)data2,numInts*sizeof(int),CPU_BUFFER,0);
        double elapsedTime = double(cpuTimer.elapsed().wall) / oneSecond;
        double bandwidth = numInts * sizeof(int) / elapsedTime / 1000000000.0;
        cout << "Read Bandwidth: " << bandwidth << " GB/sec" << endl;
    }
    for (size_t i = 0; i < numInts; ++i) {
        EXPECT_EQ(data1[i],data2[i]);
    }
    /*
    gpuBufferMgr -> checkpoint();
    gpuBufferMgr -> clear();
    cout << "Before get data" << endl;
    AbstractBuffer * diskChunk = gpuBufferMgr -> getChunk(chunkKey1,numInts*sizeof(int));
    cout << "Got data" << endl;
    int *diskData = new int [numInts];
    diskChunk -> read((int8_t *)diskData,numInts*sizeof(int),0);
    for (size_t i = 0; i < numInts; ++i) {
        EXPECT_EQ(data1[i],diskData[i]);
    }
    */

    delete [] data1;
    delete [] data2;
}

TEST_F(GpuCudaBufferMgrTest, slabAllocationTest) {
    gpuBufferMgr->clear();
    //EXPECT_EQ(gpuBufferMgr->slabs_.size(),1);
    EXPECT_EQ(gpuBufferMgr->size(), 0);
    ChunkKey chunkKey1 = {1,2,3,4};
    ChunkKey chunkKey2 = {2,3,4,5};
    ChunkKey chunkKey3 = {3,4,5,6};
    ChunkKey chunkKey4 = {4,5,6,7};
    ChunkKey chunkKey5 = {5,6,7,8};
    size_t chunkSize = slabSize - 2048;
    gpuBufferMgr -> createChunk(chunkKey1, pageSize, chunkSize);
    EXPECT_EQ(slabSize, gpuBufferMgr->size());
    AbstractBuffer *chunk1 = gpuBufferMgr -> getChunk(chunkKey1);
    chunk1 -> reserve(chunkSize+1024); // Should use existing allocation and extend it
    EXPECT_EQ(slabSize, gpuBufferMgr->size());
    gpuBufferMgr -> createChunk(chunkKey2, pageSize, chunkSize);
    EXPECT_EQ(slabSize*2, gpuBufferMgr->size());
    gpuBufferMgr -> createChunk(chunkKey3, pageSize, 2048);
    EXPECT_EQ(slabSize*2, gpuBufferMgr->size());
    gpuBufferMgr -> deleteChunk(chunkKey1);
    EXPECT_EQ(slabSize*2, gpuBufferMgr->size());
    gpuBufferMgr -> createChunk(chunkKey4, pageSize, slabSize);
    EXPECT_EQ(slabSize*2, gpuBufferMgr->size());
    gpuBufferMgr -> createChunk(chunkKey5, pageSize, slabSize);
    EXPECT_EQ(slabSize*3, gpuBufferMgr->size());
    

    //Migrate to new slab
    gpuBufferMgr->clear();
    EXPECT_EQ(gpuBufferMgr->size(), 0);
    gpuBufferMgr -> createChunk(chunkKey1, pageSize, chunkSize);
    EXPECT_EQ(slabSize, gpuBufferMgr->size());
    gpuBufferMgr -> createChunk(chunkKey3, pageSize, 2048);
    EXPECT_EQ(slabSize*1, gpuBufferMgr->size());
    chunk1 = gpuBufferMgr -> getChunk(chunkKey1);
    chunk1 -> reserve(chunkSize+1024); // Should use existing allocation and extend it
    EXPECT_EQ(slabSize*2, gpuBufferMgr->size());
}

TEST_F(GpuCudaBufferMgrTest, appendTest) {
    gpuBufferMgr->clear();
    ChunkKey chunkKey1 = {1,2,3,4};
    int numInts = 2000;
    size_t chunk1Size = numInts*sizeof(int);
    int * data1 = new int [numInts * 2];
    int * data2 = new int [numInts * 2];
    for (int i = 0; i < numInts * 2; ++i) {
        data1[i] = i;
    }
    gpuBufferMgr -> createChunk(chunkKey1, pageSize, chunk1Size);
    AbstractBuffer *chunk1 = gpuBufferMgr -> getChunk(chunkKey1);
    chunk1 -> append((int8_t *)data1,chunk1Size,Data_Namespace::CPU_BUFFER);
    chunk1 -> read((int8_t *)data2,chunk1Size, Data_Namespace::CPU_BUFFER,0);
    for (int i = 0; i < numInts; ++i) {
        EXPECT_EQ(data2[i],data1[i]);
    }

    chunk1 -> append((int8_t *)data1+numInts*sizeof(int),chunk1Size,Data_Namespace::CPU_BUFFER);
    chunk1 -> read((int8_t *)data2,chunk1Size*2, Data_Namespace::CPU_BUFFER,0);
    for (int i = 0; i < numInts * 2; ++i) {
        EXPECT_EQ(data1[i],data2[i]);
    }



}

TEST_F(GpuCudaBufferMgrTest, bufferMoveTest) {
    gpuBufferMgr->clear();
    ChunkKey chunkKey1 = {1,2,3,4};
    ChunkKey chunkKey2 = {5,6,7,8};
    size_t chunk1Size = 1000*sizeof(int);
    size_t chunk2Size = 500*sizeof(int);
    gpuBufferMgr -> createChunk(chunkKey1, pageSize, chunk1Size);
    AbstractBuffer *chunk1 = gpuBufferMgr -> getChunk(chunkKey1);
    const size_t numInts = 2000;
    int * data1 = new int [numInts];
    int * data2 = new int [numInts];
    int * data3 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    chunk1 -> append((int8_t *)data1,chunk1Size,Data_Namespace::CPU_BUFFER);
    chunk1 -> read((int8_t *)data2,chunk1Size, Data_Namespace::CPU_BUFFER,0);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(data1[i],data2[i]);
    }

    // Now create chunk right after it
    AbstractBuffer *chunk2 = gpuBufferMgr -> createChunk(chunkKey2, pageSize, chunk2Size);
    chunk2 -> append((int8_t *)data1,chunk2Size,Data_Namespace::CPU_BUFFER);

    // now add to chunk 1 - should be 2 X chunk1 now 
    chunk1 -> append((int8_t *)(data1)+chunk1Size,chunk1Size,Data_Namespace::CPU_BUFFER);
    chunk1 -> read((int8_t *)data3,chunk1Size*2, Data_Namespace::CPU_BUFFER,0);

    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(data1[i],data3[i]);
    }
}








