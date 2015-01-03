#include "gtest/gtest.h"
#include "../CpuBufferMgr.h"
#include "../../../FileMgr/FileMgr.h"

#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>

#include <iostream>

using namespace Buffer_Namespace;
using namespace std;

#define BUFFERMGR_UNIT_TESTING

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class CpuBufferMgrTest : public ::testing::Test {
    
protected:
    virtual void SetUp() {
        deleteData("data");
        fm = new File_Namespace::FileMgr("data");
        bm = new CpuBufferMgr(memSize,CPU_HOST,slabSize,pageSize,fm);
    }
    
    virtual void TearDown() {
        delete bm;
    }

    void deleteData(const std::string &dirName) {
        boost::filesystem::remove_all(dirName);
    }
    
    mapd_size_t memSize = 1 << 31; 
    mapd_size_t slabSize = 1 << 28; 
    mapd_size_t pageSize = 512;
    CpuBufferMgr *bm;
    File_Namespace::FileMgr *fm;

};


TEST_F(CpuBufferMgrTest, Constructor)
{
    ASSERT_EQ(bm->size(), 0);
    ASSERT_EQ(bm->size(), 0);
}

TEST_F(CpuBufferMgrTest, createChunk)
{
    bm->clear();
    
    ChunkKey key;

    for (int i = 0; i < 3; ++i)
        key.push_back(i);
    bm->printSegs(); 
    bm->createChunk(key, pageSize,4096);
    //bm->printSegs(); 
    bm->printMap();
    // should not be able to call createChunk with the same key again
    EXPECT_THROW(bm->createChunk(key, pageSize), std::runtime_error);
    //bm->printSegs(); 
    
    // should be able to delete the Chunk and then create it again
    EXPECT_NO_THROW(bm->deleteChunk(key));
    bm->printSegs(); 
    bm->printMap();
    EXPECT_NO_THROW(bm->createChunk(key, pageSize));
    bm->printSegs(); 
    
}

TEST_F(CpuBufferMgrTest, createChunks)
{
    bm->clear();
    
    for (int c = 0; c < 1000; c++) {
        ChunkKey key;
        for (int i = 0; i < 3; ++i) {
            key.push_back(c+i);
        }
        bm->createChunk(key,pageSize,4096);
    }
    ChunkKey key1 {1000,1001,1002};
    bm->createChunk(key1,pageSize,2048);
    ChunkKey key2 {2000,2001,2002};
    bm->createChunk(key2,pageSize,8704);
    ChunkKey key3 {3000,3001,3002};
    bm->createChunk(key3,pageSize,2500);
    ChunkKey key0 {0,1,2};
    //EXPECT_THROW(bm->deleteChunk(key0),std::runtime_error);
    bm->printSegs(); 
    bm->printMap();
}

TEST_F(CpuBufferMgrTest, readAndWrite) {
    bm->clear();
    ChunkKey chunkKey1 = {1,2,3,4};
    //ChunkKey chunkKey2 = {2,3,4,5};
    bm -> createChunk(chunkKey1,pageSize);
    //bm -> createChunk(chunkKey2,pageSize,4096*100);
    cout << "After create" << endl;
    const size_t numInts = 12500000;
    int * data1 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    cout << "After initializing data" << endl;
    AbstractBuffer *chunk1 = bm -> getChunk(chunkKey1);
    cout << "After gotChunk" << endl;
    const boost::timer::nanosecond_type oneSecond(1000000000LL);
    {
        boost::timer::cpu_timer cpuTimer;
        chunk1 -> write((mapd_addr_t)data1,numInts*sizeof(int),Memory_Namespace::CPU_BUFFER,0);
        double elapsedTime = double(cpuTimer.elapsed().wall) / oneSecond;
        double bandwidth = numInts * sizeof(int) / elapsedTime / 1000000000.0;
        cout << "Write Bandwidth: " << bandwidth << " GB/sec" << endl;
    }
    int * data2 = new int [numInts];
    {
        boost::timer::cpu_timer cpuTimer;
        chunk1 -> read((mapd_addr_t)data2,numInts*sizeof(int),Memory_Namespace::CPU_BUFFER,0);
        double elapsedTime = double(cpuTimer.elapsed().wall) / oneSecond;
        double bandwidth = numInts * sizeof(int) / elapsedTime / 1000000000.0;
        cout << "Read Bandwidth: " << bandwidth << " GB/sec" << endl;
    }
    for (size_t i = 0; i < numInts; ++i) {
        EXPECT_EQ(data1[i],data2[i]);
    }
    bm -> checkpoint();
    bm -> clear();
    cout << "Before get data" << endl;
    AbstractBuffer * diskChunk = bm -> getChunk(chunkKey1,numInts*sizeof(int));
    cout << "Got data" << endl;
    int *diskData = new int [numInts];
    diskChunk -> read((mapd_addr_t)diskData,numInts*sizeof(int),Memory_Namespace::CPU_BUFFER,0);
    for (size_t i = 0; i < numInts; ++i) {
        EXPECT_EQ(data1[i],diskData[i]);
    }

    delete [] data1;
    delete [] data2;
    delete [] diskData;
}

TEST_F(CpuBufferMgrTest, slabAllocationTest) {
    bm->clear();
    //EXPECT_EQ(bm->slabs_.size(),1);
    EXPECT_EQ(bm->size(), 0);
    ChunkKey chunkKey1 = {1,2,3,4};
    ChunkKey chunkKey2 = {2,3,4,5};
    ChunkKey chunkKey3 = {3,4,5,6};
    ChunkKey chunkKey4 = {4,5,6,7};
    ChunkKey chunkKey5 = {5,6,7,8};
    size_t chunkSize = slabSize - 2048;
    bm -> createChunk(chunkKey1, pageSize, chunkSize);
    EXPECT_EQ(slabSize, bm->size());
    AbstractBuffer *chunk1 = bm -> getChunk(chunkKey1);
    chunk1 -> reserve(chunkSize+1024); // Should use existing allocation and extend it
    EXPECT_EQ(slabSize, bm->size());
    bm -> createChunk(chunkKey2, pageSize, chunkSize);
    EXPECT_EQ(slabSize*2, bm->size());
    bm -> createChunk(chunkKey3, pageSize, 2048);
    EXPECT_EQ(slabSize*2, bm->size());
    bm -> deleteChunk(chunkKey1);
    EXPECT_EQ(slabSize*2, bm->size());
    bm -> createChunk(chunkKey4, pageSize, slabSize);
    EXPECT_EQ(slabSize*2, bm->size());
    bm -> createChunk(chunkKey5, pageSize, slabSize);
    EXPECT_EQ(slabSize*3, bm->size());
    

    //Migrate to new slab
    bm->clear();
    EXPECT_EQ(bm->size(), 0);
    bm -> createChunk(chunkKey1, pageSize, chunkSize);
    EXPECT_EQ(slabSize, bm->size());
    bm -> createChunk(chunkKey3, pageSize, 2048);
    EXPECT_EQ(slabSize*1, bm->size());
    chunk1 = bm -> getChunk(chunkKey1);
    chunk1 -> reserve(chunkSize+1024); // Should use existing allocation and extend it
    EXPECT_EQ(slabSize*2, bm->size());
}

TEST_F(CpuBufferMgrTest, appendTest) {
    bm->clear();
    ChunkKey chunkKey1 = {1,2,3,4};
    int numInts = 2000;
    size_t chunk1Size = numInts*sizeof(int);
    int * data1 = new int [numInts * 2];
    int * data2 = new int [numInts * 2];
    for (int i = 0; i < numInts * 2; ++i) {
        data1[i] = i;
    }
    bm -> createChunk(chunkKey1, pageSize, chunk1Size);
    AbstractBuffer *chunk1 = bm -> getChunk(chunkKey1);
    chunk1 -> append((mapd_addr_t)data1,chunk1Size,Memory_Namespace::CPU_BUFFER);
    chunk1 -> read((mapd_addr_t)data2,chunk1Size, Memory_Namespace::CPU_BUFFER,0);
    for (int i = 0; i < numInts; ++i) {
        EXPECT_EQ(data2[i],data1[i]);
    }

    chunk1 -> append((mapd_addr_t)data1+numInts*sizeof(int),chunk1Size,Memory_Namespace::CPU_BUFFER);
    chunk1 -> read((mapd_addr_t)data2,chunk1Size*2, Memory_Namespace::CPU_BUFFER,0);
    for (int i = 0; i < numInts * 2; ++i) {
        EXPECT_EQ(data1[i],data2[i]);
    }



}

TEST_F(CpuBufferMgrTest, bufferMoveTest) {
    bm->clear();
    ChunkKey chunkKey1 = {1,2,3,4};
    ChunkKey chunkKey2 = {5,6,7,8};
    size_t chunk1Size = 1000*sizeof(int);
    size_t chunk2Size = 500*sizeof(int);
    bm -> createChunk(chunkKey1, pageSize, chunk1Size);
    AbstractBuffer *chunk1 = bm -> getChunk(chunkKey1);
    const size_t numInts = 2000;
    int * data1 = new int [numInts];
    int * data2 = new int [numInts];
    int * data3 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    chunk1 -> append((mapd_addr_t)data1,chunk1Size,Memory_Namespace::CPU_BUFFER);
    chunk1 -> read((mapd_addr_t)data2,chunk1Size, Memory_Namespace::CPU_BUFFER,0);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(data1[i],data2[i]);
    }

    // Now create chunk right after it
    AbstractBuffer *chunk2 = bm -> createChunk(chunkKey2, pageSize, chunk2Size);
    chunk2 -> append((mapd_addr_t)data1,chunk2Size,Memory_Namespace::CPU_BUFFER);

    // now add to chunk 1 - should be 2 X chunk1 now 
    chunk1 -> append((mapd_addr_t)(data1)+chunk1Size,chunk1Size,Memory_Namespace::CPU_BUFFER);
    chunk1 -> read((mapd_addr_t)data3,chunk1Size*2, Memory_Namespace::CPU_BUFFER,0);

    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(data1[i],data3[i]);
    }
}














