#include "gtest/gtest.h"
#include "../BufferMgr.h"
#include "../../FileMgr/FileMgr.h"

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

class BufferMgrTest : public ::testing::Test {
    
protected:
    virtual void SetUp() {
        const mapd_size_t pageSize = 512;
        deleteData("data");
        fm = new File_Namespace::FileMgr("data");
        bm = new BufferMgr(memSize,slabSize,pageSize,fm);
    }
    
    virtual void TearDown() {
        delete bm;
    }

    void deleteData(const std::string &dirName) {
        boost::filesystem::remove_all(dirName);
    }
    
    mapd_size_t memSize = 1 << 31; 
    mapd_size_t slabSize = 1 << 29; 
    BufferMgr *bm;
    File_Namespace::FileMgr *fm;

};


TEST_F(BufferMgrTest, Constructor)
{
    ASSERT_EQ(bm->size(), slabSize);
}

TEST_F(BufferMgrTest, createChunk)
{
    bm->clear();
    mapd_size_t pageSize = 512;
    
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

TEST_F(BufferMgrTest, createChunks)
{
    bm->clear();
    mapd_size_t pageSize = 512;
    
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

TEST_F(BufferMgrTest, readAndWrite) {
    bm->clear();
    mapd_size_t pageSize = 512;
    ChunkKey chunkKey1 = {1,2,3,4};
    //ChunkKey chunkKey2 = {2,3,4,5};
    bm -> createChunk(chunkKey1,pageSize);
    //bm -> createChunk(chunkKey2,pageSize,4096*100);
    cout << "After create" << endl;
    const size_t numInts = 1000000;
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
        chunk1 -> write((mapd_addr_t)data1,numInts*sizeof(int),0);
        double elapsedTime = double(cpuTimer.elapsed().wall) / oneSecond;
        double bandwidth = numInts * sizeof(int) / elapsedTime / 1000000000.0;
        cout << "Write Bandwidth: " << bandwidth << " GB/sec" << endl;
    }
    int * data2 = new int [numInts];
    {
        boost::timer::cpu_timer cpuTimer;
        chunk1 -> read((mapd_addr_t)data2,numInts*sizeof(int),0);
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
    diskChunk -> read((mapd_addr_t)diskData,numInts*sizeof(int),0);
    for (size_t i = 0; i < numInts; ++i) {
        EXPECT_EQ(data1[i],diskData[i]);
    }

    delete [] data1;
    delete [] data2;
    delete [] diskData;
}








