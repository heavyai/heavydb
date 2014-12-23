#include "gtest/gtest.h"
#include "../BufferMgr.h"

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
        bm = new BufferMgr(memSize);
    }
    
    virtual void TearDown() {
        delete bm;
    }
    
    mapd_size_t memSize = 512*100; 
    BufferMgr *bm;

};

TEST_F(BufferMgrTest, Constructor)
{
    ASSERT_EQ(bm->size(), memSize);
}

TEST_F(BufferMgrTest, createChunk)
{
    mapd_size_t pageSize = 512;
    
    ChunkKey key;

    for (int i = 0; i < 3; ++i)
        key.push_back(i);
    bm->printSegs(); 
    bm->createChunk(key, pageSize,4096);
    //bm->printSegs(); 
    bm->printMap();
    // should not be able to call createChunk with the same key again
    //EXPECT_THROW(bm->createChunk(key, pageSize), std::runtime_error);
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
    mapd_size_t pageSize = 512;
    
    for (int c = 1; c <= 100; c++) {
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
    bm->printSegs(); 
    bm->printMap();
}


