#include "gtest/gtest.h"
#include "../BufferMgr.h"

using namespace Buffer_Namespace;

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
    
    mapd_size_t memSize = 1024*1048576; // 1GB
    BufferMgr *bm;

};

TEST_F(BufferMgrTest, Constructor)
{
    ASSERT_EQ(bm->size(), memSize);
}

TEST_F(BufferMgrTest, createChunk)
{
    mapd_size_t pageSize = 4096;
    
    ChunkKey key;
    for (int i = 0; i < 3; ++i)
        key.push_back(i);
    
    bm->createChunk(key, pageSize);
    
    // should not be able to call createChunk with the same key again
    EXPECT_THROW(bm->createChunk(key, pageSize), std::runtime_error);
    
    // should be able to delete the Chunk and then create it again
    EXPECT_NO_THROW(bm->deleteChunk(key));
    EXPECT_NO_THROW(bm->createChunk(key, pageSize));
    
    
    
}