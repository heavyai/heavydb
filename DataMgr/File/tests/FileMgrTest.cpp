#include "gtest/gtest.h"
#include "../FileMgr.h"

using namespace File_Namespace;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(FileInfo, Constructor)
{
    FILE *f = create(0, 4096, 1024);
    FileInfo fInfo(0, f, 4096, 1024);
}

TEST(FileMgr, ChunkBufferCopying)
{
    mapd_size_t blockSize = 4096;
    mapd_size_t bufSize = 1048576;
    mapd_size_t numInts = bufSize / sizeof(int);
    
    // initialize buffers with random integers
    int randUpper = 10;
    srand((unsigned)time(0));
    int *A = new int[numInts];
    int *B = new int[numInts];
    for (int i = 0; i < numInts; ++i) {
        A[i] = (rand() % randUpper)+1;
        B[i] = -1;
    }

    FileMgr fm(".");
    ChunkKey key = {0, 0, 0};
    
    fm.createChunk(key, blockSize);
    fm.copyBufferToChunk(key, bufSize, (mapd_addr_t)A);
    fm.copyChunkToBuffer(key, (mapd_addr_t)B);
    
    // compare buffers
    for (int i = 0; i < numInts; ++i)
        ASSERT_EQ(A[i], B[i]);
    
    delete[] A;
    delete[] B;

}