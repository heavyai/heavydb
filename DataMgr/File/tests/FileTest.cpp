#include <cstdlib>
#include <ctime>
#include "gtest/gtest.h"
#include "../File.h"

using namespace File_Namespace;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(File, CreateOpenClose)
{
    // test invalid arguments
    EXPECT_THROW(create(0, 4096, 0), std::invalid_argument);
    EXPECT_THROW(create(0, 0, 4096), std::invalid_argument);
    EXPECT_THROW(create(0, 0, 0), std::invalid_argument);
    
    // test creating a 1GB file with 4KB block size
    int fileId = 0;
    mapd_size_t blockSize = 4096;
    mapd_size_t nblocks = 262144;
    FILE *f = create(0, blockSize, nblocks);
    EXPECT_NE(f, nullptr);
    
    // try to open, verify file size, and close the file
    ASSERT_NO_THROW(f = open(fileId));
    EXPECT_EQ(fileSize(f), blockSize * nblocks);
    EXPECT_NO_THROW(close(f));
}

TEST(File, ReadWrite)
{
    int fileId = 0;
    int blockSize = 4096;
    int nblocks = 1;
    int numInts = (blockSize * nblocks) / sizeof(int);
    int randUpper = 10;
    
    // initialize buffers with random integers
    srand((unsigned)time(0));
    int *A = new int[numInts];
    int *B = new int[numInts];
    for (int i = 0; i < numInts; ++i) {
        A[i] = (rand()%randUpper)+1;
        B[i] = -1;
    }
    
    // create a file
    FILE *f = create(fileId, blockSize, nblocks);
    
    // write buffer to the file
    size_t bytesWritten = write(f, 0, blockSize * nblocks, (mapd_addr_t)A);
    ASSERT_EQ(bytesWritten, blockSize * nblocks);
    
    // read buffer
    read(f, 0, blockSize * nblocks, (mapd_addr_t)B);
    
    // compare buffers
    for (int i = 0; i < numInts; ++i)
        ASSERT_EQ(A[i], B[i]);
    
    delete[] A;
    delete[] B;
}

