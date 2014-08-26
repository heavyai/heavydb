#include "gtest/gtest.h"
#include "../File.h"

using namespace File_Namespace;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(File, create)
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

