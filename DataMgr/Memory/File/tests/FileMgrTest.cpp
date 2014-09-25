#include <cstdlib>
#include <ctime>
#include <vector>

#include "gtest/gtest.h"
#include "../FileMgr.h"

using namespace File_Namespace;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(FileMgr, getFreeBlocks)
{
    FileMgr fm(".");
    std::vector<Block> freeBlocks;
    mapd_size_t nblocks = 1024;
    mapd_size_t blockSize = 4096;
    
    EXPECT_EQ(freeBlocks.size(), 0);
    fm.requestFreeBlocks(nblocks, blockSize, freeBlocks);
    EXPECT_EQ(freeBlocks.size(), nblocks);
    
}

