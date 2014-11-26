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
    std::vector<Page> freePages;
    mapd_size_t numPages = 1024;
    mapd_size_t pageSize = 4096;
    
    EXPECT_EQ(freePages.size(), 0);
    fm.requestFreePages(numPages, pageSize, freePages);
    EXPECT_EQ(freePages.size(), numPages);
    
}

