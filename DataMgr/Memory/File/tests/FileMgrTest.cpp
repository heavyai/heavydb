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

TEST(FileMgr, getFreePages)
{
    FileMgr fm(".");
    std::vector<Page> freePages;
    mapd_size_t numPages = 2048;
    mapd_size_t pageSize = 4096;
    
    EXPECT_EQ(freePages.size(), 0);
    fm.requestFreePages(numPages, pageSize, freePages);
    EXPECT_EQ(freePages.size(), numPages);
    
}

TEST(FileMgr, getFreePage)
{
    FileMgr fm(".");
    mapd_size_t pageSize = 1024796;
    Page page = fm.requestFreePage(pageSize);
    EXPECT_EQ(page.isValid(),true);
}
