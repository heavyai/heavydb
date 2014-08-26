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