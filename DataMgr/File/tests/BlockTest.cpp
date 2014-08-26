#include "gtest/gtest.h"
#include "../Block.h"

using namespace File_Namespace;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(Block, Constructor)
{
    Block b(0, 0);
    EXPECT_EQ(b.fileId, 0);
    EXPECT_EQ(b.blockNum, 0);
    EXPECT_EQ(b.used, 0);
}

TEST(MultiBlock, Constructor)
{
    MultiBlock m(1024);
    EXPECT_EQ(m.blockSize, 1024);
    EXPECT_EQ(m.blkVersions.size(), 0);
    EXPECT_EQ(m.epochs.size(), 0);
    EXPECT_THROW(m.current(), std::runtime_error);
    EXPECT_THROW(m.pop(), std::runtime_error);
}

TEST(MultiBlock, pushAndPop) {
    MultiBlock m(1024);
    int loopCount = 2;
    
    // push()
    for (int i = 0; i < loopCount; ++i) {
        m.push(0, i, i);
        EXPECT_EQ(m.blkVersions.size(), i + 1);
        EXPECT_EQ(m.epochs.size(), i + 1);
        EXPECT_EQ(m.current()->fileId, 0);
        ASSERT_EQ(m.current()->blockNum, i);
        EXPECT_EQ(m.current()->used, 0);
    }
    
    // pop()
    for (int i = loopCount; i > 0; --i) {
        EXPECT_NO_THROW(m.pop());
        EXPECT_EQ(m.blkVersions.size(), i-1);
        EXPECT_EQ(m.epochs.size(), i-1);
    }
}