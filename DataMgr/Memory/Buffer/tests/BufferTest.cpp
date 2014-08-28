#include "gtest/gtest.h"
#include "../Buffer.h"

using namespace Buffer_Namespace;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(Buffer, Constructor)
{
    mapd_size_t numPages = 1024;
    mapd_size_t pageSize = 1048576;
    mapd_size_t memSize = numPages * pageSize;
    unsigned char *mem = new unsigned char[memSize];
    int epoch = 0;
    
    Buffer b(mem, numPages, pageSize, epoch);
    EXPECT_EQ(b.isDirty(), false);
    EXPECT_EQ(b.size(), numPages * pageSize);
    EXPECT_EQ(b.used(), 0);
}