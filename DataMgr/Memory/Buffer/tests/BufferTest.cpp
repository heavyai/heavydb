#include "gtest/gtest.h"
#include "../Buffer.h"

using namespace Buffer_Namespace;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class BufferTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        numPages = 1024;
        pageSize = 4096;
        memSize = numPages * pageSize;
        int epoch = 0;
        mem = new unsigned char[memSize];
        b1 = new Buffer(mem, numPages, pageSize, epoch);
    }
    
    virtual void TearDown() {
        delete b1;
        delete mem;
    }

    unsigned char *mem;
    Buffer *b1;
    mapd_size_t numPages;
    mapd_size_t pageSize;
    mapd_size_t memSize;
};

TEST_F(BufferTest, Constructor)
{
    EXPECT_EQ(b1->isDirty(), false);
    EXPECT_EQ(b1->size(), numPages * pageSize);
    EXPECT_EQ(b1->used(), 0);
}

TEST_F(BufferTest, readAndWrite)
{
    mapd_size_t numInts = memSize/sizeof(int);
    int randUpper = 10;

    // reading from an empty buffer should throw an exception
    int *buf = new int[numInts];
    EXPECT_THROW(b1->read((mapd_addr_t)buf, 0, 0), std::runtime_error);
    EXPECT_THROW(b1->read((mapd_addr_t)buf, 0, memSize + 1), std::runtime_error);
    delete[] buf;

    // initialize buffers with random integers
    srand((unsigned)time(0));
    int *A = new int[numInts];
    int *B = new int[numInts];
    for (int i = 0; i < numInts; ++i) {
        A[i] = (rand()%randUpper)+1;
        B[i] = -1;
    }
    
    // write A to the buffer, then read it into B
    b1->write((mapd_addr_t)A, 0, memSize);
    b1->read((mapd_addr_t)B, 0, memSize);
    
    // compare buffers
    for (int i = 0; i < numInts; ++i)
        ASSERT_EQ(A[i], B[i]);
}

