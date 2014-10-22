#include <cstdlib>
#include <ctime>
#include "gtest/gtest.h"
#include "../FileBuffer.h"
#include "../FileMgr.h"

using namespace File_Namespace;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(FileBuffer, read_and_write)
{
    mapd_size_t numPages = 1;
    mapd_size_t pageSize = 5;
    FileMgr fm(".");
    FileBuffer fb(pageSize, &fm);
    
    // create some data
    mapd_size_t numInt = numPages * (pageSize / sizeof(int));
    int data1[numInt];
    for (int i = 0; i < numInt; ++i)
        data1[i] = i*2;
    
    // write the data to the buffer
    mapd_addr_t p_data1 = (mapd_addr_t)data1;
    fb.write(p_data1, 0, numPages * pageSize);
    
    // read the data back
    //fb.read(mapd_addr_t const dst, const mapd_size_t offset, const mapd_size_t nbytes = 0);
    int data2[numInt];
    mapd_addr_t p_data2 = (mapd_addr_t)data2;
    fb.read(p_data2, 0, numPages * pageSize);
    
    // verify
    for (int i = 0; i < numInt; ++i) {
        printf("%d %d\n", data1[i], data2[i]);
        ASSERT_EQ(data1[i], data2[i]);
    }
    
}
