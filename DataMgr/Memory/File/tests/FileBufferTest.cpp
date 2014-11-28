#include <cstdlib>
#include <ctime>
#include "gtest/gtest.h"
#include "../FileBuffer.h"
#include "../FileMgr.h"
#include <algorithm>

using namespace File_Namespace;
using namespace std;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
/*
TEST(FileBuffer, read_and_write)
{
    mapd_size_t numPages = 400;
    mapd_size_t pageSize = 4096;
    FileMgr fm(".");
    FileBuffer fb(&fm,pageSize);
    
    // create some data
    mapd_size_t numInserts = numPages * (pageSize / sizeof(int));
    int data1[numInserts];
    for (int i = 0; i < numInserts; ++i)
        data1[i] = i*2;
    
    // write the data to the buffer
    mapd_addr_t p_data1 = (mapd_addr_t)data1;
    fb.write(p_data1, numPages * pageSize,0);
    
    // read the data back
    //fb.read(mapd_addr_t const dst, const mapd_size_t offset, const mapd_size_t nbytes = 0);
    int data2[numInserts];
    mapd_addr_t p_data2 = (mapd_addr_t)data2;
    fb.read(p_data2, numPages * pageSize,0);
    
    // verify
    for (int i = 0; i < numInserts; ++i) {
        //printf("%d %d\n", data1[i], data2[i]);
        ASSERT_EQ(data1[i], data2[i]);
    }
    
}
*/

TEST(FileBuffer, interleaved_read_and_write)
{
    mapd_size_t numPages = 400;
    mapd_size_t pageSize = 4096;
    FileMgr fm("data");
    FileBuffer fb1(&fm,pageSize);
    FileBuffer fb2(&fm,pageSize);
    
    // create some data
    mapd_size_t numInserts = numPages * (pageSize / sizeof(int));
    int data1In[numInserts];
    int data2In[numInserts];
    for (int i = 0; i < numInserts; ++i) {
        data1In[i] = i*2;
        data2In[i] = -i*2;
    }
    
    // write the data to the buffer
    mapd_addr_t data1InPtr = (mapd_addr_t)data1In;
    mapd_addr_t data2InPtr = (mapd_addr_t)data2In;
    int numCycles = 20;
    int numInsertsPerCycle = numInserts / numCycles;
    int numInsertsLeft = numInserts;
    int cycle = 0;
    while (numInsertsLeft > 0) {
        cout << "Num inserts left: " << numInsertsLeft << endl;
        int curNumInserts = min(numInsertsLeft,numInsertsPerCycle);
        fb1.write(data1InPtr+cycle*numInsertsPerCycle*sizeof(int),curNumInserts*sizeof(int),cycle*numInsertsPerCycle*sizeof(int));
        fb2.write(data2InPtr+cycle*numInsertsPerCycle*sizeof(int),curNumInserts*sizeof(int),cycle*numInsertsPerCycle*sizeof(int));
        numInsertsLeft -= numInsertsPerCycle;
        cycle++;
    }

    
    // read the data back
    //fb.read(mapd_addr_t const dst, const mapd_size_t offset, const mapd_size_t nbytes = 0);
    int data1Out[numInserts];
    int data2Out[numInserts];
    mapd_addr_t data1OutPtr = (mapd_addr_t)data1Out;
    mapd_addr_t data2OutPtr = (mapd_addr_t)data2Out;
    fb1.read(data1OutPtr, numPages * pageSize,0);
    fb2.read(data2OutPtr, numPages * pageSize,0);
    
    // verify
    for (int i = 0; i < numInserts; ++i) {
        //printf("%d %d\n", data1[i], data2[i]);
        ASSERT_EQ(data1In[i], data1Out[i]);
        ASSERT_EQ(data2In[i], data2Out[i]);
    }
    
}


