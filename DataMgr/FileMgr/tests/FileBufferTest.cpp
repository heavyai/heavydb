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
    size_t numPages = 400;
    size_t pageSize = 4096;
    FileMgr fm(".");
    FileBuffer fb(&fm,pageSize);
    
    // create some data
    size_t numInserts = numPages * (pageSize / sizeof(int));
    int data1[numInserts];
    for (int i = 0; i < numInserts; ++i)
        data1[i] = i*2;
    
    // write the data to the buffer
    int8_t * p_data1 = (int8_t *)data1;
    fb.write(p_data1, numPages * pageSize,0);
    
    // read the data back
    //fb.read(int8_t * const dst, const size_t offset, const size_t nbytes = 0);
    int data2[numInserts];
    int8_t * p_data2 = (int8_t *)data2;
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
    size_t numPages = 400;
    size_t pageSize = 4096;
    FileMgr fm("data");
    ChunkKey chunkKey1 = {4,2,1,5};
    FileBuffer fb1(&fm,pageSize,chunkKey1);
    ChunkKey chunkKey2 = {3,9,27,2};
    FileBuffer fb2(&fm,pageSize,chunkKey2);
    
    // create some data
    size_t numInserts = numPages * (pageSize / sizeof(int));
    int data1In[numInserts];
    int data2In[numInserts];
    for (int i = 0; i < numInserts; ++i) {
        data1In[i] = i*2;
        data2In[i] = -i*2;
    }
    
    // write the data to the buffer
    int8_t * data1InPtr = (int8_t *)data1In;
    int8_t * data2InPtr = (int8_t *)data2In;
    int numCycles = 20;
    int numInsertsPerCycle = numInserts / numCycles;
    int numInsertsLeft = numInserts;
    int cycle = 0;
    while (numInsertsLeft > 0) {
        cout << "Num inserts left: " << numInsertsLeft << endl;
        int curNumInserts = min(numInsertsLeft,numInsertsPerCycle);
        fb1.write(data1InPtr+cycle*numInsertsPerCycle*sizeof(int),curNumInserts*sizeof(int),CPU_BUFFER,cycle*numInsertsPerCycle*sizeof(int));
        ASSERT_EQ((cycle+1)*numInsertsPerCycle*sizeof(int),fb1.size());
        fb2.write(data2InPtr+cycle*numInsertsPerCycle*sizeof(int),curNumInserts*sizeof(int),CPU_BUFFER,cycle*numInsertsPerCycle*sizeof(int));
        ASSERT_EQ((cycle+1)*numInsertsPerCycle*sizeof(int),fb2.size());
        numInsertsLeft -= numInsertsPerCycle;
        cycle++;
    }

    
    // read the data back
    //fb.read(int8_t * const dst, const size_t offset, const size_t nbytes = 0);
    int data1Out[numInserts];
    int data2Out[numInserts];
    int8_t * data1OutPtr = (int8_t *)data1Out;
    int8_t * data2OutPtr = (int8_t *)data2Out;
    fb1.read(data1OutPtr, numPages * pageSize,CPU_BUFFER,0);
    fb2.read(data2OutPtr, numPages * pageSize,CPU_BUFFER,0);
    
    // verify
    for (int i = 0; i < numInserts; ++i) {
        //printf("%d %d\n", data1[i], data2[i]);
        ASSERT_EQ(data1In[i], data1Out[i]);
        ASSERT_EQ(data2In[i], data2Out[i]);
    }
    
}


