#include "LinearTablePartitioner.h"
#include "../File/FileMgr.h"
#include "../Buffer/BufferMgr.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
   
    File_Namespace::FileMgr * fileMgr = new File_Namespace::FileMgr ("data");
    Buffer_Namespace::BufferMgr bufferMgr (128*1048576, fileMgr);
    vector <ColumnInfo> columnInfoVec;
    ColumnInfo colInfo;
    colInfo.columnId_ = 0;
    colInfo.columnType_ = INT_TYPE; 
    colInfo.bitSize_ = 32;
    colInfo.insertBuffer_ = 0;
    columnInfoVec.push_back(colInfo);
    colInfo.columnId_ = 1;
    colInfo.columnType_ = FLOAT_TYPE; 
    colInfo.bitSize_ = 32;
    colInfo.insertBuffer_ = 0;
    columnInfoVec.push_back(colInfo);
    LinearTablePartitioner linearTablePartitioner(0, columnInfoVec, bufferMgr, 1048576 );

    vector <int> columnIds; 
    vector <void *> data;
    columnIds.push_back(0);
    columnIds.push_back(1);
    int intData = 3;
    data.push_back(static_cast <void *> (&intData));
    float floatData = 7.2;
    data.push_back(static_cast <void *> (&floatData));

    for (int i = 0; i < 40; ++i) {
        for (int r = 0; r < 100000; ++r) { 
            linearTablePartitioner.insertData(columnIds,data,1);
        }
        vector <PartitionInfo> partitions;
        linearTablePartitioner.getPartitionsForQuery(partitions);
        cout << endl << "Query " << i << endl;
        for (int p = 0; p < partitions.size(); ++p)
            cout << partitions[p].partitionId_ << " " << partitions[p].numTuples_ << endl;
    }




    //linearTablePartitioner.insertData








}

