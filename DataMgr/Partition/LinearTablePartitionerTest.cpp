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
    LinearTablePartitioner linearTablePartitioner(0, columnInfoVec, bufferMgr, 1048576 );

    //linearTablePartitioner.insertData








}

