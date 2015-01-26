cmake_minimum_required(VERSION 2.8)

SET (partitioner_source_files
    TablePartitionerMgr.cpp
    InsertOrderTablePartitioner.cpp
)

ADD_LIBRARY(Partitioner ${partitioner_source_files})

TARGET_LINK_LIBRARIES (Partitioner ${Boost_THREAD_LIBRARY})

