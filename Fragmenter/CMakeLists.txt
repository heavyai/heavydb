cmake_minimum_required(VERSION 2.8)

SET (fragmenter_source_files
    InsertOrderFragmenter.cpp
)

ADD_LIBRARY(Fragmenter ${fragmenter_source_files})

TARGET_LINK_LIBRARIES (Fragmenter ${Boost_THREAD_LIBRARY})

