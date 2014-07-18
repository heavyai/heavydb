cmake_minimum_required(VERSION 2.8)

SET (database_source_files
    Database.cpp
    Database.h
)

find_package(Boost 1.50.0 REQUIRED system thread)
ADD_LIBRARY (Database ${database_source_files})


TARGET_LINK_LIBRARIES (Database pthread ${Boost_THREAD_LIBRARY} )
