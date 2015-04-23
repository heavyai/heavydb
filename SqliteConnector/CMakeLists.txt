cmake_minimum_required(VERSION 2.8)

#SET (CMAKE_CXX_FLAGS "-std=c++0x -g")
SET (sqliteconnector_source_files
    SqliteConnector.cpp
		SqliteConnector.h
		sqlite3.c
		sqlite3.h
)

ADD_LIBRARY (SqliteConnector ${sqliteconnector_source_files})

TARGET_LINK_LIBRARIES (SqliteConnector ${Boost_THREAD_LIBRARY} )
