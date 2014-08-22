cmake_minimum_required(VERSION 2.8)

Set (analysis_source_files
    TypeChecker.h
    TypeChecker.cpp
)

ADD_LIBRARY (Analysis ${analysis_source_files})
