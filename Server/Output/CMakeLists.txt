cmake_minimum_required(VERSION 2.8)


SET (output_source_files
    OutputWriter.cpp
    OutputWriter.h
    OutputBuffer.cpp
    OutputBuffer.h
)

ADD_LIBRARY (Output ${output_source_files})
