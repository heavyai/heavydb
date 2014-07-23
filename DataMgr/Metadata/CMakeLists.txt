cmake_minimum_required(VERSION 2.8)

SET (catalog_source_files
    Catalog.cpp
    Catalog.h
)

ADD_LIBRARY (Catalog ${catalog_source_files})
