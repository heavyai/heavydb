cmake_minimum_required(VERSION 2.8)

ADD_LIBRARY(catalog Catalog.cpp)

PROJECT			(MapD)

ADD_EXECUTABLE(catalogTest CatalogTest.cpp)
target_link_libraries(catalogTest catalog )

