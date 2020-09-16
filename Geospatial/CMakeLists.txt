add_library(Geospatial Compression.cpp Types.cpp)
target_link_libraries(Geospatial ${GDAL_LIBRARIES} ${CURL_LIBRARIES} ${GDAL_EXTRA_LIBRARIES})
