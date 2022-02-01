add_library(Geospatial Compression.cpp Types.cpp GDAL.cpp)
target_link_libraries(Geospatial OSDependent ${GDAL_LIBRARIES} ${CURL_LIBRARIES} ${GDAL_EXTRA_LIBRARIES})
