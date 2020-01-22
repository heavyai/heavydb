set(shared_source_files
    Datum.cpp
    timegm.cpp
    mapd_glob.cpp
    StringTransform.cpp
    geo_types.cpp
    SysInfo.cpp
    File.cpp
    StackTrace.cpp
    base64.cpp
    Logger.cpp
    thread_count.cpp
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/funcannotations.h DESTINATION ${CMAKE_BINARY_DIR}/Shared/)

add_library(Shared ${shared_source_files})
target_link_libraries(Shared ${Boost_LIBRARIES} ${GDAL_LIBRARIES})

# Required by ThriftClient.cpp
add_definitions("-DTHRIFT_PACKAGE_VERSION=\"${Thrift_VERSION}\"")
add_library(ThriftClient ThriftClient.cpp)
target_link_libraries(ThriftClient  ${Thrift_LIBRARIES} ${Boost_LIBRARIES})
