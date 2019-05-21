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
)

add_library(Shared ${shared_source_files})

find_package(Boost COMPONENTS filesystem system REQUIRED QUIET)
# Required by ThriftClient.cpp
add_definitions("-DTHRIFT_PACKAGE_VERSION=\"${Thrift_VERSION}\"")
add_library(ThriftClient ThriftClient.cpp)
target_link_libraries(ThriftClient ${Thrift_LIBRARIES} ${Boost_LIBRARIES})
