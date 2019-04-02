set(shared_source_files
    Datum.cpp
    timegm.cpp
    mapd_glob.cpp
    StringTransform.cpp
    geo_types.cpp
    SysInfo.cpp
    File.cpp
)

add_library(Shared ${shared_source_files})

find_package(Boost COMPONENTS filesystem system REQUIRED QUIET)
add_library(ThriftClient ThriftClient.cpp)
target_link_libraries(ThriftClient ${Thrift_LIBRARIES} ${Boost_LIBRARIES})
