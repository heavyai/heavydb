set(utils_standalone_source_files
    StringLike.cpp
    Regexp.cpp
    ChunkIter.cpp
)
set(utils_source_files
    ChunkAccessorTable.cpp
    DdlUtils.cpp
)

add_library(Utils ${utils_source_files})
add_library(UtilsStandalone ${utils_standalone_source_files})
add_dependencies(Utils Catalog)
target_link_libraries(UtilsStandalone OSDependent ${Boost_LIBRARIES} Shared)
target_link_libraries(Utils UtilsStandalone OSDependent ${Boost_LIBRARIES} Shared)
