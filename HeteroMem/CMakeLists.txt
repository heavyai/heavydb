set(heteromem_source_files
MemResourceProvider.cpp)

add_library(HeteroMem ${heteromem_source_files})

target_link_libraries(HeteroMem ${memkind_LIBRARIES})