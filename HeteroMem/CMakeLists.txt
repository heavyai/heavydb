set(heteromem_source_files
MemResourceProvider.cpp)

add_library(HeteroMem ${heteromem_source_files})

if(ENABLE_DCPMM)
    target_link_libraries(HeteroMem ${memkind_LIBRARIES})
else()
    target_link_libraries(HeteroMem)
endif()