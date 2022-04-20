set(arrow_storage_source_files
    ArrowStorage.cpp
    ArrowStorageUtils.cpp
)

add_library(ArrowStorage ${arrow_storage_source_files})

target_link_libraries(ArrowStorage Shared SchemaMgr ${Parquet_LIBRARIES} ${Arrow_LIBRARIES} ${Boost_THREAD_LIBRARY} ${TBB_LIBS} Utils ${CMAKE_DL_LIBS})
