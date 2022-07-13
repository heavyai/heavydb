set(lock_mgr_source_files
    LockMgr.cpp
)

add_library(LockMgr ${lock_mgr_source_files})

add_dependencies(LockMgr calciteserver_thrift thrift_gen)

target_link_libraries(LockMgr Shared ${Boost_THREAD_LIBRARY})
