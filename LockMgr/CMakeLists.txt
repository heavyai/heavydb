set(lock_mgr_source_files
    TableLockMgr.cpp
)

add_library(LockMgr ${lock_mgr_source_files})

target_link_libraries(LockMgr Shared ${Boost_THREAD_LIBRARY} ${Glog_LIBRARIES})

