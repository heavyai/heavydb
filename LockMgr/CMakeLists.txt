set(lock_mgr_source_files
    LockMgr.cpp
)

add_library(LockMgr ${lock_mgr_source_files})

if(MSVC)
  # On the win32 build lockmgr 'sometimes' tries to
  # include gen-cpp/CalciteServer.h before it generated
  add_dependencies(LockMgr calciteserver_thrift)
endif()

target_link_libraries(LockMgr Shared ${Boost_THREAD_LIBRARY})
