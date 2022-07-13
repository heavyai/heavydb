set(OSDEPENDENT_SOURCE_FILES
  heavyai_glob.cpp
  heavyai_path.cpp
  heavyai_hostname.cpp
  heavyai_fs.cpp)

if(MSVC)
  add_subdirectory(Windows)
else()
  add_subdirectory(Unix)
endif()
