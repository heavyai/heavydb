set(OSDEPENDENT_SOURCE_FILES
  omnisci_glob.cpp
  omnisci_path.cpp
  omnisci_hostname.cpp
  omnisci_fs.cpp)

if(MSVC)
  add_subdirectory(Windows)
else()
  add_subdirectory(Unix)
endif()
