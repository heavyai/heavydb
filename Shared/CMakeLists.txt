set(shared_source_files
    Datum.cpp
    timegm.cpp
    mapd_glob.cpp
    StringTransform.cpp
)

add_library(Shared ${shared_source_files})
