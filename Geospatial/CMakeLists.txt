install(DIRECTORY ${CMAKE_SOURCE_DIR}/ThirdParty/gdal-data DESTINATION "ThirdParty" COMPONENT "data")
add_custom_target(gdal-data ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/ThirdParty/gdal-data" "${CMAKE_BINARY_DIR}/ThirdParty/gdal-data")

install(DIRECTORY ${CMAKE_SOURCE_DIR}/ThirdParty/geo_samples DESTINATION "ThirdParty" COMPONENT "data")
add_custom_target(geo_samples ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/ThirdParty/geo_samples" "${CMAKE_BINARY_DIR}/ThirdParty/geo_samples")

find_program(PROJINFO_EXECUTABLE NAMES projinfo)
if(NOT PROJINFO_EXECUTABLE)
  message(ERROR "projinfo not found. Will not attempt to copy proj_lib data files.")
else()
  execute_process(
    COMMAND ${PROJINFO_EXECUTABLE} --searchpaths
    OUTPUT_VARIABLE PROJ_SEARCH_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  string(REPLACE "\n" ";" PROJ_SEARCH_PATHS_SEMI "${PROJ_SEARCH_PATHS}")
  find_path(PROJ_DATA_PATH
    PATHS ${PROJ_SEARCH_PATHS_SEMI}
    NAMES "proj.db"
    PATH_SUFFIXES "proj"
    NO_DEFAULT_PATH
    )
  if(PROJ_DATA_PATH)
    install(DIRECTORY ${PROJ_DATA_PATH} DESTINATION "ThirdParty/gdal-data" COMPONENT "data")
    add_custom_target(proj-data ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJ_DATA_PATH}" "${CMAKE_BINARY_DIR}/ThirdParty/gdal-data/proj")
    add_dependencies(proj-data gdal-data)
  else()
    message(ERROR "proj data path not found. Will not attempt to copy proj_lib data files.")
  endif()
endif()

add_library(Geospatial Compression.cpp Types.cpp)
target_link_libraries(Geospatial OSDependent ${GDAL_LIBRARIES} ${CURL_LIBRARIES} ${GDAL_EXTRA_LIBRARIES})
