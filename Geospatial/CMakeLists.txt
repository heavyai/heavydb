# copy geo samples
install(DIRECTORY ${CMAKE_SOURCE_DIR}/ThirdParty/geo_samples DESTINATION "ThirdParty" COMPONENT "data")
add_custom_target(geo_samples ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/ThirdParty/geo_samples" "${CMAKE_BINARY_DIR}/ThirdParty/geo_samples")
add_dependencies(initheavy geo_samples)

# copy PROJ_DATA
find_program(PROJINFO_EXECUTABLE NAMES projinfo)
if(NOT PROJINFO_EXECUTABLE)
  message(WARNING "projinfo not found. Will not attempt to copy PROJ_DATA files.")
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
    install(DIRECTORY ${PROJ_DATA_PATH} DESTINATION "ThirdParty" COMPONENT "data")
    add_custom_target(proj-data ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJ_DATA_PATH}" "${CMAKE_BINARY_DIR}/ThirdParty/proj")
    add_dependencies(initheavy proj-data)
  else()
    message(WARNING "PROJ_DATA path not found. Will not attempt to copy PROJ_DATA files.")
  endif()
endif()

# copy GDAL_DATA
find_program(GDALCONFIG_EXECUTABLE NAMES gdal-config)
if(NOT GDALCONFIG_EXECUTABLE)
  message(WARNING "gdal-config not found. Will not attempt to copy GDAL_DATA files.")
else()
  execute_process(
    COMMAND ${GDALCONFIG_EXECUTABLE} --datadir
    OUTPUT_VARIABLE GDAL_SEARCH_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  string(REPLACE "\n" ";" GDAL_SEARCH_PATHS_SEMI "${GDAL_SEARCH_PATHS}")
  find_path(GDAL_DATA_PATH
    PATHS ${GDAL_SEARCH_PATHS_SEMI}
    NAMES "gdalicon.png"
    PATH_SUFFIXES "gdal"
    NO_DEFAULT_PATH
    )
  if(GDAL_DATA_PATH)
    install(DIRECTORY ${GDAL_DATA_PATH} DESTINATION "ThirdParty" COMPONENT "data")
    add_custom_target(gdal-data ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${GDAL_DATA_PATH}" "${CMAKE_BINARY_DIR}/ThirdParty/gdal")
    add_dependencies(initheavy gdal-data)
  else()
    message(WARNING "GDAL_DATA path not found. Will not attempt to copy GDAL_DATA files.")
  endif()
endif()

add_library(Geospatial Compression.cpp Types.cpp GDAL.cpp GeosValidation.cpp)
target_link_libraries(Geospatial OSDependent ${GDAL_LIBRARIES} ${CURL_LIBRARIES} ${GDAL_EXTRA_LIBRARIES})
