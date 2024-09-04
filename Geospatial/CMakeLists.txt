# copy geo samples
install(DIRECTORY ${CMAKE_SOURCE_DIR}/ThirdParty/geo_samples DESTINATION "ThirdParty" COMPONENT "data")
add_custom_target(geo_samples ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/ThirdParty/geo_samples" "${CMAKE_BINARY_DIR}/ThirdParty/geo_samples")
add_dependencies(initheavy geo_samples)

# copy PROJ/GDAL data files from deps
if(MAPD_DEPS_PATH)
  # copy PROJ files
  string(CONCAT PROJ_DATA_PATH ${MAPD_DEPS_PATH} "/share/proj")
  message(STATUS "Coping PROJ_DATA from " ${PROJ_DATA_PATH})
  install(DIRECTORY ${PROJ_DATA_PATH} DESTINATION "ThirdParty" COMPONENT "data")
  add_custom_target(proj-data ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJ_DATA_PATH}" "${CMAKE_BINARY_DIR}/ThirdParty/proj")
  add_dependencies(initheavy proj-data)

  # copy GDAL files
  string(CONCAT GDAL_DATA_PATH ${MAPD_DEPS_PATH} "/share/gdal")
  message(STATUS "Copying GDAL_DATA from " ${GDAL_DATA_PATH})
  install(DIRECTORY ${GDAL_DATA_PATH} DESTINATION "ThirdParty" COMPONENT "data")
  add_custom_target(gdal-data ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${GDAL_DATA_PATH}" "${CMAKE_BINARY_DIR}/ThirdParty/gdal")
  add_dependencies(initheavy gdal-data)
else()
  message(WARNING "MAPD_DEPS_PATH not set. Not copying PROJ/GDAL data files.")
endif()


add_library(Geospatial Compression.cpp Types.cpp GDAL.cpp GeosValidation.cpp)
target_link_libraries(Geospatial OSDependent ${GDAL_LIBRARIES} ${CURL_LIBRARIES} ${GDAL_EXTRA_LIBRARIES})
