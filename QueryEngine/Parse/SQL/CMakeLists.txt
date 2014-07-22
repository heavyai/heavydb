CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#MESSAGE( STATUS "CMAKE_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )

#MESSAGE( STATUS "CMAKE_CURRENT_LIST_FILE: " ${CMAKE_CURRENT_LIST_FILE} )


SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

MESSAGE( STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

find_package(Bisonpp REQUIRED)
find_package(Flexpp REQUIRED)

Set(BisonppOutput ${CMAKE_CURRENT_BINARY_DIR}/Parser.cpp)

#IF(BISONPP_EXECUTABLE)
ADD_CUSTOM_COMMAND(
    OUTPUT ${BisonppOutput}
    COMMAND ${BISONPP_EXECUTABLE}
    -d
    -h${CMAKE_CURRENT_BINARY_DIR}/parser.h
        -o${BisonppOutput}
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.y
    DEPENDS parser.y
    COMMENT "Generating Parser.cpp"
)
#ENDIF()

Set(FlexppOutput ${CMAKE_CURRENT_BINARY_DIR}/Scanner.cpp)

Set(FlexppOutput Scanner.cpp)
#IF (FLEXPP_EXECUTABLE) 
ADD_CUSTOM_COMMAND(
    OUTPUT ${FlexppOutput}
    COMMAND ${FLEXPP_EXECUTABLE}
        -d
        -i
        -o${FlexppOutput}
        ${CMAKE_CURRENT_SOURCE_DIR}/scanner.l
    DEPENDS scanner.l 
    COMMENT "Generating Scanner.cpp"
)
#ENDIF()


#SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/Parser.cpp GENERATED)
SET_SOURCE_FILES_PROPERTIES(${BisonppOutput} GENERATED)
SET_SOURCE_FILES_PROPERTIES(${FlexppOutput} GENERATED)
SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/parser.h GENERATED)

ADD_CUSTOM_TARGET(ParserFiles DEPENDS ${BisonppOutput})
ADD_CUSTOM_TARGET(ScannerFiles DEPENDS ${FlexppOutput})

#ADD_LIBRARY (Parser ${BisonppOutput} ${FlexppOutput} ${CMAKE_CURRENT_BINARY_DIR}/parser.h)
#ADD_LIBRARY (Parser ${BisonppOutput} ${FlexppOutput}  ${CMAKE_CURRENT_BINARY_DIR}/parser.h)

ADD_LIBRARY (Parser ${BisonppOutput} ${CMAKE_CURRENT_BINARY_DIR}/parser.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#ADD_LIBRARY (Scanner ${FlexppOutput} ${CMAKE_CURRENT_BINARY_DIR}/parser.h)
#ADD_LIBRARY (Scanner ${FlexppOutput})

#ADD_DEPENDENCIES (Parser ParserFiles)
#ADD_DEPENDENCIES (Scanner ScannerFiles)
