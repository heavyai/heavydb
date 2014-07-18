CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#MESSAGE( STATUS "CMAKE_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )

#MESSAGE( STATUS "CMAKE_CURRENT_LIST_FILE: " ${CMAKE_CURRENT_LIST_FILE} )

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(Bisonpp REQUIRED)
#Bisonpp_Target(Parser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
ADD_CUSTOM_TARGET(ParserFiles "Creating Parser.cpp and Parser.h")
#Set(BisonppOutput ${CMAKE_CURRE}/parser.y)
IF(Bisonpp_FOUND)
    ADD_CUSTOM_COMMAND(
    OUTPUT ${BisonppOutput}
    COMMAND ${BISONPP_EXECUTABLE}
        --defines=parser.h
        --output=${BisonppOutput}
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.y
    COMMENT "Generating parser.cpp"
    )
ENDIF()

find_package(Flexpp REQUIRED)
Set(FlexppOutput ${CMAKE_CURRENT_SOURCE_DIR}/scanner.cpp)
IF (Flexpp_FOUND) 
    ADD_CUSTOM_COMMAND(
        OUTPUT ${FlexppOutput}
        COMMAND ${FLEXPP_EXECUTABLE}
            --outfile=${FlexppOutput}
            ${CMAKE_CURRENT_SOURCE_DIR}/scanner.l
        COMMENT "Generating Scanner.cpp"
    )
ENDIF()

ADD_LIBRARY (Parser ${BisonppOutput} ${FlexppOutput})
