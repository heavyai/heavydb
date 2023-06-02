if(ENABLE_RUNTIME_LIBS)
    set(RUNTIME_TABLE_FUNCTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeLibTestTableFunctions.hpp)
    set(runtime_table_funcion_source_files ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp)

    find_package(Python COMPONENTS Interpreter REQUIRED)

    execute_process(
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/QueryEngine/scripts/generate_TableFunctionsFactory_init.py
            ${RUNTIME_TABLE_FUNCTION_HEADERS}
            ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp)

    # cpu files
    file(GLOB GENERATED_RUNTIME_TF_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_cpu*.cpp)

    # table function add functions
    file(GLOB GENERATED_RUNTIME_TF_ADD_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_add*.cpp)

    # gpu files
    file(GLOB GENERATED_RUNTIME_TF_GPU_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_gpu*.cpp)

    # set properties
    set_property(DIRECTORY APPEND PROPERTY GENERATED SOURCE
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp
    ${GENERATED_RUNTIME_TF_FILES}
    ${GENERATED_RUNTIME_TF_GPU_FILES}
    ${GENERATED_RUNTIME_TF_ADD_FILES})

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS SOURCE
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp
    ${GENERATED_RUNTIME_TF_FILES}
    ${GENERATED_RUNTIME_TF_GPU_FILES}
    ${GENERATED_RUNTIME_TF_ADD_FILES})

    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES SOURCE
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp
    ${GENERATED_RUNTIME_TF_FILES}
    ${GENERATED_RUNTIME_TF_GPU_FILES}
    ${GENERATED_RUNTIME_TF_ADD_FILES})

    add_custom_command(
        DEPENDS ${RUNTIME_TABLE_FUNCTION_HEADERS}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_cpu.hpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_gpu.hpp
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/QueryEngine/scripts/generate_TableFunctionsFactory_init.py ${RUNTIME_TABLE_FUNCTION_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp)

    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp GENERATED)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_cpu.hpp GENERATED)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_gpu.hpp GENERATED)

    add_library(RuntimeLibTestTableFunctions SHARED ${runtime_table_funcion_source_files} ${GENERATED_RUNTIME_TF_FILES} ${GENERATED_RUNTIME_TF_ADD_FILES})

    add_custom_target(RuntimeLibTestTableFunctionsFactory_init
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp ${GENERATED_RUNTIME_TF_FILES} ${GENERATED_RUNTIME_TF_GPU_FILES} ${GENERATED_RUNTIME_TF_ADD_FILES})

    add_dependencies(RuntimeLibTestTableFunctions RuntimeLibTestTableFunctionsFactory_init)

    add_library(TestRuntimeLib SHARED TestRuntimeLib.cpp)
endif()