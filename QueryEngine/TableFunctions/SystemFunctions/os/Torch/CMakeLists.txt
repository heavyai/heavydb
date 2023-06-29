if(ENABLE_SYSTEM_TFS)
  if(ENABLE_TORCH_TFS)
    list(APPEND TORCH_INCLUDES ${TORCH_INCLUDE_DIRS} ${MKL_INCLUDE_DIRS})
    list(APPEND TORCH_LIBS ${TORCH_LIBRARIES} ${MKL_LIBRARIES})

    set(torch_table_function_source_files
      TorchWrapper.cpp TorchOps.hpp TorchTableFunctions.cpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp TestTorchTableFunctions.cpp
    )

    set(TORCH_TABLE_FUNCTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/TestTorchTableFunctions.h)

    find_package(Python COMPONENTS Interpreter REQUIRED)

    execute_process(
      COMMAND ${Python_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/QueryEngine/scripts/generate_TableFunctionsFactory_init.py
        ${TORCH_TABLE_FUNCTION_HEADERS}
        ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp)

    # cpu files
    file(GLOB GENERATED_TORCH_TF_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_cpu*.cpp)

    # table function add functions
    file(GLOB GENERATED_TORCH_TF_ADD_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_add*.cpp)

    # set properties
    set_property(DIRECTORY APPEND PROPERTY GENERATED SOURCE
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp
    ${GENERATED_TORCH_TF_FILES}
    ${GENERATED_TORCH_TF_ADD_FILES})

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS SOURCE
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp
    ${GENERATED_TORCH_TF_FILES}
    ${GENERATED_TORCH_TF_ADD_FILES})

    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES SOURCE
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp
    ${GENERATED_TORCH_TF_FILES}
    ${GENERATED_TORCH_TF_ADD_FILES})

    add_custom_command(
      DEPENDS ${TORCH_TABLE_FUNCTION_HEADERS}
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_cpu.hpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_gpu.hpp
      COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/QueryEngine/scripts/generate_TableFunctionsFactory_init.py ${TORCH_TABLE_FUNCTION_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp)

    add_library(TorchTableFunctions SHARED ${torch_table_function_source_files} ${GENERATED_TORCH_TF_FILES} ${GENERATED_TORCH_TF_ADD_FILES})

    target_compile_options(TorchTableFunctions PRIVATE ${TORCH_CXX_FLAGS})
    target_include_directories(TorchTableFunctions PRIVATE ${TORCH_INCLUDES})

    add_custom_target(TorchTableFunctionsFactory_init
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp ${GENERATED_TORCH_TF_FILES} ${GENERATED_TORCH_TF_ADD_FILES})
    add_dependencies(TorchTableFunctions TorchTableFunctionsFactory_init)
  endif()
else()
  set(ENABLE_TORCH_TFS OFF CACHE BOOL "Enable (py)torch table functions" FORCE)
endif()