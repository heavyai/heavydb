add_library(DBEngine SHARED DBETypes.cpp DBEngine.cpp)

target_link_libraries(DBEngine QueryRunner QueryEngine DataMgr ${Arrow_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${TBB_LIBS})

set(INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")
install(TARGETS DBEngine DESTINATION ${INSTALL_PATH})

find_program(PYTHON "python3")
IF(PYTHON)
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/dbe.cpp")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.in.py" "${CMAKE_CURRENT_BINARY_DIR}/setup.py.in" @ONLY)
    # cannot be done in one step, splitting in configure_file and file commands
    file(GENERATE OUTPUT "${SETUP_PY}" INPUT "${CMAKE_CURRENT_BINARY_DIR}/setup.py.in")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DBEngine.h ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DBEngine.pxd ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dbe.pyx ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)

    add_custom_command(OUTPUT ${OUTPUT}
        COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${PYTHON} ${SETUP_PY} build_ext --inplace
        DEPENDS DBEngine ${SETUP_PY})

    add_custom_target(Python ALL DEPENDS ${OUTPUT})

    install(CODE execute_process("COMMAND cd ${CMAKE_CURRENT_BINARY_DIR} &&
                pip install --install-option --install-base=${INSTALL_PATH} ."))
                #--install-option --install-platlib=${INSTALL_PATH} \
                #--install-option --install-scripts=${INSTALL_PATH} \
                #--install-option --install-data=${INSTALL_PATH} \
                #--install-option --install-purelib=${INSTALL_PATH}"))

ENDIF(PYTHON)

