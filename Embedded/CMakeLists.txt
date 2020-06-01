add_library(DBEngine SHARED DBETypes.cpp DBEngine.cpp)

target_link_libraries(DBEngine QueryRunner QueryEngine DataMgr ${Arrow_LIBRARIES} ${TBB_LIBS})

set(INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS DBEngine DESTINATION ${INSTALL_PATH})

find_program(PYTHON "python3")
IF(PYTHON)
    set(SETUP_PY_IN  "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
    set(SETUP_PY     "${CMAKE_BINARY_DIR}/setup.py")
    set(OUTPUT       "${CMAKE_BINARY_DIR}/timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${PYTHON} ${SETUP_PY} build_ext --inplace
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS DBEngine)

    add_custom_target(target ALL DEPENDS ${OUTPUT})

    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install -f \
        --install-headers=${INSTALL_PATH} \
        --install-base=${INSTALL_PATH} \
        --install-platlib=${INSTALL_PATH} \
        --install-scripts=${INSTALL_PATH} \
        --install-data=${INSTALL_PATH} \
        --install-purelib=${INSTALL_PATH})")
ENDIF(PYTHON)

