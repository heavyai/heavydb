set(
    sources
    rdaddr.c
    rdavl.c
    rdcrc32.c
    rdkafka.c
    rdkafka_assignor.c
    rdkafka_broker.c
    rdkafka_buf.c
    rdkafka_cgrp.c
    rdkafka_conf.c
    rdkafka_event.c
    rdkafka_feature.c
    rdkafka_metadata.c
    rdkafka_metadata_cache.c
    rdkafka_msg.c
    rdkafka_offset.c
    rdkafka_op.c
    rdkafka_partition.c
    rdkafka_pattern.c
    rdkafka_queue.c
    rdkafka_range_assignor.c
    rdkafka_request.c
    rdkafka_roundrobin_assignor.c
    rdkafka_subscription.c
    rdkafka_timer.c
    rdkafka_topic.c
    rdkafka_transport.c
    rdlist.c
    rdlog.c
    rdports.c
    rdrand.c
    rdregex.c
    rdstring.c
    snappy.c
    tinycthread.c
    xxhash.c
    lz4.c
    lz4frame.c
    lz4hc.c
)

if(WITH_SASL)
  list(
      APPEND
      sources
      rdkafka_sasl.c
      rdkafka_sasl.h
      rdkafka_sasl_int.h
  )
  if(WIN32)
    list(APPEND sources rdkafka_sasl_win32.c)
  else()
    list(APPEND sources rdkafka_sasl_cyrus.c)
  endif()
endif()

if(WITH_ZLIB)
  list(APPEND sources rdgz.c)
endif()

if(NOT HAVE_REGEX)
  list(APPEND sources regexp.c)
endif()

add_library(rdkafka ${sources})

# Support '#include <rdkafka.h>'
target_include_directories(rdkafka PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>")

# We need 'dummy' directory to support `#include "../config.h"` path
set(dummy "${GENERATED_DIR}/dummy")
file(MAKE_DIRECTORY "${dummy}")
target_include_directories(rdkafka PUBLIC "$<BUILD_INTERFACE:${dummy}>")

if(WITH_ZLIB)
  find_package(ZLIB REQUIRED)
  target_link_libraries(rdkafka PUBLIC ZLIB::ZLIB)
endif()

if(WITH_SSL)
  if(WITH_BUNDLED_SSL) # option from 'h2o' parent project
    if(NOT TARGET bundled-ssl)
      message(FATAL_ERROR "bundled-ssl target not exist")
    endif()
    target_include_directories(rdkafka BEFORE PUBLIC ${BUNDLED_SSL_INCLUDE_DIR})
    target_link_libraries(rdkafka PUBLIC ${BUNDLED_SSL_LIBRARIES})
    add_dependencies(rdkafka bundled-ssl)
  else()
    find_package(OpenSSL REQUIRED)
    target_link_libraries(rdkafka PUBLIC OpenSSL::SSL OpenSSL::Crypto)
  endif()
endif()

if(LINK_ATOMIC)
  target_link_libraries(rdkafka PUBLIC "-latomic")
endif()

find_package(Threads REQUIRED)
target_link_libraries(rdkafka PUBLIC Threads::Threads)

if(WITH_SASL)
  pkg_check_modules(SASL REQUIRED libsasl2)
  target_link_libraries(rdkafka PUBLIC ${SASL_LIBRARIES})
endif()

include(CheckLibraryExists)
CHECK_LIBRARY_EXISTS(c clock_gettime "time.h" HAVE_CLOCK_GETTIME_C)
CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME_RT)
if ((NOT HAVE_CLOCK_GETTIME_C) AND HAVE_CLOCK_GETTIME_RT)
  target_link_libraries(rdkafka PUBLIC rt)
endif()

# install(
#     TARGETS rdkafka
#     EXPORT "${targets_export_name}"
#     LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#     ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#     RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
#     INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
# )
#
# install(
#     FILES "rdkafka.h"
#     DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/librdkafka"
# )
