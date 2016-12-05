// clang-format off
/**
 * @file		release.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Defines the release number string
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef RELEASE_H
#define RELEASE_H

#include <string>

static const int32_t MAPD_VERSION_MAJOR{@MAPD_VERSION_MAJOR@};
static const int32_t MAPD_VERSION_MINOR{@MAPD_VERSION_MINOR@};
static const int32_t MAPD_VERSION_PATCH{@MAPD_VERSION_PATCH@};
static const int32_t MAPD_VERSION{
  @MAPD_VERSION_MAJOR@ * 1000000 + @MAPD_VERSION_MINOR@ * 1000 + @MAPD_VERSION_PATCH@
};

static const std::string MAPD_VERSION_EXTRA{"@MAPD_VERSION_EXTRA@"};
static const std::string MAPD_VERSION_RAW{"@MAPD_VERSION_RAW@"};
static const std::string MAPD_BUILD_DATE{"@MAPD_BUILD_DATE@"};
static const std::string MAPD_GIT_HASH{"@MAPD_GIT_HASH@"};

static const std::string MapDRelease{"@MAPD_VERSION_RAW@-@MAPD_BUILD_DATE@-@MAPD_GIT_HASH@"};

#endif  // RELEASE_H
