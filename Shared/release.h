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

const std::string MapDRelease("@MAPD_VERSION_RAW@-@MAPD_BUILD_DATE@-@MAPD_GIT_HASH@");

#endif  // RELEASE_H
