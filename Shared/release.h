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

#define MAPD_RELEASE_NO   std::string("0.1.")
const std::string MapDRelease(MAPD_RELEASE_NO + "MAPD_GIT_HASH");

#endif // RELEASE_H
