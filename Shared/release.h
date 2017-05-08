/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
static const std::string MAPD_EDITION{"@MAPD_EDITION_LOWER@"};

static const std::string MAPD_RELEASE{"@MAPD_VERSION_RAW@-@MAPD_BUILD_DATE@-@MAPD_GIT_HASH@"};

#endif  // RELEASE_H
