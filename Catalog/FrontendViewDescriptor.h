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

#ifndef FRONTEND_VIEW_DESCRIPTOR_H
#define FRONTEND_VIEW_DESCRIPTOR_H

#include <string>
#include <cstdint>
#include "../Shared/sqldefs.h"

/**
 * @type FrontendViewDescriptor
 * @brief specifies the content in-memory of a row in the frontend view metadata view
 *
 */

struct FrontendViewDescriptor {
  int32_t viewId;       /**< viewId starts at 0 for valid views. */
  std::string viewName; /**< viewName is the name of the view view -must be unique */
  std::string viewState;
  std::string imageHash;
  std::string updateTime;
  std::string viewMetadata;
  int32_t userId;
};

#endif  // FRONTEND_VIEW_DESCRIPTOR
