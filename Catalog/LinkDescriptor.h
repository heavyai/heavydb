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

#ifndef LINK_DESCRIPTOR_H
#define LINK_DESCRIPTOR_H

#include <string>
#include <cstdint>
#include "../Shared/sqldefs.h"

/**
 * @type LinkDescriptor
 * @brief specifies the content in-memory of a row in the link metadata view
 *
 */

struct LinkDescriptor {
  int32_t linkId;
  int32_t userId;
  std::string link;
  std::string viewState;
  std::string updateTime;
  std::string viewMetadata;
};

#endif  // LINK_DESCRIPTOR
