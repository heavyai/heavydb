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

#ifndef OBJECT_ROLE_DESCRIPTOR_H
#define OBJECT_ROLE_DESCRIPTOR_H

#include <string>
#include <cstdint>
#include "DBObject.h"

/**
 * @type ObjectRoleDescriptor
 * @brief specifies the object_roles content in-memory of a row in mapd_object_permissions table
 *
 */

struct ObjectRoleDescriptor {
  std::string roleName;
  int32_t objectType;
  int32_t dbId;
  int objectId;
  AccessPrivileges privs;
  int32_t objectOwnerId;
  std::string objectName;
};

#endif  // OBJECT_ROLE_DESCRIPTOR
