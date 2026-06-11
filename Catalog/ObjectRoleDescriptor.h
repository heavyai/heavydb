/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJECT_ROLE_DESCRIPTOR_H
#define OBJECT_ROLE_DESCRIPTOR_H

#include <cstdint>
#include <string>
#include "DBObject.h"

/**
 * @type ObjectRoleDescriptor
 * @brief specifies the object_roles content in-memory of a row in mapd_object_permissions
 * table
 *
 */

struct ObjectRoleDescriptor {
  std::string roleName;
  bool roleType;
  int32_t objectType;
  int32_t dbId;
  int objectId;
  AccessPrivileges privs;
  int32_t objectOwnerId;
  std::string objectName;
  int32_t subObjectId;
};

#endif  // OBJECT_ROLE_DESCRIPTOR
