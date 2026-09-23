/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LINK_DESCRIPTOR_H
#define LINK_DESCRIPTOR_H

#include <cstdint>
#include <string>
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
