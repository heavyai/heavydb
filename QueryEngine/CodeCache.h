/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/functional/hash.hpp>
#include <memory>

#include "Shared/LruCache.h"

using CodeCacheKey = std::vector<std::string>;
template <typename CC>
using CodeCacheVal = std::shared_ptr<CC>;
template <typename CC>
using CodeCache = LruCache<CodeCacheKey, CodeCacheVal<CC>, boost::hash<CodeCacheKey>>;
