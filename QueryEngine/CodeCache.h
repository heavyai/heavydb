/*
 * Copyright 2019 OmniSci, Inc.
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

#pragma once

#include <boost/functional/hash.hpp>
#include <memory>

#include "QueryEngine/CompilationContext.h"
#include "StringDictionary/LruCache.hpp"

using CodeCacheKey = std::vector<std::string>;
template <typename CC>
using CodeCacheVal = std::shared_ptr<CC>;
template <typename CC>
using CodeCache = LruCache<CodeCacheKey, CodeCacheVal<CC>, boost::hash<CodeCacheKey>>;
