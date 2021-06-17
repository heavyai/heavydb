/*
 * Copyright 2021 MapD Technologies, Inc.
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

#include "DataMgr/ChunkMetadata.h"

namespace foreign_storage {

/**
 * Check if ChunkMetadata corresponds to a chunk for which metadata must be
 * populated.
 */
inline bool is_metadata_placeholder(const ChunkMetadata& metadata) {
  return metadata.chunkStats.min.intval > metadata.chunkStats.max.intval &&
         metadata.sqlType.is_dict_encoded_type();  // Only supported type for now
}

}  // namespace foreign_storage
