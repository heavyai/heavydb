/*
 * Copyright 2020 OmniSci, Inc.
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

/*
This file implements some of the longer functions used by the AbstractBuffer interface
outside of the header file in order to prevent unnecessary re-compilation.
*/

#include "AbstractBuffer.h"

namespace Data_Namespace {

void AbstractBuffer::initEncoder(const SQLTypeInfo& tmp_sql_type) {
  sql_type_ = tmp_sql_type;
  encoder_.reset(Encoder::Create(this, sql_type_));
  LOG_IF(FATAL, encoder_ == nullptr)
      << "Failed to create encoder for SQL Type " << sql_type_.get_type_name();
}

void AbstractBuffer::syncEncoder(const AbstractBuffer* src_buffer) {
  if (src_buffer->hasEncoder()) {
    if (!hasEncoder()) {
      initEncoder(src_buffer->sql_type_);
    }
    encoder_->copyMetadata(src_buffer->encoder_.get());
  } else {
    encoder_ = nullptr;
  }
}

void AbstractBuffer::copyTo(AbstractBuffer* destination_buffer, const size_t num_bytes) {
  size_t chunk_size = (num_bytes == 0) ? size() : num_bytes;
  destination_buffer->reserve(chunk_size);
  if (isUpdated()) {
    read(destination_buffer->getMemoryPtr(),
         chunk_size,
         0,
         destination_buffer->getType(),
         destination_buffer->getDeviceId());
  } else {
    read(destination_buffer->getMemoryPtr() + destination_buffer->size(),
         chunk_size - destination_buffer->size(),
         destination_buffer->size(),
         destination_buffer->getType(),
         destination_buffer->getDeviceId());
  }
  destination_buffer->setSize(chunk_size);
  destination_buffer->syncEncoder(this);
}

void AbstractBuffer::resetToEmpty() {
  encoder_ = nullptr;
  size_ = 0;
}
}  // namespace Data_Namespace
