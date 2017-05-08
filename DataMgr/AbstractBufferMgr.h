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

/**
 * @file    AbstractBufferMgr.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#ifndef ABSTRACTDATAMGR_H
#define ABSTRACTDATAMGR_H

#include "../Shared/types.h"
#include "AbstractBuffer.h"
#include <boost/preprocessor.hpp>

#define X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE(r, data, elem) \
  case elem:                                                               \
    return BOOST_PP_STRINGIZE(elem);

#define DEFINE_ENUM_WITH_STRING_CONVERSIONS(name, enumerators)                                      \
  enum name { BOOST_PP_SEQ_ENUM(enumerators) };                                                     \
                                                                                                    \
  inline const char* ToString(name v) {                                                             \
    switch (v) {                                                                                    \
      BOOST_PP_SEQ_FOR_EACH(X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE, name, enumerators) \
      default:                                                                                      \
        return "[Unknown " BOOST_PP_STRINGIZE(name) "]";                                            \
    }                                                                                               \
  }

DEFINE_ENUM_WITH_STRING_CONVERSIONS(MgrType, (FILE_MGR)(CPU_MGR)(GPU_MGR)(GLOBAL_FILE_MGR))

namespace Data_Namespace {

/**
 * @class   AbstractBufferMgr
 * @brief   Abstract prototype (interface) for a data manager.
 *
 * A data manager provides a common interface by inheriting the public interface
 * of this class, whose methods are pure virtual, enforcing each class that
 * implements this interfact to implement the necessary methods.
 *
 * A data manager literally manages data. One assumption about the data manager
 * is that it divides up its data into buffers of data of some kind, each of which
 * inherit the interface specified in AbstractBuffer (@see AbstractBuffer).
 */
class AbstractBufferMgr {
 public:
  virtual ~AbstractBufferMgr() {}
  AbstractBufferMgr(const int deviceId) : deviceId_(deviceId) {}

  // Chunk API
  virtual AbstractBuffer* createBuffer(const ChunkKey& key,
                                       const size_t pageSize = 0,
                                       const size_t initialSize = 0) = 0;
  virtual void deleteBuffer(const ChunkKey& key, const bool purge = true) = 0;  // purge param only used in FileMgr
  virtual void deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool purge = true) = 0;
  virtual AbstractBuffer* getBuffer(const ChunkKey& key, const size_t numBytes = 0) = 0;
  virtual void fetchBuffer(const ChunkKey& key, AbstractBuffer* destBuffer, const size_t numBytes = 0) = 0;
  // virtual AbstractBuffer* putBuffer(const ChunkKey &key, AbstractBuffer *srcBuffer, const size_t numBytes = 0) = 0;
  virtual AbstractBuffer* putBuffer(const ChunkKey& key, AbstractBuffer* srcBuffer, const size_t numBytes = 0) = 0;
  virtual void getChunkMetadataVec(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadata) = 0;
  virtual void getChunkMetadataVecForKeyPrefix(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
                                               const ChunkKey& keyPrefix) = 0;

  virtual bool isBufferOnDevice(const ChunkKey& key) = 0;
  virtual std::string printSlabs() = 0;
  virtual void clearSlabs() = 0;
  virtual size_t getMaxSize() = 0;
  virtual size_t getInUseSize() = 0;
  virtual size_t getAllocated() = 0;
  virtual bool isAllocationCapped() = 0;

  virtual void checkpoint() = 0;
  virtual void checkpoint(const int db_id, const int tb_id) = 0;

  // Buffer API
  virtual AbstractBuffer* alloc(const size_t numBytes = 0) = 0;
  virtual void free(AbstractBuffer* buffer) = 0;
  // virtual AbstractBuffer* putBuffer(AbstractBuffer *d) = 0;
  virtual MgrType getMgrType() = 0;
  virtual std::string getStringMgrType() = 0;
  virtual size_t getNumChunks() = 0;
  inline int getDeviceId() { return deviceId_; }

 protected:
  AbstractBufferMgr* parentMgr_;
  int deviceId_;
};

}  // Data_Namespace

#endif  // ABSTRACTDATAMGR_H
