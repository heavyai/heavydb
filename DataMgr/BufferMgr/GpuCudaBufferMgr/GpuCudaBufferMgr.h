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

#ifndef GPUCUDABUFFERMGR_H
#define GPUCUDABUFFERMGR_H

#include "../BufferMgr.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Buffer_Namespace {

class GpuCudaBufferMgr : public BufferMgr {
 public:
  GpuCudaBufferMgr(const int deviceId,
                   const size_t maxBufferSize,
                   CudaMgr_Namespace::CudaMgr* cudaMgr,
                   const size_t bufferAllocIncrement = 1073741824,
                   const size_t pageSize = 512,
                   AbstractBufferMgr* parentMgr = 0);
  virtual inline MgrType getMgrType() { return GPU_MGR; }
  virtual inline std::string getStringMgrType() { return ToString(GPU_MGR); }
  ~GpuCudaBufferMgr();

 private:
  virtual void addSlab(const size_t slabSize);
  virtual void freeAllMem();
  virtual void allocateBuffer(BufferList::iterator segIt, const size_t pageSize, const size_t initialSize);
  CudaMgr_Namespace::CudaMgr* cudaMgr_;
};

}  // Buffer_Namespace

#endif  // CPUBUFFERMGR_H
