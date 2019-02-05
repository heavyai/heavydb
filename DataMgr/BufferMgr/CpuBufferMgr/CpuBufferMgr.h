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

#ifndef CPUBUFFERMGR_H
#define CPUBUFFERMGR_H

#include "../BufferMgr.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Buffer_Namespace {

class CpuBufferMgr : public BufferMgr {
 public:
  CpuBufferMgr(const int deviceId,
               const size_t maxBufferSize,
               CudaMgr_Namespace::CudaMgr* cudaMgr,
               const size_t bufferAllocIncrement = 2147483648,
               const size_t pageSize = 512,
               AbstractBufferMgr* parentMgr = 0);
  inline MgrType getMgrType() override { return CPU_MGR; }
  inline std::string getStringMgrType() override { return ToString(CPU_MGR); }
  ~CpuBufferMgr() override;

 private:
  void addSlab(const size_t slabSize) override;
  void freeAllMem() override;
  void allocateBuffer(BufferList::iterator segIt,
                      const size_t pageSize,
                      const size_t initialSize) override;
  CudaMgr_Namespace::CudaMgr* cudaMgr_;
};

}  // namespace Buffer_Namespace

#endif  // CPUBUFFERMGR_H
