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
  virtual inline MgrType getMgrType() { return CPU_MGR; }
  virtual inline std::string getStringMgrType() { return ToString(CPU_MGR); }
  ~CpuBufferMgr();

 private:
  virtual void addSlab(const size_t slabSize);
  virtual void freeAllMem();
  virtual void allocateBuffer(BufferList::iterator segIt, const size_t pageSize, const size_t initialSize);
  CudaMgr_Namespace::CudaMgr* cudaMgr_;
};

}  // Buffer_Namespace

#endif  // CPUBUFFERMGR_H
