#ifndef PMM_PERSISTENT_BUFFER_MGR_H
#define PMM_PERSISTENT_BUFFER_MGR_H

#ifdef HAVE_DCPMM

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "../../Shared/types.h"
#include "../Pmem.h"
#include "FileMgr.h"

namespace File_Namespace {

#define PMM_PERSISTENT_BUFFER_POOL_MAGIC 0xBEEFBEEF
#define PMM_PERSISTENT_BUFFER_POOL_VERSION 1

/*
 * each persistent memory pool is a collection of slabs with fixed size, 4GB by default
 * each slab is a collection of pages with fixed size, 2MB by default
 * each column chunk buffer can take one or more contiguous pages
 * chunks of the same column can be in different pools
 */
struct PersistentPoolHeader {  // at the very begining of the first slab in the persistent
                               // memory pool
  uint64_t magic;              // magic word for the pool
  uint64_t version;            // pool format version
  uint64_t size;               // size of pool
  uint64_t slabSize;           // slab size
  uint64_t pageSize;           // page size
  uint64_t numSlabs;           // total number of slabs
  uint64_t descriptorHeaderOffset;  // first buffer descriptor header after pool header
                                    // and allocation map
  uint64_t checksum;                // checksum of above fields
  // size_t slabAllocationMap[8];	// immediately follows the header,  variable size but
  // at least a cache line
};

struct PersistentBufferDescriptor {  // after the pool header and buffer descritor header
  uint64_t size;                     // size of real data
  int slabSeq;       // each pool can have maximum of 2^31 slabs.  -1: invalid
                     // different from the slabNum in volatile data structure slabs_
  int startPage;     // start of page in the slab
  int numPages;      // number of pages in this buffer
  int epoch;         // epoch for consistency check
  int chunkKey[5];   // same as header in each disk file page
                     // data base id
                     // table id
                     // column id
                     // fragment id
                     // varlen offset
  int metaData[10];  // meta data with size NUM_METADATA
  char encoderMetaData[44];  // big enough for encoder meta data?
                             // make it multiples of cache-line

  int getEpoch(void) { return epoch; }
  void setEpoch(int e) {
    epoch = e;
    PmemPersist(&epoch, sizeof(epoch));
  }
};

// TODO: string dictionaries type
#define PMM_PERSISTENT_BUFFER_CHUNK_TYPE 0

// the first header is in the header slab immediately after the allocation map
// buffer descriptors immediately follows the header
struct PersistentBufferDescriptorDirectoryHeader {
  int descriptorType;
  int slabSeq;                // slab in which this header resides
  int startPage;              // starting page number of the header
  int numPages;               // number of pages the header and the descriptors take
  uint64_t nextHeaderOffset;  // point to the next slab holding the descriptors
  uint64_t
      numDescriptors;    // total number of descriptors immediately folowwing this header
  uint64_t reserved[4];  // make descriptor array cacheline aligned
};

struct PersistentBufferPool {  // volatile
  union {
    PersistentPoolHeader* header;
    int8_t* base;
  };
  int8_t* ceiling;
  uint64_t* volatile slabAllocationMap;
  // size_t size;	// total size of the pool
  // size_t slabSize;
  // size_t numSlabs;
  uint64_t volatile numFreeSlabs;
  PersistentBufferDescriptorDirectoryHeader* descriptorHeaderHead;
  std::deque<PersistentBufferDescriptor*>
      freeBufferDescriptors;  // collection of free buffer descriptors
  std::map<int64_t, std::vector<PersistentBufferDescriptor*>> tableBufferDescriptorMap;
  std::mutex allocationMutex;
  std::mutex descriptorMutex;
  std::mutex freeDescriptorMutex;
  uint64_t* volatile pageAllocationBitMap;
};

union DbTblKey {
  uint64_t dBTblKey;
  struct {
    uint dbId;   // database id
    uint tblId;  // table id
  };
};

union PageKey {
  int64_t pageKey;
  struct {
    int32_t poolId;    // pool identifier
    uint32_t pageSeq;  // page seqence number in pool
  };
};

union SlabKey {
  int64_t slabKey;
  struct {
    int32_t poolId;    // pool identifier
    uint32_t slabSeq;  // slab seqence number in pool
  };
};

class PersistentBufferMgr {
 public:
  PersistentBufferMgr(const bool pmm,
                      const std::string& pmmPath,
                      const size_t pageSize = 2 * 1024 * 1024,
                      const size_t slabSize = 4L * 1024 * 1024 * 1024);
  ~PersistentBufferMgr();

  void constructPersistentBuffers(int dbId, int tblId, FileMgr* fm);
  int8_t* getPersistentBufferAddress(int pool, PersistentBufferDescriptor* p);
  int8_t* allocatePersistentBuffer(ChunkKey key,
                                   size_t numBytes,
                                   PersistentBufferDescriptor** desc);
  void freePersistentBuffer(PersistentBufferDescriptor* p, int8_t* addr);
  void shrinkPersistentBuffer(PersistentBufferDescriptor* p, int8_t* addr);
  int8_t* reallocatePersistentBuffer(ChunkKey key,
                                     int8_t* addr,
                                     size_t numBytes,
                                     PersistentBufferDescriptor** desc);
  size_t getPersistentBufferPageSize(void) { return pageSize_; }

 private:
  int initializePersistentBufferPools(const std::string& pmmPath);
  bool allocatePersistentBufferDescriptors(int pool);
  PersistentBufferDescriptor* getPersistentBufferDescriptor(int pool);
  void putPersistentBufferDescriptor(int pool, PersistentBufferDescriptor* p);
  bool isPersistentSlabAllocatedUnlocked(int pool, int slab);
  PageKey getFreePersistentBuffer(size_t pagesNeeded, PersistentBufferDescriptor** p);
  int getPersistentBufferPoolId(int8_t* addr);
  SlabKey allocatePersistentSlab(void);

  size_t slabSize_;  // 4L * 1024 * 1024 * 1024 = 4GB
  size_t pageSize_;
  size_t pagesPerSlab_;  // slabSize_ / pageSize_;

  struct PersistentBufferPool* pmemPools_;
  int curPool_;  // current pool where buffer is alloacted. for alternating purpose
  int numPools_;
};

}  // namespace File_Namespace

#endif /* HAVE_DCPMM */

#endif  // PMM_PERSISTENT_BUFFER_MGR_H