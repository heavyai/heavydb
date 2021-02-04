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

#ifdef HAVE_DCPMM

#include "PmmPersistentBufferMgr.h"
#include <glog/logging.h>
#include "../../Shared/checked_alloc.h"
#include "../Pmem.h"

#include <errno.h>
#include <fcntl.h>
#include <memory.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// TODO: A more efficient allocation algorithm
//      The current algorithm is linear and less efficient
//      Consider a bucket based allocation algorithm
namespace File_Namespace {

PersistentBufferMgr::PersistentBufferMgr(const bool pmm,
                                         const std::string& pmmPath,
                                         // const size_t pageSize = 2 * 1024 * 1024,
                                         // const size_t slabSize = 4 * 1024 * 1024 * 1024
                                         const size_t pageSize,
                                         const size_t slabSize) {
  slabSize_ = slabSize;
  pageSize_ = pageSize;
  pagesPerSlab_ = slabSize / pageSize;
  numPools_ = 0;
  pmemPools_ = NULL;
  if (pmm) {
    initializePersistentBufferPools(pmmPath);
  }
}

PersistentBufferMgr::~PersistentBufferMgr() {
  if (pmemPools_) {
    delete[] pmemPools_;
  }
}

void PersistentBufferMgr::constructPersistentBuffers(int dbId, int tblId, FileMgr* fm) {
  DbTblKey key;

  key.dbId = dbId;
  key.tblId = tblId;

  for (int i = 0; i < numPools_; i++) {
    pmemPools_[i].descriptorMutex.lock();
    auto it = pmemPools_[i].tableBufferDescriptorMap.find(key.dBTblKey);

    if (it != pmemPools_[i].tableBufferDescriptorMap.end()) {
      for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
        PersistentBufferDescriptor* desc;

        desc = *it2;

        if (desc->chunkKey[4]) {
          ChunkKey chunkKey{desc->chunkKey[0],
                            desc->chunkKey[1],
                            desc->chunkKey[2],
                            desc->chunkKey[3],
                            desc->chunkKey[4]};
          fm->constructPersistentBuffer(
              chunkKey, desc, getPersistentBufferAddress(i, desc));
        } else {
          ChunkKey chunkKey{
              desc->chunkKey[0], desc->chunkKey[1], desc->chunkKey[2], desc->chunkKey[3]};
          fm->constructPersistentBuffer(
              chunkKey, desc, getPersistentBufferAddress(i, desc));
        }
      }
    }
    pmemPools_[i].descriptorMutex.unlock();
  }
}

bool PersistentBufferMgr::allocatePersistentBufferDescriptors(int pool) {
  bool done = false;
  size_t j, k;
  int m;
  pmemPools_[pool].allocationMutex.lock();
  for (j = 0; (j < pmemPools_[pool].header->numSlabs) && !done; j++) {
    if (isPersistentSlabAllocatedUnlocked(pool, j) == false) {
      continue;
    }

    for (k = j * pagesPerSlab_; (k < (j + 1) * pagesPerSlab_) && !done;
         k += (sizeof(uint64_t) * 8)) {
      size_t index;

      index = k >> 6;
      if (pmemPools_[pool].pageAllocationBitMap[index] == 0x0) {
        // no free pages
        continue;
      }

      uint64_t probe;
      probe = (uint64_t)0x1;
      for (m = 0; m < (int)(8 * sizeof(uint64_t)); m++) {
        if ((pmemPools_[pool].pageAllocationBitMap[index] & probe) == probe) {
          pmemPools_[pool].pageAllocationBitMap[index] ^= probe;

          size_t offset;

          offset = (k + m) * pageSize_;

          int8_t* addr = (pmemPools_[pool].base + offset);

          PersistentBufferDescriptor* desc;
          PersistentBufferDescriptorDirectoryHeader* current;

          current = (PersistentBufferDescriptorDirectoryHeader*)addr;
          current->numDescriptors =
              (pageSize_ - sizeof(PersistentBufferDescriptorDirectoryHeader)) /
              sizeof(PersistentBufferDescriptor);
          desc =
              (PersistentBufferDescriptor*)(addr +
                                            sizeof(
                                                PersistentBufferDescriptorDirectoryHeader));
          PmemMemSet((char*)desc,
                     0xFF,
                     current->numDescriptors * sizeof(PersistentBufferDescriptor));
          for (size_t n = 0; n < current->numDescriptors; n++) {
            putPersistentBufferDescriptor(pool, &(desc[n]));
          }

          PersistentBufferDescriptorDirectoryHeader* head;

          head = pmemPools_[pool].descriptorHeaderHead;

          current->nextHeaderOffset = head->nextHeaderOffset;
          PmemPersist(&(current->nextHeaderOffset), sizeof(current->nextHeaderOffset));
          head->nextHeaderOffset = offset;
          PmemPersist(&(head->nextHeaderOffset), sizeof(head->nextHeaderOffset));
          done = true;
          break;
        }
      }
    }
  }

  pmemPools_[pool].allocationMutex.unlock();
  return done;
}

PersistentBufferDescriptor* PersistentBufferMgr::getPersistentBufferDescriptor(int pool) {
  PersistentBufferDescriptor* p;

  p = NULL;

  if (pool < numPools_) {
    pmemPools_[pool].freeDescriptorMutex.lock();
    if (pmemPools_[pool].freeBufferDescriptors.size() > 0) {
      p = pmemPools_[pool].freeBufferDescriptors.front();
      pmemPools_[pool].freeBufferDescriptors.pop_front();

    } else {
      if (allocatePersistentBufferDescriptors(pool)) {
        p = pmemPools_[pool].freeBufferDescriptors.front();
        pmemPools_[pool].freeBufferDescriptors.pop_front();
      }
    }
    pmemPools_[pool].freeDescriptorMutex.unlock();
  }
  return p;
}

void PersistentBufferMgr::putPersistentBufferDescriptor(int pool,
                                                        PersistentBufferDescriptor* p) {
  if (p && (pool < numPools_)) {
    if (p->slabSeq != -1) {
      PmemMemSet((char*)p, 0xFF, sizeof(*p));
    }
    pmemPools_[pool].freeDescriptorMutex.lock();
    pmemPools_[pool].freeBufferDescriptors.push_back(p);
    pmemPools_[pool].freeDescriptorMutex.unlock();
  }
}

int8_t* PersistentBufferMgr::getPersistentBufferAddress(int pool,
                                                        PersistentBufferDescriptor* p) {
  if (pool < numPools_) {
    return (pmemPools_[pool].base + p->slabSeq * slabSize_ + p->startPage * pageSize_);
  }

  return NULL;
}

bool PersistentBufferMgr::isPersistentSlabAllocatedUnlocked(int pool, int slab) {
  return (pmemPools_[pool].slabAllocationMap[slab] == 1);
}

PageKey PersistentBufferMgr::getFreePersistentBuffer(size_t pagesNeeded,
                                                     PersistentBufferDescriptor** p) {
  if (pagesNeeded > pagesPerSlab_) {
    LOG(FATAL) << "Number of pages to allocate exceeds numbers of pages in a slab"
               << std::endl;
  }

  PageKey pkey;

  pkey.poolId = -1;

  int i;
  size_t j, k;
  size_t n;
  int m;
  PersistentBufferDescriptor* q = NULL;

  bool done = false;
  for (i = 0; (i < numPools_) && !done; i++) {
    if (q) {
      // buffer descriptor reserved for a differnt pool
      putPersistentBufferDescriptor(i - 1, q);
    }
    // reserve a descriptor
    q = getPersistentBufferDescriptor(i);
    if (q == NULL) {
      // no descriptor available in this pool
      // move to the next pool
      continue;
    }
    pmemPools_[i].allocationMutex.lock();
    for (j = 0; (j < pmemPools_[i].header->numSlabs) && !done; j++) {
      if (isPersistentSlabAllocatedUnlocked(i, j) == false) {
        continue;
      }

      n = 0;
      for (k = j * pagesPerSlab_; (k < ((j + 1) * pagesPerSlab_));
           k += (sizeof(uint64_t) * 8)) {
        size_t index;

        index = k >> 6;
        if (pmemPools_[i].pageAllocationBitMap[index] == 0x0) {
          // no free page
          n = 0;
          continue;
        }
        uint64_t probe;
        probe = (uint64_t)0x1;
        for (m = 0; m < (int)(sizeof(uint64_t) * 8); m++) {
          if ((pmemPools_[i].pageAllocationBitMap[index] & probe) == probe) {
            n++;
            if (n == pagesNeeded) {
              done = true;
              break;
            }
          } else {
            n = 0;
          }
          probe = (probe << 1);
        }
        if (done) {
          break;
        }
      }

      if (done) {
        done = false;
        for (; (k >= (j * pagesPerSlab_)) && !done; k -= (sizeof(uint64_t) * 8)) {
          size_t index;

          index = k >> 6;
          uint64_t probe;
          probe = ((uint64_t)0x1 << m);
          for (; m >= 0; m--) {
            pmemPools_[i].pageAllocationBitMap[index] ^= probe;
            probe = (probe >> 1);
            n--;
            if (n == 0) {
              done = true;
              break;
            }
          }
          if (done) {
            break;
          }
          m = sizeof(uint64_t) * 8 - 1;
        }
        pkey.poolId = i;
        pkey.pageSeq = k + m;
      }
    }
    pmemPools_[i].allocationMutex.unlock();
  }

  if (done) {
    *p = q;
    return pkey;
  }

  putPersistentBufferDescriptor(i - 1, q);

  // allocate a new slab
  SlabKey key;

  key = allocatePersistentSlab();
  if (key.poolId == -1) {
    LOG(FATAL) << "OUT OF PMEM" << std::endl;
  }

  return getFreePersistentBuffer(pagesNeeded, p);
}

int8_t* PersistentBufferMgr::allocatePersistentBuffer(ChunkKey key,
                                                      size_t numBytes,
                                                      PersistentBufferDescriptor** desc) {
  if (numBytes == 0) {  // allocate an empty buffer
    *desc = NULL;
    return NULL;
  }

  size_t numPages = (numBytes + pageSize_ - 1) / pageSize_;

  // numPages = numPages * 128;
  PersistentBufferDescriptor* p = NULL;

  auto pkey = getFreePersistentBuffer(numPages, &p);

  if (pkey.poolId == -1) {
    return NULL;
  }

  p->slabSeq = pkey.pageSeq / pagesPerSlab_;
  p->chunkKey[0] = key[0];
  p->chunkKey[1] = key[1];
  p->chunkKey[2] = key[2];
  p->chunkKey[3] = key[3];
  if (key.size() > 4) {
    p->chunkKey[4] = key[4];
  } else {
    p->chunkKey[4] = 0;
  }
  p->numPages = numPages;
  p->startPage = pkey.pageSeq % pagesPerSlab_;
  p->size = 0;
  p->epoch = -1;

  PmemPersist(p, sizeof(*p));

  *desc = p;

  return getPersistentBufferAddress(pkey.poolId, p);
}

int PersistentBufferMgr::getPersistentBufferPoolId(int8_t* addr) {
  for (int i = 0; i < numPools_; i++) {
    if ((pmemPools_[i].base <= addr) && (pmemPools_[i].ceiling > addr)) {
      return i;
    }
  }

  return -1;
}

void PersistentBufferMgr::freePersistentBuffer(PersistentBufferDescriptor* p,
                                               int8_t* addr) {
  if (p) {
    int poolId = getPersistentBufferPoolId(addr);

    int pageId = p->slabSeq * pagesPerSlab_ + p->startPage;

    int n = 0;
    pmemPools_[poolId].allocationMutex.lock();

    bool done = false;
    size_t j = pageId % (sizeof(uint64_t) * 8);
    for (int i = (pageId / (sizeof(uint64_t) * 8)); !done; i++) {
      uint64_t probe;
      probe = ((uint64_t)0x1 << j);
      for (; (j < sizeof(uint64_t) * 8); j++) {
        pmemPools_[poolId].pageAllocationBitMap[i] ^= probe;
        n++;
        if (n == p->numPages) {
          done = true;
          break;
        }
        probe = (probe << 1);
      }
      j = 0;
    }
    pmemPools_[poolId].allocationMutex.unlock();

    putPersistentBufferDescriptor(poolId, p);
  }
}

void PersistentBufferMgr::shrinkPersistentBuffer(PersistentBufferDescriptor* p,
                                                 int8_t* addr) {
  if (p) {
    int newNumPages = (p->size + pageSize_ - 1) / pageSize_;

    if (newNumPages == 0) {
      newNumPages = 1;  // at least one page
    }

    if (newNumPages >= p->numPages) {
      return;
    }

    int pageId = p->slabSeq * pagesPerSlab_ + p->startPage + newNumPages;
    int n = 0;
    int poolId = getPersistentBufferPoolId(addr);
    pmemPools_[poolId].allocationMutex.lock();

    bool done = false;
    size_t j = pageId % (sizeof(uint64_t) * 8);
    for (int i = (pageId / (sizeof(uint64_t) * 8)); !done; i++) {
      uint64_t probe;
      probe = ((uint64_t)0x1 << j);
      for (; (j < sizeof(uint64_t) * 8); j++) {
        pmemPools_[poolId].pageAllocationBitMap[i] ^= probe;
        n++;
        if (n == p->numPages - newNumPages) {
          done = true;
          break;
        }
        probe = (probe << 1);
      }
      j = 0;
    }
    pmemPools_[poolId].allocationMutex.unlock();

    p->numPages = newNumPages;

    PmemPersist(&(p->numPages), sizeof(p->numPages));
  }
}

int8_t* PersistentBufferMgr::reallocatePersistentBuffer(
    ChunkKey key,
    int8_t* addr,
    size_t numBytes,
    PersistentBufferDescriptor** desc) {
  if (numBytes > slabSize_) {
    LOG(FATAL) << "Buffer size exceeds persostent slab size " << slabSize_ << std::endl;
  }

  int pagesNeeded = (numBytes + pageSize_ - 1) / pageSize_;

  PersistentBufferDescriptor* p = *desc;
  if (p == NULL) {
    if (addr) {
      LOG(FATAL) << "Address corrupted" << std::endl;
    }
    return allocatePersistentBuffer(key, numBytes, desc);
  }

  int poolId;

  poolId = getPersistentBufferPoolId(addr);

  if (numBytes <= p->numPages * pageSize_) {
    // exisitng buffer is big enough
    return getPersistentBufferAddress(poolId, p);
  }

  // first try to extend existing buffer
  int extraPagesNeeded;

  extraPagesNeeded = pagesNeeded - p->numPages;
  int pageId = (p->slabSeq * pagesPerSlab_) + p->startPage + p->numPages;
  bool done;

  done = false;
  int m = pageId % (sizeof(uint64_t) * 8);

  int n = 0;
  pmemPools_[poolId].allocationMutex.lock();
  for (size_t i = pageId; (i < ((p->slabSeq + 1) * pagesPerSlab_)) && !done;
       i += (sizeof(uint64_t) * 8)) {
    // try to extend the same slab
    size_t index;
    index = i >> 6;
    uint64_t probe;
    probe = ((uint64_t)0x1 << m);
    for (; m < (int)(sizeof(uint64_t) * 8); m++) {
      if ((pmemPools_[poolId].pageAllocationBitMap[index] & probe) == probe) {
        n++;
        if (n == extraPagesNeeded) {
          done = true;
          break;
        }
      } else {
        done = true;
        break;
      }
      probe = (probe << 0x1);
    }
    m = 0;
  }

  done = false;                 // reset flag
  if (n == extraPagesNeeded) {  // safe to extend
    n = 0;
    m = pageId % (sizeof(uint64_t) * 8);
    for (int i = pageId / (sizeof(uint64_t) * 8); !done; i++) {
      uint64_t probe;
      probe = ((uint64_t)0x1 << m);
      for (; m < (int)(sizeof(uint64_t) * 8); m++) {
        pmemPools_[poolId].pageAllocationBitMap[i] ^= probe;
        n++;
        if (n == extraPagesNeeded) {
          done = true;
          break;
        }
        probe = (probe << 0x1);
      }
      m = 0;
    }

    p->numPages = pagesNeeded;
    PmemPersist(&(p->numPages), sizeof(p->numPages));
  }
  pmemPools_[poolId].allocationMutex.unlock();

  if (done) {
    return addr;
  }

  int8_t* newAddr;
  PersistentBufferDescriptor* q = NULL;

  newAddr = allocatePersistentBuffer(key, numBytes, &q);
  if (newAddr) {
    PmemMemCpy((char*)newAddr, (char*)addr, p->numPages * pageSize_);
    freePersistentBuffer(p, addr);

    *desc = q;
    return newAddr;
  }

  return NULL;
}

SlabKey PersistentBufferMgr::allocatePersistentSlab(void) {
  size_t i;
  int index;
  SlabKey key;

  // index = __sync_val_compare_and_swap(&curPool_, numPools_, 0);

  index = curPool_;

  for (int j = 0; j < numPools_; j++) {
    if (pmemPools_[index].numFreeSlabs) {
      pmemPools_[index].allocationMutex.lock();
      for (i = 0; i < pmemPools_[index].header->numSlabs; i++) {
        if (pmemPools_[index].slabAllocationMap[i] == 0) {
          pmemPools_[index].slabAllocationMap[i] = 1;
          // flush
          PmemPersist(&(pmemPools_[index].slabAllocationMap[i]),
                      sizeof(pmemPools_[index].slabAllocationMap[i]));

          pmemPools_[index].numFreeSlabs--;

          curPool_ = index + 1;
          ;
          if (curPool_ == numPools_) {
            curPool_ = 0;
          }

          break;
        }
      }
      pmemPools_[index].allocationMutex.unlock();
      if (i != pmemPools_[index].header->numSlabs) {
        key.poolId = index;
        key.slabSeq = i;

        return key;
      }
    }

    index++;

    if (index == numPools_)
      index = 0;
  }
#if 0
  size_t oldv;
  size_t newv;
  while (numfreeslabs) {
  for (i = 0; i < numbitmaps; i++) {
    if (bitmap[i] != ~(size_t)(0)) {
      size_t j;
      for (j = 0; j < sizeof(size_t) * 8; j++) {
        if ((i * sizeof(size_t) * 8 + j) == numslabs)
          return NULL;
        oldv = bitmap[i];
        if ((oldv & (1 << j)) == 0) {
          newv = oldv | (1 << j);
          if (__sync_bool_compare_and_swap(&(bitmap[i]), oldv, newv)) {
            size_t left;

            left = __sync_fetch_and_sub(&numfreeslabs, 1);
            printf("%ld free slabs\n", left-1);
            return (void *)((char *)base + slabSize_ * (i * sizeof(size_t) * 8 + j));
          }
        }
      }
    }
  }
  }
#endif /* 0 */

  LOG(FATAL) << "OUT OF PMEM" << std::endl;

  key.poolId = -1;
  key.slabSeq = 0;

  return key;
}

#if 0
void
PersistentBufferMgr::freePersistentSlab(void *addr)
{
  size_t i;

  //printf("free slab %p\n", addr);

  for (unsigned int j = 0; j < numPools_; j++) {
    if (((char *)addr >= (char *)(pmemPools_[j].base)) && ((char *)addr < (char *)(pmemPools_[j].ceiling))) {
      i = (((char *)addr - (char *)(pmemPools_[j].base)) / slabSize_);
      pmemPools_[j].allocationMutex.lock();
      pmemPools_[j].slabAllocationMap[i] = 0;
      //flush
      PmemPersist(&(pmemPools_[j].slabAllocationMap[i]), sizeof(pmemPools_[j].slabAllocationMap[i]));

      pmemPools_[j].numFreeSlabs++;
      pmemPools_[j].allocationMutex.unlock();
      return;
    }
  }
#if 0
  size_t j;

  i = (((char *)addr - (char *)base) / slabSize_) / (sizeof(size_t) * 8);
  j = (((char *)addr - (char *)base) /slabSize_) % (sizeof(size_t) * 8);

  while (1) {
    size_t oldv;
    size_t newv;

    oldv = bitmap[i];
    newv = oldv & (~(1 << j));
    if (__sync_bool_compare_and_swap(&(bitmap[i]), oldv, newv)) {
      __sync_fetch_and_add(&numfreeslabs, 1);
      return;
    }
  }
#endif /* 0 */
}
#endif /* 0 */

/*
 * the pmmPath is the file containing the persistent memory file folder
 * each line in the file is a directory pathname, for example:
 * /mnt/ad1/omnisci  0    // use all availabe space
 * /mnt/ad3/omnisci  128    // use no more than 128GB
 */
int PersistentBufferMgr::initializePersistentBufferPools(const std::string& pmmPath) {
  std::vector<std::string> pmem_dirs;
  std::vector<size_t> pmem_sizes;

  std::ifstream pmem_dirs_file(pmmPath);
  if (pmem_dirs_file.is_open()) {
    std::string line;
    while (!pmem_dirs_file.eof()) {
      std::getline(pmem_dirs_file, line);
      if (!line.empty()) {
        std::stringstream ss;
        std::string path;
        size_t size;

        ss << line;
        ss >> path;
        ss >> size;

        pmem_dirs.push_back(path);
        pmem_sizes.push_back(size * 1024 * 1024 * 1024);

        numPools_++;
      }
    }
    pmem_dirs_file.close();
  } else {
    LOG(INFO) << "Unable to open file " << pmmPath << std::endl;
    return -1;
  }

  pmemPools_ = new PersistentBufferPool[numPools_];

  for (int i = 0; i < numPools_; i++) {
    int fd;
    char filename[pmem_dirs[i].length() + strlen("/omnisci.data") + 1];
    struct stat statbuf;
    int ret;

    sprintf(filename, "%s/omnisci.data", pmem_dirs[i].c_str());

    if (stat(filename, &statbuf)) {
      // persistent memory pool does not exist
      // create it
      struct statfs buf;

      if (statfs(pmem_dirs[i].c_str(), &buf)) {
        std::cout << "failed to initialize pmem " << pmem_dirs[i] << std::endl;
        return -1;
      }

      if ((fd = open(filename, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)) < 0) {
        LOG(FATAL) << "Failed to create persistent memory file: " << filename
                   << std::endl;
        exit(-1);
      }

      // zero out the header
      PersistentPoolHeader h;
      memset(&h, 0, sizeof(h));

      write(fd, &h, sizeof(h));

      // pmemPools_[i].numSlabs = (buf.f_bavail * buf.f_bsize) / slabSize;
      // pmemPools_[i].size = pmemPools_[i].numSlabs * slabSize;

      size_t poolSize = (buf.f_bavail * buf.f_bsize);
      if ((poolSize > pmem_sizes[i]) && (pmem_sizes[i] != 0)) {
        poolSize = pmem_sizes[i];
      }
      poolSize = (poolSize / slabSize_) * slabSize_;

      if ((ret = posix_fallocate(fd, 0, poolSize)) != 0) {
        printf("posix_fallcoate failed err=%d\n", ret);
        exit(-1);
      }

      if ((pmemPools_[i].header = (PersistentPoolHeader*)mmap(
               NULL, poolSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) ==
          MAP_FAILED) {
        printf("mmap failed\n");
        return -1;
      }

      close(fd);

      struct PersistentPoolHeader* header;
      header = pmemPools_[i].header;
      header->numSlabs = poolSize / slabSize_;
      header->size = poolSize;
      header->slabSize = slabSize_;
      header->pageSize = pageSize_;
      header->descriptorHeaderOffset = sizeof(PersistentPoolHeader) +
                                       ((header->numSlabs + 7) / 8 * 8) * sizeof(size_t);

      pmemPools_[i].slabAllocationMap =
          (size_t*)(pmemPools_[i].base + sizeof(struct PersistentPoolHeader));

      pmemPools_[i].descriptorHeaderHead =
          (PersistentBufferDescriptorDirectoryHeader*)((char*)(pmemPools_[i].base) +
                                                       header->descriptorHeaderOffset);

      // rest of the header will be flushed last
      PmemMemSet(
          (char*)(pmemPools_[i].slabAllocationMap),
          0,
          pmemPools_[i].header->numSlabs * sizeof(pmemPools_[i].slabAllocationMap[0]));

      pmemPools_[i].slabAllocationMap[0] = 1;  // pool header takes the first slab
      PmemPersist(&(pmemPools_[i].slabAllocationMap[0]),
                  sizeof(pmemPools_[i].slabAllocationMap[0]));

      // flush cache
      //
      pmemPools_[i].numFreeSlabs =
          pmemPools_[i].header->numSlabs - 1;  // pool header takes the 1st slab

      // initialize buffer descriptor header

      PersistentBufferDescriptorDirectoryHeader* desc_header;

      desc_header = pmemPools_[i].descriptorHeaderHead;
      // desc_header->pageSize = pageSize_;

      PersistentBufferDescriptor* bufferDescriptors;

      size_t numDescriptors =
          ((char*)pmemPools_[i].base + header->pageSize -
           (char*)pmemPools_[i].descriptorHeaderHead - sizeof(*desc_header)) /
          sizeof(PersistentBufferDescriptor);

      bufferDescriptors =
          (PersistentBufferDescriptor*)((char*)desc_header +
                                        sizeof(
                                            PersistentBufferDescriptorDirectoryHeader));

      PmemMemSet((char*)bufferDescriptors,
                 0xFF,
                 numDescriptors * sizeof(PersistentBufferDescriptor));

      desc_header->descriptorType = PMM_PERSISTENT_BUFFER_CHUNK_TYPE;
      desc_header->nextHeaderOffset = 0;

      desc_header->numDescriptors = numDescriptors;
      // flush
      PmemPersist(desc_header, sizeof(*desc_header));

      //
      // populate free BufferDescriptors
      //
      for (unsigned int j = 0; j < desc_header->numDescriptors; j++) {
        pmemPools_[i].freeBufferDescriptors.push_back(&(bufferDescriptors[j]));
      }

      header->version = PMM_PERSISTENT_BUFFER_POOL_VERSION;
      // TODO: calculate checksum
      header->magic = PMM_PERSISTENT_BUFFER_POOL_MAGIC;  // must be last to update
      PmemPersist(header, sizeof(*header));

      int num = ((header->numSlabs * header->slabSize / header->pageSize) +
                 sizeof(uint64_t) * 8 - 1) /
                (sizeof(uint64_t) * 8);
      pmemPools_[i].pageAllocationBitMap =
          (uint64_t*)checked_malloc(num * sizeof(uint64_t));

      for (int k = 0; k < num; k++) {
        pmemPools_[i].pageAllocationBitMap[k] = ~(uint64_t)0x0;
      }

      // first page in first slab is used by the header and first descriptor directory
      pmemPools_[i].pageAllocationBitMap[0] ^= (uint64_t)0x1;
    } else {  // pool already exists
      if ((fd = open(filename, O_RDWR)) < 0) {
        LOG(FATAL) << "Failed to open persistent memory file" << filename << std::endl;
        return -1;
      }

      // pmemPools_[i].size = statbuf.st_size;

      if ((pmemPools_[i].header = (PersistentPoolHeader*)mmap(
               NULL, statbuf.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) ==
          MAP_FAILED) {
        LOG(FATAL) << "Failed to mmap persistent memory " << filename << std::endl;
        return -1;
      }

      close(fd);

      struct PersistentPoolHeader* header;
      header = (struct PersistentPoolHeader*)pmemPools_[i].header;

      if (header->magic != PMM_PERSISTENT_BUFFER_POOL_MAGIC) {
        LOG(FATAL) << "Not a valid OmniSci persistent memory pool " << filename
                   << std::endl;
        return -1;
      }

      // TODO: check version and checksum
      // pmemPools_[i].numSlabs = header->numSlabs;
      // pmemPools_[i].slabSize = header->slabSize;

      if (slabSize_ != header->slabSize) {
        LOG(FATAL) << "Slab size mismatch " << filename << std::endl;
        return -1;
      }
      if (pageSize_ != header->pageSize) {
        LOG(FATAL) << "Page size mismatch " << filename << std::endl;
        return -1;
      }

      pmemPools_[i].slabAllocationMap =
          (size_t*)(pmemPools_[i].base + sizeof(PersistentPoolHeader));
      pmemPools_[i].descriptorHeaderHead =
          (PersistentBufferDescriptorDirectoryHeader*)((char*)pmemPools_[i].base +
                                                       header->descriptorHeaderOffset);

      pmemPools_[i].numFreeSlabs = pmemPools_[i].header->numSlabs;
      for (unsigned int j = 0; j < pmemPools_[i].header->numSlabs; j++) {
        if (pmemPools_[i].slabAllocationMap[j])
          pmemPools_[i].numFreeSlabs--;
      }

      int num = ((header->numSlabs * header->slabSize / header->pageSize) +
                 sizeof(uint64_t) * 8 - 1) /
                (sizeof(uint64_t) * 8);
      pmemPools_[i].pageAllocationBitMap =
          (uint64_t*)checked_malloc(num * sizeof(uint64_t));

      for (int k = 0; k < num; k++) {
        pmemPools_[i].pageAllocationBitMap[k] = ~(uint64_t)0x0;
      }

      // first page in first slab is used by the header
      pmemPools_[i].pageAllocationBitMap[0] ^= (uint64_t)0x1;

      PersistentBufferDescriptorDirectoryHeader* temp;

      temp = pmemPools_[i].descriptorHeaderHead;
      do {
        int linearPageId;

        linearPageId =
            temp->slabSeq * (header->slabSize / header->pageSize) + temp->startPage;
        for (int k = 0; k < temp->numPages; k++) {
          int m = linearPageId / (sizeof(uint64_t) * 8);
          int n = linearPageId % (sizeof(uint64_t) * 8);

          pmemPools_[i].pageAllocationBitMap[m] ^= ((uint64_t)0x1 << n);
          linearPageId++;
        }

        PersistentBufferDescriptor* bufferDescriptors;

        bufferDescriptors =
            (PersistentBufferDescriptor*)((char*)temp +
                                          sizeof(
                                              PersistentBufferDescriptorDirectoryHeader));
        for (size_t j = 0; j < temp->numDescriptors; j++) {
          if (bufferDescriptors[j].slabSeq == -1) {
            pmemPools_[i].freeBufferDescriptors.push_back(&(bufferDescriptors[j]));
          } else {
            linearPageId =
                bufferDescriptors[j].slabSeq * (header->slabSize / header->pageSize) +
                bufferDescriptors[j].startPage;
            for (int k = 0; k < bufferDescriptors[j].numPages; k++) {
              int m = linearPageId / (sizeof(uint64_t) * 8);
              int n = linearPageId % (sizeof(uint64_t) * 8);

              pmemPools_[i].pageAllocationBitMap[m] ^= ((uint64_t)0x1 << n);
              linearPageId++;
            }

            DbTblKey dtkey;

            dtkey.dbId = bufferDescriptors[j].chunkKey[0];
            dtkey.tblId = bufferDescriptors[j].chunkKey[1];

            auto descIt = pmemPools_[i].tableBufferDescriptorMap.find(dtkey.dBTblKey);
            if (descIt == pmemPools_[i].tableBufferDescriptorMap.end()) {
              std::vector<PersistentBufferDescriptor*> descVec;
              descVec.push_back(&(bufferDescriptors[j]));
              pmemPools_[i].tableBufferDescriptorMap.insert(
                  std::pair<int64_t, std::vector<PersistentBufferDescriptor*>>(
                      dtkey.dBTblKey, descVec));
            } else {
              descIt->second.push_back(&(bufferDescriptors[j]));
            }
          }
        }
        if (temp->nextHeaderOffset) {
          temp = (PersistentBufferDescriptorDirectoryHeader*)(pmemPools_[i].base +
                                                              temp->nextHeaderOffset);
        } else {
          temp = NULL;
        }
      } while (temp != NULL);
    }

    pmemPools_[i].ceiling = (int8_t*)(pmemPools_[i].base) + pmemPools_[i].header->size;
  }

  curPool_ = 0;

  return 0;
}

}  // namespace File_Namespace

#endif /* HAVE_DCPMM */