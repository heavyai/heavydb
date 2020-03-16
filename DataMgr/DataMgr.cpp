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
 * @file    DataMgr.cpp
 * @author Todd Mostak <todd@mapd.com>
 */

#include "DataMgr.h"
#include "../CudaMgr/CudaMgr.h"
#include "BufferMgr/CpuBufferMgr/CpuBufferMgr.h"
#include "BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.h"
#include "FileMgr/GlobalFileMgr.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

#include <boost/filesystem.hpp>

#include <algorithm>
#include <limits>

using namespace std;
using namespace Buffer_Namespace;
using namespace File_Namespace;

namespace Data_Namespace {

DataMgr::DataMgr(const string& dataDir,
                 const MapDParameters& mapd_parameters,
                 const bool useGpus,
                 const int numGpus,
                 const int startGpu,
                 const size_t reservedGpuMem,
                 const size_t numReaderThreads)
    : dataDir_(dataDir) {
  if (useGpus) {
    try {
      cudaMgr_ = std::make_unique<CudaMgr_Namespace::CudaMgr>(numGpus, startGpu);
      reservedGpuMem_ = reservedGpuMem;
      hasGpus_ = true;
    } catch (std::runtime_error& error) {
      hasGpus_ = false;
    }
  } else {
    hasGpus_ = false;
  }

  populateMgrs(mapd_parameters, numReaderThreads);
  createTopLevelMetadata();
}

DataMgr::~DataMgr() {
  int numLevels = bufferMgrs_.size();
  for (int level = numLevels - 1; level >= 0; --level) {
    for (size_t device = 0; device < bufferMgrs_[level].size(); device++) {
      delete bufferMgrs_[level][device];
    }
  }
}

DataMgr::SystemMemoryUsage DataMgr::getSystemMemoryUsage() const {
  SystemMemoryUsage usage;

#ifdef __linux__

  // Determine Linux available memory and total memory.
  // Available memory is different from free memory because
  // when Linux sees free memory, it tries to use it for
  // stuff like disk caching. However, the memory is not
  // reserved and is still available to be allocated by
  // user processes.
  // Parsing /proc/meminfo for this info isn't very elegant
  // but as a virtual file it should be reasonably fast.
  // See also:
  //   https://github.com/torvalds/linux/commit/34e431b0ae398fc54ea69ff85ec700722c9da773
  ProcMeminfoParser mi;
  usage.free = mi["MemAvailable"];
  usage.total = mi["MemTotal"];

  // Determine process memory in use.
  // See also:
  //   https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
  //   http://man7.org/linux/man-pages/man5/proc.5.html
  int64_t size = 0;
  int64_t resident = 0;
  int64_t shared = 0;

  std::ifstream fstatm("/proc/self/statm");
  fstatm >> size >> resident >> shared;
  fstatm.close();

  long page_size =
      sysconf(_SC_PAGE_SIZE);  // in case x86-64 is configured to use 2MB pages

  usage.resident = resident * page_size;
  usage.vtotal = size * page_size;
  usage.regular = (resident - shared) * page_size;
  usage.shared = shared * page_size;

#else

  usage.total = 0;
  usage.free = 0;
  usage.resident = 0;
  usage.vtotal = 0;
  usage.regular = 0;
  usage.shared = 0;

#endif

  return usage;
}

size_t DataMgr::getTotalSystemMemory() const {
#ifdef __APPLE__
  int mib[2];
  size_t physical_memory;
  size_t length;
  // Get the Physical memory size
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  length = sizeof(size_t);
  sysctl(mib, 2, &physical_memory, &length, NULL, 0);
  return physical_memory;

#else  // Linux
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
#endif
}

void DataMgr::populateMgrs(const MapDParameters& mapd_parameters,
                           const size_t userSpecifiedNumReaderThreads) {
  bufferMgrs_.resize(2);
  bufferMgrs_[0].push_back(new GlobalFileMgr(0, dataDir_, userSpecifiedNumReaderThreads));
  levelSizes_.push_back(1);
  size_t cpuBufferSize = mapd_parameters.cpu_buffer_mem_bytes;
  if (cpuBufferSize == 0) {  // if size is not specified
    const auto total_system_memory = getTotalSystemMemory();
    VLOG(1) << "Detected " << (float)total_system_memory / (1024 * 1024)
            << "M of total system memory.";
    cpuBufferSize = total_system_memory *
                    0.8;  // should get free memory instead of this ugly heuristic
  }
  size_t cpuSlabSize = std::min(static_cast<size_t>(1L << 32), cpuBufferSize);
  // cpuSlabSize -= cpuSlabSize % 512 == 0 ? 0 : 512 - (cpuSlabSize % 512);
  cpuSlabSize = (cpuSlabSize / 512) * 512;
  LOG(INFO) << "cpuSlabSize is " << (float)cpuSlabSize / (1024 * 1024) << "M";
  LOG(INFO) << "memory pool for CPU is " << (float)cpuBufferSize / (1024 * 1024) << "M";
  if (hasGpus_) {
    LOG(INFO) << "reserved GPU memory is " << (float)reservedGpuMem_ / (1024 * 1024)
              << "M includes render buffer allocation";
    bufferMgrs_.resize(3);
    bufferMgrs_[1].push_back(new CpuBufferMgr(
        0, cpuBufferSize, cudaMgr_.get(), cpuSlabSize, 512, bufferMgrs_[0][0]));
    levelSizes_.push_back(1);
    int numGpus = cudaMgr_->getDeviceCount();
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      size_t gpuMaxMemSize =
          mapd_parameters.gpu_buffer_mem_bytes != 0
              ? mapd_parameters.gpu_buffer_mem_bytes
              : (cudaMgr_->getDeviceProperties(gpuNum)->globalMem) - (reservedGpuMem_);
      size_t gpuSlabSize = std::min(static_cast<size_t>(1L << 31), gpuMaxMemSize);
      gpuSlabSize -= gpuSlabSize % 512 == 0 ? 0 : 512 - (gpuSlabSize % 512);
      LOG(INFO) << "gpuSlabSize is " << (float)gpuSlabSize / (1024 * 1024) << "M";
      LOG(INFO) << "memory pool for GPU " << gpuNum << " is "
                << (float)gpuMaxMemSize / (1024 * 1024) << "M";
      bufferMgrs_[2].push_back(new GpuCudaBufferMgr(
          gpuNum, gpuMaxMemSize, cudaMgr_.get(), gpuSlabSize, 512, bufferMgrs_[1][0]));
    }
    levelSizes_.push_back(numGpus);
  } else {
    bufferMgrs_[1].push_back(new CpuBufferMgr(
        0, cpuBufferSize, cudaMgr_.get(), cpuSlabSize, 512, bufferMgrs_[0][0]));
    levelSizes_.push_back(1);
  }
}

void DataMgr::convertDB(const std::string basePath) {
  /* check that "mapd_data" directory exists and it's empty */
  std::string mapdDataPath(basePath + "/../mapd_data/");
  boost::filesystem::path path(mapdDataPath);
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path)) {
      LOG(FATAL) << "Path to directory mapd_data to convert DB is not a directory.";
    }
  } else {  // data directory does not exist
    LOG(FATAL) << "Path to directory mapd_data to convert DB does not exist.";
  }

  GlobalFileMgr* gfm = dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0]);
  size_t defaultPageSize = gfm->getDefaultPageSize();
  LOG(INFO) << "Database conversion started.";
  FileMgr* fm_base_db =
      new FileMgr(gfm,
                  defaultPageSize,
                  basePath);  // this call also copies data into new DB structure
  delete fm_base_db;

  /* write content of DB into newly created/converted DB structure & location */
  checkpoint();  // outputs data files as well as metadata files
  LOG(INFO) << "Database conversion completed.";
}

void DataMgr::createTopLevelMetadata()
    const {  // create metadata shared by all tables of all DBs
  ChunkKey chunkKey(2);
  chunkKey[0] = 0;  // top level db_id
  chunkKey[1] = 0;  // top level tb_id

  GlobalFileMgr* gfm = dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0]);
  auto fm_top = gfm->getFileMgr(chunkKey);
  if (dynamic_cast<File_Namespace::FileMgr*>(fm_top)) {
    static_cast<File_Namespace::FileMgr*>(fm_top)->createTopLevelMetadata();
  }
}

std::vector<MemoryInfo> DataMgr::getMemoryInfo(const MemoryLevel memLevel) {
  // TODO (vraj) : Reduce the duplicate code
  std::vector<MemoryInfo> memInfo;
  if (memLevel == MemoryLevel::CPU_LEVEL) {
    CpuBufferMgr* cpuBuffer =
        dynamic_cast<CpuBufferMgr*>(bufferMgrs_[MemoryLevel::CPU_LEVEL][0]);
    MemoryInfo mi;

    mi.pageSize = cpuBuffer->getPageSize();
    mi.maxNumPages = cpuBuffer->getMaxSize() / mi.pageSize;
    mi.isAllocationCapped = cpuBuffer->isAllocationCapped();
    mi.numPageAllocated = cpuBuffer->getAllocated() / mi.pageSize;

    const std::vector<BufferList> slab_segments = cpuBuffer->getSlabSegments();
    size_t numSlabs = slab_segments.size();

    for (size_t slabNum = 0; slabNum != numSlabs; ++slabNum) {
      for (auto segIt : slab_segments[slabNum]) {
        MemoryData md;
        md.slabNum = slabNum;
        md.startPage = segIt.start_page;
        md.numPages = segIt.num_pages;
        md.touch = segIt.last_touched;
        md.memStatus = segIt.mem_status;
        md.chunk_key.insert(
            md.chunk_key.end(), segIt.chunk_key.begin(), segIt.chunk_key.end());
        mi.nodeMemoryData.push_back(md);
      }
    }
    memInfo.push_back(mi);
  } else if (hasGpus_) {
    int numGpus = cudaMgr_->getDeviceCount();
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      GpuCudaBufferMgr* gpuBuffer =
          dynamic_cast<GpuCudaBufferMgr*>(bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]);
      MemoryInfo mi;

      mi.pageSize = gpuBuffer->getPageSize();
      mi.maxNumPages = gpuBuffer->getMaxSize() / mi.pageSize;
      mi.isAllocationCapped = gpuBuffer->isAllocationCapped();
      mi.numPageAllocated = gpuBuffer->getAllocated() / mi.pageSize;
      const std::vector<BufferList> slab_segments = gpuBuffer->getSlabSegments();
      size_t numSlabs = slab_segments.size();

      for (size_t slabNum = 0; slabNum != numSlabs; ++slabNum) {
        for (auto segIt : slab_segments[slabNum]) {
          MemoryData md;
          md.slabNum = slabNum;
          md.startPage = segIt.start_page;
          md.numPages = segIt.num_pages;
          md.touch = segIt.last_touched;
          md.chunk_key.insert(
              md.chunk_key.end(), segIt.chunk_key.begin(), segIt.chunk_key.end());
          md.memStatus = segIt.mem_status;
          mi.nodeMemoryData.push_back(md);
        }
      }
      memInfo.push_back(mi);
    }
  }
  return memInfo;
}

/*
std::vector<MemoryData> DataMgr::getGpuMemory() {
  std::vector<MemoryData> memInfo;
  if (hasGpus_) {
    int numGpus = cudaMgr_->getDeviceCount();
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      gpuMemorySummary gms;
      gms.max = bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]->getMaxSize();
      gms.inUse = bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]->getInUseSize();
      gms.allocated = bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]->getAllocated();
      gms.isAllocationCapped =
bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]->isAllocationCapped(); memInfo.push_back(gms);
    }
  }
  return memInfo;
}

*/
//  std::ostringstream tss;
//  size_t mb = 1024 * 1024;
//  tss << std::endl;
//  // tss << "CPU RAM TOTAL AVAILABLE   : "  std::fixed << setw(9) << setprecision(2) <<
//  // ((float)bufferMgrs_[MemoryLevel::CPU_LEVEL][0]->getMaxSize() / mb)
//  //    << std::endl;
//  tss << "CPU RAM IN BUFFER USE     : " << std::fixed << setw(9) << setprecision(2)
//      << ((float)bufferMgrs_[MemoryLevel::CPU_LEVEL][0]->getInUseSize() / mb) << " MB"
//      << std::endl;
//  if (hasGpus_) {
//    int numGpus = cudaMgr_->getDeviceCount();
//    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
//      tss << "GPU" << setfill(' ') << setw(2) << gpuNum << " RAM TOTAL AVAILABLE : " <<
//      std::fixed << setw(9)
//          << setprecision(2) <<
//          ((float)bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]->getMaxSize() / mb) << "
//          MB"
//          << std::endl;
//      tss << "GPU" << setfill(' ') << setw(2) << gpuNum << " RAM IN BUFFER USE   : " <<
//      std::fixed << setw(9)
//          << setprecision(2) <<
//          ((float)bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]->getInUseSize() / mb) << "
//          MB"
//          << std::endl;
//    }
//  }
//  return tss.str();
//}

std::string DataMgr::dumpLevel(const MemoryLevel memLevel) {
  // if gpu we need to iterate through all the buffermanagers for each card
  if (memLevel == MemoryLevel::GPU_LEVEL) {
    int numGpus = cudaMgr_->getDeviceCount();
    std::ostringstream tss;
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      tss << bufferMgrs_[memLevel][gpuNum]->printSlabs();
    }
    return tss.str();
  } else {
    return bufferMgrs_[memLevel][0]->printSlabs();
  }
}

void DataMgr::clearMemory(const MemoryLevel memLevel) {
  // if gpu we need to iterate through all the buffermanagers for each card
  if (memLevel == MemoryLevel::GPU_LEVEL) {
    if (cudaMgr_) {
      int numGpus = cudaMgr_->getDeviceCount();
      for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
        LOG(INFO) << "clear slabs on gpu " << gpuNum;
        bufferMgrs_[memLevel][gpuNum]->clearSlabs();
      }
    } else {
      throw std::runtime_error("Unable to clear GPU memory: No GPUs detected");
    }
  } else {
    bufferMgrs_[memLevel][0]->clearSlabs();
  }
}

bool DataMgr::isBufferOnDevice(const ChunkKey& key,
                               const MemoryLevel memLevel,
                               const int deviceId) {
  return bufferMgrs_[memLevel][deviceId]->isBufferOnDevice(key);
}

void DataMgr::getChunkMetadataVec(
    std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec) {
  // Can we always assume this will just be at the disklevel bc we just
  // started?
  bufferMgrs_[0][0]->getChunkMetadataVec(chunkMetadataVec);
}

void DataMgr::getChunkMetadataVecForKeyPrefix(
    std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
    const ChunkKey& keyPrefix) {
  bufferMgrs_[0][0]->getChunkMetadataVecForKeyPrefix(chunkMetadataVec, keyPrefix);
}

AbstractBuffer* DataMgr::createChunkBuffer(const ChunkKey& key,
                                           const MemoryLevel memoryLevel,
                                           const int deviceId,
                                           const size_t page_size) {
  int level = static_cast<int>(memoryLevel);
  return bufferMgrs_[level][deviceId]->createBuffer(key, page_size);
}

AbstractBuffer* DataMgr::getChunkBuffer(const ChunkKey& key,
                                        const MemoryLevel memoryLevel,
                                        const int deviceId,
                                        const size_t numBytes) {
  const auto level = static_cast<size_t>(memoryLevel);
  CHECK_LT(level, levelSizes_.size());     // make sure we have a legit buffermgr
  CHECK_LT(deviceId, levelSizes_[level]);  // make sure we have a legit buffermgr
  return bufferMgrs_[level][deviceId]->getBuffer(key, numBytes);
}

void DataMgr::deleteChunksWithPrefix(const ChunkKey& keyPrefix) {
  int numLevels = bufferMgrs_.size();
  for (int level = numLevels - 1; level >= 0; --level) {
    for (int device = 0; device < levelSizes_[level]; ++device) {
      bufferMgrs_[level][device]->deleteBuffersWithPrefix(keyPrefix);
    }
  }
}

// only deletes the chunks at the given memory level
void DataMgr::deleteChunksWithPrefix(const ChunkKey& keyPrefix,
                                     const MemoryLevel memLevel) {
  if (bufferMgrs_.size() <= memLevel) {
    return;
  }
  for (int device = 0; device < levelSizes_[memLevel]; ++device) {
    bufferMgrs_[memLevel][device]->deleteBuffersWithPrefix(keyPrefix);
  }
}

AbstractBuffer* DataMgr::alloc(const MemoryLevel memoryLevel,
                               const int deviceId,
                               const size_t numBytes) {
  const auto level = static_cast<int>(memoryLevel);
  CHECK_LT(deviceId, levelSizes_[level]);
  return bufferMgrs_[level][deviceId]->alloc(numBytes);
}

void DataMgr::free(AbstractBuffer* buffer) {
  int level = static_cast<int>(buffer->getType());
  bufferMgrs_[level][buffer->getDeviceId()]->free(buffer);
}

void DataMgr::freeAllBuffers() {
  ChunkKey keyPrefix = {-1};
  deleteChunksWithPrefix(keyPrefix);
}

void DataMgr::copy(AbstractBuffer* destBuffer, AbstractBuffer* srcBuffer) {
  destBuffer->write(srcBuffer->getMemoryPtr(),
                    srcBuffer->size(),
                    0,
                    srcBuffer->getType(),
                    srcBuffer->getDeviceId());
}

// could add function below to do arbitrary copies between buffers

// void DataMgr::copy(AbstractBuffer *destBuffer, const AbstractBuffer *srcBuffer, const
// size_t numBytes, const size_t destOffset, const size_t srcOffset) {
//} /

void DataMgr::checkpoint(const int db_id, const int tb_id) {
  for (auto levelIt = bufferMgrs_.rbegin(); levelIt != bufferMgrs_.rend(); ++levelIt) {
    // use reverse iterator so we start at GPU level, then CPU then DISK
    for (auto deviceIt = levelIt->begin(); deviceIt != levelIt->end(); ++deviceIt) {
      (*deviceIt)->checkpoint(db_id, tb_id);
    }
  }
}

void DataMgr::checkpoint() {
  for (auto levelIt = bufferMgrs_.rbegin(); levelIt != bufferMgrs_.rend(); ++levelIt) {
    // use reverse iterator so we start at GPU level, then CPU then DISK
    for (auto deviceIt = levelIt->begin(); deviceIt != levelIt->end(); ++deviceIt) {
      (*deviceIt)->checkpoint();
    }
  }
}

void DataMgr::removeTableRelatedDS(const int db_id, const int tb_id) {
  dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0])->removeTableRelatedDS(db_id, tb_id);
}

void DataMgr::setTableEpoch(const int db_id, const int tb_id, const int start_epoch) {
  dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0])
      ->setTableEpoch(db_id, tb_id, start_epoch);
}

size_t DataMgr::getTableEpoch(const int db_id, const int tb_id) {
  return dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0])->getTableEpoch(db_id, tb_id);
}

GlobalFileMgr* DataMgr::getGlobalFileMgr() const {
  auto global_file_mgr = dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0]);
  CHECK(global_file_mgr);
  return global_file_mgr;
}

}  // namespace Data_Namespace
