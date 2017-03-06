/**
 * @file    DataMgr.h
 * @author Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_H
#define DATAMGR_H

#include "AbstractBuffer.h"
#include "AbstractBufferMgr.h"
#include "MemoryLevel.h"

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace File_Namespace {
class FileBuffer;
}

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Data_Namespace {

struct gpuMemorySummary {
  int64_t max;
  int64_t inUse;
  int64_t allocated;
  bool isAllocationCapped;  // mean allocation request failed
};

struct memorySummary {
  int64_t cpuMemoryInUse;
  std::vector<gpuMemorySummary> gpuSummary;
};

class DataMgr {
  friend class GlobalFileMgr;

 public:
  DataMgr(const std::string& dataDir,
          const size_t cpuBufferSize /* 0 means auto set size */,
          const bool useGpus,
          const int numGpus,
          const std::string& dbConvertDir = "",
          const int startGpu = 0,
          const size_t reservedGpuMem = (1 << 27),
          const int start_epoch = -1,
          const size_t numReaderThreads = 0); /* 0 means use default for # of reader threads */
  ~DataMgr();
  AbstractBuffer* createChunkBuffer(const ChunkKey& key, const MemoryLevel memoryLevel, const int deviceId = 0);
  AbstractBuffer* getChunkBuffer(const ChunkKey& key,
                                 const MemoryLevel memoryLevel,
                                 const int deviceId = 0,
                                 const size_t numBytes = 0);
  void deleteChunksWithPrefix(const ChunkKey& keyPrefix);
  AbstractBuffer* alloc(const MemoryLevel memoryLevel, const int deviceId, const size_t numBytes);
  void free(AbstractBuffer* buffer);
  void freeAllBuffers();
  // copies one buffer to another
  void copy(AbstractBuffer* destBuffer, AbstractBuffer* srcBuffer);
  bool isBufferOnDevice(const ChunkKey& key, const MemoryLevel memLevel, const int deviceId);
  memorySummary getMemorySummary();
  std::string dumpLevel(const MemoryLevel memLevel);
  void clearMemory(const MemoryLevel memLevel);

  // const std::map<ChunkKey, File_Namespace::FileBuffer *> & getChunkMap();
  const std::map<ChunkKey, File_Namespace::FileBuffer*>& getChunkMap();
  void checkpoint(const int db_id, const int tb_id); // checkpoint for individual table of DB
  void getChunkMetadataVec(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec);
  void getChunkMetadataVecForKeyPrefix(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
                                       const ChunkKey& keyPrefix);
  inline bool gpusPresent() { return hasGpus_; }

  CudaMgr_Namespace::CudaMgr* cudaMgr_;

  // database_id, table_id, column_id, fragment_id
  std::vector<int> levelSizes_;

 private:
  size_t getTotalSystemMemory();
  void populateMgrs(const size_t userSpecifiedCpuBufferSize,
                    const size_t userSpecifiedNumReaderThreads,
                    const int start_epoch);
  void convertDB(const std::string basePath);
  void checkpoint(); // checkpoint for whole DB, called from convertDB proc only
  void createTopLevelMetadata() const;
  std::vector<std::vector<AbstractBufferMgr*>> bufferMgrs_;
  std::string dataDir_;
  bool hasGpus_;
  size_t reservedGpuMem_;
  std::string dbConvertDir_;
};
}  // Data_Namespace

#endif  // DATAMGR_H
