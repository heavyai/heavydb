/**
 * @file    DataMgr.h
 * @author Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_H
#define DATAMGR_H

#include "AbstractBufferMgr.h"
#include "AbstractBuffer.h"
#include "MemoryLevel.h"

#include <map>
#include <vector>
#include <string>

namespace File_Namespace {
class FileBuffer;
}

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Data_Namespace {

class DataMgr {
  friend class FileMgr;

 public:
  DataMgr(const std::string& dataDir,
          const size_t cpuBufferSize /* 0 means auto set size */,
          const bool useGpus,
          const int numGpus,
          const int startGpu = 0,
          const int reservedGpuMem = (1 << 27));
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

  // const std::map<ChunkKey, File_Namespace::FileBuffer *> & getChunkMap();
  const std::map<ChunkKey, File_Namespace::FileBuffer*>& getChunkMap();
  void checkpoint();
  void getChunkMetadataVec(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec);
  void getChunkMetadataVecForKeyPrefix(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
                                       const ChunkKey& keyPrefix);
  inline bool gpusPresent() { return hasGpus_; }

  CudaMgr_Namespace::CudaMgr* cudaMgr_;

  // database_id, table_id, column_id, fragment_id
  std::vector<int> levelSizes_;

 private:
  size_t getTotalSystemMemory();
  void populateMgrs(const size_t userSpecifiedCpuBufferSize);
  std::vector<std::vector<AbstractBufferMgr*>> bufferMgrs_;
  std::string dataDir_;
  bool hasGpus_;
  int reservedGpuMem_;
};
}  // Data_Namespace

#endif  // DATAMGR_H
