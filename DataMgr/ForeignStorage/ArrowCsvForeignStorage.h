#pragma once

#include "ForeignStorageInterface.h"

class ArrowCsvForeignStorage : public PersistentForeignStorageInterface {
  void append(const std::vector<ForeignStorageColumnBuffer>& column_buffers) override;

  void read(const ChunkKey& chunk_key,
            const SQLTypeInfo& sql_type,
            int8_t* dest,
            const size_t numBytes) override;

  std::string getType() const override;

 private:
  std::map<ChunkKey, std::vector<int8_t>> buffers_;
};