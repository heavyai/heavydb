#pragma once

#include "ForeignStorageInterface.h"

class ArrowCsvForeignStorage : public PersistentForeignStorageInterface {
  void append(const std::vector<ForeignStorageColumnBuffer>& column_buffers) override;

  void read(const ChunkKey& chunk_key,
            const SQLTypeInfo& sql_type,
            int8_t* dest,
            const size_t numBytes) override;

  virtual void registerTable(std::pair<int, int> table_key, const std::string &type) override;

  std::string getType() const override;
};
