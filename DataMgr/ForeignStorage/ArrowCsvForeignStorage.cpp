

#include "ArrowCsvForeignStorage.h"

#include "Shared/Logger.h"

void ArrowCsvForeignStorage::append(
    const std::vector<ForeignStorageColumnBuffer>& column_buffers) {
  CHECK(false);
}

void ArrowCsvForeignStorage::read(const ChunkKey& chunk_key,
                                  const SQLTypeInfo& sql_type,
                                  int8_t* dest,
                                  const size_t numBytes) {
  CHECK(false);
}

std::string ArrowCsvForeignStorage::getType() const {
  return "ARROWCSV";
}