#pragma once

#include "ForeignStorageInterface.h"

class ArrowCsvForeignStorage : public PersistentForeignStorageInterface {
  ~ArrowCsvForeignStorage() override;
  void append(const std::vector<ForeignStorageColumnBuffer>& column_buffers) override;

  void read(const ChunkKey& chunk_key,
            const SQLTypeInfo& sql_type,
            int8_t* dest,
            const size_t numBytes) override;

  virtual void prepareTable(const int db_id,
                            const std::string& type,
                            TableDescriptor& td,
                            std::list<ColumnDescriptor>& cols) override;
  virtual void registerTable(Catalog_Namespace::Catalog* catalog,
                             std::pair<int, int> table_key,
                             const std::string& type,
                             const TableDescriptor& td,
                             const std::list<ColumnDescriptor>& cols,
                             Data_Namespace::AbstractBufferMgr* mgr) override;

  std::string getType() const override;
};
