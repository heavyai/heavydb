#pragma once

#include "ForeignStorageInterface.h"

#include <arrow/api.h>
#include <arrow/csv/reader.h>
#include <arrow/io/file.h>
#include <arrow/util/task-group.h>
#include <arrow/util/thread-pool.h>

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

  struct ArrowFragment {
    int64_t sz;
    std::vector<std::shared_ptr<arrow::ArrayData>> chunks;
    ~ArrowFragment() { chunks.clear(); }
  };

  std::map<std::array<int, 3>, std::vector<ArrowFragment>> m_columns;
  std::map<std::array<int, 3>, StringDictionary*> m_dictionaries;
};
