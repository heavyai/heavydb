

#include "ArrowCsvForeignStorage.h"

#include "Shared/Logger.h"

#include <arrow/api.h>
#include <arrow/csv/reader.h>
#include <arrow/io/file.h>


void ArrowCsvForeignStorage::append(
    const std::vector<ForeignStorageColumnBuffer>& column_buffers) {
  printf("-- aaaaaapend!!!!\n");
  CHECK(false);
}

void ArrowCsvForeignStorage::read(const ChunkKey& chunk_key,
                                  const SQLTypeInfo& sql_type,
                                  int8_t* dest,
                                  const size_t numBytes) {
  printf("-- reading %d:%d!!!!\n", chunk_key[2], chunk_key[3]);
}

void ArrowCsvForeignStorage::registerTable(std::pair<int, int> table_key, const std::string &info,
                                            Data_Namespace::AbstractBufferMgr *mgr) {
    printf("-- registering %s!!!!\n", info.c_str());
    auto popt = arrow::csv::ParseOptions::Defaults();
    popt.quoting = false;
    popt.newlines_in_values = false;

    auto ropt = arrow::csv::ReadOptions::Defaults();
    ropt.use_threads = true;
    ropt.block_size = 1*1024*1024;

    auto copt = arrow::csv::ConvertOptions::Defaults();
    auto memp = arrow::default_memory_pool();

    std::shared_ptr<arrow::io::ReadableFile> inp;
    auto r = arrow::io::ReadableFile::Open(info.c_str(), &inp); // TODO check existence
    CHECK(r.ok());

    std::shared_ptr<arrow::csv::TableReader> trp;
    r = arrow::csv::TableReader::Make(memp, inp, ropt, popt, copt, &trp);
    CHECK(r.ok());

    std::shared_ptr<arrow::Table> out;
    r = trp->Read(&out);
    CHECK(r.ok());
    
    arrow::Table &table = *out.get();
    int num_cols = table.num_columns();
    int num_frags = table.column(0)->data()->num_chunks();
    // data comes like this - database_id, table_id, column_id, fragment_id
    ChunkKey key{table_key.first, table_key.second, 0, 0};
    for(int c = 0; c < num_cols; c++ ) {
      key[2] = c;
      for(int f = 0; f < num_frags; f++ ) {
        key[3] = f;
        mgr->createBuffer(key);
    } }
}

std::string ArrowCsvForeignStorage::getType() const {
  printf("-- getting used!!!!\n");
  return "CSV";
}
