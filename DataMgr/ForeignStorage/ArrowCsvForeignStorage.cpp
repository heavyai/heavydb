

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
}

void ArrowCsvForeignStorage::registerTable(std::pair<int, int> table_key, const std::string &info) {
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
}

std::string ArrowCsvForeignStorage::getType() const {
  printf("-- getting used!!!!\n");
  return "CSV";
}
