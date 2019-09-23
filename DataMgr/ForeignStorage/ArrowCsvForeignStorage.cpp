

#include "ArrowCsvForeignStorage.h"

#include "Shared/Logger.h"

#include <arrow/api.h>
#include <arrow/csv/reader.h>
#include <arrow/io/file.h>
#include "../DataMgr/StringNoneEncoder.h"

std::shared_ptr<arrow::Table> g_arrowTable;

void ArrowCsvForeignStorage::append(
    const std::vector<ForeignStorageColumnBuffer>& column_buffers) {
  printf("-- aaaaaapend!!!!\n");
  CHECK(false);
}

void ArrowCsvForeignStorage::read(const ChunkKey& chunk_key,
                                  const SQLTypeInfo& sql_type,
                                  int8_t* dest,
                                  const size_t numBytes) {
  printf("-- reading %d:%d<=%u\n", chunk_key[2], chunk_key[3], unsigned(numBytes));
  arrow::Table &table = *g_arrowTable.get();
  auto ch_array = table.column(chunk_key[2]-1)->data();
  auto array_data = ch_array->chunk(chunk_key[3])->data().get();
  std::shared_ptr<arrow::Buffer> bp;
  
  if(sql_type.get_type() == kTEXT) {
    // ONLY FOR NONE ENCODED STRINGS
    if(chunk_key[4] == 1) {
      bp = array_data->buffers[2];
    } else {
      bp = array_data->buffers[1];
    }
  } else {
    bp = array_data->buffers[1];
  }
  std::memcpy(dest, bp->data(), bp->size());
  CHECK_EQ(numBytes, (size_t)bp->size());
}

void ArrowCsvForeignStorage::prepareTable(const int db_id, const std::string &type, TableDescriptor& td, std::list<ColumnDescriptor>& cols) {
  td.hasDeletedCol = false;
}

void ArrowCsvForeignStorage::registerTable(std::pair<int, int> table_key, const std::string &info, const TableDescriptor& td,
                                            const std::list<ColumnDescriptor>& cols, Data_Namespace::AbstractBufferMgr *mgr) {
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

  r = trp->Read(&g_arrowTable);
  CHECK(r.ok());
  
  arrow::Table &table = *g_arrowTable.get();
  auto c0 = table.column(0)->data().get();
  int num_cols = table.num_columns();
  int num_frags = c0->num_chunks();

  // data comes like this - database_id, table_id, column_id, fragment_id
  ChunkKey key{table_key.first, table_key.second, 0, 0};
  for(auto c : cols) {
    key[2] = c.columnId;
    if(!c.isSystemCol)
      for(int f = 0; f < num_frags; f++ ) {
        key[3] = f;
        auto c0f = c0->chunk(f).get();
        if(c.columnType.get_type() == kTEXT) {
          {
            auto k = key;
            k.push_back(1);
            auto b = mgr->createBuffer(k);
            auto sz = c0f->data()->buffers[2]->size();
            b->setSize(sz);
            b->encoder = std::make_unique<StringNoneEncoder>(b);
            b->has_encoder = true;
            b->sql_type = c.columnType;
          }
          {
            auto k = key;
            k.push_back(2);
            auto &b = *mgr->createBuffer(k);
            b.sql_type = SQLTypeInfo(kINT, false);
            auto sz = c0f->length();
            b.setSize(sz);
          }
        } else {
          auto &b = *mgr->createBuffer(key);
          b.sql_type = c.columnType;
          auto sz = c0f->length();
          b.setSize(sz);
        }
        //TODO: check dynamic_cast<arrow::FixedWidthType*>(c0f->type().get())->bit_width() == b.sql_type.get_size()
  }   }
  printf("-- created %d:%d cols:frags\n", num_cols, num_frags);
}

std::string ArrowCsvForeignStorage::getType() const {
  printf("-- getting used!!!!\n");
  return "CSV";
}
