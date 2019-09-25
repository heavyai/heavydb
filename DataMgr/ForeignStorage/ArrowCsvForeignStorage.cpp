

#include "ArrowCsvForeignStorage.h"

#include "Shared/Logger.h"

#include <arrow/api.h>
#include <arrow/csv/reader.h>
#include <arrow/io/file.h>
#include "../DataMgr/StringNoneEncoder.h"

#include <array>

struct ArrowFragment {
  int64_t sz;
  std::vector<std::shared_ptr<arrow::ArrayData>> chunks;
};

std::map<std::array<int, 3>, std::vector<ArrowFragment>> g_columns;

void ArrowCsvForeignStorage::append(
    const std::vector<ForeignStorageColumnBuffer>& column_buffers) {
  printf("-- aaaaaapend!!!!\n");
  CHECK(false);
}

void ArrowCsvForeignStorage::read(const ChunkKey& chunk_key,
                                  const SQLTypeInfo& sql_type,
                                  int8_t* dest,
                                  const size_t numBytes) {
  //printf("-- reading %d:%d<=%u\n", chunk_key[2], chunk_key[3], unsigned(numBytes));
  std::array<int, 3> col_key {chunk_key[0], chunk_key[1], chunk_key[2]};
  auto &frag = g_columns.at(col_key).at(chunk_key[3]);
  int64_t sz, copied = 0;
  std::shared_ptr<arrow::ArrayData> prev_data = nullptr;
  int varlen_offset = 0;

  for( auto array_data : frag.chunks ) {
    arrow::Buffer *bp = nullptr;
    
    if(sql_type.get_type() == kTEXT) {
        CHECK_GE(array_data->buffers.size(), 3UL);
        bp = array_data->buffers[2].get();
    } else if(array_data->null_count != array_data->length) {
        CHECK_GE(array_data->buffers.size(), 2UL);
        bp = array_data->buffers[1].get();
    }
    if(bp) {
      if(chunk_key.size() == 4 && chunk_key[4] == 2) {
        const uint32_t *data = reinterpret_cast<const uint32_t *>(bp->data());
        sz = bp->size();
        // We are trying to merge separate arrow arrays with offsets to one omnisci chunk
        // So we need to know that:
        // - offsets length of every array = array.len() + 1
        // - offsets array of every array starts from zero
        // So, we calculate offset of (n+1)th array as sum of offsets of previous n arrays
        // and ignore first value of every new array offset
        if(prev_data != nullptr) {
          data++;
          sz-=sizeof(uint32_t);
        }
        std::transform(data, data+(sz / sizeof(uint32_t)), dest, [varlen_offset](uint32_t val) {
          return val + varlen_offset;
        });
        varlen_offset += data[bp->size() - 1]; // get offset of the last element of current array
      } else {
        std::memcpy(dest, bp->data(), sz = bp->size());
      }
    } else {
      // TODO: nullify?
      auto fixed_type = dynamic_cast<arrow::FixedWidthType*>(array_data->type.get());
      if(fixed_type)
        sz = array_data->length*fixed_type->bit_width()/8;
      else CHECK(false); // TODO: else???
    }
    dest += sz; copied += sz;
    prev_data = array_data;
  }
  CHECK_EQ(numBytes, size_t(copied));
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

  std::shared_ptr<arrow::Table> arrowTable;
  r = trp->Read(&arrowTable);
  CHECK(r.ok());
  
  arrow::Table &table = *arrowTable.get();
  int cln = 0, num_cols = table.num_columns();
  int arr_frags = table.column(0)->data()->num_chunks();
  arrow::ChunkedArray *c0p = table.column(0)->data().get();

  std::vector<std::pair<int,int>> fragments;
  int start = 0;
  int64_t sz = c0p->chunk(0)->length();
  // claculate size and boundaries of fragments
  for(int i = 1; i < arr_frags; i++) {
    if(sz > td.fragPageSize) {
      fragments.push_back(std::make_pair(start, i));
      start = i;
      sz = 0;
    }
    sz += c0p->chunk(i)->length();
  }
  fragments.push_back(std::make_pair(start, arr_frags));

  // data comes like this - database_id, table_id, column_id, fragment_id
  ChunkKey key{table_key.first, table_key.second, 0, 0};
  std::array<int, 3> col_key {table_key.first, table_key.second, 0};

  for(auto c : cols) {
    if(cln >= num_cols) {
      LOG(ERROR) << "Number of columns read from Arrow (" << num_cols << ") mismatch CREATE TABLE request: " << cols.size();
      break;
    }
    if(c.isSystemCol)
      continue; // must be processed by base interface implementation

    auto ctype = c.columnType.get_type();
    col_key[2] = key[2] = c.columnId;
    auto &col = g_columns[col_key];
    col.resize(fragments.size());
    auto clp = table.column(cln++)->data().get();

    // fill each fragment
    for(size_t f = 0; f < fragments.size(); f++ ) {
      key[3] = f;
      auto &frag = col[f];
      int64_t varlen = 0;
      // for each arrow chunk
      for(int i = fragments[f].first, e = fragments[f].second; i < e; i++) {
        frag.chunks.emplace_back(clp->chunk(i)->data());
        frag.sz += clp->chunk(i)->length();
        auto &buffers = clp->chunk(i)->data()->buffers;
        if(ctype == kTEXT) {
          if(buffers.size() <= 2) {
            LOG(FATAL) << "Type of column #" << cln << " does not match between Arrow and description of " << c.columnName;
            throw std::runtime_error("Column ingress mismatch: " + c.columnName);
          }
          varlen += buffers[2]->size();
        } else if(buffers.size() > 2) {
          LOG(FATAL) << "Type of column #" << cln << " does not match between Arrow and description of " << c.columnName;
          throw std::runtime_error("Column ingress mismatch: " + c.columnName);
        }
      }
      //TODO: check dynamic_cast<arrow::FixedWidthType*>(c0f->type().get())->bit_width() == b.sql_type.get_size()
      // create buffer descriptotrs
      if(ctype == kTEXT) {
        auto k = key;
        k.push_back(1);
        {
          auto b = mgr->createBuffer(k);
          b->setSize(varlen);
          b->encoder = std::make_unique<StringNoneEncoder>(b);
          b->has_encoder = true;
          b->sql_type = c.columnType;
        }
        k[4] = 2;
        {
          auto &b = *mgr->createBuffer(k);
          b.sql_type = SQLTypeInfo(kINT, false);
          b.setSize(frag.sz);
        }
      } else {
        auto &b = *mgr->createBuffer(key);
        b.sql_type = c.columnType;
        b.setSize(frag.sz);
      }
  } } // each col and fragment
  printf("-- created: %d columns, %d chunks, %d frags\n", num_cols, arr_frags, int(fragments.size()));
}

std::string ArrowCsvForeignStorage::getType() const {
  printf("CSV importer is activated. Create table `with (storage_type='CSV:path/to/file.csv');`\n");
  return "CSV";
}
