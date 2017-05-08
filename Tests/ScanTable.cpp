/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file		ScanTable.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Scan through each column of a table via Chunk iterators
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cfloat>
#include <exception>
#include <memory>
#include <random>
#include <boost/functional/hash.hpp>
#include "../Catalog/Catalog.h"
#include "../DataMgr/DataMgr.h"
#include "../Shared/sqltypes.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Chunk/Chunk.h"
#include "../Shared/measure.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Fragmenter_Namespace;
using namespace Chunk_NS;
using namespace Data_Namespace;

void scan_chunk(const ChunkMetadata& chunk_metadata, const Chunk& chunk, size_t& hash, bool use_iter) {
  ChunkIter cit = chunk.begin_iterator(chunk_metadata, 0, 1);
  VarlenDatum vd;
  bool is_end;
  const ColumnDescriptor* cd = chunk.get_column_desc();
  std::hash<std::string> string_hash;
  int nth = 0;
  while (true) {
    if (use_iter)
      ChunkIter_get_next(&cit, true, &vd, &is_end);
    else
      ChunkIter_get_nth(&cit, nth++, true, &vd, &is_end);
    if (is_end)
      break;
    switch (cd->columnType.get_type()) {
      case kSMALLINT:
        boost::hash_combine(hash, *(int16_t*)vd.pointer);
        break;
      case kINT:
        boost::hash_combine(hash, *(int32_t*)vd.pointer);
        break;
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        boost::hash_combine(hash, *(int64_t*)vd.pointer);
        break;
      case kFLOAT:
        boost::hash_combine(hash, *(float*)vd.pointer);
        break;
      case kDOUBLE:
        boost::hash_combine(hash, *(double*)vd.pointer);
        break;
      case kVARCHAR:
      case kCHAR:
      case kTEXT:
        if (cd->columnType.get_compression() == kENCODING_NONE) {
          // cout << "read string: " << string((char*)vd.pointer, vd.length) << endl;
          boost::hash_combine(hash, string_hash(string((char*)vd.pointer, vd.length)));
        }
        break;
      case kTIME:
      case kTIMESTAMP:
        if (cd->columnType.get_dimension() == 0) {
          if (sizeof(time_t) == 4)
            boost::hash_combine(hash, *(int32_t*)vd.pointer);
          else
            boost::hash_combine(hash, *(int64_t*)vd.pointer);
        } else
          assert(false);  // not supported yet
        break;
      case kDATE:
        if (sizeof(time_t) == 4)
          boost::hash_combine(hash, *(int32_t*)vd.pointer);
        else
          boost::hash_combine(hash, *(int64_t*)vd.pointer);
        break;
      default:
        assert(false);
    }
  }
}

vector<size_t> scan_table_return_hash(const string& table_name, const Catalog& cat) {
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  list<const ColumnDescriptor*> cds = cat.getAllColumnMetadataForTable(td->tableId, false, true);
  vector<size_t> col_hashs(cds.size());
  int64_t elapsed_time = 0;
  size_t total_bytes = 0;
  Fragmenter_Namespace::TableInfo query_info = td->fragmenter->getFragmentsForQuery();
  for (auto frag : query_info.fragments) {
    int i = 0;
    for (auto cd : cds) {
      auto chunk_meta_it = frag.getChunkMetadataMapPhysical().find(cd->columnId);
      ChunkKey chunk_key{cat.get_currentDB().dbId, td->tableId, cd->columnId, frag.fragmentId};
      total_bytes += chunk_meta_it->second.numBytes;
      auto ms = measure<>::execution([&]() {
        std::shared_ptr<Chunk> chunkp = Chunk::getChunk(cd,
                                                        &cat.get_dataMgr(),
                                                        chunk_key,
                                                        CPU_LEVEL,
                                                        frag.deviceIds[static_cast<int>(CPU_LEVEL)],
                                                        chunk_meta_it->second.numBytes,
                                                        chunk_meta_it->second.numElements);
        scan_chunk(chunk_meta_it->second, *chunkp, col_hashs[i], true);
        // call Chunk destructor here
      });
      elapsed_time += ms;
      i++;
    }
  }
  cout << "Scanned " << query_info.getPhysicalNumTuples() << " rows " << total_bytes << " bytes in " << elapsed_time
       << " ms. at " << (double)total_bytes / (elapsed_time / 1000.0) / 1e6 << " MB/sec." << std::endl;
  return col_hashs;
}

vector<size_t> scan_table_return_hash_non_iter(const string& table_name, const Catalog& cat) {
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  list<const ColumnDescriptor*> cds = cat.getAllColumnMetadataForTable(td->tableId, false, true);
  vector<size_t> col_hashs(cds.size());
  Fragmenter_Namespace::TableInfo query_info = td->fragmenter->getFragmentsForQuery();
  int64_t elapsed_time = 0;
  size_t total_bytes = 0;
  for (auto frag : query_info.fragments) {
    int i = 0;
    for (auto cd : cds) {
      auto chunk_meta_it = frag.getChunkMetadataMapPhysical().find(cd->columnId);
      ChunkKey chunk_key{cat.get_currentDB().dbId, td->tableId, cd->columnId, frag.fragmentId};
      total_bytes += chunk_meta_it->second.numBytes;
      auto ms = measure<>::execution([&]() {
        std::shared_ptr<Chunk> chunkp = Chunk::getChunk(cd,
                                                        &cat.get_dataMgr(),
                                                        chunk_key,
                                                        CPU_LEVEL,
                                                        frag.deviceIds[static_cast<int>(CPU_LEVEL)],
                                                        chunk_meta_it->second.numBytes,
                                                        chunk_meta_it->second.numElements);
        scan_chunk(chunk_meta_it->second, *chunkp, col_hashs[i], false);
        // call Chunk destructor here
      });
      elapsed_time += ms;
      i++;
    }
  }
  cout << "Scanned " << query_info.getPhysicalNumTuples() << " rows " << total_bytes << " bytes in " << elapsed_time
       << " ms. at " << (double)total_bytes / (elapsed_time / 1000.0) / 1e6 << " MB/sec." << std::endl;
  return col_hashs;
}
