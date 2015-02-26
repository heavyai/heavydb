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

using namespace std;
using namespace Catalog_Namespace;
using namespace Fragmenter_Namespace;
using namespace Chunk_NS;
using namespace Data_Namespace;

void
scan_chunk(const Chunk &chunk, size_t &hash)
{
	ChunkIter cit = chunk.begin_iterator(0, 1);
	VarlenDatum vd;
	bool is_end;
	const ColumnDescriptor *cd = chunk.get_column_desc();
	std::hash<std::string> string_hash;
	while (true) {
		ChunkIter_get_next(&cit, true, &vd, &is_end);
		if (is_end)
			break;
		assert(!vd.is_null);
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
				// cout << "read string: " << string((char*)vd.pointer, vd.length) << endl;
				boost::hash_combine(hash, string_hash(string((char*)vd.pointer, vd.length)));
				break;
			case kTIME:
			case kTIMESTAMP:
				if (cd->columnType.get_dimension() == 0) {
					if (sizeof(time_t) == 4)
						boost::hash_combine(hash, *(int32_t*)vd.pointer);
					else
						boost::hash_combine(hash, *(int64_t*)vd.pointer);
				} else
					assert(false); // not supported yet
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

vector<size_t>
scan_table_return_hash(const string &table_name, const Catalog &cat)
{
	const TableDescriptor *td = cat.getMetadataForTable(table_name);
	list<const ColumnDescriptor *> cds = cat.getAllColumnMetadataForTable(td->tableId);
	vector<size_t> col_hashs(cds.size());
	QueryInfo query_info;
	td->fragmenter->getFragmentsForQuery(query_info);
	for (auto frag : query_info.fragments) {
		int i = 0;
		for (auto cd : cds) {
			auto chunk_meta_it = frag.chunkMetadataMap.find(cd->columnId);
			ChunkKey chunk_key { cat.get_currentDB().dbId, td->tableId, cd->columnId, frag.fragmentId };
            //cout << "Chunk: " << cat.get_currentDB().dbId << " " <<  td->tableId << " " <<  cd->columnId << " " << frag.fragmentId << endl;
			Chunk chunk = Chunk::getChunk(cd, &cat.get_dataMgr(), chunk_key, CPU_LEVEL, frag.deviceIds[static_cast<int>(CPU_LEVEL)], chunk_meta_it->second.numBytes, chunk_meta_it->second.numElements);
			scan_chunk(chunk, col_hashs[i]);
            chunk.unpin_buffer();
			i++;
		}
	}
	return col_hashs;
}
