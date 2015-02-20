/**
 * @file		PopulateTableRandom.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Populate a table with random data
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

using namespace std;
using namespace Catalog_Namespace;
using namespace Fragmenter_Namespace;

size_t
random_fill_int16(int8_t *buf, size_t num_elems)
{
		default_random_engine gen;
    uniform_int_distribution<int16_t> dist(INT16_MIN, INT16_MAX);
		int16_t *p = (int16_t*)buf; 
		size_t hash = 0;
		for (int i = 0; i < num_elems; i++) {
			p[i] = dist(gen);
			boost::hash_combine(hash, p[i]);
		}
		return hash;
}

size_t
random_fill_int32(int8_t *buf, size_t num_elems)
{
		default_random_engine gen;
    uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);
		int32_t *p = (int32_t*)buf; 
		size_t hash = 0;
		for (int i = 0; i < num_elems; i++) {
			p[i] = dist(gen);
			boost::hash_combine(hash, p[i]);
		}
		return hash;
}

size_t
random_fill_int64(int8_t *buf, size_t num_elems)
{
		default_random_engine gen;
    uniform_int_distribution<int64_t> dist(INT64_MIN, INT64_MAX);
		int64_t *p = (int64_t*)buf; 
		size_t hash = 0;
		for (int i = 0; i < num_elems; i++) {
			p[i] = dist(gen);
			boost::hash_combine(hash, p[i]);
		}
		return hash;
}

size_t
random_fill_float(int8_t *buf, size_t num_elems)
{
		default_random_engine gen;
    uniform_real_distribution<float> dist(FLT_MIN, FLT_MAX);
		float *p = (float*)buf; 
		size_t hash = 0;
		for (int i = 0; i < num_elems; i++) {
			p[i] = dist(gen);
			boost::hash_combine(hash, p[i]);
		}
		return hash;
}

size_t
random_fill_double(int8_t *buf, size_t num_elems)
{
		default_random_engine gen;
    uniform_real_distribution<double> dist(DBL_MIN, DBL_MAX);
		double *p = (double*)buf; 
		size_t hash = 0;
		for (int i = 0; i < num_elems; i++) {
			p[i] = dist(gen);
			boost::hash_combine(hash, p[i]);
		}
		return hash;
}

size_t
random_fill_string(vector<string> &stringVec, size_t num_elems, int max_len)
{
		string chars("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");
		default_random_engine gen;
    uniform_int_distribution<> char_dist(0, chars.size() - 1);
    uniform_int_distribution<> len_dist(0, max_len);
		size_t hash = 0;
		std::hash<std::string> string_hash;
		for (int n = 0; n < num_elems; n++) {
			int len = len_dist(gen);
			string s(len, ' ');
			for (int i = 0; i < len; i++)
				s[i] = chars[char_dist(gen)];
			// cout << "insert string: " << s << endl;
			stringVec[n] = s;
			boost::hash_combine(hash, string_hash(s));
		}
		return hash;
}

#define MAX_TEXT_LEN		255

size_t
random_fill(const ColumnDescriptor *cd, DataBlockPtr p, size_t num_elems)
{
	size_t hash;
	switch (cd->columnType.type) {
		case kSMALLINT:
			hash = random_fill_int16(p.numbersPtr, num_elems);
			break;
		case kINT:
			hash = random_fill_int32(p.numbersPtr, num_elems);
			break;
		case kBIGINT:
		case kNUMERIC:
		case kDECIMAL:
			hash = random_fill_int64(p.numbersPtr, num_elems);
			break;
		case kFLOAT:
			hash = random_fill_float(p.numbersPtr, num_elems);
			break;
		case kDOUBLE:
			hash = random_fill_double(p.numbersPtr, num_elems);
			break;
		case kVARCHAR:
		case kCHAR:
			hash = random_fill_string(*p.stringsPtr, num_elems, cd->columnType.dimension);
			break;
		case kTEXT:
			hash = random_fill_string(*p.stringsPtr, num_elems, MAX_TEXT_LEN);
			break;
		case kTIME:
		case kTIMESTAMP:
			if (cd->columnType.dimension == 0) {
				if (sizeof(time_t) == 4)
					hash = random_fill_int32(p.numbersPtr, num_elems);
				else
					hash = random_fill_int64(p.numbersPtr, num_elems);
			} else
				assert(false); // not supported yet
			break;
		case kDATE:
			if (sizeof(time_t) == 4)
				hash = random_fill_int32(p.numbersPtr, num_elems);
			else
				hash = random_fill_int64(p.numbersPtr, num_elems);
			break;
		default:
			assert(false);
	}
	return hash;
}

vector<size_t>
populate_table_random(const string &table_name, const size_t num_rows, const Catalog &cat)
{
	const TableDescriptor *td = cat.getMetadataForTable(table_name);
	list<const ColumnDescriptor *> cds = cat.getAllColumnMetadataForTable(td->tableId);
	InsertData insert_data;
	insert_data.databaseId = cat.get_currentDB().dbId;
	insert_data.tableId = td->tableId;
	for (auto cd : cds) {
		insert_data.columnIds.push_back(cd->columnId);
	}
	insert_data.numRows = num_rows;
	vector<unique_ptr<int8_t>> gc_numbers;  // making sure input buffers get freed
	vector<unique_ptr<vector<string>>> gc_strings;  // making sure input vectors get freed
	DataBlockPtr p;
	// now allocate space for insert data
	for (auto cd : cds) {
		if (cd->is_varlen()) {
			vector<string> *col_vec = new vector<string>(num_rows);
			gc_strings.push_back(unique_ptr<vector<string>>(col_vec)); // add to gc list
			p.stringsPtr = col_vec;
		} else {
			int8_t *col_buf = static_cast<int8_t*>(malloc(num_rows * cd->getStorageSize()));
			gc_numbers.push_back(unique_ptr<int8_t>(col_buf)); // add to gc list
			p.numbersPtr = col_buf;
		}
		insert_data.data.push_back(p);
	}

	// fill InsertData  with random data
	vector<size_t> col_hashs(cds.size()); // compute one hash per column for the generated data
	int i = 0;
	for (auto cd : cds) {
		col_hashs[i] = random_fill(cd, insert_data.data[i], num_rows);
		i++;
	}

	// now load the data into table
	td->fragmenter->insertData(insert_data);
	// note: no checkpoint here, the inserts are not guaranteed to be persistent
	
	return col_hashs;
}
