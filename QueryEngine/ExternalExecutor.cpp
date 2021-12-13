/*
 * Copyright 2019 OmniSci, Inc.
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

#include "QueryEngine/ExternalExecutor.h"

#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "SqliteConnector/SqliteConnector.h"

namespace {

struct OmniSciVtab {
  sqlite3_vtab base;
  sqlite3* db;
  const ExternalQueryTable* external_query_table;
};

struct OmniSciCursor {
  sqlite3_vtab_cursor base;
  int count;
  int eof;
};

int vt_destructor(sqlite3_vtab* pVtab) {
  OmniSciVtab* p = reinterpret_cast<OmniSciVtab*>(pVtab);
  sqlite3_free(p);

  return 0;
}

int vt_create(sqlite3* db,
              void* p_aux,
              int argc,
              const char* const* argv,
              sqlite3_vtab** pp_vt,
              char** pzErr) {
  // Allocate the sqlite3_vtab/OmniSciVtab structure itself.
  auto p_vt = static_cast<OmniSciVtab*>(sqlite3_malloc(sizeof(OmniSciVtab)));
  if (!p_vt) {
    return SQLITE_NOMEM;
  }

  p_vt->db = db;
  p_vt->external_query_table = reinterpret_cast<const ExternalQueryTable*>(p_aux);

  std::vector<std::string> col_defs;
  std::transform(p_vt->external_query_table->schema.begin(),
                 p_vt->external_query_table->schema.end(),
                 std::back_inserter(col_defs),
                 [](const TargetMetaInfo& target_metainfo) {
                   return target_metainfo.get_resname() + " " +
                          target_metainfo.get_type_info().get_type_name();
                 });
  const auto col_defs_str = boost::algorithm::join(col_defs, ", ");
  const auto create_statement =
      "create table vtable (" + (col_defs_str.empty() ? "dummy int" : col_defs_str) + ")";

  // Declare the vtable's structure.
  int rc = sqlite3_declare_vtab(db, create_statement.c_str());

  if (rc != SQLITE_OK) {
    vt_destructor(reinterpret_cast<sqlite3_vtab*>(p_vt));
    return SQLITE_ERROR;
  }

  // Success. Set *pp_vt and return.
  *pp_vt = &p_vt->base;

  return SQLITE_OK;
}

int vt_connect(sqlite3* db,
               void* p_aux,
               int argc,
               const char* const* argv,
               sqlite3_vtab** pp_vt,
               char** pzErr) {
  return vt_create(db, p_aux, argc, argv, pp_vt, pzErr);
}

int vt_disconnect(sqlite3_vtab* pVtab) {
  return vt_destructor(pVtab);
}

int vt_destroy(sqlite3_vtab* pVtab) {
  int rc = SQLITE_OK;

  if (rc == SQLITE_OK) {
    rc = vt_destructor(pVtab);
  }

  return rc;
}

int vt_open(sqlite3_vtab* pVTab, sqlite3_vtab_cursor** pp_cursor) {
  auto p_cur = static_cast<OmniSciCursor*>(sqlite3_malloc(sizeof(OmniSciCursor)));
  *pp_cursor = reinterpret_cast<sqlite3_vtab_cursor*>(p_cur);

  return (p_cur ? SQLITE_OK : SQLITE_NOMEM);
}

int vt_close(sqlite3_vtab_cursor* cur) {
  auto p_cur = reinterpret_cast<OmniSciCursor*>(cur);
  sqlite3_free(p_cur);

  return SQLITE_OK;
}

int vt_eof(sqlite3_vtab_cursor* cur) {
  return reinterpret_cast<OmniSciCursor*>(cur)->eof;
}

int64_t get_num_rows(OmniSciCursor* p_cur) {
  auto p = reinterpret_cast<OmniSciVtab*>(p_cur->base.pVtab);
  CHECK_EQ(p->external_query_table->fetch_result.num_rows.size(), size_t(1));
  CHECK_EQ(p->external_query_table->fetch_result.num_rows.front().size(), size_t(1));
  return p->external_query_table->fetch_result.num_rows.front().front();
}

int vt_next(sqlite3_vtab_cursor* cur) {
  auto p_cur = reinterpret_cast<OmniSciCursor*>(cur);
  const auto num_rows = get_num_rows(p_cur);

  if (p_cur->count == num_rows) {
    p_cur->eof = 1;
  }

  // Increment the current row count.
  ++p_cur->count;

  return SQLITE_OK;
}

struct DecodedString {
  std::pair<const char*, size_t> payload;
  bool is_null;
};

template <class T>
DecodedString decode_string(const int8_t* column,
                            const size_t cursor,
                            StringDictionaryProxy* sdp) {
  const auto ids_column = reinterpret_cast<const T*>(column);
  const auto val = ids_column[cursor];
  DecodedString result{};
  if (val == inline_int_null_value<T>()) {
    result.is_null = true;
  } else {
    result.payload = sdp->getStringBytes(val);
  }
  return result;
}

int vt_column(sqlite3_vtab_cursor* cur, sqlite3_context* ctx, int col_idx) {
  auto p_cur = reinterpret_cast<OmniSciCursor*>(cur);
  const auto num_rows = get_num_rows(p_cur);

  auto p = reinterpret_cast<OmniSciVtab*>(p_cur->base.pVtab);
  const auto& external_query_table = *(p->external_query_table);
  CHECK_LT(static_cast<size_t>(col_idx),
           external_query_table.fetch_result.col_buffers[0].size());
  const auto column = external_query_table.fetch_result.col_buffers[0][col_idx];
  const auto& col_ti = external_query_table.schema[col_idx].get_type_info();
  switch (col_ti.get_type()) {
    case kTINYINT: {
      const auto val = column[p_cur->count - 1];
      if (val == inline_int_null_value<int8_t>()) {
        sqlite3_result_null(ctx);
      } else {
        sqlite3_result_int(ctx, val);
      }
      break;
    }
    case kSMALLINT: {
      const auto int_column = reinterpret_cast<const int16_t*>(column);
      const auto val = int_column[p_cur->count - 1];
      if (val == inline_int_null_value<int16_t>()) {
        sqlite3_result_null(ctx);
      } else {
        sqlite3_result_int(ctx, val);
      }
      break;
    }
    case kINT: {
      const auto int_column = reinterpret_cast<const int32_t*>(column);
      const auto val = int_column[p_cur->count - 1];
      if (val == inline_int_null_value<int32_t>()) {
        sqlite3_result_null(ctx);
      } else {
        sqlite3_result_int(ctx, val);
      }
      break;
    }
    case kBIGINT: {
      const auto int_column = reinterpret_cast<const int64_t*>(column);
      const auto val = int_column[p_cur->count - 1];
      if (val == inline_int_null_value<int64_t>()) {
        sqlite3_result_null(ctx);
      } else {
        sqlite3_result_int(ctx, val);
      }
      break;
    }
    case kFLOAT: {
      const auto float_column = reinterpret_cast<const float*>(column);
      const auto val = float_column[p_cur->count - 1];
      if (val == inline_fp_null_value<float>()) {
        sqlite3_result_null(ctx);
      } else {
        sqlite3_result_double(ctx, val);
      }
      break;
    }
    case kDOUBLE: {
      const auto double_column = reinterpret_cast<const double*>(column);
      const auto val = double_column[p_cur->count - 1];
      if (val == inline_fp_null_value<double>()) {
        sqlite3_result_null(ctx);
      } else {
        sqlite3_result_double(ctx, val);
      }
      break;
    }
    case kTEXT: {
      if (col_ti.get_compression() == kENCODING_DICT) {
        const auto executor = external_query_table.executor;
        const auto sdp = executor->getStringDictionaryProxy(
            col_ti.get_comp_param(), executor->getRowSetMemoryOwner(), true);
        CHECK(sdp);
        DecodedString decoded_string;
        switch (col_ti.get_size()) {
          case 1: {
            decoded_string = decode_string<uint8_t>(column, p_cur->count - 1, sdp);
            break;
          }
          case 2: {
            decoded_string = decode_string<uint16_t>(column, p_cur->count - 1, sdp);
            break;
          }
          case 4: {
            decoded_string = decode_string<int32_t>(column, p_cur->count - 1, sdp);
            break;
          }
          default: {
            decoded_string = DecodedString{};
            LOG(FATAL) << "Invalid encoding size: " << col_ti.get_size();
          }
        }
        if (decoded_string.is_null) {
          sqlite3_result_null(ctx);
        } else {
          sqlite3_result_text(
              ctx, decoded_string.payload.first, decoded_string.payload.second, nullptr);
        }
      } else {
        CHECK(col_ti.get_compression() == kENCODING_NONE);
        const auto chunk_iter =
            const_cast<ChunkIter*>(reinterpret_cast<const ChunkIter*>(column));
        VarlenDatum vd;
        bool is_end;
        ChunkIter_get_nth(chunk_iter, p_cur->count - 1, false, &vd, &is_end);
        if (vd.is_null) {
          sqlite3_result_null(ctx);
        } else {
          sqlite3_result_text(
              ctx, reinterpret_cast<const char*>(vd.pointer), vd.length, nullptr);
        }
      }
      break;
    }
    default: {
      LOG(FATAL) << "Unexpected type: " << col_ti.get_type_name();
      break;
    }
  }
  CHECK_LE(p_cur->count, num_rows);

  return SQLITE_OK;
}

int vt_rowid(sqlite3_vtab_cursor* cur, sqlite_int64* p_rowid) {
  auto p_cur = reinterpret_cast<OmniSciCursor*>(cur);
  // Just use the current row count as the rowid.
  *p_rowid = p_cur->count;
  return SQLITE_OK;
}

int vt_filter(sqlite3_vtab_cursor* p_vtc,
              int idxNum,
              const char* idxStr,
              int argc,
              sqlite3_value** argv) {
  // Initialize the cursor structure.
  auto p_cur = reinterpret_cast<OmniSciCursor*>(p_vtc);
  // Zero rows returned thus far.
  p_cur->count = 0;
  // Have not reached end of set.
  p_cur->eof = 0;
  // Move cursor to first row.
  return vt_next(p_vtc);
}

// We don't implement indexing.
int vt_best_index(sqlite3_vtab* tab, sqlite3_index_info* pIdxInfo) {
  return SQLITE_OK;
}

sqlite3_module omnisci_module = {
    0,              // iVersion
    vt_create,      // xCreate       - create a vtable
    vt_connect,     // xConnect      - associate a vtable with a connection
    vt_best_index,  // xBestIndex    - best index
    vt_disconnect,  // xDisconnect   - disassociate a vtable with a connection
    vt_destroy,     // xDestroy      - destroy a vtable
    vt_open,        // xOpen         - open a cursor
    vt_close,       // xClose        - close a cursor
    vt_filter,      // xFilter       - configure scan constraints
    vt_next,        // xNext         - advance a cursor
    vt_eof,         // xEof          - inidicate end of result set
    vt_column,      // xColumn       - read data
    vt_rowid,       // xRowid        - read data
    nullptr,        // xUpdate       - write data
    nullptr,        // xBegin        - begin transaction
    nullptr,        // xSync         - sync transaction
    nullptr,        // xCommit       - commit transaction
    nullptr,        // xRollback     - rollback transaction
    nullptr,        // xFindFunction - function overloading
    nullptr,        // xRename       - function overloading
    nullptr,        // xSavepoint    - function overloading
    nullptr,        // xRelease      - function overloading
    nullptr         // xRollbackto   - function overloading
};

std::vector<TargetMetaInfo> create_table_schema(const PlanState* plan_state) {
  std::map<size_t, TargetMetaInfo> schema_map;
  const auto catalog = plan_state->executor_->getCatalog();
  for (const auto& kv : plan_state->global_to_local_col_ids_) {
    const int table_id = kv.first.getTableId();
    const int column_id = kv.first.getColId();
    SQLTypeInfo column_type;
    if (table_id < 0) {
      const auto& table =
          get_temporary_table(plan_state->executor_->getTemporaryTables(), table_id);
      column_type = table.getColType(column_id);
    } else {
      const auto cd = catalog->getMetadataForColumn(table_id, column_id);
      column_type = cd->columnType;
    }
    if (!is_supported_type_for_extern_execution(column_type)) {
      throw std::runtime_error("Type not supported yet for extern execution: " +
                               column_type.get_type_name());
    }
    const auto column_ref = serialize_column_ref(table_id, column_id, catalog);
    const auto it_ok =
        schema_map.emplace(kv.second, TargetMetaInfo(column_ref, column_type));
    CHECK(it_ok.second);
  }
  std::vector<TargetMetaInfo> schema;
  for (const auto& kv : schema_map) {
    schema.push_back(kv.second);
  }
  return schema;
}

}  // namespace

SqliteMemDatabase::SqliteMemDatabase(const ExternalQueryTable& external_query_table)
    : external_query_table_(external_query_table) {
  int status = sqlite3_open(":memory:", &db_);
  CHECK_EQ(status, SQLITE_OK);
  status = sqlite3_create_module(db_, "omnisci", &omnisci_module, &external_query_table_);
  CHECK_EQ(status, SQLITE_OK);
}

SqliteMemDatabase::~SqliteMemDatabase() {
  std::lock_guard session_lock(session_mutex_);
  int status = sqlite3_close(db_);
  CHECK_EQ(status, SQLITE_OK);
}

void SqliteMemDatabase::run(const std::string& sql) {
  std::lock_guard session_lock(session_mutex_);
  char* msg;
  int status = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &msg);
  CHECK_EQ(status, SQLITE_OK);
}

namespace {

int64_t* get_scan_output_slot(int64_t* output_buffer,
                              const size_t output_buffer_entry_count,
                              const size_t pos,
                              const size_t row_size_quad) {
  const auto off = pos * row_size_quad;
  CHECK_LT(pos, output_buffer_entry_count);
  output_buffer[off] = off;
  return output_buffer + off + 1;
}

}  // namespace

std::unique_ptr<ResultSet> SqliteMemDatabase::runSelect(
    const std::string& sql,
    const ExternalQueryOutputSpec& output_spec) {
  SqliteConnector connector(db_);
  connector.query(sql);
  auto query_mem_desc = output_spec.query_mem_desc;
  const auto num_rows = connector.getNumRows();
  query_mem_desc.setEntryCount(num_rows);
  auto rs = std::make_unique<ResultSet>(output_spec.target_infos,
                                        ExecutorDeviceType::CPU,
                                        query_mem_desc,
                                        output_spec.executor->getRowSetMemoryOwner(),
                                        nullptr,
                                        -1,
                                        0,
                                        0);
  const auto storage = rs->allocateStorage();
  auto output_buffer = storage->getUnderlyingBuffer();
  CHECK(!num_rows || output_buffer);
  for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
    auto row = get_scan_output_slot(reinterpret_cast<int64_t*>(output_buffer),
                                    num_rows,
                                    row_idx,
                                    query_mem_desc.getRowSize() / sizeof(int64_t));
    CHECK_EQ(output_spec.target_infos.size(), connector.getNumCols());
    size_t slot_idx = 0;
    for (size_t col_idx = 0; col_idx < connector.getNumCols(); ++col_idx, ++slot_idx) {
      const auto& col_type = output_spec.target_infos[col_idx].sql_type;
      const int sqlite_col_type = connector.columnTypes[col_idx];
      switch (col_type.get_type()) {
        case kBOOLEAN:
        case kTINYINT:
        case kSMALLINT:
        case kINT:
        case kBIGINT: {
          static const std::string overflow_message{"Overflow or underflow"};
          if (sqlite_col_type != SQLITE_INTEGER && sqlite_col_type != SQLITE_NULL) {
            throw std::runtime_error(overflow_message);
          }
          if (!connector.isNull(row_idx, col_idx)) {
            const auto limits = inline_int_max_min(col_type.get_logical_size());
            const auto val = connector.getData<int64_t>(row_idx, col_idx);
            if (val > limits.first || val < limits.second) {
              throw std::runtime_error(overflow_message);
            }
            row[slot_idx] = val;
          } else {
            row[slot_idx] = inline_int_null_val(col_type);
          }
          break;
        }
        case kFLOAT: {
          CHECK(sqlite_col_type == SQLITE_FLOAT || sqlite_col_type == SQLITE_NULL);
          if (!connector.isNull(row_idx, col_idx)) {
            reinterpret_cast<double*>(row)[slot_idx] =
                connector.getData<double>(row_idx, col_idx);
          } else {
            reinterpret_cast<double*>(row)[slot_idx] = inline_fp_null_value<float>();
          }
          break;
        }
        case kDOUBLE: {
          CHECK(sqlite_col_type == SQLITE_FLOAT || sqlite_col_type == SQLITE_NULL);
          if (!connector.isNull(row_idx, col_idx)) {
            reinterpret_cast<double*>(row)[slot_idx] =
                connector.getData<double>(row_idx, col_idx);
          } else {
            reinterpret_cast<double*>(row)[slot_idx] = inline_fp_null_value<double>();
          }
          break;
        }
        case kCHAR:
        case kTEXT:
        case kVARCHAR: {
          CHECK(sqlite_col_type == SQLITE_TEXT || sqlite_col_type == SQLITE_NULL);
          if (!connector.isNull(row_idx, col_idx)) {
            const auto str = connector.getData<std::string>(row_idx, col_idx);
            const auto owned_str =
                output_spec.executor->getRowSetMemoryOwner()->addString(str);
            row[slot_idx] = reinterpret_cast<int64_t>(owned_str->c_str());
            row[++slot_idx] = str.size();
          } else {
            row[slot_idx] = 0;
            row[++slot_idx] = 0;
          }
          break;
        }
        default: {
          LOG(FATAL) << "Unexpected type: " << col_type.get_type_name();
          break;
        }
      }
    }
  }
  return rs;
}

std::mutex SqliteMemDatabase::session_mutex_;

std::unique_ptr<ResultSet> run_query_external(
    const ExecutionUnitSql& sql,
    const FetchResult& fetch_result,
    const PlanState* plan_state,
    const ExternalQueryOutputSpec& output_spec) {
  ExternalQueryTable external_query_table{fetch_result,
                                          create_table_schema(plan_state),
                                          sql.from_table,
                                          output_spec.executor};
  SqliteMemDatabase db(external_query_table);
  const auto create_table = "create virtual table " + sql.from_table + " using omnisci";
  db.run(create_table);
  return db.runSelect(sql.query, output_spec);
}

bool is_supported_type_for_extern_execution(const SQLTypeInfo& ti) {
  return ti.is_integer() || ti.is_fp() || ti.is_string();
}
