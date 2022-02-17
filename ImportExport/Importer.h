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

/*
 * @file Importer.h
 * @author Wei Hong < wei@mapd.com>
 * @brief Importer class for table import from file
 */
#ifndef _IMPORTER_H_
#define _IMPORTER_H_

#include <gdal.h>
#include <ogrsf_frmts.h>

#include <atomic>
#include <boost/filesystem.hpp>
#include <boost/noncopyable.hpp>
#include <boost/tokenizer.hpp>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "ImportExport/AbstractImporter.h"
#include "Catalog/Catalog.h"
#include "Catalog/TableDescriptor.h"
#include "DataMgr/Chunk/Chunk.h"
#include "Fragmenter/Fragmenter.h"
#include "ImportExport/CopyParams.h"
#include "Logger/Logger.h"
#include "Shared/ThreadController.h"
#include "Shared/checked_alloc.h"
#include "Shared/fixautotools.h"
#include "ImportExport/TypedImportBuffer.h"

// Some builds of boost::geometry require iostream, but don't explicitly include it.
// Placing in own section to ensure it's included after iostream.
#include <boost/geometry/index/rtree.hpp>

class TDatum;
class TColumn;

namespace Catalog_Namespace {

struct UserMetadata;

}

namespace arrow {

class Array;

}  // namespace arrow

namespace import_export {

class Loader {
  using LoadCallbackType =
      std::function<bool(const std::vector<std::unique_ptr<TypedImportBuffer>>&,
                         std::vector<DataBlockPtr>&,
                         size_t)>;

 public:
  // TODO: Remove the `use_catalog_locks` parameter once Loader is refactored out of
  // ParquetDataWrapper
  Loader(Catalog_Namespace::Catalog& c,
         const TableDescriptor* t,
         LoadCallbackType load_callback = nullptr,
         bool use_catalog_locks = true)
      : catalog_(c)
      , table_desc_(t)
      , column_descs_(
            use_catalog_locks
                ? c.getAllColumnMetadataForTable(t->tableId, false, false, true)
                : c.getAllColumnMetadataForTableUnlocked(t->tableId, false, false, true))
      , load_callback_(load_callback) {
    init(use_catalog_locks);
  }

  virtual ~Loader() {}

  Catalog_Namespace::Catalog& getCatalog() const { return catalog_; }
  const TableDescriptor* getTableDesc() const { return table_desc_; }
  const std::list<const ColumnDescriptor*>& get_column_descs() const {
    return column_descs_;
  }

  StringDictionary* getStringDict(const ColumnDescriptor* cd) const {
    if ((cd->columnType.get_type() != kARRAY ||
         !IS_STRING(cd->columnType.get_subtype())) &&
        (!cd->columnType.is_string() ||
         cd->columnType.get_compression() != kENCODING_DICT)) {
      return nullptr;
    }
    return dict_map_.at(cd->columnId);
  }

  virtual bool load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                    const size_t row_count,
                    const Catalog_Namespace::SessionInfo* session_info);
  virtual bool loadNoCheckpoint(
      const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
      const size_t row_count,
      const Catalog_Namespace::SessionInfo* session_info);
  virtual void checkpoint();
  virtual std::vector<Catalog_Namespace::TableEpochInfo> getTableEpochs() const;
  virtual void setTableEpochs(
      const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs);

  void setAddingColumns(const bool adding_columns) { adding_columns_ = adding_columns; }
  bool isAddingColumns() const { return adding_columns_; }
  void dropColumns(const std::vector<int>& columns);
  std::string getErrorMessage() { return error_msg_; };

 protected:
  void init(const bool use_catalog_locks);

  virtual bool loadImpl(
      const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
      size_t row_count,
      bool checkpoint,
      const Catalog_Namespace::SessionInfo* session_info);

  using OneShardBuffers = std::vector<std::unique_ptr<TypedImportBuffer>>;

  Catalog_Namespace::Catalog& catalog_;
  const TableDescriptor* table_desc_;
  std::list<const ColumnDescriptor*> column_descs_;
  LoadCallbackType load_callback_;
  Fragmenter_Namespace::InsertData insert_data_;
  std::map<int, StringDictionary*> dict_map_;

 private:
  bool loadToShard(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
                   size_t row_count,
                   const TableDescriptor* shard_table,
                   bool checkpoint,
                   const Catalog_Namespace::SessionInfo* session_info);
  void fillShardRow(const size_t row_index,
                    OneShardBuffers& shard_output_buffers,
                    const OneShardBuffers& import_buffers);

  bool adding_columns_ = false;
  std::mutex loader_mutex_;
  std::string error_msg_;
};

struct ImportStatus {
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  size_t rows_completed;
  size_t rows_estimated;
  size_t rows_rejected;
  std::chrono::duration<size_t, std::milli> elapsed;
  bool load_failed = false;
  std::string load_msg;
  int thread_id;  // to recall thread_id after thread exit
  ImportStatus()
      : start(std::chrono::steady_clock::now())
      , rows_completed(0)
      , rows_estimated(0)
      , rows_rejected(0)
      , elapsed(0)
      , thread_id(0) {}

  ImportStatus& operator+=(const ImportStatus& is) {
    rows_completed += is.rows_completed;
    rows_rejected += is.rows_rejected;
    if (is.load_failed) {
      load_failed = true;
      load_msg = is.load_msg;
    }

    return *this;
  }
};

class DataStreamSink {
 public:
  DataStreamSink() {}
  DataStreamSink(const CopyParams& copy_params, const std::string file_path)
      : copy_params(copy_params), file_path(file_path) {}
  virtual ~DataStreamSink() {}
  virtual ImportStatus importDelimited(
      const std::string& file_path,
      const bool decompressed,
      const Catalog_Namespace::SessionInfo* session_info) = 0;
#ifdef ENABLE_IMPORT_PARQUET
  virtual void import_parquet(std::vector<std::string>& file_paths,
                              const Catalog_Namespace::SessionInfo* session_info);
  virtual void import_local_parquet(
      const std::string& file_path,
      const Catalog_Namespace::SessionInfo* session_info) = 0;
#endif
  const CopyParams& get_copy_params() const { return copy_params; }
  void import_compressed(std::vector<std::string>& file_paths,
                         const Catalog_Namespace::SessionInfo* session_info);

 protected:
  ImportStatus archivePlumber(const Catalog_Namespace::SessionInfo* session_info);

  CopyParams copy_params;
  const std::string file_path;
  FILE* p_file = nullptr;
  ImportStatus import_status_;
  mapd_shared_mutex import_mutex_;
  size_t total_file_size{0};
  std::vector<size_t> file_offsets;
  std::mutex file_offsets_mutex;
};

class Detector : public DataStreamSink {
 public:
  Detector(const boost::filesystem::path& fp, CopyParams& cp)
      : DataStreamSink(cp, fp.string()), file_path(fp) {
    read_file();
    init();
  };
#ifdef ENABLE_IMPORT_PARQUET
  void import_local_parquet(const std::string& file_path,
                            const Catalog_Namespace::SessionInfo* session_info) override;
#endif
  static SQLTypes detect_sqltype(const std::string& str);
  std::vector<std::string> get_headers();
  std::vector<std::vector<std::string>> raw_rows;
  std::vector<std::vector<std::string>> get_sample_rows(size_t n);
  std::vector<SQLTypes> best_sqltypes;
  std::vector<EncodingType> best_encodings;
  bool has_headers = false;

 private:
  void init();
  void read_file();
  void detect_row_delimiter();
  void split_raw_data();
  std::vector<SQLTypes> detect_column_types(const std::vector<std::string>& row);
  static bool more_restrictive_sqltype(const SQLTypes a, const SQLTypes b);
  void find_best_sqltypes();
  std::vector<SQLTypes> find_best_sqltypes(
      const std::vector<std::vector<std::string>>& raw_rows,
      const CopyParams& copy_params);
  std::vector<SQLTypes> find_best_sqltypes(
      const std::vector<std::vector<std::string>>::const_iterator& row_begin,
      const std::vector<std::vector<std::string>>::const_iterator& row_end,
      const CopyParams& copy_params);

  std::vector<EncodingType> find_best_encodings(
      const std::vector<std::vector<std::string>>::const_iterator& row_begin,
      const std::vector<std::vector<std::string>>::const_iterator& row_end,
      const std::vector<SQLTypes>& best_types);

  bool detect_headers(const std::vector<SQLTypes>& first_types,
                      const std::vector<SQLTypes>& rest_types);
  void find_best_sqltypes_and_headers();
  ImportStatus importDelimited(
      const std::string& file_path,
      const bool decompressed,
      const Catalog_Namespace::SessionInfo* session_info) override;
  std::string raw_data;
  boost::filesystem::path file_path;
  std::chrono::duration<double> timeout{1};
  std::string line1;
};

class ImporterUtils {
 public:
  static ArrayDatum composeNullArray(const SQLTypeInfo& ti);
  static ArrayDatum composeNullPointCoords(const SQLTypeInfo& coords_ti,
                                           const SQLTypeInfo& geo_ti);
};

class Importer : public DataStreamSink, public AbstractImporter {
 public:
  Importer(Catalog_Namespace::Catalog& c,
           const TableDescriptor* t,
           const std::string& f,
           const CopyParams& p);
  Importer(Loader* providedLoader, const std::string& f, const CopyParams& p);
  ~Importer() override;
  ImportStatus import(const Catalog_Namespace::SessionInfo* session_info) override;
  ImportStatus importDelimited(
      const std::string& file_path,
      const bool decompressed,
      const Catalog_Namespace::SessionInfo* session_info) override;
  const CopyParams& get_copy_params() const { return copy_params; }
  const std::list<const ColumnDescriptor*>& get_column_descs() const {
    return loader->get_column_descs();
  }
  void load(const std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
            size_t row_count,
            const Catalog_Namespace::SessionInfo* session_info);
  std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>>& get_import_buffers_vec() {
    return import_buffers_vec;
  }
  std::vector<std::unique_ptr<TypedImportBuffer>>& get_import_buffers(int i) {
    return import_buffers_vec[i];
  }
  const bool* get_is_array() const { return is_array_a.get(); }
#ifdef ENABLE_IMPORT_PARQUET
  void import_local_parquet(const std::string& file_path,
                            const Catalog_Namespace::SessionInfo* session_info) override;
#endif
  static ImportStatus get_import_status(const std::string& id);
  static void set_import_status(const std::string& id, const ImportStatus is);
  static std::vector<std::string> gdalGetAllFilesInArchive(
      const std::string& archive_path,
      const CopyParams& copy_params);
  enum class GeoFileLayerContents { EMPTY, GEO, NON_GEO, UNSUPPORTED_GEO };
  struct GeoFileLayerInfo {
    GeoFileLayerInfo(const std::string& name_, GeoFileLayerContents contents_)
        : name(name_), contents(contents_) {}
    std::string name;
    GeoFileLayerContents contents;
  };
  Catalog_Namespace::Catalog& getCatalog() { return loader->getCatalog(); }
  static void set_geo_physical_import_buffer(
      const Catalog_Namespace::Catalog& catalog,
      const ColumnDescriptor* cd,
      std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
      size_t& col_idx,
      std::vector<double>& coords,
      std::vector<double>& bounds,
      std::vector<int>& ring_sizes,
      std::vector<int>& poly_rings,
      int render_group);
  static void set_geo_physical_import_buffer_columnar(
      const Catalog_Namespace::Catalog& catalog,
      const ColumnDescriptor* cd,
      std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffers,
      size_t& col_idx,
      std::vector<std::vector<double>>& coords_column,
      std::vector<std::vector<double>>& bounds_column,
      std::vector<std::vector<int>>& ring_sizes_column,
      std::vector<std::vector<int>>& poly_rings_column,
      std::vector<int>& render_groups_column);
  void checkpoint(const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs);
  auto getLoader() const { return loader.get(); }

 private:
  static void setGDALAuthorizationTokens(const CopyParams& copy_params);
  std::string import_id;
  size_t file_size;
  size_t max_threads;
  char* buffer[2];
  std::vector<std::vector<std::unique_ptr<TypedImportBuffer>>> import_buffers_vec;
  std::unique_ptr<Loader> loader;
  std::unique_ptr<bool[]> is_array_a;
  static std::mutex init_gdal_mutex;
};

std::vector<std::unique_ptr<TypedImportBuffer>> setup_column_loaders(
    const TableDescriptor* td,
    Loader* loader);

}  // namespace import_export

#endif  // _IMPORTER_H_
