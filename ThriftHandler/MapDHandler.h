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
 * File:   MapDHandler.h
 * Author: michael
 *
 * Created on Jan 1, 2017, 12:40 PM
 */

#ifndef MAPDHANDLER_H
#define MAPDHANDLER_H

#include "LeafAggregator.h"
#ifdef HAVE_PROFILER
#include <gperftools/heap-profiler.h>
#endif  // HAVE_PROFILER
#include <thrift/concurrency/ThreadManager.h>
#include <thrift/concurrency/PlatformThreadFactory.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/server/TThreadPoolServer.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/THttpServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

#include "MapDRelease.h"

#include "Calcite/Calcite.h"

#include "Catalog/Catalog.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Import/Importer.h"
#include "Parser/parser.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "Planner/Planner.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/TableGenerations.h"
#include "Shared/geosupport.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/measure.h"
#include "Shared/scope.h"
#include "Shared/StringTransform.h"
#include "Shared/MapDParameters.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <memory>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <random>
#include <map>
#include <cmath>
#include <typeinfo>
#include <thread>
#include <glog/logging.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <regex>

#include "gen-cpp/MapD.h"

typedef std::map<TSessionId, std::shared_ptr<Catalog_Namespace::SessionInfo>> SessionMap;

class MapDHandler : public MapDIf {
 public:
  MapDHandler(const std::vector<LeafHostInfo>& db_leaves,
              const std::vector<LeafHostInfo>& string_leaves,
              const std::string& base_data_path,
              const std::string& executor_device,
              const bool allow_multifrag,
              const bool jit_debug,
              const bool read_only,
              const bool allow_loop_joins,
              const bool enable_rendering,
              const size_t cpu_buffer_mem_bytes,
              const size_t render_mem_bytes,
              const int num_gpus,
              const int start_gpu,
              const size_t reserved_gpu_mem,
              const size_t num_reader_threads,
              const LdapMetadata ldapMetadata,
              const MapDParameters& mapd_parameters,
              const std::string& db_convert_dir,
              const bool legacy_syntax,
              const bool access_priv_check);

  ~MapDHandler();

  void connect(TSessionId& session, const std::string& user, const std::string& passwd, const std::string& dbname);
  void internal_connect(TSessionId& session, const std::string& user, const std::string& dbname);
  void connectImpl(TSessionId& session,
                   const std::string& user,
                   const std::string& passwd,
                   const std::string& dbname,
                   Catalog_Namespace::UserMetadata& user_meta);
  void disconnect(const TSessionId& session);
  void interrupt(const TSessionId& session);
  void get_server_status(TServerStatus& _return, const TSessionId& session);
  void get_status(std::vector<TServerStatus>& _return, const TSessionId& session);
  void sql_execute(TQueryResult& _return,
                   const TSessionId& session,
                   const std::string& query,
                   const bool column_format,
                   const std::string& nonce,
                   const int32_t first_n);
  // TODO(miyu): merge the following two data frame APIs.
  void sql_execute_df(TDataFrame& _return,
                      const TSessionId& session,
                      const std::string& query,
                      const TDeviceType::type device_type,
                      const int32_t device_id,
                      const int32_t first_n);
  void sql_execute_gdf(TDataFrame& _return,
                       const TSessionId& session,
                       const std::string& query,
                       const int32_t device_id,
                       const int32_t first_n);
  void sql_validate(TTableDescriptor& _return, const TSessionId& session, const std::string& query);
  void get_table_descriptor(TTableDescriptor& _return, const TSessionId& session, const std::string& table_name);
  void get_row_descriptor(TRowDescriptor& _return, const TSessionId& session, const std::string& table_name);
  void get_frontend_view(TFrontendView& _return, const TSessionId& session, const std::string& view_name);
  void delete_frontend_view(const TSessionId& session, const std::string& view_name);
  void get_tables(std::vector<std::string>& _return, const TSessionId& session);
  void get_users(std::vector<std::string>& _return, const TSessionId& session);
  void get_databases(std::vector<TDBInfo>& _return, const TSessionId& session);
  void get_frontend_views(std::vector<TFrontendView>& _return, const TSessionId& session);
  void set_execution_mode(const TSessionId& session, const TExecuteMode::type mode);
  void get_version(std::string& _return);
  void get_memory(std::vector<TNodeMemoryInfo>& _return, const TSessionId& session, const std::string& memory_level);
  void load_table_binary(const TSessionId& session, const std::string& table_name, const std::vector<TRow>& rows);

  void prepare_columnar_loader(const TSessionId& session,
                               const std::string& table_name,
                               size_t num_cols,
                               std::unique_ptr<Importer_NS::Loader>* loader,
                               std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>* import_buffers);

  void load_table_binary_columnar(const TSessionId& session,
                                  const std::string& table_name,
                                  const std::vector<TColumn>& cols);
  void load_table_binary_arrow(const TSessionId& session,
                               const std::string& table_name,
                               const std::string& arrow_stream);

  void load_table(const TSessionId& session, const std::string& table_name, const std::vector<TStringRow>& rows);
  void render(TRenderResult& _return,
              const TSessionId& session,
              const std::string& query,
              const std::string& render_type,
              const std::string& nonce);
  void render_vega(TRenderResult& _return,
                   const TSessionId& session,
                   const int64_t widget_id,
                   const std::string& vega_json,
                   const int32_t compression_level,
                   const std::string& nonce);
  void create_frontend_view(const TSessionId& session,
                            const std::string& view_name,
                            const std::string& view_state,
                            const std::string& image_hash,
                            const std::string& view_metadata);
  void detect_column_types(TDetectResult& _return,
                           const TSessionId& session,
                           const std::string& file_name,
                           const TCopyParams& copy_params);
  void create_table(const TSessionId& session,
                    const std::string& table_name,
                    const TRowDescriptor& row_desc,
                    const TTableType::type table_type);
  void import_table(const TSessionId& session,
                    const std::string& table_name,
                    const std::string& file_name,
                    const TCopyParams& copy_params);
  void import_table_status(TImportStatus& _return, const TSessionId& session, const std::string& import_id);
  void get_link_view(TFrontendView& _return, const TSessionId& session, const std::string& link);
  void create_link(std::string& _return,
                   const TSessionId& session,
                   const std::string& view_state,
                   const std::string& view_metadata);
  void get_rows_for_pixels(TPixelResult& _return,
                           const TSessionId& session,
                           const int64_t widget_id,
                           const std::vector<TPixel>& pixels,
                           const std::string& table_name,
                           const std::vector<std::string>& col_names,
                           const bool column_format,
                           const std::string& nonce);
  void get_row_for_pixel(TPixelRowResult& _return,
                         const TSessionId& session,
                         const int64_t widget_id,
                         const TPixel& pixel,
                         const std::string& table_name,
                         const std::vector<std::string>& col_names,
                         const bool column_format,
                         const int32_t pixelRadius,
                         const std::string& nonce);
  void get_result_row_for_pixel(TPixelTableRowResult& _return,
                                const TSessionId& session,
                                const int64_t widget_id,
                                const TPixel& pixel,
                                const std::map<std::string, std::vector<std::string>>& table_col_names,
                                const bool column_format,
                                const int32_t pixelRadius,
                                const std::string& nonce);
  void start_heap_profile(const TSessionId& session);
  void stop_heap_profile(const TSessionId& session);
  void get_heap_profile(std::string& _return, const TSessionId& session);
  void import_geo_table(const TSessionId& session,
                        const std::string& table_name,
                        const std::string& file_name,
                        const TCopyParams& copy_params,
                        const TRowDescriptor& row_desc);
  void start_query(TPendingQuery& _return,
                   const TSessionId& session,
                   const std::string& query_ra,
                   const bool just_explain);
  void execute_first_step(TStepResult& _return, const TPendingQuery& pending_query);
  void broadcast_serialized_rows(const std::string& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const TQueryId query_id);
  void insert_data(const TSessionId& session, const TInsertData& insert_data);
  void render_vega_raw_pixels(TRawPixelDataResult& _return,
                              const TSessionId& session,
                              const int64_t widget_id,
                              const int16_t node_idx,
                              const std::string& vega_json);
  void checkpoint(const TSessionId& session, const int32_t db_id, const int32_t table_id);
  void get_table_details(TTableDetails& _return, const TSessionId& session, const std::string& table_name);
  void get_internal_table_details(TTableDetails& _return, const TSessionId& session, const std::string& table_name);
  void clear_gpu_memory(const TSessionId& session);
  void clear_cpu_memory(const TSessionId& session);
  TSessionId getInvalidSessionId() const;
  void rollback_table_epoch(const TSessionId& session, const int db_id, const int table_id, const int new_epoch);

  std::unique_ptr<Catalog_Namespace::SysCatalog> sys_cat_;
  std::shared_ptr<Data_Namespace::DataMgr> data_mgr_;
  std::map<TSessionId, std::shared_ptr<Catalog_Namespace::SessionInfo>> sessions_;
  std::map<std::string, std::shared_ptr<Catalog_Namespace::Catalog>> cat_map_;

  LeafAggregator leaf_aggregator_;
  const std::vector<LeafHostInfo> string_leaves_;
  const std::string base_data_path_;
  boost::filesystem::path import_path_;
  ExecutorDeviceType executor_device_type_;
  std::default_random_engine random_gen_;
  std::uniform_int_distribution<int64_t> session_id_dist_;
  const bool jit_debug_;
  bool allow_multifrag_;
  const bool read_only_;
  const bool allow_loop_joins_;
  bool cpu_mode_only_;
  mapd_shared_mutex sessions_mutex_;
  std::mutex render_mutex_;
  int64_t start_time_;
  const MapDParameters& mapd_parameters_;
  bool enable_rendering_;
  std::shared_ptr<Calcite> calcite_;
  const bool legacy_syntax_;
  Catalog_Namespace::SessionInfo get_session(const TSessionId& session);

 private:
  void check_table_load_privileges(const TSessionId& session, const std::string& table_name);
  void get_table_details_impl(TTableDetails& _return,
                              const TSessionId& session,
                              const std::string& table_name,
                              const bool get_system);
  void check_read_only(const std::string& str);
  SessionMap::iterator get_session_it(const TSessionId& session);
  static void value_to_thrift_column(const TargetValue& tv, const SQLTypeInfo& ti, TColumn& column);
  static TDatum value_to_thrift(const TargetValue& tv, const SQLTypeInfo& ti);
  std::string parse_to_ra(const std::string& query_str, const Catalog_Namespace::SessionInfo& session_info);

  void sql_execute_impl(TQueryResult& _return,
                        const Catalog_Namespace::SessionInfo& session_info,
                        const std::string& query_str,
                        const bool column_format,
                        const std::string& nonce,
                        const ExecutorDeviceType executor_device_type,
                        const int32_t first_n);

  void execute_distributed_copy_statement(Parser::CopyTableStmt*, const Catalog_Namespace::SessionInfo& session_info);

  void validate_rel_alg(TTableDescriptor& _return,
                        const std::string& query_str,
                        const Catalog_Namespace::SessionInfo& session_info);
  void execute_rel_alg(TQueryResult& _return,
                       const std::string& query_ra,
                       const bool column_format,
                       const Catalog_Namespace::SessionInfo& session_info,
                       const ExecutorDeviceType executor_device_type,
                       const int32_t first_n,
                       const bool just_explain,
                       const bool just_validate) const;
  void execute_rel_alg_df(TDataFrame& _return,
                          const std::string& query_ra,
                          const Catalog_Namespace::SessionInfo& session_info,
                          const ExecutorDeviceType device_type,
                          const size_t device_id,
                          const int32_t first_n) const;
  TColumnType populateThriftColumnType(const Catalog_Namespace::Catalog* cat, const ColumnDescriptor* cd);
  TRowDescriptor fixup_row_descriptor(const TRowDescriptor& row_desc, const Catalog_Namespace::Catalog& cat);
  void set_execution_mode_nolock(Catalog_Namespace::SessionInfo* session_ptr, const TExecuteMode::type mode);
  char unescape_char(std::string str);
  Importer_NS::CopyParams thrift_to_copyparams(const TCopyParams& cp);
  TCopyParams copyparams_to_thrift(const Importer_NS::CopyParams& cp);
  void check_geospatial_files(const boost::filesystem::path file_path);
  std::string sanitize_name(const std::string& name);
  void render_rel_alg(TRenderResult& _return,
                      const std::string& query_ra,
                      const std::string& query_str,
                      const Catalog_Namespace::SessionInfo& session_info,
                      const std::string& render_type,
                      const bool is_projection_query);

  TColumnType create_array_column(const TDatumType::type type, const std::string& name);
  void throw_profile_exception(const std::string& error_msg);

  void convert_explain(TQueryResult& _return, const ResultSet& results, const bool column_format) const;
  void convert_result(TQueryResult& _return, const ResultSet& results, const bool column_format) const;

  template <class R>
  void convert_rows(TQueryResult& _return,
                    const std::vector<TargetMetaInfo>& targets,
                    const R& results,
                    const bool column_format,
                    const int32_t first_n) const;

  void create_simple_result(TQueryResult& _return,
                            const ResultSet& results,
                            const bool column_format,
                            const std::string label) const;

  void execute_root_plan(TQueryResult& _return,
                         const Planner::RootPlan* root_plan,
                         const bool column_format,
                         const Catalog_Namespace::SessionInfo& session_info,
                         const ExecutorDeviceType executor_device_type,
                         const int32_t first_n) const;

  std::vector<TargetMetaInfo> getTargetMetaInfo(
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const;

  std::vector<std::string> getTargetNames(const std::vector<TargetMetaInfo>& targets) const;
  std::vector<std::string> getTargetNames(const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const;

  void render_root_plan(TRenderResult& _return,
                        Planner::RootPlan* root_plan,
                        const std::string& query_str,
                        const Catalog_Namespace::SessionInfo& session_info,
                        const std::string& render_type,
                        const bool is_projection_query);

  TRowDescriptor convert_target_metainfo(const std::vector<TargetMetaInfo>& targets) const;

  Planner::RootPlan* parse_to_plan(const std::string& query_str, const Catalog_Namespace::SessionInfo& session_info);

  std::vector<TColumnRange> column_ranges_to_thrift(const AggregatedColRange& column_ranges);

  std::vector<TDictionaryGeneration> string_dictionary_generations_to_thrift(
      const StringDictionaryGenerations& dictionary_generations);

  static std::vector<TTableGeneration> table_generations_to_thrift(const TableGenerations& table_generations);

  Planner::RootPlan* parse_to_plan_legacy(const std::string& query_str,
                                          const Catalog_Namespace::SessionInfo& session_info,
                                          const std::string& action /* render or validate */);

  bool super_user_rights_;  // default is "false"; setting to "true" ignores passwd checks in "connect(..)" method
  const bool access_priv_check_;
  friend void run_warmup_queries(boost::shared_ptr<MapDHandler> handler,
                                 std::string base_path,
                                 std::string query_file_path);
};

#endif /* MAPDHANDLER_H */
