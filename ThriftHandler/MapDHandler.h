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
#include <thrift/concurrency/PlatformThreadFactory.h>
#include <thrift/concurrency/ThreadManager.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/THttpServer.h>
#include <thrift/transport/TServerSocket.h>

#include "MapDRelease.h"

#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"
#include "DataMgr/LockMgr.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Import/Importer.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "Parser/parser.h"
#include "Planner/Planner.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/RelAlgExecutionDescriptor.h"
#include "QueryEngine/TableGenerations.h"
#include "Shared/ConfigResolve.h"
#include "Shared/MapDParameters.h"
#include "Shared/StringTransform.h"
#include "Shared/geosupport.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/mapd_shared_ptr.h"
#include "Shared/measure.h"
#include "Shared/scope.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include "gen-cpp/MapD.h"

using namespace std::string_literals;

class MapDRenderHandler;
class MapDAggHandler;
class MapDLeafHandler;

enum GetTablesType { GET_PHYSICAL_TABLES_AND_VIEWS, GET_PHYSICAL_TABLES, GET_VIEWS };

using SessionMap = std::map<TSessionId, std::shared_ptr<Catalog_Namespace::SessionInfo>>;
using permissionFuncPtr = bool (*)(const AccessPrivileges&, const TDBObjectPermissions&);

class MapDHandler : public MapDIf {
 public:
  MapDHandler(const std::vector<LeafHostInfo>& db_leaves,
              const std::vector<LeafHostInfo>& string_leaves,
              const std::string& base_data_path,
              const bool cpu_only,
              const bool allow_multifrag,
              const bool jit_debug,
              const bool read_only,
              const bool allow_loop_joins,
              const bool enable_rendering,
              const size_t render_mem_bytes,
              const int num_gpus,
              const int start_gpu,
              const size_t reserved_gpu_mem,
              const size_t num_reader_threads,
              const AuthMetadata authMetadata,
              const MapDParameters& mapd_parameters,
              const bool legacy_syntax,
              const bool access_priv_check,
              const int idle_session_duration,
              const int max_session_duration);

  ~MapDHandler();

  static inline size_t max_bytes_for_thrift() { return 2 * 1000 * 1000 * 1000L; }

  // Important ****
  //         This block must be keep in sync with mapd.thrift and HAHandler.h
  //         Please keep in same order for easy check and cut and paste
  // Important ****

  // connection, admin
  void connect(TSessionId& session,
               const std::string& user,
               const std::string& passwd,
               const std::string& dbname);
  void disconnect(const TSessionId& session);
  void get_server_status(TServerStatus& _return, const TSessionId& session);
  void get_status(std::vector<TServerStatus>& _return, const TSessionId& session);
  void get_hardware_info(TClusterHardwareInfo& _return, const TSessionId& session);

  bool hasTableAccessPrivileges(const TableDescriptor* td, const TSessionId& session);
  void get_tables(std::vector<std::string>& _return, const TSessionId& session);
  void get_physical_tables(std::vector<std::string>& _return, const TSessionId& session);
  void get_views(std::vector<std::string>& _return, const TSessionId& session);
  void get_tables_meta(std::vector<TTableMeta>& _return, const TSessionId& session);
  void get_table_details(TTableDetails& _return,
                         const TSessionId& session,
                         const std::string& table_name);
  void get_internal_table_details(TTableDetails& _return,
                                  const TSessionId& session,
                                  const std::string& table_name);
  void get_users(std::vector<std::string>& _return, const TSessionId& session);
  void get_databases(std::vector<TDBInfo>& _return, const TSessionId& session);

  void get_version(std::string& _return);
  void start_heap_profile(const TSessionId& session);
  void stop_heap_profile(const TSessionId& session);
  void get_heap_profile(std::string& _return, const TSessionId& session);
  void get_memory(std::vector<TNodeMemoryInfo>& _return,
                  const TSessionId& session,
                  const std::string& memory_level);
  void clear_cpu_memory(const TSessionId& session);
  void clear_gpu_memory(const TSessionId& session);
  void set_table_epoch(const TSessionId& session,
                       const int db_id,
                       const int table_id,
                       const int new_epoch);
  void set_table_epoch_by_name(const TSessionId& session,
                               const std::string& table_name,
                               const int new_epoch);
  int32_t get_table_epoch(const TSessionId& session,
                          const int32_t db_id,
                          const int32_t table_id);
  int32_t get_table_epoch_by_name(const TSessionId& session,
                                  const std::string& table_name);
  void get_session_info(TSessionInfo& _return, const TSessionId& session);
  // query, render
  void sql_execute(TQueryResult& _return,
                   const TSessionId& session,
                   const std::string& query,
                   const bool column_format,
                   const std::string& nonce,
                   const int32_t first_n,
                   const int32_t at_most_n);
  void get_completion_hints(std::vector<TCompletionHint>& hints,
                            const TSessionId& session,
                            const std::string& sql,
                            const int cursor);
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
  void deallocate_df(const TSessionId& session,
                     const TDataFrame& df,
                     const TDeviceType::type device_type,
                     const int32_t device_id);
  void interrupt(const TSessionId& session);
  void sql_validate(TTableDescriptor& _return,
                    const TSessionId& session,
                    const std::string& query);
  void set_execution_mode(const TSessionId& session, const TExecuteMode::type mode);
  void render_vega(TRenderResult& _return,
                   const TSessionId& session,
                   const int64_t widget_id,
                   const std::string& vega_json,
                   const int32_t compression_level,
                   const std::string& nonce);
  void get_result_row_for_pixel(
      TPixelTableRowResult& _return,
      const TSessionId& session,
      const int64_t widget_id,
      const TPixel& pixel,
      const std::map<std::string, std::vector<std::string>>& table_col_names,
      const bool column_format,
      const int32_t pixel_radius,
      const std::string& nonce);
  // Immerse
  void get_frontend_view(TFrontendView& _return,
                         const TSessionId& session,
                         const std::string& view_name);
  void get_frontend_views(std::vector<TFrontendView>& _return, const TSessionId& session);
  void create_frontend_view(const TSessionId& session,
                            const std::string& view_name,
                            const std::string& view_state,
                            const std::string& image_hash,
                            const std::string& view_metadata);
  void delete_frontend_view(const TSessionId& session, const std::string& view_name);

  // dashboards
  void get_dashboard(TDashboard& _return,
                     const TSessionId& session,
                     int32_t dashboard_id);
  void get_dashboards(std::vector<TDashboard>& _return, const TSessionId& session);
  int32_t create_dashboard(const TSessionId& session,
                           const std::string& dashboard_name,
                           const std::string& dashboard_state,
                           const std::string& image_hash,
                           const std::string& dashboard_metadata);
  void replace_dashboard(const TSessionId& session,
                         const int32_t dashboard_id,
                         const std::string& dashboard_name,
                         const std::string& dashboard_owner,
                         const std::string& dashboard_state,
                         const std::string& image_hash,
                         const std::string& dashboard_metadata);
  void delete_dashboard(const TSessionId& session, const int32_t dashboard_id);
  void share_dashboard(const TSessionId& session,
                       const int32_t dashboard_id,
                       const std::vector<std::string>& groups,
                       const std::vector<std::string>& objects,
                       const TDashboardPermissions& permissions);
  void unshare_dashboard(const TSessionId& session,
                         const int32_t dashboard_id,
                         const std::vector<std::string>& groups,
                         const std::vector<std::string>& objects,
                         const TDashboardPermissions& permissions);
  void get_dashboard_grantees(std::vector<TDashboardGrantees>& _return,
                              const TSessionId& session,
                              int32_t dashboard_id);

  void get_link_view(TFrontendView& _return,
                     const TSessionId& session,
                     const std::string& link);
  void create_link(std::string& _return,
                   const TSessionId& session,
                   const std::string& view_state,
                   const std::string& view_metadata);
  // import
  void load_table_binary(const TSessionId& session,
                         const std::string& table_name,
                         const std::vector<TRow>& rows);

  void prepare_columnar_loader(
      const Catalog_Namespace::SessionInfo& session_info,
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

  void load_table(const TSessionId& session,
                  const std::string& table_name,
                  const std::vector<TStringRow>& rows);
  void detect_column_types(TDetectResult& _return,
                           const TSessionId& session,
                           const std::string& file_name,
                           const TCopyParams& copy_params);
  void create_table(const TSessionId& session,
                    const std::string& table_name,
                    const TRowDescriptor& row_desc,
                    const TTableType::type table_type,
                    const TCreateParams& create_params);
  void import_table(const TSessionId& session,
                    const std::string& table_name,
                    const std::string& file_name,
                    const TCopyParams& copy_params);
  void import_geo_table(const TSessionId& session,
                        const std::string& table_name,
                        const std::string& file_name,
                        const TCopyParams& copy_params,
                        const TRowDescriptor& row_desc);
  void import_table_status(TImportStatus& _return,
                           const TSessionId& session,
                           const std::string& import_id);
  void get_first_geo_file_in_archive(std::string& _return,
                                     const TSessionId& session,
                                     const std::string& archive_path,
                                     const TCopyParams& copy_params);
  void get_all_files_in_archive(std::vector<std::string>& _return,
                                const TSessionId& session,
                                const std::string& archive_path,
                                const TCopyParams& copy_params);
  // distributed
  void check_table_consistency(TTableMeta& _return,
                               const TSessionId& session,
                               const int32_t table_id);
  void start_query(TPendingQuery& _return,
                   const TSessionId& session,
                   const std::string& query_ra,
                   const bool just_explain);
  void execute_first_step(TStepResult& _return, const TPendingQuery& pending_query);
  void broadcast_serialized_rows(const std::string& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const int64_t result_size,
                                 const TQueryId query_id);

  void start_render_query(TPendingRenderQuery& _return,
                          const TSessionId& session,
                          const int64_t widget_id,
                          const int16_t node_idx,
                          const std::string& vega_json);
  void execute_next_render_step(TRenderStepResult& _return,
                                const TPendingRenderQuery& pending_render,
                                const TRenderAggDataMap& merged_data);

  void insert_data(const TSessionId& session, const TInsertData& insert_data);
  void checkpoint(const TSessionId& session, const int32_t db_id, const int32_t table_id);
  // deprecated
  void get_table_descriptor(TTableDescriptor& _return,
                            const TSessionId& session,
                            const std::string& table_name);
  void get_row_descriptor(TRowDescriptor& _return,
                          const TSessionId& session,
                          const std::string& table_name);
  // DB Object Privileges
  void get_roles(std::vector<std::string>& _return, const TSessionId& session);
  bool has_object_privilege(const TSessionId& sessionId,
                            const std::string& granteeName,
                            const std::string& objectName,
                            const TDBObjectType::type object_type,
                            const TDBObjectPermissions& permissions);
  void get_db_objects_for_grantee(std::vector<TDBObject>& _return,
                                  const TSessionId& session,
                                  const std::string& roleName);
  void get_db_object_privs(std::vector<TDBObject>& _return,
                           const TSessionId& session,
                           const std::string& objectName,
                           const TDBObjectType::type type);
  void get_all_roles_for_user(std::vector<std::string>& _return,
                              const TSessionId& session,
                              const std::string& granteeName);
  std::vector<std::string> get_valid_groups(const TSessionId& session,
                                            int32_t dashboard_id,
                                            std::vector<std::string> groups);
  // licensing
  void set_license_key(TLicenseInfo& _return,
                       const TSessionId& session,
                       const std::string& key,
                       const std::string& nonce);
  void get_license_claims(TLicenseInfo& _return,
                          const TSessionId& session,
                          const std::string& nonce);
  void shutdown();
  // end of sync block for HAHandler and mapd.thrift

  TSessionId getInvalidSessionId() const;

  void internal_connect(TSessionId& session,
                        const std::string& user,
                        const std::string& dbname);

  std::shared_ptr<Data_Namespace::DataMgr> data_mgr_;
  std::map<TSessionId, std::shared_ptr<Catalog_Namespace::SessionInfo>> sessions_;

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
  std::unique_ptr<MapDRenderHandler> render_handler_;
  std::unique_ptr<MapDAggHandler> agg_handler_;
  std::unique_ptr<MapDLeafHandler> leaf_handler_;
  std::shared_ptr<Calcite> calcite_;
  const bool legacy_syntax_;
  Catalog_Namespace::SessionInfo get_session(const TSessionId& session);

 private:
  void connect_impl(TSessionId& session,
                    const std::string& passwd,
                    const std::string& dbname,
                    Catalog_Namespace::UserMetadata& user_meta,
                    std::shared_ptr<Catalog_Namespace::Catalog> cat);
  void disconnect_impl(const SessionMap::iterator& session_it);
  void check_table_load_privileges(const TSessionId& session,
                                   const std::string& table_name);
  void check_table_load_privileges(const Catalog_Namespace::SessionInfo& session_info,
                                   const std::string& table_name);
  void get_tables_impl(std::vector<std::string>& table_names,
                       const TSessionId& session,
                       const GetTablesType get_tables_type);
  void get_table_details_impl(TTableDetails& _return,
                              const TSessionId& session,
                              const std::string& table_name,
                              const bool get_system,
                              const bool get_physical);
  void check_read_only(const std::string& str);
  void check_session_exp(const SessionMap::iterator& session_it);
  SessionMap::iterator get_session_it(const TSessionId& session);
  static void value_to_thrift_column(const TargetValue& tv,
                                     const SQLTypeInfo& ti,
                                     TColumn& column);
  static TDatum value_to_thrift(const TargetValue& tv, const SQLTypeInfo& ti);
  static std::string apply_copy_to_shim(const std::string& query_str);

  std::string parse_to_ra(const std::string& query_str,
                          const std::vector<TFilterPushDownInfo>& filter_push_down_info,
                          const Catalog_Namespace::SessionInfo& session_info,
                          std::map<std::string, bool>* tableNames = nullptr);

  void sql_execute_impl(TQueryResult& _return,
                        const Catalog_Namespace::SessionInfo& session_info,
                        const std::string& query_str,
                        const bool column_format,
                        const std::string& nonce,
                        const ExecutorDeviceType executor_device_type,
                        const int32_t first_n,
                        const int32_t at_most_n);

  void execute_distributed_copy_statement(
      Parser::CopyTableStmt*,
      const Catalog_Namespace::SessionInfo& session_info);

  void validate_rel_alg(TTableDescriptor& _return,
                        const std::string& query_str,
                        const Catalog_Namespace::SessionInfo& session_info);
  std::vector<PushedDownFilterInfo> execute_rel_alg(
      TQueryResult& _return,
      const std::string& query_ra,
      const bool column_format,
      const Catalog_Namespace::SessionInfo& session_info,
      const ExecutorDeviceType executor_device_type,
      const int32_t first_n,
      const int32_t at_most_n,
      const bool just_explain,
      const bool just_validate,
      const bool find_push_down_candidates,
      const bool just_calcite_explain) const;

  void execute_rel_alg_with_filter_push_down(
      TQueryResult& _return,
      std::string& query_ra,
      const bool column_format,
      const Catalog_Namespace::SessionInfo& session_info,
      const ExecutorDeviceType executor_device_type,
      const int32_t first_n,
      const int32_t at_most_n,
      const bool just_explain,
      const bool just_calcite_explain,
      const std::string& query_str,
      const std::vector<PushedDownFilterInfo> filter_push_down_requests);

  void execute_rel_alg_df(TDataFrame& _return,
                          const std::string& query_ra,
                          const Catalog_Namespace::SessionInfo& session_info,
                          const ExecutorDeviceType device_type,
                          const size_t device_id,
                          const int32_t first_n) const;
  TColumnType populateThriftColumnType(const Catalog_Namespace::Catalog* cat,
                                       const ColumnDescriptor* cd);
  TRowDescriptor fixup_row_descriptor(const TRowDescriptor& row_desc,
                                      const Catalog_Namespace::Catalog& cat);
  void set_execution_mode_nolock(Catalog_Namespace::SessionInfo* session_ptr,
                                 const TExecuteMode::type mode);
  char unescape_char(std::string str);
  Importer_NS::CopyParams thrift_to_copyparams(const TCopyParams& cp);
  TCopyParams copyparams_to_thrift(const Importer_NS::CopyParams& cp);
  void check_geospatial_files(const boost::filesystem::path file_path,
                              const Importer_NS::CopyParams& copy_params);
  void render_rel_alg(TRenderResult& _return,
                      const std::string& query_ra,
                      const std::string& query_str,
                      const Catalog_Namespace::SessionInfo& session_info,
                      const std::string& render_type,
                      const bool is_projection_query);

  TColumnType create_geo_column(const TDatumType::type type,
                                const std::string& name,
                                const bool is_array);

  void convert_explain(TQueryResult& _return,
                       const ResultSet& results,
                       const bool column_format) const;
  void convert_result(TQueryResult& _return,
                      const ResultSet& results,
                      const bool column_format) const;

  template <class R>
  void convert_rows(TQueryResult& _return,
                    const std::vector<TargetMetaInfo>& targets,
                    const R& results,
                    const bool column_format,
                    const int32_t first_n,
                    const int32_t at_most_n) const;

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

  std::vector<std::string> getTargetNames(
      const std::vector<TargetMetaInfo>& targets) const;

  std::vector<std::string> getTargetNames(
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const;

  TColumnType convert_target_metainfo(const TargetMetaInfo& target,
                                      const size_t idx) const;

  TRowDescriptor convert_target_metainfo(
      const std::vector<TargetMetaInfo>& targets) const;

  void get_completion_hints_unsorted(std::vector<TCompletionHint>& hints,
                                     std::vector<std::string>& visible_tables,
                                     const TSessionId& session,
                                     const std::string& sql,
                                     const int cursor);
  void get_token_based_completions(std::vector<TCompletionHint>& hints,
                                   const TSessionId& session,
                                   const std::vector<std::string>& visible_tables,
                                   const std::string& sql,
                                   const int cursor);
  Planner::RootPlan* parse_to_plan(const std::string& query_str,
                                   const Catalog_Namespace::SessionInfo& session_info);
  Planner::RootPlan* parse_to_plan_legacy(
      const std::string& query_str,
      const Catalog_Namespace::SessionInfo& session_info,
      const std::string& action /* render or validate */);

  std::unordered_map<std::string, std::unordered_set<std::string>>
  fill_column_names_by_table(const std::vector<std::string>& table_names,
                             const TSessionId& session);

  static bool has_database_permission(const AccessPrivileges& privs,
                                      const TDBObjectPermissions& permissions);
  static bool has_table_permission(const AccessPrivileges& privs,
                                   const TDBObjectPermissions& permission);
  static bool has_dashboard_permission(const AccessPrivileges& privs,
                                       const TDBObjectPermissions& permissions);
  static bool has_view_permission(const AccessPrivileges& privs,
                                  const TDBObjectPermissions& permissions);

  // For the provided upper case column names `uc_column_names`, return the tables
  // from `table_names` which contain at least one of them. Used to rank the TABLE
  // auto-completion hints by the columns specified in the projection.
  std::unordered_set<std::string> get_uc_compatible_table_names_by_column(
      const std::unordered_set<std::string>& uc_column_names,
      const std::vector<std::string>& table_names,
      const TSessionId& session);

  bool super_user_rights_;  // default is "false"; setting to "true" ignores passwd checks
                            // in "connect(..)" method
  const bool access_priv_check_;
  const int idle_session_duration_;  // max duration of idle session
  const int max_session_duration_;   // max duration of session

  bool _was_geo_copy_from;
  std::string _geo_copy_from_table;
  std::string _geo_copy_from_file_name;
  Importer_NS::CopyParams _geo_copy_from_copy_params;

  // Only for IPC device memory deallocation
  mutable std::mutex handle_to_dev_ptr_mutex_;
  mutable std::unordered_map<std::string, int8_t*> ipc_handle_to_dev_ptr_;

  friend void run_warmup_queries(mapd::shared_ptr<MapDHandler> handler,
                                 std::string base_path,
                                 std::string query_file_path);

  friend class MapDRenderHandler;
  friend class MapDAggHandler;
  friend class MapDLeafHandler;

  std::map<const std::string, const permissionFuncPtr> permissionFuncMap_ = {
      {"database"s, has_database_permission},
      {"dashboard"s, has_dashboard_permission},
      {"table"s, has_table_permission},
      {"view"s, has_view_permission}};
};

#endif /* MAPDHANDLER_H */
