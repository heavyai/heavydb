/*
 * Copyright 2020 OmniSci, Inc.
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
 * File:   DBHandler.h
 * Author: michael
 *
 * Created on Jan 1, 2017, 12:40 PM
 */

#pragma once

#include "LeafAggregator.h"

#ifdef HAVE_PROFILER
#include <gperftools/heap-profiler.h>
#endif  // HAVE_PROFILER

#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Geospatial/Transforms.h"
#include "ImportExport/Importer.h"
#include "LockMgr/LockMgr.h"
#include "Logger/Logger.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "Parser/parser.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/TableGenerations.h"
#include "Shared/StringTransform.h"
#include "Shared/SystemParameters.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/mapd_shared_ptr.h"
#include "Shared/measure.h"
#include "Shared/scope.h"
#include "StringDictionary/StringDictionaryClient.h"
#include "ThriftHandler/ConnectionInfo.h"
#include "ThriftHandler/DistributedValidate.h"
#include "ThriftHandler/QueryState.h"
#include "ThriftHandler/RenderHandler.h"

#include <sys/types.h>
#include <thrift/server/TServer.h>
#include <thrift/transport/THttpTransport.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransport.h>
#include <atomic>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/noncopyable.hpp>
#include <boost/none_t.hpp>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <cmath>
#include <csignal>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>

#include "gen-cpp/OmniSci.h"
#include "gen-cpp/extension_functions_types.h"

using namespace std::string_literals;

class MapDAggHandler;
class MapDLeafHandler;

// Multiple concurrent requests for the same session can occur.  For that reason, each
// request briefly takes a lock to make a copy of the appropriate SessionInfo object. Then
// it releases the lock and uses the copy for the remainder of the request.
using SessionMap = std::map<TSessionId, std::shared_ptr<Catalog_Namespace::SessionInfo>>;
using PermissionFuncPtr = bool (*)(const AccessPrivileges&, const TDBObjectPermissions&);
using query_state::QueryStateProxy;

class TrackingProcessor : public OmniSciProcessor {
 public:
  TrackingProcessor(mapd::shared_ptr<OmniSciIf> handler, const bool check_origin)
      : OmniSciProcessor(handler), check_origin_(check_origin) {}

  bool process(mapd::shared_ptr<::apache::thrift::protocol::TProtocol> in,
               mapd::shared_ptr<::apache::thrift::protocol::TProtocol> out,
               void* connectionContext) {
    using namespace ::apache::thrift;

    auto transport = in->getTransport();
    if (transport && check_origin_) {
      static std::mutex processor_mutex;
      std::lock_guard lock(processor_mutex);
      const auto origin_str = transport->getOrigin();
      std::vector<std::string> origins;
      boost::split(origins, origin_str, boost::is_any_of(","));
      if (origins.empty()) {
        TrackingProcessor::client_address = origin_str;
      } else {
        // Take the first origin, which should be the client IP before any intermediate
        // servers (e.g. the web server)
        auto trimmed_origin = origins.front();
        boost::algorithm::trim(trimmed_origin);
        TrackingProcessor::client_address = trimmed_origin;
      }
      if (dynamic_cast<transport::THttpTransport*>(transport.get())) {
        TrackingProcessor::client_protocol = ClientProtocol::HTTP;
      } else if (dynamic_cast<transport::TBufferedTransport*>(transport.get())) {
        TrackingProcessor::client_protocol = ClientProtocol::TCP;
      } else {
        TrackingProcessor::client_protocol = ClientProtocol::Other;
      }
    } else {
      TrackingProcessor::client_address = "";
    }

    return OmniSciProcessor::process(in, out, connectionContext);
  }

  static thread_local std::string client_address;
  static thread_local ClientProtocol client_protocol;

 private:
  const bool check_origin_;
};

struct DiskCacheConfig;

class DBHandler : public OmniSciIf {
 public:
  DBHandler(const std::vector<LeafHostInfo>& db_leaves,
            const std::vector<LeafHostInfo>& string_leaves,
            const std::string& base_data_path,
            const bool cpu_only,
            const bool allow_multifrag,
            const bool jit_debug,
            const bool intel_jit_profile,
            const bool read_only,
            const bool allow_loop_joins,
            const bool enable_rendering,
            const bool renderer_use_vulkan_driver,
            const bool enable_auto_clear_render_mem,
            const int render_oom_retry_threshold,
            const size_t render_mem_bytes,
            const size_t max_concurrent_render_sessions,
            const int num_gpus,
            const int start_gpu,
            const size_t reserved_gpu_mem,
            const bool render_compositor_use_last_gpu,
            const size_t num_reader_threads,
            const AuthMetadata& authMetadata,
            const SystemParameters& system_parameters,
            const bool legacy_syntax,
            const int idle_session_duration,
            const int max_session_duration,
            const bool enable_runtime_udf_registration,
            const std::string& udf_filename,
            const std::string& clang_path,
            const std::vector<std::string>& clang_options,
#ifdef ENABLE_GEOS
            const std::string& libgeos_so_filename,
#endif
            const DiskCacheConfig& disk_cache_config);

  ~DBHandler() override;

  static inline size_t max_bytes_for_thrift() { return 2 * 1000 * 1000 * 1000L; }

  // Important ****
  //         This block must be keep in sync with mapd.thrift and HAHandler.h
  //         Please keep in same order for easy check and cut and paste
  // Important ****
  static void parser_with_error_handler(
      const std::string& query_str,
      std::list<std::unique_ptr<Parser::Stmt>>& parse_trees);

  void krb5_connect(TKrb5Session& session,
                    const std::string& token,
                    const std::string& dbname) override;
  // connection, admin
  void connect(TSessionId& session,
               const std::string& username,
               const std::string& passwd,
               const std::string& dbname) override;
  void disconnect(const TSessionId& session) override;
  void switch_database(const TSessionId& session, const std::string& dbname) override;
  void clone_session(TSessionId& session2, const TSessionId& session1) override;
  void get_server_status(TServerStatus& _return, const TSessionId& session) override;
  void get_status(std::vector<TServerStatus>& _return,
                  const TSessionId& session) override;
  void get_hardware_info(TClusterHardwareInfo& _return,
                         const TSessionId& session) override;

  bool hasTableAccessPrivileges(const TableDescriptor* td,
                                const Catalog_Namespace::SessionInfo& session_info);
  void get_tables(std::vector<std::string>& _return, const TSessionId& session) override;
  void get_physical_tables(std::vector<std::string>& _return,
                           const TSessionId& session) override;
  void get_views(std::vector<std::string>& _return, const TSessionId& session) override;
  void get_tables_meta(std::vector<TTableMeta>& _return,
                       const TSessionId& session) override;
  void get_table_details(TTableDetails& _return,
                         const TSessionId& session,
                         const std::string& table_name) override;
  void get_internal_table_details(TTableDetails& _return,
                                  const TSessionId& session,
                                  const std::string& table_name) override;
  void get_users(std::vector<std::string>& _return, const TSessionId& session) override;
  void get_databases(std::vector<TDBInfo>& _return, const TSessionId& session) override;

  void get_version(std::string& _return) override;
  void start_heap_profile(const TSessionId& session) override;
  void stop_heap_profile(const TSessionId& session) override;
  void get_heap_profile(std::string& _return, const TSessionId& session) override;
  void get_memory(std::vector<TNodeMemoryInfo>& _return,
                  const TSessionId& session,
                  const std::string& memory_level) override;
  void clear_cpu_memory(const TSessionId& session) override;
  void clear_gpu_memory(const TSessionId& session) override;
  void set_table_epoch(const TSessionId& session,
                       const int db_id,
                       const int table_id,
                       const int new_epoch) override;
  void set_table_epoch_by_name(const TSessionId& session,
                               const std::string& table_name,
                               const int new_epoch) override;
  int32_t get_table_epoch(const TSessionId& session,
                          const int32_t db_id,
                          const int32_t table_id) override;
  int32_t get_table_epoch_by_name(const TSessionId& session,
                                  const std::string& table_name) override;
  void get_table_epochs(std::vector<TTableEpochInfo>& _return,
                        const TSessionId& session,
                        const int32_t db_id,
                        const int32_t table_id) override;
  void set_table_epochs(const TSessionId& session,
                        const int32_t db_id,
                        const std::vector<TTableEpochInfo>& table_epochs) override;

  void get_session_info(TSessionInfo& _return, const TSessionId& session) override;
  // query, render
  void sql_execute(TQueryResult& _return,
                   const TSessionId& session,
                   const std::string& query,
                   const bool column_format,
                   const std::string& nonce,
                   const int32_t first_n,
                   const int32_t at_most_n) override;
  void get_completion_hints(std::vector<TCompletionHint>& hints,
                            const TSessionId& session,
                            const std::string& sql,
                            const int cursor) override;
  // TODO(miyu): merge the following two data frame APIs.
  void sql_execute_df(TDataFrame& _return,
                      const TSessionId& session,
                      const std::string& query,
                      const TDeviceType::type device_type,
                      const int32_t device_id,
                      const int32_t first_n,
                      const TArrowTransport::type transport_method) override;
  void sql_execute_gdf(TDataFrame& _return,
                       const TSessionId& session,
                       const std::string& query,
                       const int32_t device_id,
                       const int32_t first_n) override;
  void deallocate_df(const TSessionId& session,
                     const TDataFrame& df,
                     const TDeviceType::type device_type,
                     const int32_t device_id) override;
  void interrupt(const TSessionId& query_session,
                 const TSessionId& interrupt_session) override;
  void sql_validate(TRowDescriptor& _return,
                    const TSessionId& session,
                    const std::string& query) override;

  void set_execution_mode(const TSessionId& session,
                          const TExecuteMode::type mode) override;
  void render_vega(TRenderResult& _return,
                   const TSessionId& session,
                   const int64_t widget_id,
                   const std::string& vega_json,
                   const int32_t compression_level,
                   const std::string& nonce) override;
  void get_result_row_for_pixel(
      TPixelTableRowResult& _return,
      const TSessionId& session,
      const int64_t widget_id,
      const TPixel& pixel,
      const std::map<std::string, std::vector<std::string>>& table_col_names,
      const bool column_format,
      const int32_t pixel_radius,
      const std::string& nonce) override;

  // dashboards
  void get_dashboard(TDashboard& _return,
                     const TSessionId& session,
                     int32_t dashboard_id) override;
  void get_dashboards(std::vector<TDashboard>& _return,
                      const TSessionId& session) override;
  int32_t create_dashboard(const TSessionId& session,
                           const std::string& dashboard_name,
                           const std::string& dashboard_state,
                           const std::string& image_hash,
                           const std::string& dashboard_metadata) override;
  void replace_dashboard(const TSessionId& session,
                         const int32_t dashboard_id,
                         const std::string& dashboard_name,
                         const std::string& dashboard_owner,
                         const std::string& dashboard_state,
                         const std::string& image_hash,
                         const std::string& dashboard_metadata) override;
  void delete_dashboard(const TSessionId& session, const int32_t dashboard_id) override;
  void share_dashboards(const TSessionId& session,
                        const std::vector<int32_t>& dashboard_ids,
                        const std::vector<std::string>& groups,
                        const TDashboardPermissions& permissions) override;
  void delete_dashboards(const TSessionId& session,
                         const std::vector<int32_t>& dashboard_ids) override;
  void share_dashboard(const TSessionId& session,
                       const int32_t dashboard_id,
                       const std::vector<std::string>& groups,
                       const std::vector<std::string>& objects,
                       const TDashboardPermissions& permissions,
                       const bool grant_role) override;
  void unshare_dashboards(const TSessionId& session,
                          const std::vector<int32_t>& dashboard_ids,
                          const std::vector<std::string>& groups,
                          const TDashboardPermissions& permissions) override;
  void unshare_dashboard(const TSessionId& session,
                         const int32_t dashboard_id,
                         const std::vector<std::string>& groups,
                         const std::vector<std::string>& objects,
                         const TDashboardPermissions& permissions) override;
  void get_dashboard_grantees(std::vector<TDashboardGrantees>& _return,
                              const TSessionId& session,
                              int32_t dashboard_id) override;

  void get_link_view(TFrontendView& _return,
                     const TSessionId& session,
                     const std::string& link) override;
  void create_link(std::string& _return,
                   const TSessionId& session,
                   const std::string& view_state,
                   const std::string& view_metadata) override;
  // import
  void load_table_binary(const TSessionId& session,
                         const std::string& table_name,
                         const std::vector<TRow>& rows,
                         const std::vector<std::string>& column_names) override;

  std::unique_ptr<lockmgr::AbstractLockContainer<const TableDescriptor*>>
  prepare_columnar_loader(
      const Catalog_Namespace::SessionInfo& session_info,
      const std::string& table_name,
      size_t num_cols,
      std::unique_ptr<import_export::Loader>* loader,
      std::vector<std::unique_ptr<import_export::TypedImportBuffer>>* import_buffers,
      const std::vector<std::string>& column_names);

  void load_table_binary_columnar(const TSessionId& session,
                                  const std::string& table_name,
                                  const std::vector<TColumn>& cols,
                                  const std::vector<std::string>& column_names) override;
  void load_table_binary_arrow(const TSessionId& session,
                               const std::string& table_name,
                               const std::string& arrow_stream,
                               const bool use_column_names) override;

  void load_table(const TSessionId& session,
                  const std::string& table_name,
                  const std::vector<TStringRow>& rows,
                  const std::vector<std::string>& column_names) override;
  void detect_column_types(TDetectResult& _return,
                           const TSessionId& session,
                           const std::string& file_name,
                           const TCopyParams& copy_params) override;
  void create_table(const TSessionId& session,
                    const std::string& table_name,
                    const TRowDescriptor& row_desc,
                    const TFileType::type file_type,
                    const TCreateParams& create_params) override;
  void import_table(const TSessionId& session,
                    const std::string& table_name,
                    const std::string& file_name,
                    const TCopyParams& copy_params) override;
  void import_geo_table(const TSessionId& session,
                        const std::string& table_name,
                        const std::string& file_name,
                        const TCopyParams& copy_params,
                        const TRowDescriptor& row_desc,
                        const TCreateParams& create_params) override;
  void import_table_status(TImportStatus& _return,
                           const TSessionId& session,
                           const std::string& import_id) override;
  void get_first_geo_file_in_archive(std::string& _return,
                                     const TSessionId& session,
                                     const std::string& archive_path,
                                     const TCopyParams& copy_params) override;
  void get_all_files_in_archive(std::vector<std::string>& _return,
                                const TSessionId& session,
                                const std::string& archive_path,
                                const TCopyParams& copy_params) override;
  void get_layers_in_geo_file(std::vector<TGeoFileLayerInfo>& _return,
                              const TSessionId& session,
                              const std::string& file_name,
                              const TCopyParams& copy_params) override;
  // distributed
  int64_t query_get_outer_fragment_count(const TSessionId& session,
                                         const std::string& select_query) override;

  void check_table_consistency(TTableMeta& _return,
                               const TSessionId& session,
                               const int32_t table_id) override;
  void start_query(TPendingQuery& _return,
                   const TSessionId& leaf_session,
                   const TSessionId& parent_session,
                   const std::string& query_ra,
                   const bool just_explain,
                   const std::vector<int64_t>& outer_fragment_indices) override;
  void execute_query_step(TStepResult& _return,
                          const TPendingQuery& pending_query,
                          const TSubqueryId subquery_id) override;
  void broadcast_serialized_rows(const TSerializedRows& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const TQueryId query_id,
                                 const TSubqueryId subquery_id,
                                 const bool is_final_subquery_result) override;

  void start_render_query(TPendingRenderQuery& _return,
                          const TSessionId& session,
                          const int64_t widget_id,
                          const int16_t node_idx,
                          const std::string& vega_json) override;
  void execute_next_render_step(TRenderStepResult& _return,
                                const TPendingRenderQuery& pending_render,
                                const TRenderAggDataMap& merged_data) override;

  void insert_data(const TSessionId& session, const TInsertData& insert_data) override;
  void checkpoint(const TSessionId& session, const int32_t table_id) override;
  // DB Object Privileges
  void get_roles(std::vector<std::string>& _return, const TSessionId& session) override;
  bool has_role(const TSessionId& sessionId,
                const std::string& granteeName,
                const std::string& roleName) override;
  bool has_object_privilege(const TSessionId& sessionId,
                            const std::string& granteeName,
                            const std::string& objectName,
                            const TDBObjectType::type object_type,
                            const TDBObjectPermissions& permissions) override;
  void get_db_objects_for_grantee(std::vector<TDBObject>& _return,
                                  const TSessionId& session,
                                  const std::string& roleName) override;
  void get_db_object_privs(std::vector<TDBObject>& _return,
                           const TSessionId& session,
                           const std::string& objectName,
                           const TDBObjectType::type type) override;
  void get_all_roles_for_user(std::vector<std::string>& _return,
                              const TSessionId& session,
                              const std::string& granteeName) override;
  std::vector<std::string> get_valid_groups(const TSessionId& session,
                                            int32_t dashboard_id,
                                            std::vector<std::string> groups);
  // licensing
  void set_license_key(TLicenseInfo& _return,
                       const TSessionId& session,
                       const std::string& key,
                       const std::string& nonce) override;
  void get_license_claims(TLicenseInfo& _return,
                          const TSessionId& session,
                          const std::string& nonce) override;
  // user-defined functions
  /*
    Returns a mapping of device (CPU, GPU) parameters (name, LLVM IR
    triplet, features, etc)
   */
  void get_device_parameters(std::map<std::string, std::string>& _return,
                             const TSessionId& session) override;

  /*
    Register Runtime Extension Functions (UDFs, UDTFs) with given
    signatures. The extension functions implementations are given in a
    mapping of a device and the corresponding LLVM/NVVM IR string.
   */

  void register_runtime_extension_functions(
      const TSessionId& session,
      const std::vector<TUserDefinedFunction>& udfs,
      const std::vector<TUserDefinedTableFunction>& udtfs,
      const std::map<std::string, std::string>& device_ir_map) override;

  // end of sync block for HAHandler and mapd.thrift

  void shutdown();
  void emergency_shutdown();

  TSessionId getInvalidSessionId() const;

  void internal_connect(TSessionId& session,
                        const std::string& username,
                        const std::string& dbname);

  std::shared_ptr<Data_Namespace::DataMgr> data_mgr_;

  LeafAggregator leaf_aggregator_;
  const std::vector<LeafHostInfo> string_leaves_;
  const std::string base_data_path_;
  boost::filesystem::path import_path_;
  ExecutorDeviceType executor_device_type_;
  std::default_random_engine random_gen_;
  std::uniform_int_distribution<int64_t> session_id_dist_;
  const bool jit_debug_;
  const bool intel_jit_profile_;
  bool allow_multifrag_;
  const bool read_only_;
  const bool allow_loop_joins_;
  bool cpu_mode_only_;
  mapd_shared_mutex sessions_mutex_;
  std::mutex render_mutex_;
  int64_t start_time_;
  const AuthMetadata& authMetadata_;
  const SystemParameters& system_parameters_;
  std::unique_ptr<RenderHandler> render_handler_;
  std::unique_ptr<MapDAggHandler> agg_handler_;
  std::unique_ptr<MapDLeafHandler> leaf_handler_;
  std::shared_ptr<Calcite> calcite_;
  const bool legacy_syntax_;

  std::unique_ptr<QueryDispatchQueue> dispatch_queue_;

  template <typename... ARGS>
  std::shared_ptr<query_state::QueryState> create_query_state(ARGS&&... args) {
    return query_states_.create(std::forward<ARGS>(args)...);
  }

  // Exactly one immutable SessionInfo copy should be taken by a typical request.
  Catalog_Namespace::SessionInfo get_session_copy(const TSessionId& session);
  std::shared_ptr<Catalog_Namespace::SessionInfo> get_session_copy_ptr(
      const TSessionId& session);

  void get_tables_meta_impl(std::vector<TTableMeta>& _return,
                            QueryStateProxy query_state_proxy,
                            const Catalog_Namespace::SessionInfo& session_info,
                            const bool with_table_locks = true);

 private:
  std::shared_ptr<Catalog_Namespace::SessionInfo> create_new_session(
      TSessionId& session,
      const std::string& dbname,
      const Catalog_Namespace::UserMetadata& user_meta,
      std::shared_ptr<Catalog_Namespace::Catalog> cat);
  void connect_impl(TSessionId& session,
                    const std::string& passwd,
                    const std::string& dbname,
                    const Catalog_Namespace::UserMetadata& user_meta,
                    std::shared_ptr<Catalog_Namespace::Catalog> cat,
                    query_state::StdLog& stdlog);
  void disconnect_impl(const SessionMap::iterator& session_it,
                       mapd_unique_lock<mapd_shared_mutex>& write_lock);
  void check_table_load_privileges(const TSessionId& session,
                                   const std::string& table_name);
  void check_table_load_privileges(const Catalog_Namespace::SessionInfo& session_info,
                                   const std::string& table_name);
  void get_tables_impl(std::vector<std::string>& table_names,
                       const Catalog_Namespace::SessionInfo&,
                       const GetTablesType get_tables_type);
  void get_table_details_impl(TTableDetails& _return,
                              query_state::StdLog& stdlog,
                              const std::string& table_name,
                              const bool get_system,
                              const bool get_physical);
  void check_read_only(const std::string& str);
  void check_session_exp_unsafe(const SessionMap::iterator& session_it);
  void validateGroups(const std::vector<std::string>& groups);
  void validateDashboardIdsForSharing(const Catalog_Namespace::SessionInfo& session_info,
                                      const std::vector<int32_t>& dashboard_ids);
  void shareOrUnshareDashboards(const TSessionId& session,
                                const std::vector<int32_t>& dashboard_ids,
                                const std::vector<std::string>& groups,
                                const TDashboardPermissions& permissions,
                                const bool do_share);

  // Use get_session_copy() or get_session_copy_ptr() instead of get_const_session_ptr()
  // unless you know what you are doing. If you need to save a SessionInfo beyond the
  // present Thrift call, then the safe way to do this is by saving the return value of
  // get_const_session_ptr() as a std::weak_ptr.
  // Returns empty std::shared_ptr if session.empty().
  std::shared_ptr<const Catalog_Namespace::SessionInfo> get_const_session_ptr(
      const TSessionId& session);
  std::shared_ptr<Catalog_Namespace::SessionInfo> get_session_ptr(
      const TSessionId& session_id);
  template <typename SESSION_MAP_LOCK>
  SessionMap::iterator get_session_it_unsafe(const TSessionId& session,
                                             SESSION_MAP_LOCK& lock);
  static void value_to_thrift_column(const TargetValue& tv,
                                     const SQLTypeInfo& ti,
                                     TColumn& column);
  static TDatum value_to_thrift(const TargetValue& tv, const SQLTypeInfo& ti);
  static std::string apply_copy_to_shim(const std::string& query_str);

  std::pair<TPlanResult, lockmgr::LockedTableDescriptors> parse_to_ra(
      QueryStateProxy,
      const std::string& query_str,
      const std::vector<TFilterPushDownInfo>& filter_push_down_info,
      const bool acquire_locks,
      const SystemParameters system_parameters,
      bool check_privileges = true);

  void sql_execute_impl(TQueryResult& _return,
                        QueryStateProxy,
                        const bool column_format,
                        const std::string& nonce,
                        const ExecutorDeviceType executor_device_type,
                        const int32_t first_n,
                        const int32_t at_most_n);

  bool user_can_access_table(const Catalog_Namespace::SessionInfo&,
                             const TableDescriptor* td,
                             const AccessPrivileges acess_priv);

  void execute_distributed_copy_statement(
      Parser::CopyTableStmt*,
      const Catalog_Namespace::SessionInfo& session_info);

  TQueryResult validate_rel_alg(const std::string& query_ra, QueryStateProxy);

  std::vector<PushedDownFilterInfo> execute_rel_alg(
      TQueryResult& _return,
      QueryStateProxy,
      const std::string& query_ra,
      const bool column_format,
      const ExecutorDeviceType executor_device_type,
      const int32_t first_n,
      const int32_t at_most_n,
      const bool just_validate,
      const bool find_push_down_candidates,
      const ExplainInfo& explain_info,
      const std::optional<size_t> executor_index = std::nullopt) const;

  void execute_rel_alg_with_filter_push_down(
      TQueryResult& _return,
      QueryStateProxy,
      std::string& query_ra,
      const bool column_format,
      const ExecutorDeviceType executor_device_type,
      const int32_t first_n,
      const int32_t at_most_n,
      const bool just_explain,
      const bool just_calcite_explain,
      const std::vector<PushedDownFilterInfo>& filter_push_down_requests);

  void execute_rel_alg_df(TDataFrame& _return,
                          const std::string& query_ra,
                          QueryStateProxy query_state_proxy,
                          const Catalog_Namespace::SessionInfo& session_info,
                          const ExecutorDeviceType device_type,
                          const size_t device_id,
                          const int32_t first_n,
                          const TArrowTransport::type transport_method) const;

  void executeDdl(TQueryResult& _return,
                  const std::string& query_ra,
                  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  TColumnType populateThriftColumnType(const Catalog_Namespace::Catalog* cat,
                                       const ColumnDescriptor* cd);
  TRowDescriptor fixup_row_descriptor(const TRowDescriptor& row_desc,
                                      const Catalog_Namespace::Catalog& cat);
  void set_execution_mode_nolock(Catalog_Namespace::SessionInfo* session_ptr,
                                 const TExecuteMode::type mode);
  char unescape_char(std::string str);
  import_export::CopyParams thrift_to_copyparams(const TCopyParams& cp);
  TCopyParams copyparams_to_thrift(const import_export::CopyParams& cp);
  void check_geospatial_files(const boost::filesystem::path file_path,
                              const import_export::CopyParams& copy_params);
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

  void convert_rows(TQueryResult& _return,
                    QueryStateProxy query_state_proxy,
                    const std::vector<TargetMetaInfo>& targets,
                    const ResultSet& results,
                    const bool column_format,
                    const int32_t first_n,
                    const int32_t at_most_n) const;

  void create_simple_result(TQueryResult& _return,
                            const ResultSet& results,
                            const bool column_format,
                            const std::string label) const;

  std::vector<TargetMetaInfo> getTargetMetaInfo(
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const;

  std::vector<std::string> getTargetNames(
      const std::vector<TargetMetaInfo>& targets) const;

  std::vector<std::string> getTargetNames(
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const;

  void get_completion_hints_unsorted(std::vector<TCompletionHint>& hints,
                                     std::vector<std::string>& visible_tables,
                                     query_state::StdLog& stdlog,
                                     const std::string& sql,
                                     const int cursor);
  void get_token_based_completions(std::vector<TCompletionHint>& hints,
                                   query_state::StdLog& stdlog,
                                   std::vector<std::string>& visible_tables,
                                   const std::string& sql,
                                   const int cursor);

  std::unordered_map<std::string, std::unordered_set<std::string>>
  fill_column_names_by_table(std::vector<std::string>& table_names,
                             query_state::StdLog& stdlog);

  ConnectionInfo getConnectionInfo() const;

  static bool has_database_permission(const AccessPrivileges& privs,
                                      const TDBObjectPermissions& permissions);
  static bool has_table_permission(const AccessPrivileges& privs,
                                   const TDBObjectPermissions& permission);
  static bool has_dashboard_permission(const AccessPrivileges& privs,
                                       const TDBObjectPermissions& permissions);
  static bool has_view_permission(const AccessPrivileges& privs,
                                  const TDBObjectPermissions& permissions);
  static bool has_server_permission(const AccessPrivileges& privs,
                                    const TDBObjectPermissions& permissions);
  // For the provided upper case column names `uc_column_names`, return
  // the tables from `table_names` which contain at least one of them.
  // Used to rank the TABLE auto-completion hints by the columns
  // specified in the projection.
  std::unordered_set<std::string> get_uc_compatible_table_names_by_column(
      const std::unordered_set<std::string>& uc_column_names,
      std::vector<std::string>& table_names,
      query_state::StdLog& stdlog);

  query_state::QueryStates query_states_;
  SessionMap sessions_;

  bool super_user_rights_;           // default is "false"; setting to "true"
                                     // ignores passwd checks in "connect(..)"
                                     // method
  const int idle_session_duration_;  // max duration of idle session
  const int max_session_duration_;   // max duration of session

  const bool runtime_udf_registration_enabled_;

  struct GeoCopyFromState {
    std::string geo_copy_from_table;
    std::string geo_copy_from_file_name;
    import_export::CopyParams geo_copy_from_copy_params;
    std::string geo_copy_from_partitions;
  };

  struct GeoCopyFromSessions {
    std::unordered_map<std::string, GeoCopyFromState> was_geo_copy_from;
    std::mutex geo_copy_from_mutex;

    std::optional<GeoCopyFromState> operator()(const std::string& session_id) {
      std::lock_guard<std::mutex> map_lock(geo_copy_from_mutex);
      auto itr = was_geo_copy_from.find(session_id);
      if (itr == was_geo_copy_from.end()) {
        return std::nullopt;
      }
      return itr->second;
    }

    void add(const std::string& session_id, const GeoCopyFromState& state) {
      std::lock_guard<std::mutex> map_lock(geo_copy_from_mutex);
      const auto ret = was_geo_copy_from.insert(std::make_pair(session_id, state));
      CHECK(ret.second);
    }

    void remove(const std::string& session_id) {
      std::lock_guard<std::mutex> map_lock(geo_copy_from_mutex);
      was_geo_copy_from.erase(session_id);
    }
  };
  GeoCopyFromSessions geo_copy_from_sessions;

  // Only for IPC device memory deallocation
  mutable std::mutex handle_to_dev_ptr_mutex_;
  mutable std::unordered_map<std::string, std::string> ipc_handle_to_dev_ptr_;

  friend void run_warmup_queries(mapd::shared_ptr<DBHandler> handler,
                                 std::string base_path,
                                 std::string query_file_path);

  friend class RenderHandler::Impl;
  friend class MapDAggHandler;
  friend class MapDLeafHandler;

  std::map<const std::string, const PermissionFuncPtr> permissionFuncMap_ = {
      {"database"s, has_database_permission},
      {"dashboard"s, has_dashboard_permission},
      {"table"s, has_table_permission},
      {"view"s, has_view_permission},
      {"server"s, has_server_permission}};

  void check_and_invalidate_sessions(Parser::DDLStmt* ddl);

  template <typename STMT_TYPE>
  void invalidate_sessions(std::string& name, STMT_TYPE* stmt) {
    using namespace Parser;
    auto is_match = [&](auto session_it) {
      if (ShouldInvalidateSessionsByDB<STMT_TYPE>()) {
        return boost::iequals(name,
                              session_it->second->getCatalog().getCurrentDB().dbName);
      } else if (ShouldInvalidateSessionsByUser<STMT_TYPE>()) {
        return boost::iequals(name, session_it->second->get_currentUser().userName);
      }
      return false;
    };
    auto check_and_remove_sessions = [&]() {
      for (auto it = sessions_.begin(); it != sessions_.end();) {
        if (is_match(it)) {
          it = sessions_.erase(it);
        } else {
          ++it;
        }
      }
    };
    check_and_remove_sessions();
  }

  std::string const createInMemoryCalciteSession(
      const std::shared_ptr<Catalog_Namespace::Catalog>& catalog_ptr);
  bool isInMemoryCalciteSession(const Catalog_Namespace::UserMetadata user_meta);
  void removeInMemoryCalciteSession(const std::string& session_id);

  void getUserSessions(const Catalog_Namespace::SessionInfo& session_info,
                       TQueryResult& _return);

  // this function returns a set of queries queued in the DB
  // that belongs to the same DB in the caller's session
  void getQueries(const Catalog_Namespace::SessionInfo& session_info,
                  TQueryResult& _return);

  // this function passes the interrupt request to the DB executor
  void interruptQuery(const Catalog_Namespace::SessionInfo& session_info,
                      const std::string& target_session);
};
