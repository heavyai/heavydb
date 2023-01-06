/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @file   DBHandler.h
 * @brief
 *
 */

#pragma once

#include "LeafAggregator.h"

#ifdef HAVE_PROFILER
#include <gperftools/heap-profiler.h>
#endif  // HAVE_PROFILER

#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"
#include "Catalog/SessionsStore.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Geospatial/Transforms.h"
#include "ImportExport/Importer.h"
#include "ImportExport/RenderGroupAnalyzer.h"
#include "LockMgr/LockMgr.h"
#include "Logger/Logger.h"
#include "Parser/ParserNode.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryEngine/TableGenerations.h"
#include "Shared/StringTransform.h"
#include "Shared/SystemParameters.h"
#include "Shared/clean_boost_regex.hpp"
#include "Shared/heavyai_shared_mutex.h"
#include "Shared/measure.h"
#include "Shared/scope.h"
#include "StringDictionary/StringDictionaryClient.h"
#include "ThriftHandler/ConnectionInfo.h"
#include "ThriftHandler/QueryState.h"
#include "ThriftHandler/RenderHandler.h"
#include "ThriftHandler/SystemValidator.h"

#include <sys/types.h>
#include <thrift/server/TServer.h>
#include <thrift/transport/THttpClient.h>
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
#include <boost/tokenizer.hpp>
#include <cmath>
#include <csignal>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>

#include "gen-cpp/Heavy.h"
#include "gen-cpp/extension_functions_types.h"

using namespace std::string_literals;

class HeavyDBAggHandler;
class HeavyDBLeafHandler;

// Multiple concurrent requests for the same session can occur.  For that reason, each
// request briefly takes a lock to make a copy of the appropriate SessionInfo object. Then
// it releases the lock and uses the copy for the remainder of the request.
using SessionMap = std::map<TSessionId, std::shared_ptr<Catalog_Namespace::SessionInfo>>;
using PermissionFuncPtr = bool (*)(const AccessPrivileges&, const TDBObjectPermissions&);
using query_state::QueryStateProxy;

namespace dbhandler {
bool is_info_schema_db(const std::string& db_name);

void check_not_info_schema_db(const std::string& db_name,
                              bool throw_db_exception = false);
}  // namespace dbhandler

class TrackingProcessor : public HeavyProcessor {
 public:
  TrackingProcessor(std::shared_ptr<HeavyIf> handler, const bool check_origin)
      : HeavyProcessor(handler), check_origin_(check_origin) {}

  bool process(std::shared_ptr<::apache::thrift::protocol::TProtocol> in,
               std::shared_ptr<::apache::thrift::protocol::TProtocol> out,
               void* connectionContext) override {
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

    return HeavyProcessor::process(in, out, connectionContext);
  }

  static thread_local std::string client_address;
  static thread_local ClientProtocol client_protocol;

 private:
  const bool check_origin_;
};

namespace File_Namespace {
struct DiskCacheConfig;
}

class DBHandler : public HeavyIf {
 public:
  DBHandler(const std::vector<LeafHostInfo>& db_leaves,
            const std::vector<LeafHostInfo>& string_leaves,
            const std::string& base_data_path,
            const bool allow_multifrag,
            const bool jit_debug,
            const bool intel_jit_profile,
            const bool read_only,
            const bool allow_loop_joins,
            const bool enable_rendering,
            const bool renderer_use_ppll_polys,
            const bool renderer_prefer_igpu,
            const unsigned renderer_vulkan_timeout_ms,
            const bool renderer_use_parallel_executors,
            const bool enable_auto_clear_render_mem,
            const int render_oom_retry_threshold,
            const size_t render_mem_bytes,
            const size_t max_concurrent_render_sessions,
            const size_t reserved_gpu_mem,
            const bool render_compositor_use_last_gpu,
            const size_t num_reader_threads,
            const AuthMetadata& authMetadata,
            SystemParameters& system_parameters,
            const bool legacy_syntax,
            const int idle_session_duration,
            const int max_session_duration,
            const std::string& udf_filename,
            const std::string& clang_path,
            const std::vector<std::string>& clang_options,
#ifdef ENABLE_GEOS
            const std::string& libgeos_so_filename,
#endif
            const File_Namespace::DiskCacheConfig& disk_cache_config,
            const bool is_new_db);
  void initialize(const bool is_new_db);
  ~DBHandler() override;

  static inline size_t max_bytes_for_thrift() { return 2 * 1000 * 1000 * 1000LL; }

  // Important ****
  //         This block must be keep in sync with mapd.thrift and HAHandler.h
  //         Please keep in same order for easy check and cut and paste
  // Important ****

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
  void get_tables_for_database(std::vector<std::string>& _return,
                               const TSessionId& session,
                               const std::string& database_name) override;
  void get_physical_tables(std::vector<std::string>& _return,
                           const TSessionId& session) override;
  void get_views(std::vector<std::string>& _return, const TSessionId& session) override;
  void get_tables_meta(std::vector<TTableMeta>& _return,
                       const TSessionId& session) override;
  void get_table_details(TTableDetails& _return,
                         const TSessionId& session,
                         const std::string& table_name) override;
  void get_table_details_for_database(TTableDetails& _return,
                                      const TSessionId& session,
                                      const std::string& table_name,
                                      const std::string& database_name) override;
  void get_internal_table_details(TTableDetails& _return,
                                  const TSessionId& session,
                                  const std::string& table_name,
                                  const bool include_system_columns) override;
  void get_internal_table_details_for_database(TTableDetails& _return,
                                               const TSessionId& session,
                                               const std::string& table_name,
                                               const std::string& database_name) override;
  void get_users(std::vector<std::string>& _return, const TSessionId& session) override;
  void put_immerse_users_metadata(
      const TSessionId& session,
      const std::vector<TImmerseUserMetadata>& immerse_user_metadata_list) override;
  void get_users_info(std::vector<TUserInfo>& return_list,
                      const TSessionId& session) override;
  void put_immerse_database_metadata(const TSessionId& session,
                                     const std::string& immerse_metadata_json,
                                     const std::string& database_name) override;
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
  void clearRenderMemory(const TSessionId& session);  // it's not declared on thrifth
                                                      // and on persisten leaf client
  void set_cur_session(const TSessionId& parent_session,
                       const TSessionId& leaf_session,
                       const std::string& start_time_str,
                       const std::string& label,
                       bool for_running_query_kernel) override;
  void invalidate_cur_session(const TSessionId& parent_session,
                              const TSessionId& leaf_session,
                              const std::string& start_time_str,
                              const std::string& label,
                              bool for_running_query_kernel) override;
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

  void set_leaf_info(const TSessionId& session, const TLeafInfo& info) override;

  void sql_execute(ExecutionResult& _return,
                   const TSessionId& session,
                   const std::string& query,
                   const bool column_format,
                   const int32_t first_n,
                   const int32_t at_most_n,
                   lockmgr::LockedTableDescriptors& locks);
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
  TExecuteMode::type getExecutionMode(const TSessionId& session);
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

  // custom expressions
  int32_t create_custom_expression(const TSessionId& session,
                                   const TCustomExpression& custom_expression) override;
  void get_custom_expressions(std::vector<TCustomExpression>& _return,
                              const TSessionId& session) override;
  void update_custom_expression(const TSessionId& session,
                                const int32_t id,
                                const std::string& expression_json) override;
  void delete_custom_expressions(const TSessionId& session,
                                 const std::vector<int32_t>& custom_expression_ids,
                                 const bool do_soft_delete) override;

  // dashboards
  void get_dashboard(TDashboard& _return,
                     const TSessionId& session,
                     const int32_t dashboard_id) override;
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
                              const int32_t dashboard_id) override;

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

  void load_table_binary_columnar(const TSessionId& session,
                                  const std::string& table_name,
                                  const std::vector<TColumn>& cols,
                                  const std::vector<std::string>& column_names) override;
  void load_table_binary_columnar_polys(const TSessionId& session,
                                        const std::string& table_name,
                                        const std::vector<TColumn>& cols,
                                        const std::vector<std::string>& column_names,
                                        const bool assign_render_groups) override;
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
                   const std::string& serialized_rel_alg_dag,
                   const std::string& start_time_str,
                   const bool just_explain,
                   const std::vector<int64_t>& outer_fragment_indices) override;
  void execute_query_step(TStepResult& _return,
                          const TPendingQuery& pending_query,
                          const TSubqueryId subquery_id,
                          const std::string& start_time_str) override;
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
  void insert_chunks(const TSessionId& session,
                     const TInsertChunks& insert_chunks) override;
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
  void get_all_effective_roles_for_user(std::vector<std::string>& _return,
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

  /*
    Returns a list of User-Defined Function names available
   */
  void get_function_names(std::vector<std::string>& _return,
                          const TSessionId& session) override;

  /*
    Returns a list of runtime User-Defined Function names available
   */
  void get_runtime_function_names(std::vector<std::string>& _return,
                                  const TSessionId& session) override;

  /*
    Returns a list of runtime User-Defined Function names available
   */
  void get_function_details(std::vector<TUserDefinedFunction>& _return,
                            const TSessionId& session,
                            const std::vector<std::string>& udf_names) override;

  /*
    Returns a list of User-Defined Table Function names available
   */
  void get_table_function_names(std::vector<std::string>& _return,
                                const TSessionId& session) override;

  /*
    Returns a list of runtime User-Defined Table Function names available
   */
  void get_runtime_table_function_names(std::vector<std::string>& _return,
                                        const TSessionId& session) override;

  /*
    Returns a list of User-Defined Table Function details
   */
  void get_table_function_details(std::vector<TUserDefinedTableFunction>& _return,
                                  const TSessionId& session,
                                  const std::vector<std::string>& udtf_names) override;

  // end of sync block for HAHandler and mapd.thrift

  void shutdown();
  void emergency_shutdown();

  TSessionId getInvalidSessionId() const;

  void internal_connect(TSessionId& session,
                        const std::string& username,
                        const std::string& dbname);

  bool isAggregator() const;

  bool checkInMemorySystemTableQuery(
      const std::unordered_set<shared::TableKey>& tables_selected_from) const;

  std::shared_ptr<Data_Namespace::DataMgr> data_mgr_;

  LeafAggregator leaf_aggregator_;
  std::vector<LeafHostInfo> db_leaves_;
  std::vector<LeafHostInfo> string_leaves_;
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
  heavyai::shared_mutex sessions_mutex_;
  std::mutex render_mutex_;
  int64_t start_time_;
  const AuthMetadata& authMetadata_;
  SystemParameters& system_parameters_;
  std::shared_ptr<QueryEngine> query_engine_;
  std::unique_ptr<RenderHandler> render_handler_;
  std::unique_ptr<HeavyDBAggHandler> agg_handler_;
  std::unique_ptr<HeavyDBLeafHandler> leaf_handler_;
  std::shared_ptr<Calcite> calcite_;
  const bool legacy_syntax_;

  std::unique_ptr<QueryDispatchQueue> dispatch_queue_;

  template <typename... ARGS>
  std::shared_ptr<query_state::QueryState> create_query_state(ARGS&&... args) {
    return query_states_.create(std::forward<ARGS>(args)...);
  }

  // Exactly one immutable SessionInfo copy should be taken by a typical request.
  Catalog_Namespace::SessionInfo get_session_copy(const TSessionId& session_id);

  void get_tables_meta_impl(std::vector<TTableMeta>& _return,
                            QueryStateProxy query_state_proxy,
                            const Catalog_Namespace::SessionInfo& session_info,
                            const bool with_table_locks = true);

  // Visible for use in tests.
  void resizeDispatchQueue(size_t queue_size);

 protected:
  // Returns empty std::shared_ptr if session.empty().
  std::shared_ptr<Catalog_Namespace::SessionInfo> get_session_ptr(
      const TSessionId& session_id);

  ConnectionInfo getConnectionInfo() const;

 private:
  std::atomic<bool> initialized_{false};
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
  void disconnect_impl(Catalog_Namespace::SessionInfoPtr& session_ptr);
  void check_table_load_privileges(const Catalog_Namespace::SessionInfo& session_info,
                                   const std::string& table_name);
  void get_tables_impl(std::vector<std::string>& table_names,
                       const Catalog_Namespace::SessionInfo&,
                       const GetTablesType get_tables_type,
                       const std::string& database_name = {});
  void get_table_details_impl(TTableDetails& _return,
                              query_state::StdLog& stdlog,
                              const std::string& table_name,
                              const bool get_system,
                              const bool get_physical,
                              const std::string& database_name = {});
  void getAllRolesForUserImpl(
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr,
      std::vector<std::string>& roles,
      const std::string& granteeName,
      bool effective);
  void check_read_only(const std::string& str);
  void validateGroups(const std::vector<std::string>& groups);
  void validateDashboardIdsForSharing(const Catalog_Namespace::SessionInfo& session_info,
                                      const std::vector<int32_t>& dashboard_ids);
  void shareOrUnshareDashboards(const TSessionId& session,
                                const std::vector<int32_t>& dashboard_ids,
                                const std::vector<std::string>& groups,
                                const TDashboardPermissions& permissions,
                                const bool do_share);

  static void value_to_thrift_column(const TargetValue& tv,
                                     const SQLTypeInfo& ti,
                                     TColumn& column);
  static TDatum value_to_thrift(const TargetValue& tv, const SQLTypeInfo& ti);

  std::pair<TPlanResult, lockmgr::LockedTableDescriptors> parse_to_ra(
      QueryStateProxy,
      const std::string& query_str,
      const std::vector<TFilterPushDownInfo>& filter_push_down_info,
      const bool acquire_locks,
      const SystemParameters& system_parameters,
      bool check_privileges = true);

  void sql_execute_local(
      TQueryResult& _return,
      const QueryStateProxy& query_state_proxy,
      const std::shared_ptr<Catalog_Namespace::SessionInfo> session_ptr,
      const std::string& query_str,
      const bool column_format,
      const std::string& nonce,
      const int32_t first_n,
      const int32_t at_most_n,
      const bool use_calcite);

  int64_t process_deferred_copy_from(const TSessionId& session_id);

  static void convertData(TQueryResult& _return,
                          ExecutionResult& result,
                          const QueryStateProxy& query_state_proxy,
                          const bool column_format,
                          const int32_t first_n,
                          const int32_t at_most_n);

  void sql_execute_impl(ExecutionResult& _return,
                        QueryStateProxy,
                        const bool column_format,
                        const ExecutorDeviceType executor_device_type,
                        const int32_t first_n,
                        const int32_t at_most_n,
                        const bool use_calcite,
                        lockmgr::LockedTableDescriptors& locks);

  bool user_can_access_table(const Catalog_Namespace::SessionInfo&,
                             const TableDescriptor* td,
                             const AccessPrivileges acess_priv);

  void execute_distributed_copy_statement(
      Parser::CopyTableStmt*,
      const Catalog_Namespace::SessionInfo& session_info);

  TPlanResult processCalciteRequest(
      QueryStateProxy,
      const std::shared_ptr<Catalog_Namespace::Catalog>& cat,
      const std::string& query_str,
      const std::vector<TFilterPushDownInfo>& filter_push_down_info,
      const SystemParameters& system_parameters,
      const bool check_privileges);

  TRowDescriptor validateRelAlg(const std::string& query_ra,
                                QueryStateProxy query_state_proxy);

  void dispatch_query_task(std::shared_ptr<QueryDispatchQueue::Task> query_task,
                           const bool is_update_delete);

  std::vector<PushedDownFilterInfo> execute_rel_alg(
      ExecutionResult& _return,
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
      ExecutionResult& _return,
      QueryStateProxy,
      std::string& query_ra,
      const bool column_format,
      const ExecutorDeviceType executor_device_type,
      const int32_t first_n,
      const int32_t at_most_n,
      const bool just_explain,
      const bool just_calcite_explain,
      const std::vector<PushedDownFilterInfo>& filter_push_down_requests);

  void executeDdl(TQueryResult& _return,
                  const std::string& query_ra,
                  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  void executeDdl(ExecutionResult& _return,
                  const std::string& query_ra,
                  std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  TColumnType populateThriftColumnType(const Catalog_Namespace::Catalog* cat,
                                       const ColumnDescriptor* cd);

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

  static void convertExplain(TQueryResult& _return,
                             const ResultSet& results,
                             const bool column_format);
  static void convertResult(TQueryResult& _return,
                            const ResultSet& results,
                            const bool column_format);

  static void convertRows(TQueryResult& _return,
                          QueryStateProxy query_state_proxy,
                          const std::vector<TargetMetaInfo>& targets,
                          const ResultSet& results,
                          const bool column_format,
                          const int32_t first_n,
                          const int32_t at_most_n);

  // Use ExecutionResult to populate a TQueryResult
  //    calls convertRows, but after some setup using session_info
  void convertResultSet(ExecutionResult& result,
                        const Catalog_Namespace::SessionInfo& session_info,
                        const std::string& query_state_str,
                        TQueryResult& _return);

  static void createSimpleResult(TQueryResult& _return,
                                 const ResultSet& results,
                                 const bool column_format,
                                 const std::string label);

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

  TDashboard get_dashboard_impl(
      const std::shared_ptr<Catalog_Namespace::SessionInfo const>& session_ptr,
      Catalog_Namespace::UserMetadata& user_meta,
      const DashboardDescriptor* dash,
      const bool populate_state = true);

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

  std::unique_ptr<lockmgr::AbstractLockContainer<const TableDescriptor*>>
  prepare_loader_generic(
      const Catalog_Namespace::SessionInfo& session_info,
      const std::string& table_name,
      size_t num_cols,
      std::unique_ptr<import_export::Loader>* loader,
      std::vector<std::unique_ptr<import_export::TypedImportBuffer>>* import_buffers,
      const std::vector<std::string>& column_names,
      std::string load_type);

  void fillGeoColumns(
      const TSessionId& session,
      const Catalog_Namespace::Catalog& catalog,
      std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers,
      const ColumnDescriptor* cd,
      size_t& col_idx,
      size_t num_rows,
      const std::string& table_name,
      bool assign_render_groups);

  void fillMissingBuffers(
      const TSessionId& session,
      const Catalog_Namespace::Catalog& catalog,
      std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers,
      const std::list<const ColumnDescriptor*>& cds,
      const std::vector<int>& desc_id_to_column_id,
      size_t num_rows,
      const std::string& table_name,
      bool assign_render_groups);

  query_state::QueryStates query_states_;
  std::unordered_map<std::string, Catalog_Namespace::SessionInfoPtr> calcite_sessions_;
  mutable heavyai::shared_mutex calcite_sessions_mtx_;

  Catalog_Namespace::SessionInfoPtr findCalciteSession(TSessionId const&) const;

  bool super_user_rights_;           // default is "false"; setting to "true"
                                     // ignores passwd checks in "connect(..)"
                                     // method
  const int idle_session_duration_;  // max duration of idle session
  const int max_session_duration_;   // max duration of session

  const bool enable_rendering_;
  const bool renderer_use_ppll_polys_;
  const bool renderer_prefer_igpu_;
  const unsigned renderer_vulkan_timeout_;
  const bool renderer_use_parallel_executors_;
  const bool enable_auto_clear_render_mem_;
  const int render_oom_retry_threshold_;
  const size_t max_concurrent_render_sessions_;
  const size_t reserved_gpu_mem_;
  const bool render_compositor_use_last_gpu_;
  const size_t render_mem_bytes_;
  const size_t num_reader_threads_;
#ifdef ENABLE_GEOS
  const std::string& libgeos_so_filename_;
#endif
  const File_Namespace::DiskCacheConfig& disk_cache_config_;
  const std::string& udf_filename_;
  const std::string& clang_path_;
  const std::vector<std::string>& clang_options_;
  int32_t max_num_sessions_{-1};
  std::unique_ptr<Catalog_Namespace::SessionsStore> sessions_store_;

  struct DeferredCopyFromState {
    std::string table;
    std::string file_name;
    import_export::CopyParams copy_params;
    std::string partitions;
  };

  struct DeferredCopyFromSessions {
    std::unordered_map<std::string, DeferredCopyFromState> was_deferred_copy_from;
    std::mutex deferred_copy_from_mutex;

    std::optional<DeferredCopyFromState> operator()(const std::string& session_id) {
      std::lock_guard<std::mutex> map_lock(deferred_copy_from_mutex);
      auto itr = was_deferred_copy_from.find(session_id);
      if (itr == was_deferred_copy_from.end()) {
        return std::nullopt;
      }
      return itr->second;
    }

    void add(const std::string& session_id, const DeferredCopyFromState& state) {
      std::lock_guard<std::mutex> map_lock(deferred_copy_from_mutex);
      const auto ret = was_deferred_copy_from.insert(std::make_pair(session_id, state));
      CHECK(ret.second);
    }

    void remove(const std::string& session_id) {
      std::lock_guard<std::mutex> map_lock(deferred_copy_from_mutex);
      was_deferred_copy_from.erase(session_id);
    }
  };
  DeferredCopyFromSessions deferred_copy_from_sessions;

  // Only for IPC device memory deallocation
  mutable std::mutex handle_to_dev_ptr_mutex_;
  mutable std::unordered_map<std::string, std::string> ipc_handle_to_dev_ptr_;

  friend void run_warmup_queries(std::shared_ptr<DBHandler> handler,
                                 std::string base_path,
                                 std::string query_file_path);

  friend class RenderHandler::Impl;
  friend class HeavyDBAggHandler;
  friend class HeavyDBLeafHandler;

  std::map<const std::string, const PermissionFuncPtr> permissionFuncMap_ = {
      {"database"s, has_database_permission},
      {"dashboard"s, has_dashboard_permission},
      {"table"s, has_table_permission},
      {"view"s, has_view_permission},
      {"server"s, has_server_permission}};

  void check_and_invalidate_sessions(Parser::DDLStmt* ddl);

  std::string const createInMemoryCalciteSession(
      const std::shared_ptr<Catalog_Namespace::Catalog>& catalog_ptr);
  void removeInMemoryCalciteSession(const std::string& session_id);

  ExecutionResult getUserSessions(
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  // getQueries returns a set of queries queued in the DB
  //    that belongs to the same DB in the caller's session

  ExecutionResult getQueries(
      std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr);

  void get_queries_info(std::vector<TQueryInfo>& _return,
                        const TSessionId& session) override;

  // this function passes the interrupt request to the DB executor
  void interruptQuery(const Catalog_Namespace::SessionInfo& session_info,
                      const std::string& target_session);

  void alterSystemClear(const std::string& sesson_id,
                        ExecutionResult& result,
                        const std::string& cache_type,
                        int64_t& execution_time_ms);

  void alterSession(const std::string& sesson_id,
                    ExecutionResult& result,
                    const std::pair<std::string, std::string>& session_parameter,
                    int64_t& execution_time_ms);

  // render group assignment

  enum class AssignRenderGroupsMode { kNone, kAssign, kCleanUp };

  void loadTableBinaryColumnarInternal(
      const TSessionId& session,
      const std::string& table_name,
      const std::vector<TColumn>& cols,
      const std::vector<std::string>& column_names,
      const AssignRenderGroupsMode assign_render_groups_mode);

  TRole::type getServerRole() const;

  using RenderGroupAssignmentColumnMap =
      std::unordered_map<std::string,
                         std::unique_ptr<import_export::RenderGroupAnalyzer>>;
  using RenderGroupAssignmentTableMap =
      std::unordered_map<std::string, RenderGroupAssignmentColumnMap>;
  using RenderGroupAnalyzerSessionMap =
      std::unordered_map<TSessionId, RenderGroupAssignmentTableMap>;
  RenderGroupAnalyzerSessionMap render_group_assignment_map_;
  std::mutex render_group_assignment_mutex_;
  heavyai::shared_mutex custom_expressions_mutex_;

  void importGeoTableGlobFilterSort(const TSessionId& session,
                                    const std::string& table_name,
                                    const std::string& file_name,
                                    const import_export::CopyParams& copy_params,
                                    const TRowDescriptor& row_desc,
                                    const TCreateParams& create_params);

  void importGeoTableSingle(const TSessionId& session,
                            const std::string& table_name,
                            const std::string& file_name,
                            const import_export::CopyParams& copy_params,
                            const TRowDescriptor& row_desc,
                            const TCreateParams& create_params);

  void resetSessionsStore();
};
