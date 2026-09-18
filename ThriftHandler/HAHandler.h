/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HAHANDLER_H
#define HAHANDLER_H

#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.

#include "Shared/RemoteHostInfo.h"
#include "Shared/SystemParameters.h"
#include "ThriftHandler/DBHandler.h"
#include "gen-cpp/Heavy.h"

class HAHandler : public DBHandler {
 public:
  HAHandler(const std::string& base_data_path,
            const bool allow_multifrag,
            const bool jit_debug,
            const bool intel_jit_profile,
            const bool read_only,
            const bool allow_loop_joins,
            const bool enable_rendering,
            const bool renderer_prefer_igpu,
            const unsigned renderer_vulkan_timeout_ms,
            const bool renderer_use_parallel_executors,
            const bool enable_auto_clear_render_mem,
            const int render_oom_retry_threshold,
            const size_t render_mem_bytes,
            const size_t max_concurrent_render_sessions,
            const size_t reserved_gpu_mem,
            const bool render_compositor_use_last_gpu,
            const bool renderer_enable_slab_allocation,
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
#ifdef HAVE_TORCH_TFS
            const std::string& torch_lib_path,
#endif
            const File_Namespace::DiskCacheConfig& disk_cache_config,
            const bool is_new_db);

  void sql_execute(TQueryResult& _return,
                   const TSessionId& session,
                   const std::string& query_str,
                   const bool column_format,
                   const std::string& nonce,
                   const int32_t first_n,
                   const int32_t at_most_n) override;

  void connect(TSessionId& session,
               const std::string& username,
               const std::string& passwd,
               const std::string& dbname) override;

  void disconnect(const TSessionId& session) override;

  void switch_database(const TSessionId& session_id_or_json,
                       const std::string& dbname) override;

  void clone_session(TSessionId& session2_id,
                     const TSessionId& session1_id_or_json) override;

  void get_server_status(TServerStatus& _return, const TSessionId& session) override;

  void get_status(std::vector<TServerStatus>& _return,
                  const TSessionId& session) override;

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

  void get_db_object_privs(std::vector<TDBObject>& TDBObjects,
                           const TSessionId& sessionId,
                           const std::string& objectName,
                           const TDBObjectType::type type) override;

  void get_queries_info(std::vector<TQueryInfo>& _return,
                        const TSessionId& session) override;

 private:
  static constexpr int const& MAX_RETRY = 3;
  static constexpr int const& RETRY_SECONDS = 5;

  // TODO MAT unify this structure into shared code.
  struct Credentials {
    Credentials(const std::string user,
                const std::string passwd,
                const std::string dbname)
        : user(user), passwd(passwd), dbname(dbname) {}
    const std::string user;
    const std::string passwd;
    const std::string dbname;
  };

  std::unique_ptr<HeavyClient> getClient();
  const TSessionId& getMasterSession(const TSessionId& session);
  std::shared_ptr<HAHandler::Credentials> getSessionCredentials(
      const TSessionId& session);
  void connectToMaster(const TSessionId& session,
                       const std::string& username,
                       const std::string& passwd,
                       const std::string& dbname);
  void resetMasterSession(const TSessionId& session);

  std::shared_ptr<ThriftClientConnection> connMgr_;
  std::shared_ptr<RemoteHostInfo> master_host_;
  std::unordered_map<TSessionId, TSessionId> master_session_map_;
  heavyai::shared_mutex master_session_map_mutex_;
  std::unordered_map<TSessionId, std::shared_ptr<Credentials>>
      master_session_credentials_map_;
};

#endif /* HAHANDLER_H */
