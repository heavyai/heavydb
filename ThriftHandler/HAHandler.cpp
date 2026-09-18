/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "HAHandler.h"
#include "Shared/ThriftClient.h"
#include "ThriftHandler/RequestInfo.h"

#include <thrift/protocol/TBinaryProtocol.h>

#define THROW_DB_EXCEPTION(errstr) \
  {                                \
    TDBException ex;               \
    ex.error_msg = errstr;         \
    LOG(ERROR) << ex.error_msg;    \
    throw ex;                      \
  }

HAHandler::HAHandler(const std::string& base_data_path,
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
                     const bool is_new_db)
    : DBHandler(base_data_path,
                allow_multifrag,
                jit_debug,
                intel_jit_profile,
                read_only,
                allow_loop_joins,
                enable_rendering,
                renderer_prefer_igpu,
                renderer_vulkan_timeout_ms,
                renderer_use_parallel_executors,
                enable_auto_clear_render_mem,
                render_oom_retry_threshold,
                render_mem_bytes,
                max_concurrent_render_sessions,
                reserved_gpu_mem,
                render_compositor_use_last_gpu,
                renderer_enable_slab_allocation,
                num_reader_threads,
                authMetadata,
                system_parameters,
                legacy_syntax,
                idle_session_duration,
                max_session_duration,
                udf_filename,
                clang_path,
                clang_options,
#ifdef ENABLE_GEOS
                libgeos_so_filename,
#endif
#ifdef HAVE_TORCH_TFS
                torch_lib_path,
#endif
                disk_cache_config,
                is_new_db) {
  LOG(INFO) << " HeavyDB server running as HAHandler";
  connMgr_ = std::make_shared<ThriftClientConnection>();
  master_host_ = std::make_shared<RemoteHostInfo>(system_parameters.master_address,
                                                  system_parameters.master_port,
                                                  true,    // keep_alive
                                                  20000,   // connect
                                                  300000,  // recv
                                                  300000,  // send
                                                  system_parameters.ssl_trust_ca_file);
}

std::unique_ptr<HeavyClient> HAHandler::getClient() {
  for (auto i = 1; i <= MAX_RETRY; i++) {
    try {
      const auto transport =
          connMgr_->open_buffered_client_transport(master_host_->getHost(),
                                                   master_host_->getPort(),
                                                   master_host_->getCACertFile(),
                                                   // with_timeout_,
                                                   false,
                                                   master_host_->getWithKeepAlive(),
                                                   master_host_->getConnectTimeout(),
                                                   master_host_->getRecvTimeout(),
                                                   master_host_->getSendTimeout());

      try {
        transport->open();
      } catch (const apache::thrift::TException& e) {
        throw apache::thrift::TException(std::string(e.what()) + ": host " +
                                         master_host_->getHost() + ", port " +
                                         std::to_string(master_host_->getPort()));
      }
      const auto protocol = std::make_shared<TBinaryProtocol>(transport);
      return std::make_unique<HeavyClient>(protocol);
    } catch (const std::exception& e) {
      // If we get an error trying to establish a connection for a client we can assume
      // something has gone wrong with master.
      // we will clear the session connection map and force them to reconnect
      {
        heavyai::unique_lock<heavyai::shared_mutex> write_lock(master_session_map_mutex_);
        master_session_map_.clear();
      }
      if (i != (MAX_RETRY)) {
        LOG(WARNING) << "Could not create client connection to master.  Issue was "
                     << e.what() << " will retry " << std::to_string(MAX_RETRY - i)
                     << " more times";
        std::this_thread::sleep_for(std::chrono::seconds(RETRY_SECONDS));
        continue;
      }
      throw;
    }
  }
  CHECK(true);
  return nullptr;  // not possible to get here
}

std::shared_ptr<HAHandler::Credentials> HAHandler::getSessionCredentials(
    const TSessionId& session) {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(master_session_map_mutex_);
  auto session_it = master_session_credentials_map_.find(session);
  if (session_it != master_session_credentials_map_
                        .end()) {  // should be impossible not to have credentials
    return session_it->second;
  } else {
    THROW_DB_EXCEPTION(
        std::string("Could not find credentials for session, should not be possible, is "
                    "load balancing being used without sticky sessions"));
  }
}

void HAHandler::connectToMaster(const TSessionId& session,
                                const std::string& username,
                                const std::string& passwd,
                                const std::string& dbname) {
  TSessionId session1;
  try {
    auto client = getClient();
    client->connect(session1, username, passwd, dbname);
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(std::string("Could not Connect to master node.  Issue was ") +
                       e.what());
  }
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(master_session_map_mutex_);
  master_session_map_[session] = session1;
}

// we will call this any time we have an issue on a connection to master
// it will catch the case where the master has restarted and we no longer
// have valid session, but we dont want to do it to all sessions, as some already
// may be connected to new server
void HAHandler::resetMasterSession(const TSessionId& session) {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(master_session_map_mutex_);
  auto it = master_session_map_.find(session);
  if (it != master_session_map_.end()) {
    try {
      getClient()->disconnect(it->second);
    } catch (...) {
      // we assume ther eis nothing to disconnect but just in case
    };
    master_session_map_.erase(it);
  }
}

const TSessionId& HAHandler::getMasterSession(const TSessionId& session) {
  {
    heavyai::shared_lock<heavyai::shared_mutex> read_lock(master_session_map_mutex_);
    auto it = master_session_map_.find(session);
    if (it != master_session_map_.end()) {
      return it->second;
    }
  }
  // we have no session to the master yet in the map, lets try a connect now
  // this condition is acceptable as we let the RO connection happen and some work
  // to proceed before we have to have a connection
  auto creds = getSessionCredentials(session);
  connectToMaster(session, creds->user, creds->passwd, creds->dbname);
  // if we get to here we didnt get exception so check the map for session
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(master_session_map_mutex_);
  auto it = master_session_map_.find(session);
  if (it != master_session_map_.end()) {
    return it->second;
  } else {
    THROW_DB_EXCEPTION(
        std::string("Could not connect to master node. No session in map"));
  }
}

void HAHandler::connect(TSessionId& session,
                        const std::string& username,
                        const std::string& passwd,
                        const std::string& dbname) {
  DBHandler::connect(session, username, passwd, dbname);
  // now capture the session details and also connect to the master node
  {
    heavyai::unique_lock<heavyai::shared_mutex> write_lock(master_session_map_mutex_);
    master_session_credentials_map_[session] =
        std::make_shared<Credentials>(username, passwd, dbname);
  }

  // try to connect to master, if it fails warn but continue and we will connect later
  try {
    connectToMaster(session, username, passwd, dbname);
  } catch (const TDBException& e) {
    LOG(WARNING) << e.error_msg;
  } catch (const std::exception& e) {
    LOG(WARNING) << e.what();
  }
}

void HAHandler::disconnect(const TSessionId& session) {
  // make sure we remove the entries
  // and disconnect from master regardless of issue locally
  ScopeGuard session_guard = [&] {
    heavyai::unique_lock<heavyai::shared_mutex> write_lock(master_session_map_mutex_);
    auto credential_it = master_session_credentials_map_.find(session);
    if (credential_it != master_session_credentials_map_.end()) {
      master_session_credentials_map_.erase(credential_it);
    }

    auto connect_it = master_session_map_.find(session);
    if (connect_it != master_session_map_.end()) {
      // we may not have been able to establish a connection to the master
      auto master_session = connect_it->second;
      master_session_map_.erase(connect_it);
      try {
        getClient()->disconnect(master_session);
      } catch (const std::exception& e) {
        LOG(ERROR) << "Could not disconnect from master node.  Issue was " << e.what();
      }
    }
  };

  DBHandler::disconnect(session);
}

void HAHandler::clone_session(TSessionId& session2_id,
                              const TSessionId& session1_id_or_json) {
  DBHandler::clone_session(session2_id, session1_id_or_json);

  heavyai::RequestInfo const request_info(session1_id_or_json);
  try {
    TSessionId session2_id_master;
    getClient()->clone_session(session2_id_master,
                               getMasterSession(request_info.sessionId()));

    heavyai::unique_lock<heavyai::shared_mutex> write_lock(master_session_map_mutex_);
    master_session_map_[session2_id] = session2_id_master;

  } catch (const std::exception& e) {
    resetMasterSession(request_info.sessionId());
    THROW_DB_EXCEPTION(
        std::string("Could not clone_session from master node.  Issue was ") + e.what());
  }
}

void HAHandler::switch_database(const TSessionId& session_id_or_json,
                                const std::string& dbname) {
  DBHandler::switch_database(session_id_or_json, dbname);

  heavyai::RequestInfo const request_info(session_id_or_json);
  try {
    getClient()->switch_database(getMasterSession(request_info.sessionId()), dbname);
  } catch (const std::exception& e) {
    resetMasterSession(request_info.sessionId());
    THROW_DB_EXCEPTION(
        std::string("Could not switch_database from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::sql_execute(TQueryResult& _return,
                            const TSessionId& session,
                            const std::string& query_str,
                            const bool column_format,
                            const std::string& nonce,
                            const int32_t first_n,
                            const int32_t at_most_n) {
  LOG(INFO) << " HAHandler sql_execute";

  DBHandler::sql_execute(
      _return, session, query_str, column_format, nonce, first_n, at_most_n);

  // this is the session keep alive ping to master
  try {
    TServerStatus junk_return;
    getClient()->get_server_status(junk_return, getMasterSession(session));
  } catch (const std::exception& e) {
    LOG(WARNING) << "Warning: Ping call could not get_server_status from master node.  "
                    "Issue was "
                 << e.what();
    resetMasterSession(session);
  }
}

void HAHandler::get_server_status(TServerStatus& _return, const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session, true));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->get_server_status(_return, getMasterSession(session));
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not get_server_status from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::get_status(std::vector<TServerStatus>& _return,
                           const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session, true));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->get_status(_return, getMasterSession(session));
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(std::string("Could not get_status from master node.  Issue was ") +
                       e.what());
  }
}

void HAHandler::get_dashboard(TDashboard& _return,
                              const TSessionId& session,
                              const int32_t dashboard_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->get_dashboard(_return, getMasterSession(session), dashboard_id);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not get_dashboard from master node.  Issue was ") + e.what());
  }
}

void HAHandler::get_dashboards(std::vector<TDashboard>& _return,
                               const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->get_dashboards(_return, getMasterSession(session));
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not get_dashboards from master node.  Issue was ") + e.what());
  }
}

int32_t HAHandler::create_dashboard(const TSessionId& session,
                                    const std::string& dashboard_name,
                                    const std::string& dashboard_state,
                                    const std::string& image_hash,
                                    const std::string& dashboard_metadata) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    return getClient()->create_dashboard(getMasterSession(session),
                                         dashboard_name,
                                         dashboard_state,
                                         image_hash,
                                         dashboard_metadata);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not create_dashboard from master node.  Issue was ") +
        e.what());
  }
}
void HAHandler::replace_dashboard(const TSessionId& session,
                                  const int32_t dashboard_id,
                                  const std::string& dashboard_name,
                                  const std::string& dashboard_owner,
                                  const std::string& dashboard_state,
                                  const std::string& image_hash,
                                  const std::string& dashboard_metadata) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->replace_dashboard(getMasterSession(session),
                                   dashboard_id,
                                   dashboard_name,
                                   dashboard_owner,
                                   dashboard_state,
                                   image_hash,
                                   dashboard_metadata);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not replace_dashboard from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::delete_dashboard(const TSessionId& session, const int32_t dashboard_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->delete_dashboard(getMasterSession(session), dashboard_id);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not delete_dashboard from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::share_dashboards(const TSessionId& session,
                                 const std::vector<int32_t>& dashboard_ids,
                                 const std::vector<std::string>& groups,
                                 const TDashboardPermissions& permissions) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->share_dashboards(
        getMasterSession(session), dashboard_ids, groups, permissions);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not share_dashboards from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::delete_dashboards(const TSessionId& session,
                                  const std::vector<int32_t>& dashboard_ids) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->delete_dashboards(getMasterSession(session), dashboard_ids);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not delete_dashboards from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::share_dashboard(const TSessionId& session,
                                const int32_t dashboard_id,
                                const std::vector<std::string>& groups,
                                const std::vector<std::string>& objects,
                                const TDashboardPermissions& permissions,
                                const bool grant_role) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->share_dashboard(getMasterSession(session),
                                 dashboard_id,
                                 groups,
                                 objects,
                                 permissions,
                                 grant_role);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not share_dashboard from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::unshare_dashboards(const TSessionId& session,
                                   const std::vector<int32_t>& dashboard_ids,
                                   const std::vector<std::string>& groups,
                                   const TDashboardPermissions& permissions) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->unshare_dashboards(
        getMasterSession(session), dashboard_ids, groups, permissions);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not unshare_dashboards from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::unshare_dashboard(const TSessionId& session,
                                  const int32_t dashboard_id,
                                  const std::vector<std::string>& groups,
                                  const std::vector<std::string>& objects,
                                  const TDashboardPermissions& permissions) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->unshare_dashboard(
        getMasterSession(session), dashboard_id, groups, objects, permissions);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not unshare_dashboard from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::get_dashboard_grantees(std::vector<TDashboardGrantees>& _return,
                                       const TSessionId& session,
                                       const int32_t dashboard_id) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->get_dashboard_grantees(_return, getMasterSession(session), dashboard_id);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not get_dashboard_grantees from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::get_db_object_privs(std::vector<TDBObject>& TDBObjects,
                                    const TSessionId& sessionId,
                                    const std::string& objectName,
                                    const TDBObjectType::type type) {
  auto stdlog = STDLOG(get_session_ptr(sessionId));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->get_db_object_privs(
        TDBObjects, getMasterSession(sessionId), objectName, type);
  } catch (const std::exception& e) {
    resetMasterSession(sessionId);
    THROW_DB_EXCEPTION(
        std::string("Could not get_db_object_privs from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::get_queries_info(std::vector<TQueryInfo>& _return,
                                 const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->get_queries_info(_return, getMasterSession(session));
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not get_queries_info from master node.  Issue was ") +
        e.what());
  }
}

int32_t HAHandler::create_custom_expression(const TSessionId& session,
                                            const TCustomExpression& custom_expression) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    return getClient()->create_custom_expression(getMasterSession(session),
                                                 custom_expression);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not create_custom_expression from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::get_custom_expressions(std::vector<TCustomExpression>& _return,
                                       const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->get_custom_expressions(_return, getMasterSession(session));
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not get_custom_expressions from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::update_custom_expression(const TSessionId& session,
                                         const int32_t id,
                                         const std::string& expression_json) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->update_custom_expression(getMasterSession(session), id, expression_json);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not update_custom_expression from master node.  Issue was ") +
        e.what());
  }
}

void HAHandler::delete_custom_expressions(
    const TSessionId& session,
    const std::vector<int32_t>& custom_expression_ids,
    const bool do_soft_delete) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  try {
    getClient()->delete_custom_expressions(
        getMasterSession(session), custom_expression_ids, do_soft_delete);
  } catch (const std::exception& e) {
    resetMasterSession(session);
    THROW_DB_EXCEPTION(
        std::string("Could not delete_custom_expressions from master node.  Issue was ") +
        e.what());
  }
}
