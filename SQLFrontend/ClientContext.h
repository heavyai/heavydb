#ifndef CLIENTCONTEXT_H
#define CLIENTCONTEXT_H

#include "gen-cpp/mapd_types.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

static std::string const INVALID_SESSION_ID("");
static std::string const MAPD_ROOT_USER("mapd");
static std::string const MAPD_DEFAULT_ROOT_USER_ROLE("mapd_default_suser_role");

struct ClientContext {
  std::string user_name;
  std::string passwd;
  std::string db_name;
  std::string server_host;
  int port;
  bool http;
  TTransport& transport;
  MapDClient& client;
  TSessionId session;
  TQueryResult query_return;
  std::vector<std::string> names_return;
  std::vector<TDBInfo> dbinfos_return;
  TExecuteMode::type execution_mode;
  std::string version;
  std::vector<TNodeMemoryInfo> gpu_memory;
  std::vector<TNodeMemoryInfo> cpu_memory;
  TTableDetails table_details;
  std::string table_name;
  std::string file_name;
  TCopyParams copy_params;
  int db_id;
  int table_id;
  int epoch_value;
  TServerStatus server_status;
  TClusterHardwareInfo cluster_hardware_info;
  std::vector<TServerStatus> cluster_status;
  std::string view_name;
  std::string view_state;
  std::string view_metadata;
  TFrontendView view_return;
  std::string privs_role_name;
  std::string privs_user_name;
  std::string privs_object_name;
  bool userPrivateRole;
  std::vector<std::string> role_names;
  std::vector<TAccessPrivileges> object_privileges;
  std::vector<TDBObject> db_objects;
  std::string license_key;
  TLicenseInfo license_info;

  ClientContext(TTransport& t, MapDClient& c)
      : transport(t), client(c), session(INVALID_SESSION_ID), execution_mode(TExecuteMode::GPU) {}
};

#endif
