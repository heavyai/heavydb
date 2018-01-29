#ifndef THRIFTWITHRETRY_H
#define THRIFTWITHRETRY_H

#include <iostream>
#include <cmath>
#include "gen-cpp/mapd_types.h"

template <typename SERVICE_ENUM, typename CLIENT_CONTEXT>
bool thrift_with_retry(SERVICE_ENUM which_service, CLIENT_CONTEXT& context, char const* arg, const int try_count = 1) {
  using TException = ::apache::thrift::TException;

  int max_reconnect = 4;
  int con_timeout_base = 1;
  if (try_count > max_reconnect) {
    return false;
  }
  try {
    switch (which_service) {
      case kCONNECT:
        context.client.connect(context.session, context.user_name, context.passwd, context.db_name);
        break;
      case kDISCONNECT:
        context.client.disconnect(context.session);
        break;
      case kINTERRUPT:
        context.client.interrupt(context.session);
        break;
      case kSQL:
        context.client.sql_execute(context.query_return, context.session, arg, true, "", -1, -1);
        break;
      case kGET_TABLES:
        context.client.get_tables(context.names_return, context.session);
        break;
      case kGET_PHYSICAL_TABLES:
        context.client.get_physical_tables(context.names_return, context.session);
        break;
      case kGET_VIEWS:
        context.client.get_views(context.names_return, context.session);
        break;
      case kGET_DATABASES:
        context.client.get_databases(context.dbinfos_return, context.session);
        break;
      case kGET_USERS:
        context.client.get_users(context.names_return, context.session);
        break;
      case kSET_EXECUTION_MODE:
        context.client.set_execution_mode(context.session, context.execution_mode);
        break;
      case kGET_VERSION:
        context.client.get_version(context.version);
        break;
      case kGET_MEMORY_GPU:
        context.client.get_memory(context.gpu_memory, context.session, "gpu");
        break;
      case kGET_MEMORY_CPU:
        context.client.get_memory(context.cpu_memory, context.session, "cpu");
        break;
      case kGET_MEMORY_SUMMARY:
        context.client.get_memory(context.gpu_memory, context.session, "gpu");
        context.client.get_memory(context.cpu_memory, context.session, "cpu");
        break;
      case kGET_TABLE_DETAILS:
        context.client.get_table_details(context.table_details, context.session, arg);
        break;
      case kCLEAR_MEMORY_GPU:
        context.client.clear_gpu_memory(context.session);
        break;
      case kCLEAR_MEMORY_CPU:
        context.client.clear_cpu_memory(context.session);
        break;
      case kGET_HARDWARE_INFO:
        context.client.get_hardware_info(context.cluster_hardware_info, context.session);
        break;
      case kIMPORT_GEO_TABLE:
        context.client.import_geo_table(
            context.session, context.table_name, context.file_name, context.copy_params, TRowDescriptor());
        break;
      case kSET_TABLE_EPOCH:
        context.client.set_table_epoch(context.session, context.db_id, context.table_id, context.epoch_value);
        break;
      case kSET_TABLE_EPOCH_BY_NAME:
        context.client.set_table_epoch_by_name(context.session, context.table_name, context.epoch_value);
        break;
      case kGET_TABLE_EPOCH:
        context.epoch_value = context.client.get_table_epoch(context.session, context.db_id, context.table_id);
        break;
      case kGET_TABLE_EPOCH_BY_NAME:
        context.epoch_value = context.client.get_table_epoch_by_name(context.session, context.table_name);
        break;
      case kGET_SERVER_STATUS:
        context.client.get_status(context.cluster_status, context.session);
        break;
      case kIMPORT_DASHBOARD:
        context.client.create_frontend_view(
            context.session, context.view_name, context.view_state, "", context.view_metadata);
        break;
      case kEXPORT_DASHBOARD:
        context.client.get_frontend_view(context.view_return, context.session, context.view_name);
        break;
      case kGET_ROLE:
        context.client.get_role(context.role_names, context.session, context.privs_role_name, context.userPrivateRole);
        break;
      case kGET_ALL_ROLES:
        context.client.get_all_roles(context.role_names, context.session, context.userPrivateRole);
        break;
      case kGET_OBJECTS_FOR_ROLE:
        context.client.get_db_objects_for_role(context.db_objects, context.session, context.privs_role_name);
        break;
      case kGET_OBJECT_PRIVS:
        context.client.get_db_object_privs(context.db_objects, context.session, context.privs_object_name);
        break;
      case kGET_ROLES_FOR_USER:
        context.client.get_all_roles_for_user(context.role_names, context.session, context.privs_user_name);
        break;
      case kSET_LICENSE_KEY:
        context.client.set_license_key(context.license_info, context.session, context.license_key, "");
        break;
      case kGET_LICENSE_CLAIMS:
        context.client.get_license_claims(context.license_info, context.session, "");
        break;
    }
  } catch (TMapDException& e) {
    std::cerr << e.error_msg << std::endl;
    return false;
  } catch (TException& te) {
    try {
      context.transport.open();
      if (which_service == kDISCONNECT)
        return false;
      sleep(con_timeout_base * pow(2, try_count));
      if (which_service != kCONNECT) {
        if (!thrift_with_retry(kCONNECT, context, nullptr, try_count + 1))
          return false;
      }
      return thrift_with_retry(which_service, context, arg, try_count + 1);
    } catch (TException& te1) {
      std::cerr << "Thrift error: " << te1.what() << std::endl;
      return false;
    }
  }
  return true;
}

#endif
