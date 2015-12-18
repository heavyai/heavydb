#ifndef QUERY_RUNNER_H
#define QUERY_RUNNER_H

#include "../Catalog/Catalog.h"
#include "../QueryEngine/Execute.h"

#include <memory>
#include <string>

Catalog_Namespace::SessionInfo* get_session(const char* db_path);

ResultRows run_multiple_agg(const std::string& query_str,
                            const bool use_calcite,
                            const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
                            const ExecutorDeviceType device_type,
                            const NVVMBackend nvvm_backend);

#endif  // QUERY_RUNNER_H
