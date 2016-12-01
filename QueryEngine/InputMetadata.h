#ifndef QUERYENGINE_INPUTMETADATA_H
#define QUERYENGINE_INPUTMETADATA_H

#include "InputDescriptors.h"
#include "IteratorTable.h"

#include <unordered_map>

namespace Catalog_Namespace {
class Catalog;
}  // Catalog_Namespace

class Executor;

typedef std::unordered_map<int, const ResultPtr&> TemporaryTables;

struct InputTableInfo {
  int table_id;
  Fragmenter_Namespace::TableInfo info;
};

class InputTableInfoCache {
 public:
  InputTableInfoCache(Executor* executor);

  Fragmenter_Namespace::TableInfo getTableInfo(const int table_id);

  void clear();

 private:
  std::unordered_map<int, Fragmenter_Namespace::TableInfo> cache_;
  Executor* executor_;
};

size_t get_frag_count_of_table(const int table_id, Executor* executor);

std::vector<InputTableInfo> get_table_infos(const std::vector<InputDescriptor>& input_descs, Executor* executor);

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit, Executor* executor);

#endif  // QUERYENGINE_INPUTMETADATA_H
