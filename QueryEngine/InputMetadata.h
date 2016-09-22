#ifndef QUERYENGINE_INPUTMETADATA_H
#define QUERYENGINE_INPUTMETADATA_H

#include "InputDescriptors.h"
#include "IteratorTable.h"

#include <unordered_map>

namespace Catalog_Namespace {
class Catalog;
}  // Catalog_Namespace

typedef std::unordered_map<int, const ResultPtr&> TemporaryTables;

struct InputTableInfo {
  int table_id;
  Fragmenter_Namespace::TableInfo info;
};

size_t get_frag_count_of_table(const int table_id,
                               const Catalog_Namespace::Catalog& cat,
                               const TemporaryTables& temporary_tables);

std::vector<InputTableInfo> get_table_infos(const std::vector<InputDescriptor>& input_descs,
                                            const Catalog_Namespace::Catalog& cat,
                                            const TemporaryTables& temporary_tables);

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit,
                                            const Catalog_Namespace::Catalog& cat,
                                            const TemporaryTables& temporary_tables);

#endif  // QUERYENGINE_INPUTMETADATA_H
