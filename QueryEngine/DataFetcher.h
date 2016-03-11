#ifndef QUERYENGINE_DATAFETCHER_H
#define QUERYENGINE_DATAFETCHER_H

#include "ScanDescriptors.h"

#include <unordered_map>

namespace Catalog_Namespace {
class Catalog;
}  // Catalog_Namespace

class ResultRows;

typedef std::unordered_map<int, const ResultRows*> TemporaryTables;

std::vector<Fragmenter_Namespace::TableInfo> get_table_infos(const std::vector<ScanDescriptor>& scan_descs,
                                                             const Catalog_Namespace::Catalog& cat,
                                                             const TemporaryTables& temporary_tables) noexcept;

#endif  // QUERYENGINE_DATAFETCHER_H
