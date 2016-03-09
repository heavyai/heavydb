#ifndef QUERYENGINE_DATAFETCHER_H
#define QUERYENGINE_DATAFETCHER_H

#include "ScanDescriptors.h"

namespace Catalog_Namespace {
class Catalog;
}  // Catalog_Namespace

std::vector<Fragmenter_Namespace::QueryInfo> get_query_infos(const std::vector<ScanDescriptor>& scan_ids,
                                                             const Catalog_Namespace::Catalog& cat);

std::deque<Fragmenter_Namespace::FragmentInfo> get_fragments_meta(const ResultRows*);

#endif  // QUERYENGINE_DATAFETCHER_H
